"""
Training modular, pencatatan metrik per epoch, evaluasi ROUGE, dan pembuatan
tabel/grafik perbandingan konfigurasi eksperimen.
"""

from __future__ import annotations

import gc
import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from config import DataConfig, ExperimentConfig
from data import PreparedData, describe_data, make_tf_dataset, prepare_data
from inference import generate_summaries
from model import build_model, compile_model
from preprocessing import Vocabulary
from rouge import corpus_rouge


class EpochTimer(keras.callbacks.Callback):
    """Mencatat execution time tiap epoch (termasuk validasi)."""

    def __init__(self) -> None:
        super().__init__()
        self.epoch_times: List[float] = []
        self._start: Optional[float] = None

    def on_train_begin(self, logs=None) -> None:
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None) -> None:
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None) -> None:
        if self._start is not None:
            self.epoch_times.append(time.perf_counter() - self._start)


def _to_python(value):
    """Konversi numpy scalar/array agar aman disimpan ke JSON."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _json_dump(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_to_python)


def _model_config_payload(exp_cfg: ExperimentConfig, prepared: PreparedData) -> dict:
    return {
        "experiment": exp_cfg.to_dict(),
        "data": {
            "max_article_len": prepared.config.max_article_len,
            "max_summary_len": prepared.config.max_summary_len,
            "max_dec_len": prepared.max_dec_len,
            "max_vocab_size": prepared.config.max_vocab_size,
            "min_freq": prepared.config.min_freq,
            "reverse_source": prepared.config.reverse_source,
        },
        "vocab_size": len(prepared.vocab),
        "stats": prepared.stats,
    }


def _history_dataframe(history: keras.callbacks.History, epoch_times: Sequence[float]) -> pd.DataFrame:
    h = history.history
    n = len(h.get("loss", []))
    data: Dict[str, List[float]] = {"epoch": list(range(1, n + 1))}
    mapping = {
        "train_loss": "loss",
        "val_loss": "val_loss",
        "train_accuracy": "masked_token_accuracy",
        "val_accuracy": "val_masked_token_accuracy",
    }
    for out_key, hist_key in mapping.items():
        values = h.get(hist_key)
        data[out_key] = list(values) if values is not None else [math.nan] * n
    times = list(epoch_times)
    if len(times) < n:
        times += [math.nan] * (n - len(times))
    data["epoch_time_sec"] = times[:n]
    return pd.DataFrame(data)


def train_one_experiment(
    exp_cfg: ExperimentConfig,
    prepared: PreparedData,
    output_root: str | Path = "outputs/experiments",
    verbose: int = 2,
) -> dict:
    """Latih satu konfigurasi, simpan model/artefak, lalu evaluasi ROUGE."""
    output_root = Path(output_root)
    out_dir = output_root / exp_cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    keras.utils.set_random_seed(exp_cfg.seed)
    model = build_model(
        vocab_size=len(prepared.vocab),
        exp_cfg=exp_cfg,
        max_article_len=prepared.config.max_article_len,
        max_dec_len=prepared.max_dec_len,
    )
    compile_model(model, exp_cfg)

    train_ds = make_tf_dataset(
        prepared.train,
        batch_size=exp_cfg.batch_size,
        shuffle=True,
        seed=exp_cfg.seed,
    )
    has_val = len(prepared.val) > 0
    val_ds = (
        make_tf_dataset(
            prepared.val,
            batch_size=exp_cfg.batch_size,
            shuffle=False,
            seed=exp_cfg.seed,
        )
        if has_val
        else None
    )

    timer = EpochTimer()
    checkpoint_path = out_dir / "best.weights.h5"
    callbacks: List[keras.callbacks.Callback] = [timer]
    if has_val:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=True,
                verbose=0,
            )
        )

    print(f"\n{'=' * 78}")
    print(f"Training: {exp_cfg.name}")
    print(
        f"optimizer={exp_cfg.optimizer} lr={exp_cfg.learning_rate} "
        f"emb={exp_cfg.embedding_dim} hidden={exp_cfg.hidden_dim} "
        f"batch={exp_cfg.batch_size} epochs={exp_cfg.epochs}"
    )
    print(f"params={model.count_params():,}")
    print(f"{'=' * 78}")

    start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=exp_cfg.epochs,
        callbacks=callbacks,
        verbose=verbose,
    )
    total_time = time.perf_counter() - start

    if has_val and checkpoint_path.exists():
        model.load_weights(str(checkpoint_path))

    # Simpan model final (bobot terbaik jika validation tersedia).
    model.save_weights(str(out_dir / "model.weights.h5"))
    prepared.vocab.save(out_dir / "vocab.json")
    _json_dump(_model_config_payload(exp_cfg, prepared), out_dir / "model_config.json")
    _json_dump(asdict(prepared.config), out_dir / "data_config.json")

    hist_df = _history_dataframe(history, timer.epoch_times)
    hist_df.to_csv(out_dir / "history.csv", index=False)

    # Evaluasi pada test set dengan greedy decoding.
    predictions = generate_summaries(
        model=model,
        encoder_inputs=prepared.test.encoder_inputs,
        vocab=prepared.vocab,
        max_dec_len=prepared.max_dec_len,
        batch_size=exp_cfg.batch_size,
    )
    references = [pair.summary_text for pair in prepared.test_pairs]
    rouge = corpus_rouge(predictions, references)

    pred_df = pd.DataFrame(
        {
            "article": [pair.article_text for pair in prepared.test_pairs],
            "reference": references,
            "prediction": predictions,
        }
    )
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    val_losses = [v for v in hist_df["val_loss"].tolist() if not math.isnan(v)] if has_val else []
    best_val_loss = float(min(val_losses)) if val_losses else float("nan")
    final_train_loss = float(hist_df["train_loss"].iloc[-1]) if len(hist_df) else float("nan")
    final_val_loss = float(hist_df["val_loss"].iloc[-1]) if len(hist_df) else float("nan")

    result = {
        "name": exp_cfg.name,
        "config": exp_cfg.to_dict(),
        "model": {
            "vocab_size": len(prepared.vocab),
            "trainable_params": int(model.count_params()),
            "max_article_len": prepared.config.max_article_len,
            "max_dec_len": prepared.max_dec_len,
        },
        "data_stats": prepared.stats,
        "training_time_sec": float(total_time),
        "avg_epoch_time_sec": float(np.mean(timer.epoch_times)) if timer.epoch_times else float("nan"),
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "history": hist_df.to_dict(orient="records"),
        "rouge": rouge,
        "predictions": predictions[:5],
        "references": references[:5],
    }
    _json_dump(result, out_dir / "result.json")

    print(
        f"Selesai {exp_cfg.name} | total={total_time:.2f}s | "
        f"best_val_loss={best_val_loss:.4f} | "
        f"ROUGE-1={rouge.get('rouge1_f1', 0):.4f} "
        f"ROUGE-2={rouge.get('rouge2_f1', 0):.4f} "
        f"ROUGE-L={rouge.get('rougeL_f1', 0):.4f}"
    )
    return result


def build_comparison(results: Sequence[dict]) -> pd.DataFrame:
    """Bentuk tabel ringkas siap dimasukkan ke laporan."""
    rows = []
    for r in results:
        cfg = r["config"]
        rouge = r.get("rouge", {})
        rows.append(
            {
                "config": r["name"],
                "optimizer": cfg["optimizer"],
                "lr": cfg["learning_rate"],
                "emb_dim": cfg["embedding_dim"],
                "hidden_dim": cfg["hidden_dim"],
                "batch": cfg["batch_size"],
                "epochs": cfg["epochs"],
                "params": r["model"]["trainable_params"],
                "final_train_loss": r["final_train_loss"],
                "final_val_loss": r["final_val_loss"],
                "best_val_loss": r["best_val_loss"],
                "total_time_sec": r["training_time_sec"],
                "avg_epoch_time_sec": r["avg_epoch_time_sec"],
                "rouge1_f1": rouge.get("rouge1_f1", math.nan),
                "rouge2_f1": rouge.get("rouge2_f1", math.nan),
                "rougeL_f1": rouge.get("rougeL_f1", math.nan),
            }
        )
    return pd.DataFrame(rows)


def _df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Markdown table sederhana tanpa dependensi tabulate."""
    def fmt(value) -> str:
        if isinstance(value, (float, np.floating)):
            if math.isnan(float(value)):
                return "NaN"
            if float(value).is_integer() and abs(float(value)) < 1e12:
                return str(int(value))
            return format(float(value), floatfmt)
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def save_per_epoch_tables(results: Sequence[dict], output_dir: str | Path) -> None:
    """Simpan tabel per-epoch: training loss, validation loss, dan waktu eksekusi."""
    output_dir = Path(output_dir)
    records = []
    for r in results:
        for row in r["history"]:
            records.append(
                {
                    "config": r["name"],
                    "epoch": int(row["epoch"]),
                    "train_loss": row.get("train_loss"),
                    "val_loss": row.get("val_loss"),
                    "epoch_time_sec": row.get("epoch_time_sec"),
                }
            )
    if not records:
        return
    long_df = pd.DataFrame(records)
    for metric in ["train_loss", "val_loss", "epoch_time_sec"]:
        pivot = long_df.pivot(index="epoch", columns="config", values=metric)
        pivot.to_csv(output_dir / f"per_epoch_{metric}.csv")
        table = pivot.reset_index()
        table["epoch"] = table["epoch"].astype(int)
        (output_dir / f"per_epoch_{metric}.md").write_text(
            _df_to_markdown(table) + "\n", encoding="utf-8"
        )


def save_comparison(
    results: Sequence[dict],
    output_dir: str | Path = "outputs",
    best_dir: str | Path = "outputs/best_model",
) -> pd.DataFrame:
    """Simpan CSV/Markdown tabel perbandingan, grafik, dan model terbaik."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = build_comparison(results)
    df.to_csv(output_dir / "comparison.csv", index=False)
    (output_dir / "comparison.md").write_text(
        _df_to_markdown(df) + "\n", encoding="utf-8"
    )
    save_per_epoch_tables(results, output_dir)

    plot_loss_curves(results, output_dir / "loss_curves.png")
    plot_epoch_times(results, output_dir / "epoch_times.png")
    plot_rouge(results, output_dir / "rouge_comparison.png")

    # Pilih model terbaik: ROUGE-L tertinggi, lalu validation loss terendah.
    def score(r: dict):
        rouge_l = r.get("rouge", {}).get("rougeL_f1", float("-inf"))
        val = r.get("best_val_loss", float("inf"))
        if math.isnan(val):
            val = float("inf")
        return (-rouge_l, val)

    best = sorted(results, key=score)[0]
    best_src = output_dir / "experiments" / best["name"]
    best_dst = Path(best_dir)
    best_dst.mkdir(parents=True, exist_ok=True)
    for filename in ["model.weights.h5", "vocab.json", "model_config.json", "result.json"]:
        src = best_src / filename
        if src.exists():
            shutil.copy2(src, best_dst / filename)
    _json_dump(
        {
            "best_experiment": best["name"],
            "selection": "ROUGE-L F1 tertinggi; tie-break validation loss terendah",
            "rougeL_f1": best.get("rouge", {}).get("rougeL_f1"),
            "best_val_loss": best.get("best_val_loss"),
        },
        best_dst / "best_experiment.json",
    )

    print("\n=== TABEL PERBANDINGAN ===")
    print(df.to_string(index=False))
    print(f"\nModel terbaik: {best['name']} -> {best_dst}")
    print(f"Tabel: {output_dir / 'comparison.csv'}")
    print(f"Grafik: {output_dir / 'loss_curves.png'}")
    return df


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_loss_curves(results: Sequence[dict], path: str | Path) -> None:
    _plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    for r in results:
        hist = pd.DataFrame(r["history"])
        label = r["name"]
        axes[0].plot(hist["epoch"], hist["train_loss"], marker="o", linewidth=1.5, label=label)
        if hist["val_loss"].notna().any():
            axes[1].plot(hist["epoch"], hist["val_loss"], marker="o", linewidth=1.5, label=label)
    axes[0].set_title("Training Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Masked cross-entropy")
    axes[1].set_title("Validation Loss per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Masked cross-entropy")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_epoch_times(results: Sequence[dict], path: str | Path) -> None:
    _plot_style()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for r in results:
        hist = pd.DataFrame(r["history"])
        ax.plot(
            hist["epoch"],
            hist["epoch_time_sec"],
            marker="o",
            linewidth=1.5,
            label=r["name"],
        )
    ax.set_title("Execution Time per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Detik")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_rouge(results: Sequence[dict], path: str | Path) -> None:
    _plot_style()
    labels = [r["name"] for r in results]
    metrics = ["rouge1_f1", "rouge2_f1", "rougeL_f1"]
    titles = ["ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1"]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 4.2))
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        values = [r.get("rouge", {}).get(metric, 0.0) for r in results]
        ax.bar(x + (i - 1) * width, values, width=width, label=title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("F1")
    ax.set_title("Perbandingan ROUGE pada Test Set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_experiments(
    experiments: Sequence[ExperimentConfig],
    data_cfg: Optional[DataConfig] = None,
    output_dir: str | Path = "outputs",
    verbose: int = 2,
    clean: bool = False,
) -> tuple[List[dict], PreparedData]:
    """Jalankan seluruh konfigurasi pada data yang sama."""
    output_dir = Path(output_dir)
    experiments_dir = output_dir / "experiments"
    if clean:
        if experiments_dir.exists():
            shutil.rmtree(experiments_dir)
        for name in ["comparison.csv", "comparison.md"]:
            p = output_dir / name
            if p.exists():
                p.unlink()

    prepared = prepare_data(data_cfg)
    print(describe_data(prepared))

    results: List[dict] = []
    for exp_cfg in experiments:
        keras.backend.clear_session()
        gc.collect()
        result = train_one_experiment(
            exp_cfg=exp_cfg,
            prepared=prepared,
            output_root=experiments_dir,
            verbose=verbose,
        )
        results.append(result)

    save_comparison(results, output_dir=output_dir, best_dir=output_dir / "best_model")
    return results, prepared
