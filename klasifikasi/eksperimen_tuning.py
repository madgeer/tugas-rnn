"""
Eksperimen Pembanding dan Tuning Hyperparameter
================================================

Skrip ini melatih beberapa konfigurasi model RNN pada split data yang sama
dengan eksperimen utama, lalu membandingkan performanya pada data validasi.
Data uji hanya digunakan untuk evaluasi akhir konfigurasi terbaik.

Kelompok eksperimen:
1. Optimizer      : Adam, AdamW, RMSprop, SGD
2. Arsitektur     : SimpleRNN, GRU, LSTM, BiSimpleRNN, BiGRU, BiLSTM
3. Jumlah unit    : 16, 32, 64
4. Embedding dim  : 32, 64, 128
5. Dropout        : 0.3, 0.5, 0.7
6. Recurrent drop : 0.0, 0.2, 0.4
7. Batch size     : 8, 16, 32
8. Learning rate  : 1e-4, 5e-4, 1e-3
9. L2 embedding   : 0, 1e-4, 1e-3

Setiap konfigurasi dijalankan dengan tiga random seed (42, 123, 2024).
Pemilihan konfigurasi terbaik dilakukan berdasarkan rata-rata validation
F1-score, dengan validation accuracy dan validation loss sebagai tie-breaker.

Output:
    output_rnn/tuning/hasil_tuning_per_seed.csv
    output_rnn/tuning/hasil_tuning_per_grup.csv
    output_rnn/tuning/ringkasan_tuning.csv
    output_rnn/tuning/perbandingan_optimizer.png
    output_rnn/tuning/perbandingan_arsitektur.png
    output_rnn/tuning/tuning_hyperparameter.png
    output_rnn/tuning/top_konfigurasi.png
    output_rnn/tuning/model_terbaik/kurva_model_tuning_terbaik.png
    output_rnn/tuning/model_terbaik/confusion_matrix.png
    output_rnn/tuning/model_terbaik/hasil_evaluasi.txt
    output_rnn/tuning/model_terbaik/model_tuning_terbaik.keras
    output_rnn/tuning/model_terbaik/konfigurasi_terbaik.json
    output_rnn/tuning/model_terbaik/perbandingan_baseline_vs_terbaik.csv
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

import rnn.klasifikasi.klasifikasi_sentimen_rnn as base

# ============================== KONFIGURASI ==============================
SEEDS = [42, 123, 2024]
EPOCHS = 60
TUNING_DIR = base.OUTPUT_DIR / "tuning"
FINAL_DIR = TUNING_DIR / "model_terbaik"

BASE_CONFIG = {
    "rnn": "simple_rnn",
    "units": 32,
    "emb": 64,
    "dropout": 0.5,
    "recurrent_dropout": 0.2,
    "dense": 32,
    "dense_dropout": 0.5,
    "l2": 1e-4,
    "lr": 5e-4,
    "batch": 16,
    "optimizer": "adam",
    "bidirectional": False,
    "momentum": 0.9,
}
# ========================================================================


def build_model(cfg: dict, vocab_size: int, max_length: int) -> tf.keras.Model:
    """Membangun model RNN many-to-one sesuai konfigurasi."""
    rnn_layers = {
        "simple_rnn": tf.keras.layers.SimpleRNN,
        "gru": tf.keras.layers.GRU,
        "lstm": tf.keras.layers.LSTM,
    }
    rnn_type = cfg["rnn"].lower()
    if rnn_type not in rnn_layers:
        raise ValueError(f"rnn='{rnn_type}' tidak dikenal")

    embedding_regularizer = (
        tf.keras.regularizers.l2(cfg["l2"]) if cfg["l2"] > 0 else None
    )

    inputs = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name="token_ids")
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=cfg["emb"],
        mask_zero=True,
        embeddings_regularizer=embedding_regularizer,
        name="embedding",
    )(inputs)

    recurrent = rnn_layers[rnn_type](
        units=cfg["units"],
        dropout=cfg["dropout"],
        recurrent_dropout=cfg["recurrent_dropout"],
    )
    if cfg["bidirectional"]:
        recurrent = tf.keras.layers.Bidirectional(recurrent)

    x = recurrent(x)
    x = tf.keras.layers.Dense(
        cfg["dense"], activation="relu", name="dense_hidden"
    )(x)
    x = tf.keras.layers.Dropout(cfg["dense_dropout"], name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="sentimen")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="rnn_tuning")
    model.compile(
        optimizer=make_optimizer(cfg),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_optimizer(cfg: dict) -> tf.keras.optimizers.Optimizer:
    """Membuat optimizer dengan learning rate yang sesuai."""
    name = cfg["optimizer"].lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=cfg["lr"])
    if name == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=cfg["lr"], weight_decay=1e-4
        )
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=cfg["lr"])
    if name == "sgd":
        return tf.keras.optimizers.SGD(
            learning_rate=cfg["lr"],
            momentum=cfg["momentum"],
            nesterov=True,
        )
    raise ValueError(f"optimizer='{name}' tidak dikenal")


def config_signature(cfg: dict) -> tuple:
    """Signature unik untuk mendeteksi konfigurasi yang sama."""
    return tuple((key, cfg[key]) for key in sorted(cfg))


def describe_config(cfg: dict) -> str:
    """Label ringkas untuk menampilkan konfigurasi."""
    name = {
        "simple_rnn": "SimpleRNN",
        "gru": "GRU",
        "lstm": "LSTM",
    }.get(cfg["rnn"].lower(), cfg["rnn"])
    if cfg["bidirectional"]:
        name = f"Bi{name}"
    return (
        f"{name} u{cfg['units']} e{cfg['emb']} d{cfg['dropout']} "
        f"rd{cfg['recurrent_dropout']} b{cfg['batch']} "
        f"lr{cfg['lr']} {cfg['optimizer']} l2={cfg['l2']}"
    )


def run_single(cfg: dict, seed: int, data: dict) -> dict:
    """Melatih satu konfigurasi dengan satu seed dan mengembalikan metrik validasi."""
    tf.keras.backend.clear_session()
    gc.collect()
    base.set_seed(seed)

    model = build_model(cfg, data["vocab_size"], data["max_length"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=0,
        ),
    ]

    start = time.perf_counter()
    history = model.fit(
        data["X_train"],
        data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=EPOCHS,
        batch_size=cfg["batch"],
        callbacks=callbacks,
        verbose=0,
    )
    train_time = time.perf_counter() - start

    val_probability = model.predict(data["X_val"], verbose=0).ravel()
    val_prediction = (val_probability >= 0.5).astype(int)

    finite_losses = [
        value for value in history.history["val_loss"] if np.isfinite(value)
    ]
    val_loss = float(min(finite_losses)) if finite_losses else float("nan")
    best_epoch = (
        int(np.argmin(history.history["val_loss"])) + 1
        if finite_losses
        else -1
    )

    metrics = {
        "val_accuracy": float(accuracy_score(data["y_val"], val_prediction)),
        "val_precision": float(
            precision_score(
                data["y_val"], val_prediction, average="weighted", zero_division=0
            )
        ),
        "val_recall": float(
            recall_score(
                data["y_val"], val_prediction, average="weighted", zero_division=0
            )
        ),
        "val_f1": float(
            f1_score(
                data["y_val"], val_prediction, average="weighted", zero_division=0
            )
        ),
        "val_loss": val_loss,
        "best_epoch": best_epoch,
        "epochs_run": len(history.history["loss"]),
        "train_time": train_time,
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "n_params": int(model.count_params()),
    }

    del model
    gc.collect()
    return metrics


def build_experiment_list() -> list[dict]:
    """Mendefinisikan seluruh eksperimen pembanding."""
    experiments: list[dict] = []

    def add(group: str, label: str, overrides: dict, x_value: object) -> None:
        config = {**BASE_CONFIG, **overrides}
        experiments.append(
            {
                "group": group,
                "label": label,
                "config": config,
                "x": x_value,
            }
        )

    # 1. Optimizer.
    add("Optimizer", "Adam", {}, "Adam")
    add("Optimizer", "AdamW", {"optimizer": "adamw"}, "AdamW")
    add(
        "Optimizer",
        "RMSprop",
        {"optimizer": "rmsprop", "lr": 1e-3},
        "RMSprop",
    )
    add(
        "Optimizer",
        "SGD",
        {"optimizer": "sgd", "lr": 5e-3},
        "SGD",
    )

    # 2. Arsitektur RNN.
    add("Arsitektur", "SimpleRNN", {}, "SimpleRNN")
    add("Arsitektur", "GRU", {"rnn": "gru"}, "GRU")
    add("Arsitektur", "LSTM", {"rnn": "lstm"}, "LSTM")
    add(
        "Arsitektur",
        "BiSimpleRNN",
        {"rnn": "simple_rnn", "bidirectional": True},
        "BiSimpleRNN",
    )
    add(
        "Arsitektur",
        "BiGRU",
        {"rnn": "gru", "bidirectional": True},
        "BiGRU",
    )
    add(
        "Arsitektur",
        "BiLSTM",
        {"rnn": "lstm", "bidirectional": True},
        "BiLSTM",
    )

    # 3. Jumlah unit RNN.
    add("Unit RNN", "16", {"units": 16}, 16)
    add("Unit RNN", "32", {"units": 32}, 32)
    add("Unit RNN", "64", {"units": 64}, 64)

    # 4. Dimensi embedding.
    add("Embedding", "32", {"emb": 32}, 32)
    add("Embedding", "64", {"emb": 64}, 64)
    add("Embedding", "128", {"emb": 128}, 128)

    # 5. Dropout pada RNN.
    add("Dropout", "0.3", {"dropout": 0.3}, 0.3)
    add("Dropout", "0.5", {"dropout": 0.5}, 0.5)
    add("Dropout", "0.7", {"dropout": 0.7}, 0.7)

    # 6. Recurrent dropout.
    add("Recurrent Dropout", "0.0", {"recurrent_dropout": 0.0}, 0.0)
    add("Recurrent Dropout", "0.2", {"recurrent_dropout": 0.2}, 0.2)
    add("Recurrent Dropout", "0.4", {"recurrent_dropout": 0.4}, 0.4)

    # 7. Batch size.
    add("Batch Size", "8", {"batch": 8}, 8)
    add("Batch Size", "16", {"batch": 16}, 16)
    add("Batch Size", "32", {"batch": 32}, 32)

    # 8. Learning rate.
    add("Learning Rate", "1e-4", {"lr": 1e-4}, 1e-4)
    add("Learning Rate", "5e-4", {"lr": 5e-4}, 5e-4)
    add("Learning Rate", "1e-3", {"lr": 1e-3}, 1e-3)

    # 9. Regularisasi L2 pada embedding.
    add("L2 Embedding", "0", {"l2": 0.0}, 0.0)
    add("L2 Embedding", "1e-4", {"l2": 1e-4}, 1e-4)
    add("L2 Embedding", "1e-3", {"l2": 1e-3}, 1e-3)

    return experiments


def prepare_data() -> dict:
    """Menyiapkan data dengan split yang sama seperti eksperimen utama."""
    df = base.preprocess_dataset(base.load_dataset(base.DATASET_PATH))
    label_names = sorted(df["label"].unique().tolist())
    label_to_id = {name: index for index, name in enumerate(label_names)}

    y = df["label"].map(label_to_id).to_numpy(dtype=np.int32)
    X = df["text_clean"].to_numpy(dtype=object)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=base.TEST_SIZE,
        random_state=base.SEED,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=base.VAL_SIZE,
        random_state=base.SEED,
        stratify=y_train_full,
    )

    max_length = base.determine_sequence_length(X_train)
    vectorizer = base.build_vectorizer(X_train, max_length)

    data = {
        "X_train": base.vectorize_texts(vectorizer, X_train),
        "X_val": base.vectorize_texts(vectorizer, X_val),
        "X_test": base.vectorize_texts(vectorizer, X_test),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "label_names": label_names,
        "label_to_id": label_to_id,
        "max_length": max_length,
        "vocab_size": vectorizer.vocabulary_size(),
        "vectorizer": vectorizer,
    }

    print("\nBentuk data:")
    print(f"  Train : {data['X_train'].shape}")
    print(f"  Val   : {data['X_val'].shape}")
    print(f"  Test  : {data['X_test'].shape}")
    print(f"  Vocab : {data['vocab_size']}")
    return data


def make_summaries(
    runs_df: pd.DataFrame, group_runs_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Membuat ringkasan per konfigurasi dan per kelompok eksperimen."""
    config_fields = ["config_id"] + list(BASE_CONFIG.keys())

    config_summary = (
        runs_df.groupby("config_id")
        .agg(
            val_accuracy_mean=("val_accuracy", "mean"),
            val_accuracy_std=("val_accuracy", "std"),
            val_precision_mean=("val_precision", "mean"),
            val_recall_mean=("val_recall", "mean"),
            val_f1_mean=("val_f1", "mean"),
            val_f1_std=("val_f1", "std"),
            val_loss_mean=("val_loss", "mean"),
            val_loss_std=("val_loss", "std"),
            train_time_mean=("train_time", "mean"),
            n_params=("n_params", "first"),
            n_runs=("seed", "count"),
        )
        .reset_index()
    )
    config_info = runs_df.drop_duplicates("config_id")[config_fields]
    config_summary = config_info.merge(config_summary, on="config_id")
    config_summary["label"] = config_summary.apply(
        lambda row: describe_config(row.to_dict()), axis=1
    )

    group_summary = (
        group_runs_df.groupby(["group", "label"], sort=False)
        .agg(
            x=("x", "first"),
            val_accuracy_mean=("val_accuracy", "mean"),
            val_accuracy_std=("val_accuracy", "std"),
            val_f1_mean=("val_f1", "mean"),
            val_f1_std=("val_f1", "std"),
            val_loss_mean=("val_loss", "mean"),
            val_loss_std=("val_loss", "std"),
            train_time_mean=("train_time", "mean"),
            n_runs=("seed", "count"),
        )
        .reset_index()
    )

    return config_summary, group_summary


def plot_group_comparison(
    group_summary: pd.DataFrame,
    group_name: str,
    output_path: Path,
    title: str,
) -> None:
    """Membuat grafik batang perbandingan accuracy dan F1 untuk satu kelompok."""
    subset = group_summary[group_summary["group"] == group_name].copy()
    if subset.empty:
        return

    x = np.arange(len(subset))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(
        x - width / 2,
        subset["val_accuracy_mean"],
        width,
        yerr=subset["val_accuracy_std"],
        capsize=4,
        label="Validation Accuracy",
        color="#4C72B0",
    )
    ax.bar(
        x + width / 2,
        subset["val_f1_mean"],
        width,
        yerr=subset["val_f1_std"],
        capsize=4,
        label="Validation F1-Score",
        color="#DD8452",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(subset["label"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Skor")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafik disimpan di: {output_path}")


def plot_hyperparameter_panels(
    group_summary: pd.DataFrame,
    baseline_f1: float,
    output_path: Path,
) -> None:
    """Membuat panel pengaruh tiap hyperparameter terhadap validation F1."""
    groups = [
        "Unit RNN",
        "Embedding",
        "Dropout",
        "Recurrent Dropout",
        "Batch Size",
        "Learning Rate",
        "L2 Embedding",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), sharey=True)
    axes = axes.ravel()

    for axis, group in zip(axes, groups):
        subset = group_summary[group_summary["group"] == group].copy()
        positions = np.arange(len(subset))
        axis.errorbar(
            positions,
            subset["val_f1_mean"],
            yerr=subset["val_f1_std"],
            marker="o",
            capsize=4,
            linewidth=2,
            color="#4C72B0",
        )
        axis.axhline(
            baseline_f1,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.8,
            label="Baseline",
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(subset["label"])
        axis.set_title(group)
        axis.set_ylim(0.40, 1.0)
        axis.grid(alpha=0.3)
        axis.set_xlabel("Nilai")
        axis.set_ylabel("Validation F1")

    axes[0].legend(loc="lower right", fontsize=8)
    axes[-1].axis("off")
    axes[-1].text(
        0.02,
        0.95,
        "Catatan:\n"
        "- Error bar: simpangan baku 3 seed\n"
        "- Validation set: 24 tweet\n"
        "- Garis putus-putus: baseline F1\n"
        "- Selisih < 0.05 belum konklusif",
        transform=axes[-1].transAxes,
        va="top",
        fontsize=9,
        color="#333333",
    )
    fig.suptitle(
        "Pengaruh Hyperparameter terhadap Validation F1 "
        "(mean ± std, 3 seed)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafik hyperparameter disimpan di: {output_path}")


def plot_top_configs(
    config_summary: pd.DataFrame, output_path: Path
) -> None:
    """Membuat grafik horizontal 10 konfigurasi terbaik."""
    top = config_summary.sort_values(
        ["val_f1_mean", "val_accuracy_mean", "val_loss_mean"],
        ascending=[False, False, True],
    ).head(10)
    top = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh(
        top["label"],
        top["val_f1_mean"],
        xerr=top["val_f1_std"],
        capsize=4,
        color="#55A868",
    )
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Validation F1-Score (mean ± std)")
    ax.set_title("10 Konfigurasi Terbaik Berdasarkan Validation F1")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafik konfigurasi terbaik disimpan di: {output_path}")


def parse_baseline_metrics() -> dict:
    """Membaca metrik baseline dari hasil eksperimen utama bila tersedia."""
    report_path = base.OUTPUT_DIR / "hasil_evaluasi.txt"
    metrics = {
        "accuracy": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
    }
    if not report_path.exists():
        return metrics

    text = report_path.read_text(encoding="utf-8")
    patterns = {
        "accuracy": r"Accuracy\s*:\s*([0-9.]+)",
        "precision": r"Precision\s*:\s*([0-9.]+)",
        "recall": r"Recall\s*:\s*([0-9.]+)",
        "f1": r"F1-Score\s*:\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def main() -> None:
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("=" * 70)
    print("EKSPERIMEN PEMBANDING DAN TUNING HYPERPARAMETER")
    print("=" * 70)

    data = prepare_data()
    experiments = build_experiment_list()

    # Deduplikasi konfigurasi yang muncul di beberapa kelompok.
    unique_configs: dict[tuple, dict] = {}
    for experiment in experiments:
        signature = config_signature(experiment["config"])
        if signature not in unique_configs:
            unique_configs[signature] = {
                "config_id": f"cfg_{len(unique_configs) + 1:02d}",
                "config": experiment["config"],
            }

    print(f"\nJumlah eksperimen unik : {len(unique_configs)}")
    print(f"Jumlah seed            : {len(SEEDS)}")
    print(f"Total run              : {len(unique_configs) * len(SEEDS)}")
    print("Pemilihan model        : validation F1 (mean 3 seed)\n")

    results_path = TUNING_DIR / "hasil_tuning_per_seed.csv"
    existing_results: dict[tuple[str, int], dict] = {}
    if results_path.exists():
        existing_df = pd.read_csv(results_path)
        if not existing_df.empty:
            for _, row in existing_df.iterrows():
                existing_results[
                    (str(row["config_id"]), int(row["seed"]))
                ] = row.to_dict()
            print(
                f"Melanjutkan dari {len(existing_results)} hasil run "
                f"yang sudah tersimpan.\n"
            )

    runs = []
    total_runs = len(unique_configs) * len(SEEDS)
    run_index = 0

    for signature, info in unique_configs.items():
        cfg = info["config"]
        for seed in SEEDS:
            run_index += 1
            cache_key = (info["config_id"], seed)
            if cache_key in existing_results:
                row = existing_results[cache_key]
                runs.append(row)
                print(
                    f"[{run_index:02d}/{total_runs:02d}] {info['config_id']} "
                    f"{describe_config(cfg)} | seed={seed} | "
                    f"skip (sudah ada, val_f1={row['val_f1']:.4f})",
                    flush=True,
                )
                continue

            metrics = run_single(cfg, seed, data)
            row = {
                "config_id": info["config_id"],
                "seed": seed,
                **cfg,
                **metrics,
            }
            runs.append(row)
            print(
                f"[{run_index:02d}/{total_runs:02d}] {info['config_id']} "
                f"{describe_config(cfg)} | seed={seed} | "
                f"val_acc={metrics['val_accuracy']:.4f} "
                f"val_f1={metrics['val_f1']:.4f} "
                f"val_loss={metrics['val_loss']:.4f} "
                f"({metrics['train_time']:.1f}s)",
                flush=True,
            )

            # Simpan hasil secara bertahap agar aman jika proses terhenti.
            pd.DataFrame(runs).to_csv(results_path, index=False)

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(results_path, index=False)

    # Satu konfigurasi dapat muncul di beberapa kelompok (mis. baseline).
    # Bentuk ulang hasil per konfigurasi menjadi hasil per kelompok.
    group_rows = []
    for experiment in experiments:
        signature = config_signature(experiment["config"])
        config_id = unique_configs[signature]["config_id"]
        subset = runs_df[runs_df["config_id"] == config_id]
        for _, row in subset.iterrows():
            group_row = row.to_dict()
            group_row["group"] = experiment["group"]
            group_row["label"] = experiment["label"]
            group_row["x"] = experiment["x"]
            group_rows.append(group_row)

    group_runs_df = pd.DataFrame(group_rows)
    group_runs_df.to_csv(
        TUNING_DIR / "hasil_tuning_per_grup.csv", index=False
    )

    config_summary, group_summary = make_summaries(
        runs_df, group_runs_df
    )
    config_summary = config_summary.sort_values(
        ["val_f1_mean", "val_accuracy_mean", "val_loss_mean"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    config_summary.to_csv(TUNING_DIR / "ringkasan_tuning.csv", index=False)

    # Grafik perbandingan optimizer dan arsitektur.
    plot_group_comparison(
        group_summary,
        "Optimizer",
        TUNING_DIR / "perbandingan_optimizer.png",
        "Perbandingan Optimizer terhadap Validation Accuracy dan F1",
    )
    plot_group_comparison(
        group_summary,
        "Arsitektur",
        TUNING_DIR / "perbandingan_arsitektur.png",
        "Perbandingan Arsitektur RNN terhadap Validation Accuracy dan F1",
    )

    # Baseline F1 untuk garis referensi pada panel hyperparameter.
    baseline_signature = config_signature(BASE_CONFIG)
    baseline_id = unique_configs[baseline_signature]["config_id"]
    baseline_f1 = float(
        config_summary.loc[
            config_summary["config_id"] == baseline_id, "val_f1_mean"
        ].iloc[0]
    )

    plot_hyperparameter_panels(
        group_summary,
        baseline_f1,
        TUNING_DIR / "tuning_hyperparameter.png",
    )
    plot_top_configs(
        config_summary,
        TUNING_DIR / "top_konfigurasi.png",
    )

    # Tampilkan 10 konfigurasi terbaik.
    print("\n10 Konfigurasi Terbaik (berdasarkan mean validation F1):")
    display_columns = [
        "config_id",
        "label",
        "val_accuracy_mean",
        "val_f1_mean",
        "val_loss_mean",
        "train_time_mean",
    ]
    print(config_summary[display_columns].head(10).to_string(index=False))

    best = config_summary.iloc[0]
    best_id = best["config_id"]
    best_signature = next(
        signature
        for signature, info in unique_configs.items()
        if info["config_id"] == best_id
    )
    best_config = unique_configs[best_signature]["config"]

    print("\nKonfigurasi terbaik:")
    print(f"  ID       : {best_id}")
    print(f"  Deskripsi: {best['label']}")
    print(f"  Val F1   : {best['val_f1_mean']:.4f} ± {best['val_f1_std']:.4f}")
    print(
        f"  Val Acc  : {best['val_accuracy_mean']:.4f} "
        f"± {best['val_accuracy_std']:.4f}"
    )

    # Latih ulang konfigurasi terbaik pada data latih, lalu evaluasi data uji.
    print("\nMelatih ulang konfigurasi terbaik pada data latih...")
    tf.keras.backend.clear_session()
    gc.collect()
    base.set_seed(base.SEED)
    final_model = build_model(
        best_config, data["vocab_size"], data["max_length"]
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1,
        ),
    ]
    history = final_model.fit(
        data["X_train"],
        data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=EPOCHS,
        batch_size=best_config["batch"],
        callbacks=callbacks,
        verbose=2,
    )

    base.plot_training_history(
        history, FINAL_DIR / "kurva_model_tuning_terbaik.png"
    )
    test_metrics = base.evaluate_model(
        final_model,
        data["X_test"],
        data["y_test"],
        data["label_names"],
        FINAL_DIR,
    )

    final_model.save(FINAL_DIR / "model_tuning_terbaik.keras")
    (FINAL_DIR / "konfigurasi_terbaik.json").write_text(
        json.dumps(
            {
                "config_id": best_id,
                "description": best["label"],
                "config": best_config,
                "validation_accuracy_mean": float(best["val_accuracy_mean"]),
                "validation_accuracy_std": float(best["val_accuracy_std"]),
                "validation_f1_mean": float(best["val_f1_mean"]),
                "validation_f1_std": float(best["val_f1_std"]),
                "validation_loss_mean": float(best["val_loss_mean"]),
                "test_metrics": test_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Perbandingan baseline vs konfigurasi terbaik pada data uji.
    baseline_metrics = parse_baseline_metrics()
    comparison_rows = [
        {
            "model": "Baseline SimpleRNN (Adam, lr=5e-4)",
            "test_accuracy": baseline_metrics["accuracy"],
            "test_precision": baseline_metrics["precision"],
            "test_recall": baseline_metrics["recall"],
            "test_f1": baseline_metrics["f1"],
        },
        {
            "model": f"Tuning terbaik ({best['label']})",
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
        },
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(
        FINAL_DIR / "perbandingan_baseline_vs_terbaik.csv", index=False
    )
    print("\nPerbandingan baseline vs konfigurasi terbaik pada data uji:")
    print(comparison_df.to_string(index=False))

    print("\nSelesai. Semua hasil tuning tersimpan di:", TUNING_DIR)


if __name__ == "__main__":
    main()
