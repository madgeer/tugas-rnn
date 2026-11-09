"""
Tampilkan tabel evaluasi ROUGE dan metrik training dari hasil eksperimen.

Contoh:
    python evaluate.py --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"


def collect_results(output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    rows = []
    for result_path in sorted((output_dir / "experiments").glob("*/result.json")):
        with result_path.open("r", encoding="utf-8") as f:
            r = json.load(f)
        cfg = r["config"]
        rouge = r.get("rouge", {})
        rows.append(
            {
                "config": r["name"],
                "optimizer": cfg["optimizer"],
                "lr": cfg["learning_rate"],
                "emb_dim": cfg["embedding_dim"],
                "hidden_dim": cfg["hidden_dim"],
                "epochs": cfg["epochs"],
                "best_val_loss": r["best_val_loss"],
                "total_time_sec": r["training_time_sec"],
                "avg_epoch_time_sec": r["avg_epoch_time_sec"],
                "rouge1_f1": rouge.get("rouge1_f1"),
                "rouge2_f1": rouge.get("rouge2_f1"),
                "rougeL_f1": rouge.get("rougeL_f1"),
                "n_test": rouge.get("n_samples"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tampilkan hasil evaluasi eksperimen")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    df = collect_results(args.output_dir)
    if df.empty:
        raise SystemExit(f"Tidak ada result.json di {args.output_dir}/experiments")
    print(df.to_string(index=False))
    out = Path(args.output_dir) / "evaluation_table.csv"
    df.to_csv(out, index=False)
    print(f"\nTabel disimpan ke {out}")


if __name__ == "__main__":
    main()
