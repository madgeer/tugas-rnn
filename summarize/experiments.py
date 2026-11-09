"""
CLI menjalankan seluruh eksperimen perbandingan Seq2Seq Vanilla RNN.

Contoh:
    python experiments.py --max-samples 6000 --epochs 5
    python experiments.py --configs adam_lr0.001_h128,rmsprop_lr0.001_h128 --epochs 3
    python experiments.py --max-samples 0            # seluruh dataset (lama)
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import List

from config import DEFAULT_EXPERIMENTS, DataConfig, DEFAULT_OUTPUT_DIR
from training import run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eksperimen Text Summarization Seq2Seq Vanilla RNN (TensorFlow/Keras)"
    )
    parser.add_argument("--data-path", type=str, default=DataConfig().data_path)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DataConfig().max_samples,
        help="Jumlah pasangan maksimum; 0 = seluruh dataset",
    )
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-article-len", type=int, default=DataConfig().max_article_len)
    parser.add_argument("--max-summary-len", type=int, default=DataConfig().max_summary_len)
    parser.add_argument("--max-vocab-size", type=int, default=DataConfig().max_vocab_size)
    parser.add_argument("--min-freq", type=int, default=DataConfig().min_freq)
    parser.add_argument("--val-ratio", type=float, default=DataConfig().val_ratio)
    parser.add_argument("--test-ratio", type=float, default=DataConfig().test_ratio)
    parser.add_argument("--seed", type=int, default=DataConfig().seed)
    parser.add_argument(
        "--reverse-source",
        action="store_true",
        help="Balik urutan token artikel sebelum masuk encoder (opsional)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override jumlah epoch")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout")
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Nama konfigurasi dipisah koma; default semua konfigurasi",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--clean", action="store_true", help="Hapus hasil eksperimen lama")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_cfg = DataConfig(
        data_path=args.data_path,
        max_samples=None if args.max_samples == 0 else args.max_samples,
        max_pages=args.max_pages,
        max_article_len=args.max_article_len,
        max_summary_len=args.max_summary_len,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        reverse_source=args.reverse_source,
    )

    experiments = list(DEFAULT_EXPERIMENTS)
    if args.configs:
        wanted = {name.strip() for name in args.configs.split(",") if name.strip()}
        experiments = [cfg for cfg in experiments if cfg.name in wanted]
        missing = wanted - {cfg.name for cfg in experiments}
        if missing:
            raise SystemExit(
                f"Konfigurasi tidak ditemukan: {sorted(missing)}. "
                f"Pilihan: {[cfg.name for cfg in DEFAULT_EXPERIMENTS]}"
            )

    if args.epochs is not None or args.batch_size is not None or args.dropout is not None:
        experiments = [
            replace(
                cfg,
                epochs=args.epochs if args.epochs is not None else cfg.epochs,
                batch_size=args.batch_size if args.batch_size is not None else cfg.batch_size,
                dropout=args.dropout if args.dropout is not None else cfg.dropout,
            )
            for cfg in experiments
        ]

    run_experiments(
        experiments=experiments,
        data_cfg=data_cfg,
        output_dir=args.output_dir,
        verbose=args.verbose,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
