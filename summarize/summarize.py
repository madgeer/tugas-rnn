"""
Fungsi inference tingkat tinggi: summarize(text).

Fungsi ini memuat model terbaik (default: outputs/best_model) dan menghasilkan
ringkasan secara autoregressive.

Contoh:
    from summarize import summarize
    print(summarize("Langkah-langkah agar akur dengan pasangan ..."))
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DEFAULT_OUTPUT_DIR
from inference import Summarizer

_DEFAULT_EXPERIMENT = str(DEFAULT_OUTPUT_DIR / "best_model")
_CACHE: dict[str, Summarizer] = {}


def get_summarizer(experiment_dir: str | Path = _DEFAULT_EXPERIMENT) -> Summarizer:
    key = str(Path(experiment_dir).resolve())
    if key not in _CACHE:
        _CACHE[key] = Summarizer.from_experiment_dir(key)
    return _CACHE[key]


def summarize(
    text: str,
    experiment_dir: str | Path = _DEFAULT_EXPERIMENT,
    batch_size: int = 64,
) -> str:
    """Hasilkan ringkasan dari satu teks artikel."""
    summarizer = get_summarizer(experiment_dir)
    return summarizer.summarize_batch([text], batch_size=batch_size)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ringkas satu teks dengan model Seq2Seq")
    parser.add_argument("--experiment", type=str, default=_DEFAULT_EXPERIMENT)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()

    if args.text:
        text = args.text
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = input("Masukkan teks artikel: ").strip()

    result = summarize(text, experiment_dir=args.experiment)
    print("\n=== ARTIKEL ===")
    print(text)
    print("\n=== RINGKASAN ===")
    print(result if result else "(model tidak menghasilkan token)")


if __name__ == "__main__":
    main()
