"""
Konfigurasi global untuk eksperimen Text Summarization Seq2Seq Vanilla RNN.

Semua path diturunkan dari lokasi file ini agar skrip dapat dijalankan dari
direktori mana pun.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "wikihow.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"


@dataclass
class DataConfig:
    """Konfigurasi pembacaan data dan preprocessing."""

    data_path: str = str(DEFAULT_DATA_PATH)
    # None = gunakan seluruh dataset. Untuk eksperimen cepat gunakan mis. 6000.
    max_samples: Optional[int] = 6000
    # Batas jumlah halaman yang dibaca (None = semua).
    max_pages: Optional[int] = None
    max_article_len: int = 128
    # Panjang maksimum sequence decoder, TERMASUK token <SOS> dan <EOS>.
    max_summary_len: int = 26
    # Trik Seq2Seq: balik urutan token artikel di encoder. Opsional; default
    # False agar arsitektur tetap mengikuti Seq2Seq vanilla standar.
    reverse_source: bool = False
    min_article_tokens: int = 8
    min_summary_tokens: int = 3
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    seed: int = 42
    min_freq: int = 1
    max_vocab_size: int = 20000
    # Jika True, split dilakukan per halaman agar tidak ada kebocoran data
    # (steps dari halaman yang sama tidak muncul di train dan test sekaligus).
    split_by_page: bool = True


@dataclass
class ExperimentConfig:
    """Hyperparameter satu konfigurasi eksperimen."""

    name: str
    optimizer: str = "adam"          # adam | rmsprop | sgd
    learning_rate: float = 1e-3
    embedding_dim: int = 128
    hidden_dim: int = 128
    batch_size: int = 64
    epochs: int = 5
    clipnorm: float = 5.0
    dropout: float = 0.0
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


# Empat konfigurasi yang membandingkan optimizer, learning rate, dan hidden dim.
# Silakan kurangi/hapus salah satu untuk mempercepat eksperimen.
DEFAULT_EXPERIMENTS: List[ExperimentConfig] = [
    ExperimentConfig(
        name="adam_lr0.001_h128",
        optimizer="adam",
        learning_rate=1e-3,
        embedding_dim=128,
        hidden_dim=128,
        batch_size=64,
        epochs=8,
        dropout=0.2,
    ),
    ExperimentConfig(
        name="rmsprop_lr0.001_h128",
        optimizer="rmsprop",
        learning_rate=1e-3,
        embedding_dim=128,
        hidden_dim=128,
        batch_size=64,
        epochs=8,
        dropout=0.2,
    ),
    ExperimentConfig(
        name="adam_lr0.0001_h128",
        optimizer="adam",
        learning_rate=1e-4,
        embedding_dim=128,
        hidden_dim=128,
        batch_size=64,
        epochs=8,
        dropout=0.2,
    ),
    ExperimentConfig(
        name="adam_lr0.001_h256",
        optimizer="adam",
        learning_rate=1e-3,
        embedding_dim=128,
        hidden_dim=256,
        batch_size=64,
        epochs=8,
        dropout=0.2,
    ),
]
