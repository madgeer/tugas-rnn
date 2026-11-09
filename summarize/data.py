"""
Pembacaan dataset WikiHow, split train/validation/test, pembentukan vocabulary,
encoding teks menjadi tensor, dan padding sequence.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from config import DataConfig
from preprocessing import (
    PAD_ID,
    Vocabulary,
    detokenize,
    pad_sequences,
    tokenize,
)


@dataclass
class TextPair:
    """Satu pasangan artikel (input) dan ringkasan/abstract (target)."""

    page_id: int
    article_text: str
    summary_text: str
    article_tokens: List[str]
    summary_tokens: List[str]


@dataclass
class SplitArrays:
    """Array input/target siap dipakai model untuk satu split."""

    encoder_inputs: np.ndarray
    decoder_inputs: np.ndarray
    decoder_targets: np.ndarray

    def __len__(self) -> int:
        return int(self.encoder_inputs.shape[0])


@dataclass
class PreparedData:
    """Kumpulan data, vocabulary, dan statistik untuk seluruh eksperimen."""

    config: DataConfig
    vocab: Vocabulary
    train_pairs: List[TextPair]
    val_pairs: List[TextPair]
    test_pairs: List[TextPair]
    train: SplitArrays
    val: SplitArrays
    test: SplitArrays
    max_dec_len: int
    stats: Dict[str, object] = field(default_factory=dict)


def _load_pages(path: str | Path) -> List[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        pages = json.load(f)
    if not isinstance(pages, list):
        raise ValueError("Dataset harus berupa list halaman JSON.")
    return pages


def _page_to_pairs(
    page: dict,
    page_id: int,
    cfg: DataConfig,
) -> List[TextPair]:
    pairs: List[TextPair] = []
    for step in page.get("steps", []):
        article_tokens = tokenize(step.get("article", ""))
        summary_tokens = tokenize(step.get("abstract", ""))
        if len(article_tokens) < cfg.min_article_tokens:
            continue
        if len(summary_tokens) < cfg.min_summary_tokens:
            continue
        pairs.append(
            TextPair(
                page_id=page_id,
                article_text=detokenize(article_tokens),
                summary_text=detokenize(summary_tokens),
                article_tokens=article_tokens,
                summary_tokens=summary_tokens,
            )
        )
    return pairs


def _split_by_page_groups(
    groups: List[Tuple[int, List[TextPair]]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[TextPair], List[TextPair], List[TextPair]]:
    rng = random.Random(seed)
    groups = list(groups)
    rng.shuffle(groups)

    n = len(groups)
    if n == 0:
        return [], [], []

    n_test = max(1, round(n * test_ratio)) if test_ratio > 0 and n >= 3 else 0
    n_val = max(1, round(n * val_ratio)) if val_ratio > 0 and n >= 3 else 0
    if n_test + n_val >= n:
        n_test = min(n_test, max(0, n - 2))
        n_val = min(n_val, max(0, n - n_test - 1))
    n_train = n - n_val - n_test

    train_groups = groups[:n_train]
    val_groups = groups[n_train : n_train + n_val]
    test_groups = groups[n_train + n_val :]

    def flatten(gs: Sequence[Tuple[int, List[TextPair]]]) -> List[TextPair]:
        return [pair for _, pairs in gs for pair in pairs]

    return flatten(train_groups), flatten(val_groups), flatten(test_groups)


def _split_by_pair(
    pairs: List[TextPair],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[TextPair], List[TextPair], List[TextPair]]:
    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)
    n = len(pairs)
    n_test = int(round(n * test_ratio)) if test_ratio > 0 else 0
    n_val = int(round(n * val_ratio)) if val_ratio > 0 else 0
    n_test = min(n_test, max(0, n))
    n_val = min(n_val, max(0, n - n_test))
    n_train = n - n_val - n_test
    return pairs[:n_train], pairs[n_train : n_train + n_val], pairs[n_train + n_val :]


def load_and_split_pairs(cfg: DataConfig) -> Tuple[List[TextPair], List[TextPair], List[TextPair], Dict[str, object]]:
    """Baca wikihow.json, bersihkan, tokenisasi, lalu split data."""
    pages = _load_pages(cfg.data_path)
    rng = random.Random(cfg.seed)
    rng.shuffle(pages)
    if cfg.max_pages is not None:
        pages = pages[: int(cfg.max_pages)]

    groups: List[Tuple[int, List[TextPair]]] = []
    total_pairs = 0
    for page_id, page in enumerate(pages):
        pairs = _page_to_pairs(page, page_id=page_id, cfg=cfg)
        if not pairs:
            continue
        groups.append((page_id, pairs))
        total_pairs += len(pairs)
        if cfg.max_samples is not None and total_pairs >= int(cfg.max_samples):
            break

    if cfg.split_by_page:
        train_pairs, val_pairs, test_pairs = _split_by_page_groups(
            groups, cfg.val_ratio, cfg.test_ratio, cfg.seed
        )
    else:
        all_pairs = [pair for _, pairs in groups for pair in pairs]
        train_pairs, val_pairs, test_pairs = _split_by_pair(
            all_pairs, cfg.val_ratio, cfg.test_ratio, cfg.seed
        )

    stats = {
        "pages_read": len(groups),
        "pairs_before_split": total_pairs,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
    }
    return train_pairs, val_pairs, test_pairs, stats


def _build_split_arrays(
    pairs: Sequence[TextPair],
    vocab: Vocabulary,
    max_article_len: int,
    max_dec_len: int,
    reverse_source: bool = False,
) -> SplitArrays:
    encoder_ids = []
    for p in pairs:
        # Potong dulu, baru balik urutannya (jika opsi reverse aktif).
        source_tokens = list(p.article_tokens[:max_article_len])
        if reverse_source:
            source_tokens = list(reversed(source_tokens))
        encoder_ids.append(vocab.encode_tokens(source_tokens))
    summary_ids = [
        vocab.encode_tokens(
            p.summary_tokens,
            add_sos=True,
            add_eos=True,
            max_length=max_dec_len,
        )
        for p in pairs
    ]

    # Teacher forcing yang benar:
    #   summary_ids = [<SOS>, t1, t2, ..., <EOS>]
    #   decoder_input  = [<SOS>, t1, t2, ..., t_{n-1}]
    #   decoder_target = [t1, t2, ..., t_{n-1}, <EOS>]
    # Target TIDAK boleh memuat <SOS>; jika memuat, model belajar mengeluarkan
    # <SOS> pada langkah pertama dan seluruh prediksi bergeser satu langkah.
    decoder_input_ids = [ids[:-1] for ids in summary_ids]
    decoder_target_ids = [ids[1:] for ids in summary_ids]

    encoder_inputs = pad_sequences(encoder_ids, max_article_len, pad_id=PAD_ID)
    decoder_inputs = pad_sequences(decoder_input_ids, max_dec_len, pad_id=PAD_ID)
    decoder_targets = pad_sequences(decoder_target_ids, max_dec_len, pad_id=PAD_ID)

    return SplitArrays(
        encoder_inputs=encoder_inputs,
        decoder_inputs=decoder_inputs,
        decoder_targets=decoder_targets,
    )


def prepare_data(cfg: Optional[DataConfig] = None) -> PreparedData:
    """Pipeline lengkap: baca -> split -> vocabulary -> encode -> padding."""
    cfg = cfg or DataConfig()
    train_pairs, val_pairs, test_pairs, stats = load_and_split_pairs(cfg)
    if not train_pairs:
        raise RuntimeError("Train set kosong. Periksa filter panjang atau path dataset.")

    train_tokens = [
        tokens
        for pair in train_pairs
        for tokens in (pair.article_tokens, pair.summary_tokens)
    ]
    vocab = Vocabulary.build(
        train_tokens,
        min_freq=cfg.min_freq,
        max_size=cfg.max_vocab_size,
    )

    max_dec_len = int(cfg.max_summary_len)
    train = _build_split_arrays(
        train_pairs, vocab, cfg.max_article_len, max_dec_len, cfg.reverse_source
    )
    val = _build_split_arrays(
        val_pairs, vocab, cfg.max_article_len, max_dec_len, cfg.reverse_source
    )
    test = _build_split_arrays(
        test_pairs, vocab, cfg.max_article_len, max_dec_len, cfg.reverse_source
    )

    stats.update(
        {
            "vocab_size": len(vocab),
            "max_article_len": cfg.max_article_len,
            "max_dec_len": max_dec_len,
            "min_freq": cfg.min_freq,
            "max_vocab_size": cfg.max_vocab_size,
            "split_by_page": cfg.split_by_page,
            "reverse_source": cfg.reverse_source,
        }
    )

    return PreparedData(
        config=cfg,
        vocab=vocab,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train=train,
        val=val,
        test=test,
        max_dec_len=max_dec_len,
        stats=stats,
    )


def make_tf_dataset(
    split: SplitArrays,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Bungkus array menjadi tf.data.Dataset dengan batch dan prefetch."""
    ds = tf.data.Dataset.from_tensor_slices(
        ((split.encoder_inputs, split.decoder_inputs), split.decoder_targets)
    )
    if shuffle:
        buffer_size = min(len(split), 10000)
        ds = ds.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    return ds.prefetch(tf.data.AUTOTUNE)


def describe_data(prepared: PreparedData) -> str:
    s = prepared.stats
    return (
        "Dataset | pages={pages_read} | pairs={pairs_before_split} | "
        "train={train_pairs} | val={val_pairs} | test={test_pairs} | "
        "vocab={vocab_size} | max_article_len={max_article_len} | "
        "max_dec_len={max_dec_len} | split_by_page={split_by_page} | "
        "reverse_source={reverse_source}"
    ).format(**s)
