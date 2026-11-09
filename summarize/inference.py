"""
Inference autoregressive (greedy decoding) untuk model Seq2Seq Vanilla RNN.

Contoh pemakaian:
    python summarize.py --experiment outputs/experiments/adam_lr0.001_h128 \
        --text "Cara agar akur dengan pasangan adalah ..."

Atau dari Python:
    from inference import Summarizer
    s = Summarizer.from_experiment_dir("outputs/experiments/adam_lr0.001_h128")
    print(s.summarize("..."))
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import tensorflow as tf

from config import DEFAULT_OUTPUT_DIR, ExperimentConfig
from model import build_model
from preprocessing import PAD_ID, SOS_ID, Vocabulary, pad_sequences, tokenize


class Summarizer:
    """Wrapper model + vocabulary + konfigurasi untuk inference."""

    def __init__(
        self,
        model,
        vocab: Vocabulary,
        max_article_len: int,
        max_dec_len: int,
        experiment_config: Optional[ExperimentConfig] = None,
        reverse_source: bool = False,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.max_article_len = int(max_article_len)
        self.max_dec_len = int(max_dec_len)
        self.experiment_config = experiment_config
        self.reverse_source = bool(reverse_source)

    @classmethod
    def from_experiment_dir(cls, experiment_dir: str | Path) -> "Summarizer":
        experiment_dir = Path(experiment_dir)
        with (experiment_dir / "model_config.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)

        exp_dict = meta["experiment"]
        valid_fields = {f.name for f in fields(ExperimentConfig)}
        exp_dict = {k: v for k, v in exp_dict.items() if k in valid_fields}
        exp_cfg = ExperimentConfig(**exp_dict)

        vocab = Vocabulary.load(experiment_dir / "vocab.json")
        max_article_len = int(meta["data"]["max_article_len"])
        max_dec_len = int(meta["data"]["max_dec_len"])
        reverse_source = bool(meta["data"].get("reverse_source", False))

        model = build_model(
            vocab_size=len(vocab),
            exp_cfg=exp_cfg,
            max_article_len=max_article_len,
            max_dec_len=max_dec_len,
        )
        model.load_weights(str(experiment_dir / "model.weights.h5"))
        return cls(
            model,
            vocab,
            max_article_len,
            max_dec_len,
            exp_cfg,
            reverse_source=reverse_source,
        )

    def _encode_articles(self, texts: Sequence[str]) -> np.ndarray:
        all_tokens = [tokenize(text) for text in texts]
        ids = []
        for tokens in all_tokens:
            tokens = list(tokens[: self.max_article_len])
            if self.reverse_source:
                tokens = list(reversed(tokens))
            ids.append(self.vocab.encode_tokens(tokens))
        return pad_sequences(ids, self.max_article_len, pad_id=PAD_ID)

    def summarize_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
    ) -> List[str]:
        encoder_inputs = self._encode_articles(texts)
        return generate_summaries(
            model=self.model,
            encoder_inputs=encoder_inputs,
            vocab=self.vocab,
            max_dec_len=self.max_dec_len,
            batch_size=batch_size,
        )

    def summarize(self, text: str) -> str:
        return self.summarize_batch([text], batch_size=1)[0]


def generate_summaries(
    model,
    encoder_inputs: np.ndarray,
    vocab: Vocabulary,
    max_dec_len: int,
    batch_size: int = 64,
) -> List[str]:
    """
    Greedy autoregressive decoding secara batch.

    Pada setiap langkah, token hasil prediksi sebelumnya diumpankan kembali ke
    decoder bersama hidden state terakhir (many-to-many decoding).
    """
    results: List[str] = []
    n_total = len(encoder_inputs)
    max_steps = max(0, int(max_dec_len) - 1)

    for start in range(0, n_total, max(1, batch_size)):
        batch = encoder_inputs[start : start + batch_size]
        n = len(batch)
        if n == 0:
            continue

        encoder_batch = tf.convert_to_tensor(batch, dtype=tf.int32)
        state = model.encoder(encoder_batch, training=False)

        token_ids = np.full((n, 1), SOS_ID, dtype=np.int32)
        finished = np.zeros((n,), dtype=bool)
        generated: List[List[int]] = [[] for _ in range(n)]

        for _ in range(max_steps):
            if bool(finished.all()):
                break
            tokens = tf.convert_to_tensor(token_ids, dtype=tf.int32)
            logits, state = model.decoder(
                tokens, state, training=False, return_state=True
            )
            next_ids = tf.argmax(logits[:, -1, :], axis=-1).numpy().astype(np.int32)

            for i, token_id in enumerate(next_ids):
                if finished[i]:
                    continue
                token_id = int(token_id)
                if token_id == 0:  # <PAD>
                    finished[i] = True
                elif token_id == 1:  # <UNK>; tetap dicatat sebagai token UNK
                    generated[i].append(token_id)
                elif token_id == 3:  # <EOS>
                    finished[i] = True
                else:
                    generated[i].append(token_id)

            token_ids = next_ids.reshape(n, 1)

        results.extend(vocab.decode(tokens, skip_special=True) for tokens in generated)

    return results


def _read_input_text(args) -> str:
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    return input("Masukkan teks artikel: ").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference ringkasan Seq2Seq Vanilla RNN")
    parser.add_argument(
        "--experiment",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "best_model"),
        help="Direktori hasil eksperimen (berisi model.weights.h5, vocab.json, model_config.json)",
    )
    parser.add_argument("--text", type=str, default=None, help="Teks artikel")
    parser.add_argument("--file", type=str, default=None, help="File teks artikel")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    summarizer = Summarizer.from_experiment_dir(args.experiment)
    text = _read_input_text(args)
    summary = summarizer.summarize(text)
    print("\n=== ARTIKEL ===")
    print(text)
    print("\n=== RINGKASAN ===")
    print(summary if summary else "(model tidak menghasilkan token)")


if __name__ == "__main__":
    main()
