"""
Pembersihan teks, tokenisasi sederhana Bahasa Indonesia, dan Vocabulary.

Token khusus:
    <PAD> = 0, <UNK> = 1, <SOS> = 2, <EOS> = 3
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_IDS = {PAD_ID, UNK_ID, SOS_ID, EOS_ID}

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# Karakter yang dipertahankan: huruf/angka (Unicode), whitespace, dan tanda baca umum.
_ALLOWED_RE = re.compile(r"[^\w\s\.\,\?\!\:\;\-\%\/]", re.UNICODE)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def clean_text(text: object) -> str:
    """Bersihkan teks: HTML, URL, email, karakter aneh, dan whitespace."""
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _ALLOWED_RE.sub(" ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize(text: object) -> List[str]:
    """
    Tokenisasi sederhana: lowercase + regex kata Unicode.

    Contoh:
        "Cobalah untuk belajar, dengan kekurangan!" ->
        ["cobalah", "untuk", "belajar", "dengan", "kekurangan"]
    """
    return _TOKEN_RE.findall(clean_text(text))


def detokenize(tokens: Sequence[str]) -> str:
    """Gabungkan token menjadi teks biasa (tanpa tanda baca)."""
    return " ".join(tokens).strip()


class Vocabulary:
    """Vocabulary dua arah token <-> id dengan token khusus Seq2Seq."""

    def __init__(
        self,
        token_to_id: Optional[Dict[str, int]] = None,
        id_to_token: Optional[Dict[int, str]] = None,
    ) -> None:
        if token_to_id is None:
            token_to_id = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        if id_to_token is None:
            id_to_token = {idx: tok for tok, idx in token_to_id.items()}
        self.token_to_id = token_to_id
        self.id_to_token = {int(k): v for k, v in id_to_token.items()}

        # Pastikan token khusus selalu ada dan memakai id yang benar.
        for idx, tok in enumerate(SPECIAL_TOKENS):
            self.token_to_id.setdefault(tok, idx)
            self.id_to_token.setdefault(idx, tok)

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def vocab_size(self) -> int:
        return len(self)

    @classmethod
    def build(
        cls,
        token_sequences: Iterable[Sequence[str]],
        min_freq: int = 1,
        max_size: Optional[int] = 20000,
    ) -> "Vocabulary":
        """Bangun vocabulary dari kumpulan sequence token (idealnya hanya train set)."""
        counter: Counter[str] = Counter()
        for tokens in token_sequences:
            counter.update(tokens)

        items = [
            (tok, freq)
            for tok, freq in counter.items()
            if freq >= max(1, min_freq) and tok not in SPECIAL_TOKENS
        ]
        # Urutkan: frekuensi tertinggi, lalu alfabetis agar deterministik.
        items.sort(key=lambda item: (-item[1], item[0]))

        if max_size is not None:
            max_size = max(int(max_size), len(SPECIAL_TOKENS))
            items = items[: max_size - len(SPECIAL_TOKENS)]

        token_to_id = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        for tok, _ in items:
            token_to_id[tok] = len(token_to_id)
        id_to_token = {idx: tok for tok, idx in token_to_id.items()}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    def encode_tokens(
        self,
        tokens: Sequence[str],
        add_sos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Ubah token menjadi id.

        Jika max_length diberikan, panjang total (termasuk SOS/EOS) tidak
        melebihi max_length dan EOS tetap ditempatkan di akhir.
        """
        body = [self.token_to_id.get(tok, UNK_ID) for tok in tokens]
        n_special = int(add_sos) + int(add_eos)
        if max_length is not None:
            body = body[: max(0, int(max_length) - n_special)]
        ids: List[int] = []
        if add_sos:
            ids.append(SOS_ID)
        ids.extend(body)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def encode_text(
        self,
        text: str,
        add_sos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        return self.encode_tokens(
            tokenize(text),
            add_sos=add_sos,
            add_eos=add_eos,
            max_length=max_length,
        )

    def decode(
        self,
        ids: Sequence[int],
        skip_special: bool = True,
    ) -> str:
        tokens: List[str] = []
        for idx in ids:
            idx = int(idx)
            if skip_special and idx in SPECIAL_IDS:
                continue
            tokens.append(self.id_to_token.get(idx, "<UNK>"))
        return detokenize(tokens)

    def decode_tokens(self, ids: Sequence[int], skip_special: bool = True) -> List[str]:
        tokens: List[str] = []
        for idx in ids:
            idx = int(idx)
            if skip_special and idx in SPECIAL_IDS:
                continue
            tokens.append(self.id_to_token.get(idx, "<UNK>"))
        return tokens

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "special_tokens": SPECIAL_TOKENS,
            "token_to_id": self.token_to_id,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        token_to_id = {str(k): int(v) for k, v in payload["token_to_id"].items()}
        return cls(token_to_id=token_to_id)


def pad_sequences(
    sequences: Sequence[Sequence[int]],
    max_length: int,
    pad_id: int = PAD_ID,
    dtype=np.int32,
) -> np.ndarray:
    """Padding/truncation sequence menjadi array 2D dengan panjang tetap."""
    max_length = int(max_length)
    arr = np.full((len(sequences), max_length), pad_id, dtype=dtype)
    for i, seq in enumerate(sequences):
        n = min(len(seq), max_length)
        if n:
            arr[i, :n] = np.asarray(seq[:n], dtype=dtype)
    return arr
