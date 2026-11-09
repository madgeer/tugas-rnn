"""
Implementasi ROUGE-1, ROUGE-2, dan ROUGE-L tanpa dependensi eksternal.

Skor yang dilaporkan adalah rata-rata F1 per sampel (macro average), sesuai
praktik umum untuk tabel evaluasi ringkasan. Precision dan recall juga
disediakan agar dapat dianalisis lebih lanjut.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence

from preprocessing import tokenize


def _as_tokens(text_or_tokens) -> List[str]:
    if isinstance(text_or_tokens, str):
        return tokenize(text_or_tokens)
    return [str(tok) for tok in text_or_tokens]


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def rouge_n(hypothesis, reference, n: int) -> Dict[str, float]:
    """ROUGE-N dengan clipping n-gram terhadap referensi."""
    hyp = _as_tokens(hypothesis)
    ref = _as_tokens(reference)
    hyp_ngrams = _ngrams(hyp, n)
    ref_ngrams = _ngrams(ref, n)

    overlap = sum(min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items())
    precision = overlap / max(1, sum(hyp_ngrams.values()))
    recall = overlap / max(1, sum(ref_ngrams.values()))
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Panjang Longest Common Subsequence dengan O(min(n,m)) memori."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(hypothesis, reference) -> Dict[str, float]:
    """ROUGE-L berbasis LCS."""
    hyp = _as_tokens(hypothesis)
    ref = _as_tokens(reference)
    lcs = lcs_length(hyp, ref)
    precision = lcs / max(1, len(hyp))
    recall = lcs / max(1, len(ref))
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def rouge_scores(hypothesis, reference) -> Dict[str, float]:
    """ROUGE-1, ROUGE-2, dan ROUGE-L untuk satu pasangan teks."""
    r1 = rouge_n(hypothesis, reference, 1)
    r2 = rouge_n(hypothesis, reference, 2)
    rl = rouge_l(hypothesis, reference)
    return {
        "rouge1_precision": r1["precision"],
        "rouge1_recall": r1["recall"],
        "rouge1_f1": r1["f1"],
        "rouge2_precision": r2["precision"],
        "rouge2_recall": r2["recall"],
        "rouge2_f1": r2["f1"],
        "rougeL_precision": rl["precision"],
        "rougeL_recall": rl["recall"],
        "rougeL_f1": rl["f1"],
    }


def corpus_rouge(
    hypotheses: Iterable,
    references: Iterable,
) -> Dict[str, float]:
    """Rata-rata skor ROUGE di seluruh sampel uji."""
    hyp_list = list(hypotheses)
    ref_list = list(references)
    if len(hyp_list) != len(ref_list):
        raise ValueError("Jumlah hypothesis dan reference harus sama.")
    if not hyp_list:
        return {
            "rouge1_precision": 0.0,
            "rouge1_recall": 0.0,
            "rouge1_f1": 0.0,
            "rouge2_precision": 0.0,
            "rouge2_recall": 0.0,
            "rouge2_f1": 0.0,
            "rougeL_precision": 0.0,
            "rougeL_recall": 0.0,
            "rougeL_f1": 0.0,
            "n_samples": 0,
        }

    rows = [rouge_scores(h, r) for h, r in zip(hyp_list, ref_list)]
    keys = list(rows[0].keys())
    result = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    result["n_samples"] = len(rows)
    return result
