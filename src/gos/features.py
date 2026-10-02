"""The production detector's 47 CPU summary features."""
from __future__ import annotations

from collections import Counter
import gzip
import itertools
import lzma
import math
from numbers import Real
from typing import Any

import numpy as np

from .preprocessing import PreparedSequence, ScoringWindow, make_multiscale_subwindows, prepare_sequence

MULTISCALE_SUBWINDOW_FEATURE_NAMES = (
    "gzip_payload_ratio",
    "lzma_payload_ratio",
    "stop_density",
    "nt_entropy",
    "dimer_entropy",
    "trimer_entropy",
    "sixmer_entropy",
    "sixmer_unique_fraction",
    "linear_kmer_periodicity",
)


MULTISCALE_SUMMARY_NAMES = (
    "multiscale_available",
    "multiscale_n_windows",
) + tuple(
    f"multiscale_{statistic}_{feature}"
    for statistic in ("mean", "max", "std", "min", "top_k_mean")
    for feature in MULTISCALE_SUBWINDOW_FEATURE_NAMES
)


_BPE_LAGS = (5, 6, 10, 12, 15, 18, 21, 24, 30)


_NON_BPE_LAGS = (2, 3, 4, 7, 8, 9, 11, 13, 14)


_STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})


_COMPLEMENT = str.maketrans("ACGT", "TGCA")


_SIXMER_ALPHABET = tuple(
    "".join(kmer) for kmer in itertools.product("ACGT", repeat=6)
)


_GZIP_EMPTY_BYTES = len(gzip.compress(b"", mtime=0))


_LZMA_EMPTY_BYTES = len(lzma.compress(b""))


def linear_kmer_periodicity(seq: str) -> float:
    """Return the diagnostic BPE-lag contrast using linear autocorrelation."""

    if not isinstance(seq, str):
        raise TypeError("seq must be a string")
    sequence = seq.upper()
    n_bases = len(sequence)
    if n_bases < 2:
        return 0.0

    mean_autocorrelation = np.zeros(n_bases, dtype=float)
    for base in "ACGT":
        channel = np.fromiter((letter == base for letter in sequence), dtype=float)
        channel -= channel.mean()
        correlation = np.correlate(channel, channel, mode="full")
        zero_lag = correlation[n_bases - 1]
        if zero_lag > 0.0:
            mean_autocorrelation += correlation[n_bases - 1 :] / zero_lag
    mean_autocorrelation /= 4.0

    bpe = [mean_autocorrelation[lag] for lag in _BPE_LAGS if lag < n_bases]
    non_bpe = [mean_autocorrelation[lag] for lag in _NON_BPE_LAGS if lag < n_bases]
    if not bpe or not non_bpe:
        return 0.0
    return float(np.mean(bpe) - np.mean(non_bpe))


def _gzip_payload_ratio(seq: str) -> float:
    return (len(gzip.compress(seq.encode("ascii"), mtime=0)) - _GZIP_EMPTY_BYTES) / len(seq)


def _lzma_payload_ratio(seq: str) -> float:
    return (len(lzma.compress(seq.encode("ascii"))) - _LZMA_EMPTY_BYTES) / len(seq)


def _stop_density(seq: str) -> float:
    def count_stops(sequence: str) -> tuple[int, int]:
        stops = 0
        codons = 0
        for frame in range(3):
            for offset in range(frame, len(sequence) - 2, 3):
                codons += 1
                stops += sequence[offset : offset + 3] in _STOP_CODONS
        return stops, codons

    forward_stops, forward_codons = count_stops(seq)
    reverse_stops, reverse_codons = count_stops(seq.translate(_COMPLEMENT)[::-1])
    total_codons = forward_codons + reverse_codons
    return (forward_stops + reverse_stops) / total_codons if total_codons else 0.0


def _shannon_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return float(
        -sum((count / total) * math.log2(count / total) for count in counts.values())
    )


def _kmer_counts(sequence: str, k: int) -> Counter[str]:
    return Counter(
        sequence[offset : offset + k] for offset in range(len(sequence) - k + 1)
    )


def _sixmer_unique_fraction(sequence: str) -> float:
    if len(sequence) < 6:
        return 0.0
    return len(_kmer_counts(sequence, 6)) / len(_SIXMER_ALPHABET)


def _multiscale_subwindow_features(window: ScoringWindow) -> np.ndarray:
    sequence = window.sequence
    sixmer_counts = _kmer_counts(sequence, 6)
    return np.asarray(
        [
            _gzip_payload_ratio(sequence),
            _lzma_payload_ratio(sequence),
            _stop_density(sequence),
            _shannon_entropy(Counter(sequence)),
            _shannon_entropy(_kmer_counts(sequence, 2)),
            _shannon_entropy(_kmer_counts(sequence, 3)),
            _shannon_entropy(sixmer_counts),
            _sixmer_unique_fraction(sequence),
            linear_kmer_periodicity(sequence),
        ],
        dtype=float,
    )


def _missing_multiscale_features() -> dict[str, Any]:
    return {
        name: 0.0 if name in {"multiscale_available", "multiscale_n_windows"} else None
        for name in MULTISCALE_SUMMARY_NAMES
    }


def multiscale_summary_features(
    prepared: PreparedSequence,
    *,
    window_size: int = 500,
    stride: int = 250,
    top_k: int = 3,
) -> dict[str, Any]:
    """Summarize length-normalized 500 bp CPU features over sub-windows."""

    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    windows = make_multiscale_subwindows(
        prepared,
        size=window_size,
        stride=stride,
        min_length=window_size,
    )
    if not windows:
        return _missing_multiscale_features()

    matrix = np.vstack([_multiscale_subwindow_features(window) for window in windows])
    summaries: dict[str, Any] = {
        "multiscale_available": 1.0,
        "multiscale_n_windows": int(matrix.shape[0]),
    }
    sorted_values = np.sort(matrix, axis=0)
    top_values = sorted_values[-min(top_k, matrix.shape[0]) :, :]
    statistics = {
        "mean": np.mean(matrix, axis=0),
        "max": np.max(matrix, axis=0),
        "std": np.std(matrix, axis=0),
        "min": np.min(matrix, axis=0),
        "top_k_mean": np.mean(top_values, axis=0),
    }
    for statistic, values in statistics.items():
        for feature, value in zip(
            MULTISCALE_SUBWINDOW_FEATURE_NAMES,
            values,
            strict=True,
        ):
            summaries[f"multiscale_{statistic}_{feature}"] = float(value)
    return summaries



def extract_cpu_features(sequence: str) -> np.ndarray:
    """Match production's float32 feature boundary before float64 scaling."""
    features = multiscale_summary_features(prepare_sequence(sequence))
    values = [features[name] for name in MULTISCALE_SUMMARY_NAMES]
    if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) for v in values):
        raise ValueError("record has no eligible 500-character fragment or has non-finite features")
    return np.asarray(values, dtype=np.float32)
