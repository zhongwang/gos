"""Short-window composition and entropy features for v4 routing."""

from __future__ import annotations

from collections import Counter
import itertools
import math
import zlib
from typing import Any

import numpy as np

from .preprocessing import PreparedSequence


LOCAL_COMPLEXITY_FEATURE_NAMES = (
    "local_complexity_available",
    "local_gc_fraction",
    "local_nt_entropy_mean",
    "local_nt_entropy_min",
    "local_dimer_entropy_mean",
    "local_dimer_entropy_min",
    "local_trimer_entropy_mean",
    "local_trimer_entropy_min",
    "local_zlib_ratio_mean",
    "local_zlib_ratio_min",
)

_COMPLEMENT = str.maketrans("ACGT", "TGCA")
_CANONICAL_KMERS = {
    kmer: min(kmer, kmer.translate(_COMPLEMENT)[::-1])
    for size in (2, 3)
    for kmer in ("".join(item) for item in itertools.product("ACGT", repeat=size))
}


def _reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _subwindow_offsets(length: int, *, size: int, stride: int) -> tuple[int, ...]:
    if length < size:
        return ()
    offsets = list(range(0, length - size + 1, stride))
    terminal = length - size
    if terminal not in offsets:
        offsets.append(terminal)
    return tuple(offsets)


def _shannon_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return float(
        -sum((count / total) * math.log2(count / total) for count in counts.values())
    )


def _canonical_kmer_entropy(seq: str, k: int) -> float:
    return _shannon_entropy(
        Counter(
            _CANONICAL_KMERS[seq[offset : offset + k]]
            for offset in range(len(seq) - k + 1)
        )
    )


def _raw_zlib_ratio(seq: str) -> float:
    return len(zlib.compress(seq.encode("ascii"), level=9, wbits=-15)) / len(seq)


def _rc_symmetric_zlib_ratio(seq: str) -> float:
    return 0.5 * (_raw_zlib_ratio(seq) + _raw_zlib_ratio(_reverse_complement(seq)))


def _missing_features() -> dict[str, Any]:
    return {
        name: 0.0 if name == "local_complexity_available" else None
        for name in LOCAL_COMPLEXITY_FEATURE_NAMES
    }


def local_complexity_features(
    prepared: PreparedSequence, *, window_size: int = 100, stride: int = 50
) -> dict[str, Any]:
    """Summarize local, reverse-complement-invariant complexity evidence."""

    if not isinstance(prepared, PreparedSequence):
        raise TypeError("prepared must be a PreparedSequence")
    if type(window_size) is not int or window_size < 3:
        raise ValueError("window_size must be an integer of at least 3")
    if type(stride) is not int or stride < 1:
        raise ValueError("stride must be a positive integer")

    subwindows: list[str] = []
    fragments: list[str] = []
    for fragment in prepared.fragments:
        fragments.append(fragment.sequence)
        for offset in _subwindow_offsets(
            fragment.length, size=window_size, stride=stride
        ):
            subwindows.append(fragment.sequence[offset : offset + window_size])

    if not subwindows:
        return _missing_features()

    nt_entropy: list[float] = []
    dimer_entropy: list[float] = []
    trimer_entropy: list[float] = []
    zlib_ratio: list[float] = []
    for window in subwindows:
        nt_entropy.append(_shannon_entropy(Counter(window)))
        dimer_entropy.append(_canonical_kmer_entropy(window, 2))
        trimer_entropy.append(_canonical_kmer_entropy(window, 3))
        zlib_ratio.append(_rc_symmetric_zlib_ratio(window))

    acgt_sequence = "".join(fragments)
    gc_count = acgt_sequence.count("G") + acgt_sequence.count("C")
    return {
        "local_complexity_available": 1.0,
        "local_gc_fraction": gc_count / len(acgt_sequence),
        "local_nt_entropy_mean": float(np.mean(nt_entropy)),
        "local_nt_entropy_min": float(np.min(nt_entropy)),
        "local_dimer_entropy_mean": float(np.mean(dimer_entropy)),
        "local_dimer_entropy_min": float(np.min(dimer_entropy)),
        "local_trimer_entropy_mean": float(np.mean(trimer_entropy)),
        "local_trimer_entropy_min": float(np.min(trimer_entropy)),
        "local_zlib_ratio_mean": float(np.mean(zlib_ratio)),
        "local_zlib_ratio_min": float(np.min(zlib_ratio)),
    }
