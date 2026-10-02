"""Alphabet-aware preparation and coordinate-safe sequence windows."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .types import Span


_ACGT_FRAGMENT = re.compile(r"[ACGT]+")
_SUPPORTED_POLICIES = frozenset({"split", "reject"})


@dataclass(frozen=True)
class PreparedFragment:
    """One eligible ACGT-only region, located in the original query."""

    span: Span
    sequence: str

    @property
    def length(self) -> int:
        return self.span.length


@dataclass(frozen=True)
class IgnoredSpan:
    """One original-query region excluded from scoring, with its reason."""

    span: Span
    reason: str


@dataclass(frozen=True)
class PreparedSequence:
    """Prepared ACGT fragments and coverage accounting for one query."""

    fragments: tuple[PreparedFragment, ...]
    ignored_spans: tuple[IgnoredSpan, ...]
    acgt_bases: int
    scorable_bases: int
    acgt_coverage: float
    scorable_coverage: float
    length: int


@dataclass(frozen=True)
class ScoringWindow:
    """A scoreable original-query window from one prepared fragment."""

    span: Span
    sequence: str
    length_bin: str

    @property
    def length(self) -> int:
        return self.span.length


def prepare_sequence(
    seq: str, *, policy: str = "split", min_fragment_length: int = 500
) -> PreparedSequence:
    """Normalize one query and select scoreable ACGT-only fragments."""

    if not isinstance(seq, str):
        raise TypeError("seq must be a string")
    if not seq:
        raise ValueError("sequence must not be empty")
    if policy not in _SUPPORTED_POLICIES:
        raise ValueError(f"unknown preprocessing policy: {policy!r}")
    if not isinstance(min_fragment_length, int) or min_fragment_length < 1:
        raise ValueError("min_fragment_length must be a positive integer")

    normalized = seq.upper()
    invalid_positions = [
        index for index, base in enumerate(normalized) if base not in {"A", "C", "G", "T"}
    ]
    if policy == "reject" and invalid_positions:
        raise ValueError(f"invalid base at position {invalid_positions[0]}")

    fragments: list[PreparedFragment] = []
    ignored_spans: list[IgnoredSpan] = []
    acgt_bases = 0
    cursor = 0
    for match in _ACGT_FRAGMENT.finditer(normalized):
        start, end = match.span()
        if cursor < start:
            ignored_spans.append(IgnoredSpan(Span(cursor, start), "invalid_alphabet"))
        acgt_bases += end - start
        if end - start >= min_fragment_length:
            fragments.append(PreparedFragment(Span(start, end), match.group()))
        else:
            ignored_spans.append(
                IgnoredSpan(Span(start, end), "below_min_fragment_length")
            )
        cursor = end
    if cursor < len(normalized):
        ignored_spans.append(
            IgnoredSpan(Span(cursor, len(normalized)), "invalid_alphabet")
        )

    scorable_bases = sum(fragment.length for fragment in fragments)
    return PreparedSequence(
        fragments=tuple(fragments),
        ignored_spans=tuple(ignored_spans),
        acgt_bases=acgt_bases,
        scorable_bases=scorable_bases,
        acgt_coverage=acgt_bases / len(normalized),
        scorable_coverage=scorable_bases / len(normalized),
        length=len(normalized),
    )


def make_windows(
    prepared: PreparedSequence, *, size: int = 1000, min_length: int = 500
) -> tuple[ScoringWindow, ...]:
    """Return non-overlapping scoreable windows without changing coordinates."""

    if not isinstance(prepared, PreparedSequence):
        raise TypeError("prepared must be a PreparedSequence")
    if type(min_length) is not int or min_length < 500:
        raise ValueError("min_length must be an integer of at least 500")
    if type(size) is not int or not min_length <= size <= 1000:
        raise ValueError("size must be an integer between min_length and 1000")

    windows: list[ScoringWindow] = []
    for fragment in prepared.fragments:
        for offset in range(0, fragment.length, size):
            window_length = min(size, fragment.length - offset)
            if window_length < min_length:
                continue
            windows.append(
                ScoringWindow(
                    span=Span(fragment.span.start + offset, fragment.span.start + offset + window_length),
                    sequence=fragment.sequence[offset : offset + window_length],
                    length_bin=_length_bin(window_length),
                )
            )
    return tuple(windows)


def pav_candidate_windows(
    prepared: PreparedSequence,
    *,
    size: int = 1000,
    stride: int = 500,
    min_length: int = 500,
) -> tuple[ScoringWindow, ...]:
    """Return the deterministic pAV candidates used by training and inference.

    Long fragments use half-overlapping windows plus a terminal window.  The
    terminal candidate makes every eligible base part of at least one pAV
    attempt, including a residual shorter than ``min_length`` at the end of a
    long fragment.  Short eligible fragments are attempted once in full.
    """

    if not isinstance(prepared, PreparedSequence):
        raise TypeError("prepared must be a PreparedSequence")
    if type(min_length) is not int or min_length < 500:
        raise ValueError("min_length must be an integer of at least 500")
    if type(size) is not int or not min_length <= size <= 1000:
        raise ValueError("size must be an integer between min_length and 1000")
    if type(stride) is not int or not 1 <= stride <= size:
        raise ValueError("stride must be an integer between 1 and size")

    windows: list[ScoringWindow] = []
    for fragment in prepared.fragments:
        if fragment.length < min_length:
            continue
        if fragment.length <= size:
            offsets = [0]
        else:
            offsets = list(range(0, fragment.length - size + 1, stride))
            terminal = fragment.length - size
            if terminal not in offsets:
                offsets.append(terminal)
        for offset in offsets:
            length = min(size, fragment.length - offset)
            windows.append(
                ScoringWindow(
                    span=Span(
                        fragment.span.start + offset,
                        fragment.span.start + offset + length,
                    ),
                    sequence=fragment.sequence[offset : offset + length],
                    length_bin=_length_bin(length),
                )
            )
    return tuple(windows)


def make_multiscale_subwindows(
    prepared: PreparedSequence,
    *,
    size: int = 500,
    stride: int = 250,
    min_length: int = 500,
) -> tuple[ScoringWindow, ...]:
    """Return deterministic sliding sub-windows for local CPU summaries."""

    if not isinstance(prepared, PreparedSequence):
        raise TypeError("prepared must be a PreparedSequence")
    if type(min_length) is not int or min_length < 1:
        raise ValueError("min_length must be a positive integer")
    if type(size) is not int or size < min_length:
        raise ValueError("size must be an integer of at least min_length")
    if type(stride) is not int or not 1 <= stride <= size:
        raise ValueError("stride must be an integer between 1 and size")

    windows: list[ScoringWindow] = []
    for fragment in prepared.fragments:
        if fragment.length < min_length:
            continue
        if fragment.length <= size:
            offsets = [0]
        else:
            offsets = list(range(0, fragment.length - size + 1, stride))
            terminal = fragment.length - size
            if terminal not in offsets:
                offsets.append(terminal)
        for offset in offsets:
            length = min(size, fragment.length - offset)
            if length < min_length:
                continue
            windows.append(
                ScoringWindow(
                    span=Span(
                        fragment.span.start + offset,
                        fragment.span.start + offset + length,
                    ),
                    sequence=fragment.sequence[offset : offset + length],
                    length_bin=_length_bin(length),
                )
            )
    return tuple(windows)


def _length_bin(length: int) -> str:
    if length >= 1000:
        return "1000"
    if length >= 750:
        return "750-999"
    return "500-749"
