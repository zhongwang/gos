"""CPU sequence features and calibrated, coordinate-preserving evidence."""

from __future__ import annotations

import gzip
from collections import Counter
import itertools
import lzma
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import ArtifactCompatibilityError, artifact_is_certified, validate_artifact
from .preprocessing import (
    PreparedSequence,
    ScoringWindow,
    make_multiscale_subwindows,
    make_windows,
)
from .types import SegmentEvidence, Span, StageEvidence, StageStatus, WindowEvidence


_FEATURE_NAMES = ("gzip_ratio", "lzma_ratio", "stop_density")
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


@dataclass(frozen=True)
class ScoreAggregates:
    """Summary statistics for calibrated scores, retaining local evidence."""

    n_eligible: int
    n_windows: int
    length_bin_counts: Mapping[str, int]
    available: bool
    mean: float | None
    maximum: float | None
    top_k_mean: float | None
    count_over: int | None
    fraction_over: float | None
    longest_run_over: int | None


def _as_score_array(scores: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    return values


def aggregate_scores(scores: Sequence[float] | np.ndarray, threshold: float) -> ScoreAggregates:
    """Aggregate calibrated scores without allowing sparse hits to disappear."""

    values = _as_score_array(scores)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not len(values):
        return ScoreAggregates(0, 0, {}, False, None, None, None, None, None, None)

    over = values >= threshold
    longest_run = 0
    current_run = 0
    for hit in over:
        current_run = current_run + 1 if hit else 0
        longest_run = max(longest_run, current_run)
    top_k = np.partition(values, len(values) - min(3, len(values)))[-min(3, len(values)) :]
    return ScoreAggregates(
        n_eligible=int(len(values)),
        n_windows=int(len(values)),
        length_bin_counts={},
        available=True,
        mean=float(values.mean()),
        maximum=float(values.max()),
        top_k_mean=float(top_k.mean()),
        count_over=int(over.sum()),
        fraction_over=float(over.mean()),
        longest_run_over=int(longest_run),
    )


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


def _gzip_ratio(seq: str) -> float:
    return len(gzip.compress(seq.encode("ascii"), mtime=0)) / len(seq)


def _lzma_ratio(seq: str) -> float:
    return len(lzma.compress(seq.encode("ascii"))) / len(seq)


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


def _window_features(window: ScoringWindow) -> np.ndarray:
    return np.array(
        [_gzip_ratio(window.sequence), _lzma_ratio(window.sequence), _stop_density(window.sequence)],
        dtype=float,
    )


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


def merge_local_segments(
    windows: Sequence[WindowEvidence], local_threshold: float | Mapping[str, float]
) -> tuple[SegmentEvidence, ...]:
    """Merge adjacent high-score windows while preserving original coordinates."""

    if isinstance(local_threshold, Mapping):
        thresholds = {
            length_bin: _number(value, field=f"local threshold for {length_bin!r}")
            for length_bin, value in local_threshold.items()
        }
        if not thresholds:
            raise ValueError("local_threshold must not be empty")
    else:
        if not np.isfinite(local_threshold):
            raise ValueError("local_threshold must be finite")
        thresholds = None

    def threshold_for(window: WindowEvidence) -> float | None:
        if not window.eligible or window.score is None:
            return None
        return (
            thresholds.get(window.length_bin) if thresholds is not None else local_threshold
        )

    suspicious = sorted(
        (
            (window, threshold_for(window))
            for window in windows
            if threshold_for(window) is not None
            and window.score is not None
            and window.score >= threshold_for(window)
        ),
        key=lambda item: (item[0].span.start, item[0].span.end),
    )
    segments: list[SegmentEvidence] = []
    for window, threshold in suspicious:
        assert window.score is not None and threshold is not None
        strength = max(0.0, float(window.score - threshold))
        if segments and window.span.start <= segments[-1].span.end:
            previous = segments[-1]
            segments[-1] = SegmentEvidence(
                Span(previous.span.start, max(previous.span.end, window.span.end)),
                max(previous.score, window.score),
                max(previous.calibrated_strength, strength),
            )
        else:
            segments.append(
                SegmentEvidence(window.span, float(window.score), strength)
            )
    return tuple(segments)


def _artifact_error(message: str) -> ArtifactCompatibilityError:
    return ArtifactCompatibilityError(f"GOS v4 sequence artifact compatibility error: {message}")


def _number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise _artifact_error(f"{field} must be a finite number")
    return float(value)


def _parse_calibration_thresholds(
    metadata: Mapping[str, Any], calibration_kind: str
) -> dict[str, float]:
    thresholds = metadata.get("thresholds")
    calibration = (
        thresholds.get("calibration") if isinstance(thresholds, Mapping) else None
    )
    raw = (
        calibration.get(calibration_kind)
        if isinstance(calibration, Mapping)
        else None
    )
    if not isinstance(raw, Mapping) or not raw:
        raise _artifact_error(
            f"metadata must include per-length-bin {calibration_kind} calibration"
        )
    parsed: dict[str, float] = {}
    for length_bin, record in raw.items():
        if not isinstance(length_bin, str) or not length_bin:
            raise _artifact_error("length-bin threshold names must be non-empty strings")
        if not isinstance(record, Mapping):
            raise _artifact_error(
                f"calibration for length bin {length_bin!r} must be a mapping"
            )
        parsed[length_bin] = _number(
            record.get("threshold"),
            field=f"calibrated threshold for length bin {length_bin!r}",
        )
    return parsed


def parse_window_thresholds(metadata: Mapping[str, Any]) -> dict[str, float]:
    """Return canonical calibrated window thresholds by length bin."""

    return _parse_calibration_thresholds(metadata, "window")


class SequenceStage:
    """Score prepared windows with a validated, calibrated CPU artifact."""

    def __init__(
        self, artifact: Mapping[str, Any], *, allow_uncertified: bool = False
    ):
        if type(allow_uncertified) is not bool:
            raise TypeError("allow_uncertified must be an exact boolean")
        artifact = validate_artifact(artifact, expected_kind="sequence")
        self.certified = artifact_is_certified(artifact)
        if not self.certified and not allow_uncertified:
            raise _artifact_error(
                "uncertified sequence artifact requires allow_uncertified=True"
            )
        if artifact.get("feature_names") != list(_FEATURE_NAMES):
            raise _artifact_error("feature_names must be gzip_ratio, lzma_ratio, stop_density in order")

        metadata = artifact.get("metadata")
        payload = artifact.get("payload")
        if not isinstance(metadata, Mapping) or not isinstance(payload, Mapping):
            raise _artifact_error("metadata and payload must be mappings")
        self._artifact = artifact
        self._thresholds = self._parse_thresholds(metadata)
        self._local_thresholds = _parse_calibration_thresholds(metadata, "local")

        self._estimator = payload.get("estimator", payload.get("model"))
        self._scaler = payload.get("scaler")
        if not callable(getattr(self._estimator, "predict_proba", None)):
            raise _artifact_error("payload estimator must provide predict_proba")
        if not callable(getattr(self._scaler, "transform", None)):
            raise _artifact_error("payload scaler must provide transform")
        self._clip_lower = self._clip_vector(payload, "clip_lower", "clip_low")
        self._clip_upper = self._clip_vector(payload, "clip_upper", "clip_high")
        if np.any(self._clip_lower > self._clip_upper):
            raise _artifact_error("clip_lower must not exceed clip_upper")

    @staticmethod
    def _clip_vector(payload: Mapping[str, Any], primary: str, alternate: str) -> np.ndarray:
        raw = payload.get(primary, payload.get(alternate))
        try:
            vector = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as error:
            raise _artifact_error(f"{primary} must be a three-value numeric vector") from error
        if vector.shape != (len(_FEATURE_NAMES),) or not np.all(np.isfinite(vector)):
            raise _artifact_error(f"{primary} must be a three-value finite numeric vector")
        return vector

    @staticmethod
    def _parse_thresholds(metadata: Mapping[str, Any]) -> dict[str, float]:
        return parse_window_thresholds(metadata)

    def _score_window(self, window: ScoringWindow) -> float:
        features = np.clip(_window_features(window), self._clip_lower, self._clip_upper).reshape(1, -1)
        scaled = np.asarray(self._scaler.transform(features), dtype=float)
        if scaled.shape != features.shape or not np.all(np.isfinite(scaled)):
            raise ValueError("artifact scaler returned invalid feature values")
        probabilities = np.asarray(self._estimator.predict_proba(scaled), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[0] != 1:
            raise ValueError("artifact estimator returned invalid probabilities")
        classes = getattr(self._estimator, "classes_", None)
        class_index = 1
        if classes is not None:
            matches = np.flatnonzero(np.asarray(classes) == 1)
            if len(matches) != 1:
                raise ValueError("artifact estimator must expose one AI class labelled 1")
            class_index = int(matches[0])
        if probabilities.shape[1] <= class_index:
            raise ValueError("artifact estimator is missing the AI probability column")
        score = float(probabilities[0, class_index])
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("artifact estimator returned an invalid AI probability")
        return score

    @staticmethod
    def _aggregate_windows(windows: Sequence[WindowEvidence]) -> ScoreAggregates:
        eligible = [window for window in windows if window.eligible and window.score is not None]
        length_bin_counts: dict[str, int] = {}
        for window in windows:
            if window.length_bin is not None:
                length_bin_counts[window.length_bin] = length_bin_counts.get(window.length_bin, 0) + 1
        if not eligible:
            return replace(
                aggregate_scores(np.array([], dtype=float), threshold=0.0),
                n_windows=len(windows),
                length_bin_counts=length_bin_counts,
            )
        summary = aggregate_scores(np.array([window.score for window in eligible]), threshold=0.0)
        longest_run = 0
        current_run = 0
        previous_end: int | None = None
        for window in windows:
            hit = window.eligible and window.score is not None and window.over_threshold
            physically_adjacent = previous_end is not None and previous_end == window.span.start
            current_run = current_run + 1 if hit and physically_adjacent else int(hit)
            longest_run = max(longest_run, current_run)
            previous_end = window.span.end
        over = [window.over_threshold for window in eligible]
        return replace(
            summary,
            n_windows=len(windows),
            length_bin_counts=length_bin_counts,
            count_over=sum(over),
            fraction_over=sum(over) / len(over),
            longest_run_over=longest_run,
        )

    def score(self, prepared: PreparedSequence) -> StageEvidence:
        if not isinstance(prepared, PreparedSequence):
            raise TypeError("prepared must be a PreparedSequence")
        windows: list[WindowEvidence] = []
        failures = 0
        for window in make_windows(prepared):
            threshold = self._thresholds.get(window.length_bin)
            if threshold is None:
                windows.append(
                    WindowEvidence(window.span, None, eligible=False, length_bin=window.length_bin)
                )
                continue
            try:
                score = self._score_window(window)
            except Exception:
                failures += 1
                windows.append(
                    WindowEvidence(window.span, None, eligible=True, length_bin=window.length_bin)
                )
                continue
            windows.append(
                WindowEvidence(
                    window.span,
                    score,
                    over_threshold=score >= threshold,
                    eligible=True,
                    length_bin=window.length_bin,
                )
            )

        aggregates = self._aggregate_windows(windows)
        n_scored = aggregates.n_eligible
        if not aggregates.available:
            reason = "no_calibrated_windows" if windows else "no_scorable_windows"
            status = StageStatus.FAILED if failures else StageStatus.UNAVAILABLE
            return StageEvidence(
                status=status,
                reason=reason,
                aggregates=aggregates,
                windows=tuple(windows),
                n_scored=n_scored,
                n_failed=failures,
            )
        segments = merge_local_segments(windows, self._local_thresholds)
        warnings = ("some_window_scores_failed",) if failures else ()
        return StageEvidence(
            status=StageStatus.OK,
            aggregates=aggregates,
            windows=tuple(windows),
            segments=segments,
            n_scored=n_scored,
            n_failed=failures,
            warnings=warnings,
        )
