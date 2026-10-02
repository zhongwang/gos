"""Source-disjoint splits and conservative calibration for GOS v4.

This module deliberately keeps source membership, operating-point selection,
and natural-support checks separate from model fitting.  Its outputs are
metadata, not provenance predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, sqrt
from numbers import Real
from typing import Iterable, Sequence

import numpy as np


SPLIT_ROLES = frozenset({"train", "calibration", "test"})
_WILSON_Z_95 = 1.959963984540054
_RATE_TOLERANCE_ULPS = 16.0


class CalibrationError(ValueError):
    """Raised when a requested calibration claim lacks sufficient evidence."""


@dataclass(frozen=True)
class ThresholdCalibration:
    """Auditable empirical operating point selected on held-out scores."""

    threshold: float
    target_fpr: float
    achieved_fpr: float
    sample_count: int
    record_count: int
    sample_unit: str
    confidence_interval: tuple[float, float]
    confidence_method: str
    quantile_method: str
    certified: bool
    required_sample_count: int

    @property
    def n(self) -> int:
        """Compatibility shorthand for the number of calibration scores."""

        return self.sample_count

    @property
    def wilson_interval(self) -> tuple[float, float]:
        """Backward-compatible name for the recorded 95% interval."""

        return self.confidence_interval

    @property
    def certification_status(self) -> dict[str, bool | int]:
        """JSON-ready certification metadata for a versioned artifact."""

        return {
            "certified": self.certified,
            "required_sample_count": self.required_sample_count,
        }


@dataclass(frozen=True)
class NaturalThresholdCalibration:
    """Auditable natural-call boundary constrained by false omission."""

    threshold: float
    target_false_omission_rate: float
    achieved_false_omission_rate: float
    natural_precision: float
    sample_count: int
    calibration_source_count: int
    record_count: int
    natural_call_count: int
    false_omission_count: int
    natural_call_source_count: int
    sample_unit: str
    confidence_interval: tuple[float, float]
    confidence_method: str
    quantile_method: str
    certified: bool
    required_sample_count: int

    @property
    def target_fnr(self) -> float:
        """Compatibility alias; the value is a false-omission target, not FNR."""

        return self.target_false_omission_rate

    @property
    def achieved_fnr(self) -> float:
        """Compatibility alias; the value is false omission, not AI-class FNR."""

        return self.achieved_false_omission_rate

    @property
    def wilson_interval(self) -> tuple[float, float]:
        return self.confidence_interval

    @property
    def certification_status(self) -> dict[str, bool | int]:
        return {
            "certified": self.certified,
            "required_sample_count": self.required_sample_count,
        }


@dataclass(frozen=True)
class ManifestRecord:
    """One source sequence in a training/calibration/test manifest."""

    path: str
    label: int
    source_id: str
    role: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("path must be a non-empty string")
        if type(self.label) is not int or self.label not in {0, 1}:
            raise ValueError("label must be an exact integer in {0, 1}")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be a non-empty string")
        if self.role is not None and self.role not in SPLIT_ROLES:
            raise ValueError("role must be train, calibration, test, or None")


def validate_split_roles(records: Iterable[ManifestRecord]) -> tuple[ManifestRecord, ...]:
    """Validate that each source belongs to exactly one explicit split role."""

    materialized = tuple(records)
    source_roles: dict[str, str] = {}
    source_labels: dict[str, int] = {}
    for record in materialized:
        if not isinstance(record, ManifestRecord):
            raise TypeError("records must contain ManifestRecord instances")
        if record.role is None:
            raise ValueError(f"source {record.source_id!r} has no split role")
        previous_role = source_roles.setdefault(record.source_id, record.role)
        if previous_role != record.role:
            raise ValueError(
                f"source_id {record.source_id!r} appears in both {previous_role!r} "
                f"and {record.role!r} roles"
            )
        previous_label = source_labels.setdefault(record.source_id, record.label)
        if previous_label != record.label:
            raise ValueError(f"source_id {record.source_id!r} has conflicting labels")
    missing_roles = sorted(SPLIT_ROLES - set(source_roles.values()))
    if missing_roles:
        raise ValueError(f"manifest is missing required split roles: {', '.join(missing_roles)}")
    return materialized


def grouped_split(records: Iterable[ManifestRecord], seed: int) -> tuple[ManifestRecord, ...]:
    """Assign unsplit source groups to train/calibration/test at 60/20/20.

    Source identifiers are sorted before the seeded NumPy permutation, so the
    assignment does not depend on manifest row ordering.  For three or more
    groups, floor cut points at 60% and 80% keep every role non-empty.
    """

    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")
    materialized = tuple(records)
    if any(not isinstance(record, ManifestRecord) for record in materialized):
        raise TypeError("records must contain ManifestRecord instances")
    roles = {record.role for record in materialized}
    if not roles or roles == {None}:
        source_ids = sorted({record.source_id for record in materialized})
        if len(source_ids) < 3:
            raise ValueError("at least three source groups are required for a 60/20/20 split")
        shuffled = np.random.default_rng(seed).permutation(source_ids)
        train_end = int(np.floor(0.6 * len(shuffled)))
        calibration_end = int(np.floor(0.8 * len(shuffled)))
        assigned_roles = {
            source_id: (
                "train"
                if index < train_end
                else "calibration"
                if index < calibration_end
                else "test"
            )
            for index, source_id in enumerate(shuffled.tolist())
        }
        split = tuple(replace(record, role=assigned_roles[record.source_id]) for record in materialized)
        return validate_split_roles(split)
    if None in roles:
        raise ValueError("split roles must be either all assigned or all absent")
    return validate_split_roles(materialized)


def wilson_interval(k: int, n: int) -> tuple[float, float]:
    """Return the two-sided 95% Wilson interval for ``k / n``."""

    if type(k) is not int or type(n) is not int:
        raise TypeError("k and n must be exact integers")
    if n <= 0:
        raise ValueError("n must be positive")
    if k < 0 or k > n:
        raise ValueError("k must satisfy 0 <= k <= n")
    observed = k / n
    z_squared = _WILSON_Z_95**2
    denominator = 1.0 + z_squared / n
    center = (observed + z_squared / (2.0 * n)) / denominator
    radius = (_WILSON_Z_95 / denominator) * sqrt(
        observed * (1.0 - observed) / n + z_squared / (4.0 * n**2)
    )
    # Avoid exposing floating-point crumbs such as 2e-19 at the exact
    # binomial boundaries.  Besides being the mathematically correct closed
    # endpoint, this keeps serialized calibration intervals auditable against
    # an achieved rate of exactly zero or one.
    lower = 0.0 if k == 0 else max(0.0, center - radius)
    upper = 1.0 if k == n else min(1.0, center + radius)
    return lower, upper


def _validate_target_fpr(target_fpr: Real) -> float:
    if isinstance(target_fpr, (bool, np.bool_)) or not isinstance(target_fpr, Real):
        raise TypeError("target_fpr must be a real number, not a boolean")
    value = float(target_fpr)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("target_fpr must be finite and strictly between 0 and 1")
    return value


def rate_tolerance(target: Real) -> float:
    """Return the machine-scale tolerance used for empirical rate constraints.

    Equal-weight summation can place an exact count boundary a few ULPs above
    its mathematical value (for example, 10/1000 as
    ``0.010000000000000002``).  Sixteen float epsilons are enough to absorb
    that arithmetic noise without relaxing a scientific operating point.
    """

    value = float(target)
    return _RATE_TOLERANCE_ULPS * np.finfo(float).eps * max(1.0, abs(value))


def _rate_at_or_below(achieved: float, target: float) -> bool:
    return achieved <= target + rate_tolerance(target)


def _natural_scores(scores: object, *, name: str = "natural_scores") -> np.ndarray:
    values = np.asarray(scores)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if _contains_boolean(values):
        raise TypeError(f"{name} must be numeric probabilities, not booleans")
    try:
        values = values.astype(float, copy=False)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric probabilities") from error
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} must be finite probabilities in [0, 1]")
    return values


def _contains_boolean(values: np.ndarray) -> bool:
    """Reject booleans even when mixed into an object-dtype NumPy array."""

    return bool(
        np.issubdtype(values.dtype, np.bool_)
        or (values.dtype == object and any(isinstance(item, (bool, np.bool_)) for item in values.flat))
    )


def calibrate_ai_threshold(
    natural_scores: object,
    target_fpr: Real,
    allow_uncertified: bool = False,
    *,
    source_ids: Sequence[str] | None = None,
    sample_unit: str = "full_sequence_record",
) -> ThresholdCalibration:
    """Calibrate an AI-call threshold from held-out full natural sequences.

    The ``higher`` quantile is intentionally retained in the returned metadata.
    Scores equal to the threshold count as AI calls, so the achieved empirical
    FPR is measured with ``>=`` and may exceed the target when scores tie.
    """

    if type(allow_uncertified) is not bool:
        raise TypeError("allow_uncertified must be an exact boolean")
    target = _validate_target_fpr(target_fpr)
    values = _natural_scores(natural_scores)
    blocks, weights = _source_blocks(source_ids, len(values))
    if not isinstance(sample_unit, str) or not sample_unit:
        raise ValueError("sample_unit must be a non-empty string")
    required = ceil(10.0 / target)
    enough_samples = len(blocks) >= required
    if not enough_samples and not allow_uncertified:
        raise CalibrationError(
            f"at least {required} natural calibration sequences are required "
            f"to certify target FPR {target:g}; received {len(blocks)}"
        )
    threshold, achieved, empirical_candidate = _constrained_upper_threshold(
        values, weights, target
    )
    original_achieved = achieved
    if not empirical_candidate and allow_uncertified and float(np.max(values)) < 1.0:
        threshold = float(np.nextafter(np.max(values), np.inf))
        achieved = 0.0
    source_rates = np.asarray(
        [float(np.mean(values[indexes] >= threshold)) for indexes in blocks.values()],
        dtype=float,
    )
    interval, interval_method = _source_rate_interval(source_rates)
    meets_constraint = _rate_at_or_below(achieved, target)
    if not empirical_candidate and not allow_uncertified:
        raise CalibrationError(
            f"achieved FPR {original_achieved:g} exceeds target FPR {target:g}; "
            "ties leave no certifiable threshold"
        )
    return ThresholdCalibration(
        threshold=threshold,
        target_fpr=target,
        achieved_fpr=achieved,
        sample_count=len(blocks),
        record_count=int(values.size),
        sample_unit=sample_unit,
        confidence_interval=interval,
        confidence_method=interval_method,
        quantile_method=(
            "higher" if source_ids is None else "source_equal_weighted_higher"
        ),
        certified=bool(enough_samples and meets_constraint and empirical_candidate),
        required_sample_count=required,
    )


def calibrate_natural_threshold(
    scores: object,
    labels: object,
    target_false_omission_rate: Real,
    allow_uncertified: bool = False,
    *,
    source_ids: Sequence[str] | None = None,
    sample_unit: str = "full_sequence_record",
    max_threshold: Real | None = None,
) -> NaturalThresholdCalibration:
    """Calibrate natural precision on combined labeled calibration records.

    The false-omission rate is ``AI called natural / all records called
    natural``.  This is distinct from an AI-only false-negative rate and
    therefore requires both labels at calibration time.
    """

    if type(allow_uncertified) is not bool:
        raise TypeError("allow_uncertified must be an exact boolean")
    target = _validate_target_fpr(target_false_omission_rate)
    values = _natural_scores(scores, name="scores")
    observed_labels = np.asarray(labels)
    if observed_labels.ndim != 1 or observed_labels.shape != values.shape:
        raise ValueError("labels must have one entry per calibration score")
    if _contains_boolean(observed_labels) or not np.all(np.isin(observed_labels, (0, 1))):
        raise ValueError("labels must contain exact natural=0 and AI=1 values")
    observed_labels = observed_labels.astype(int, copy=False)
    if set(observed_labels.tolist()) != {0, 1}:
        raise ValueError("combined calibration scores must contain both natural and AI labels")
    blocks, weights = _source_blocks(source_ids, len(values))
    if not isinstance(sample_unit, str) or not sample_unit:
        raise ValueError("sample_unit must be a non-empty string")
    threshold_limit = 1.0
    if max_threshold is not None:
        if isinstance(max_threshold, (bool, np.bool_)) or not isinstance(max_threshold, Real):
            raise TypeError("max_threshold must be a real number")
        threshold_limit = float(max_threshold)
        if not np.isfinite(threshold_limit) or not 0.0 <= threshold_limit <= 1.0:
            raise ValueError("max_threshold must be finite and in [0, 1]")

    candidates = np.unique(values[values <= threshold_limit])
    if not candidates.size:
        raise CalibrationError("no natural threshold is available below the AI threshold")
    selected: tuple[float, float, np.ndarray, float] | None = None
    exploratory: tuple[float, float, np.ndarray, float] | None = None
    for candidate in candidates:
        called = values <= candidate
        natural_call_count = int(np.count_nonzero(called))
        false_omission_count = int(
            np.count_nonzero(called & (observed_labels == 1))
        )
        achieved = false_omission_count / natural_call_count
        called_weight = float(weights[called].sum())
        if called_weight <= 0.0:
            continue
        source_weighted_achieved = float(
            weights[called & (observed_labels == 1)].sum() / called_weight
        )
        current = (float(candidate), achieved, called, source_weighted_achieved)
        if exploratory is None or max(achieved, source_weighted_achieved) < max(
            exploratory[1], exploratory[3]
        ):
            exploratory = current
        if _rate_at_or_below(achieved, target) and _rate_at_or_below(
            source_weighted_achieved, target
        ):
            selected = current
    meets_constraint = selected is not None
    if selected is None:
        assert exploratory is not None
        selected = exploratory
        if not allow_uncertified:
            raise CalibrationError(
                f"false-omission rate {max(selected[1], selected[3]):g} "
                f"exceeds target {target:g}; "
                "ties leave no certifiable natural threshold"
            )
    threshold, achieved, called, source_weighted_achieved = selected
    called_sources = {
        source_id
        for source_id, indexes in blocks.items()
        if bool(np.any(called[indexes]))
    }
    required = ceil(10.0 / target)
    enough_samples = len(called_sources) >= required
    if not enough_samples and not allow_uncertified:
        raise CalibrationError(
            f"at least {required} independently sourced natural calls are required "
            f"to certify target false-omission rate {target:g}; received {len(called_sources)}"
        )
    natural_call_count = int(np.count_nonzero(called))
    false_omission_count = int(
        np.count_nonzero(called & (observed_labels == 1))
    )
    interval = wilson_interval(false_omission_count, natural_call_count)
    return NaturalThresholdCalibration(
        threshold=threshold,
        target_false_omission_rate=target,
        achieved_false_omission_rate=achieved,
        natural_precision=1.0 - achieved,
        sample_count=len(called_sources),
        calibration_source_count=len(blocks),
        record_count=int(values.size),
        natural_call_count=natural_call_count,
        false_omission_count=false_omission_count,
        natural_call_source_count=len(called_sources),
        sample_unit=sample_unit,
        confidence_interval=interval,
        confidence_method="wilson_95",
        quantile_method="largest_threshold_meeting_false_omission_constraint",
        certified=bool(
            enough_samples
            and meets_constraint
            and _rate_at_or_below(achieved, target)
            and _rate_at_or_below(source_weighted_achieved, target)
        ),
        required_sample_count=required,
    )


def _source_blocks(
    source_ids: Sequence[str] | None, record_count: int
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return independent-source blocks and equal-source record weights."""

    if source_ids is None:
        identities = tuple(f"record-{index}" for index in range(record_count))
    else:
        identities = tuple(source_ids)
        if len(identities) != record_count:
            raise ValueError("source_ids must have one entry per calibration score")
        if any(not isinstance(source_id, str) or not source_id for source_id in identities):
            raise ValueError("source_ids must contain non-empty strings")
    grouped: dict[str, list[int]] = {}
    for index, source_id in enumerate(identities):
        grouped.setdefault(source_id, []).append(index)
    blocks = {
        source_id: np.asarray(indexes, dtype=int) for source_id, indexes in grouped.items()
    }
    weights = np.zeros(record_count, dtype=float)
    source_weight = 1.0 / len(blocks)
    for indexes in blocks.values():
        weights[indexes] = source_weight / len(indexes)
    return blocks, weights


def _constrained_upper_threshold(
    values: np.ndarray, weights: np.ndarray, target: float
) -> tuple[float, float, bool]:
    """Select the lowest observed threshold whose inclusive tail meets target."""

    for candidate in np.unique(values):
        achieved = min(1.0, max(0.0, float(weights[values >= candidate].sum())))
        if _rate_at_or_below(achieved, target):
            return float(candidate), achieved, True
    fallback = float(np.quantile(values, 1.0 - target, method="higher"))
    return (
        fallback,
        min(1.0, max(0.0, float(weights[values >= fallback].sum()))),
        False,
    )


def _source_rate_interval(source_rates: np.ndarray) -> tuple[tuple[float, float], str]:
    if np.all(np.isin(source_rates, (0.0, 1.0))):
        return wilson_interval(int(source_rates.sum()), len(source_rates)), "wilson_95"
    if len(source_rates) < 2:
        value = float(source_rates[0])
        return (value, value), "source_cluster_normal_95"
    mean = float(source_rates.mean())
    radius = _WILSON_Z_95 * float(source_rates.std(ddof=1)) / sqrt(len(source_rates))
    return (max(0.0, mean - radius), min(1.0, mean + radius)), "source_cluster_normal_95"


def _ratio_cluster_interval(
    blocks: dict[str, np.ndarray],
    called: np.ndarray,
    errors: np.ndarray,
    weights: np.ndarray,
    achieved: float,
) -> tuple[float, float]:
    if len(blocks) < 2:
        return achieved, achieved
    total_called = float(weights[called].sum())
    total_errors = float(weights[called & errors].sum())
    leave_one_out: list[float] = []
    for indexes in blocks.values():
        remaining_called = total_called - float(weights[indexes][called[indexes]].sum())
        remaining_errors = total_errors - float(
            weights[indexes][called[indexes] & errors[indexes]].sum()
        )
        if remaining_called > 0.0:
            leave_one_out.append(remaining_errors / remaining_called)
    if len(leave_one_out) < 2:
        return achieved, achieved
    estimates = np.asarray(leave_one_out, dtype=float)
    center = float(estimates.mean())
    standard_error = sqrt(
        (len(estimates) - 1) / len(estimates) * float(np.sum((estimates - center) ** 2))
    )
    radius = _WILSON_Z_95 * standard_error
    return max(0.0, achieved - radius), min(1.0, achieved + radius)


def _validate_unit_interval(value: Real, name: str, *, include_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not a boolean")
    result = float(value)
    if not np.isfinite(result) or not (0.0 <= result < 1.0 if include_zero else 0.0 < result < 1.0):
        boundary = "[0, 1)" if include_zero else "(0, 1)"
        raise ValueError(f"{name} must be finite and in {boundary}")
    return result


@dataclass(frozen=True)
class NaturalSupportModel:
    """Regularized natural-feature support, intentionally not an AI classifier."""

    center: np.ndarray
    covariance_inverse: np.ndarray
    threshold: float
    quantile: float
    shrinkage: float

    @classmethod
    def fit(
        cls,
        natural_features: object,
        quantile: Real = 0.99,
        shrinkage: Real = 0.1,
    ) -> "NaturalSupportModel":
        """Fit a diagonal-shrunk Mahalanobis support boundary on natural data."""

        quantile_value = _validate_unit_interval(quantile, "quantile")
        shrinkage_value = _validate_unit_interval(shrinkage, "shrinkage", include_zero=True)
        values = np.asarray(natural_features)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
            raise ValueError("natural_features must be a finite 2D array with at least two rows")
        if _contains_boolean(values):
            raise TypeError("natural_features must be numeric, not booleans")
        try:
            values = values.astype(float, copy=False)
        except (TypeError, ValueError) as error:
            raise TypeError("natural_features must be numeric") from error
        if not np.all(np.isfinite(values)):
            raise ValueError("natural_features must contain only finite values")

        center = values.mean(axis=0)
        covariance = np.atleast_2d(np.cov(values, rowvar=False, bias=False))
        diagonal = np.diag(np.diag(covariance))
        regularized = (1.0 - shrinkage_value) * covariance + shrinkage_value * diagonal
        scale = max(float(np.max(np.abs(np.diag(regularized)))), 1.0)
        regularized = regularized + np.eye(values.shape[1]) * (np.finfo(float).eps * scale)
        inverse = np.linalg.inv(regularized)
        centered = values - center
        distances = np.sqrt(np.einsum("ij,jk,ik->i", centered, inverse, centered).clip(min=0.0))
        threshold = float(np.quantile(distances, quantile_value, method="higher"))
        return cls(
            center=center,
            covariance_inverse=inverse,
            threshold=threshold,
            quantile=quantile_value,
            shrinkage=shrinkage_value,
        )

    def distance(self, features: object) -> float:
        """Return support distance only; callers must not interpret it as AI probability."""

        values = np.asarray(features)
        if values.ndim != 1 or values.shape != self.center.shape:
            raise ValueError(f"features must be a finite vector with shape {self.center.shape}")
        if _contains_boolean(values):
            raise TypeError("features must be numeric, not booleans")
        try:
            values = values.astype(float, copy=False)
        except (TypeError, ValueError) as error:
            raise TypeError("features must be numeric") from error
        if not np.all(np.isfinite(values)):
            raise ValueError("features must contain only finite values")
        delta = values - self.center
        return float(sqrt(max(0.0, float(delta @ self.covariance_inverse @ delta))))

    def is_out_of_support(self, features: object) -> bool:
        """Return a boolean support flag without asserting AI provenance."""

        return bool(self.distance(features) > self.threshold)
