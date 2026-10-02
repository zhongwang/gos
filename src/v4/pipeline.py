"""Tri-state GOS v4 routing and evidence orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Real
from typing import Any, Mapping

import numpy as np

from .artifacts import (
    ArtifactCompatibilityError,
    validate_artifact,
    validate_artifact_compatibility,
)
from .calibration import NaturalSupportModel
from .cds import CDSStage
from .complexity import LOCAL_COMPLEXITY_FEATURE_NAMES, local_complexity_features
from .pav import PAVStage
from .preprocessing import PreparedSequence, prepare_sequence
from .sequence import MULTISCALE_SUMMARY_NAMES, SequenceStage, multiscale_summary_features
from .types import ScanResult, StageEvidence, StageStatus


CPU_FEATURE_NAMES = (
    "sequence_mean",
    "sequence_max",
    "sequence_top_k_mean",
    "sequence_count_over",
    "sequence_fraction_over",
    "sequence_longest_run_over",
    "sequence_n_windows",
    "sequence_n_eligible",
    "sequence_score_coverage",
    "sequence_length_bin_500_749",
    "sequence_length_bin_750_999",
    "sequence_length_bin_1000",
    "sequence_available",
    "cds_mean",
    "cds_max",
    "cds_top_k_mean",
    "cds_count_over",
    "cds_fraction_over",
    "cds_n_orfs",
    "cds_n_eligible",
    "cds_covered_coding_bases",
    "cds_coding_coverage",
    "cds_available",
    "length",
    "scorable_coverage",
    "acgt_coverage",
    "ambiguous_fraction",
    "out_of_support",
)
PAV_FEATURE_NAMES = (
    "pav_mean",
    "pav_max",
    "pav_n_windows",
    "pav_n_scored",
    "pav_scored_coverage",
    "pav_available",
)
FULL_FEATURE_NAMES = CPU_FEATURE_NAMES + PAV_FEATURE_NAMES
LAYER24_MEAN_FEATURE_NAMES = tuple(f"layer24_mean_{index}" for index in range(2560))
ATTENTION_ENTROPY_FEATURE_NAMES = tuple(
    f"attention_entropy_{index}" for index in range(640)
)
STRATEGY_B_FEATURE_NAMES = (
    ("pav_available",)
    + LAYER24_MEAN_FEATURE_NAMES
    + ATTENTION_ENTROPY_FEATURE_NAMES
    + MULTISCALE_SUMMARY_NAMES
)
_PAV_OBSERVER_FEATURE_PREFIXES = ("pav_", "layer24_mean_", "attention_entropy_")
_LENGTH_BIN_FEATURES = (
    "sequence_length_bin_500_749",
    "sequence_length_bin_750_999",
    "sequence_length_bin_1000",
)
_SUPPORTED_FEATURES = frozenset(
    FULL_FEATURE_NAMES
    + _LENGTH_BIN_FEATURES
    + LOCAL_COMPLEXITY_FEATURE_NAMES
    + MULTISCALE_SUMMARY_NAMES
)


def _supported_feature(name: str) -> bool:
    if name in _SUPPORTED_FEATURES:
        return True
    for prefix in (
        "pav_transformed_mean_",
        "layer24_mean_",
        "attention_entropy_",
    ):
        if name.startswith(prefix):
            return name[len(prefix) :].isdigit()
    return False


def _uses_pav_observer_features(feature_names: tuple[str, ...]) -> bool:
    return any(
        name.startswith(_PAV_OBSERVER_FEATURE_PREFIXES)
        for name in feature_names
    )


def _artifact_error(message: str) -> ArtifactCompatibilityError:
    return ArtifactCompatibilityError(f"GOS v4 router artifact compatibility error: {message}")


def _stage_certification(stage: Any, name: str, *, allow_uncertified: bool) -> bool:
    certified = getattr(stage, "certified", None)
    if type(certified) is not bool:
        raise _artifact_error(f"{name} stage must carry boolean certification")
    if not certified and not allow_uncertified:
        raise _artifact_error(
            f"uncertified {name} stage requires allow_uncertified=True"
        )
    return certified


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise _artifact_error(f"{field} must be a finite number")
    return float(value)


def _aggregate_value(evidence: StageEvidence, name: str) -> Any:
    aggregates = evidence.aggregates
    if aggregates is None:
        return None
    if isinstance(aggregates, Mapping):
        return aggregates.get(name)
    return getattr(aggregates, name, None)


def _stage_available(evidence: StageEvidence) -> float:
    available = _aggregate_value(evidence, "available")
    if available is None:
        available = evidence.status is StageStatus.OK
    return float(bool(available) and evidence.status is StageStatus.OK)


def _sequence_score_coverage(
    evidence: StageEvidence, total_scorable_bases: int
) -> float:
    if total_scorable_bases <= 0:
        return 0.0
    scored_bases = sum(
        window.span.length
        for window in evidence.windows
        if window.eligible and window.score is not None and np.isfinite(window.score)
    )
    return float(min(1.0, scored_bases / total_scorable_bases))


def _cds_natural_block_reason(evidence: StageEvidence) -> str | None:
    if evidence.status is StageStatus.FAILED or evidence.n_failed > 0:
        return "cds_failed"
    if evidence.status is not StageStatus.OK and not (
        evidence.status is StageStatus.UNAVAILABLE and evidence.reason == "no_orfs"
    ):
        return "cds_unavailable"
    return None


def _can_route_without_sequence_stage(
    sequence: StageEvidence, router: "_RouterProfile"
) -> bool:
    """Allow calibrated short-fragment routers to cover a missing 500 bp score."""

    return (
        router.can_route_short_fragment
        and sequence.status is StageStatus.UNAVAILABLE
        and sequence.reason in {"no_scorable_windows", "no_calibrated_windows"}
    )


def _feature_values(
    prepared: PreparedSequence,
    sequence: StageEvidence,
    cds: StageEvidence,
    pav: StageEvidence | None,
    *,
    out_of_support: bool,
) -> dict[str, Any]:
    sequence_bins = _aggregate_value(sequence, "length_bin_counts") or {}
    local_features = local_complexity_features(prepared)
    multiscale_features = multiscale_summary_features(prepared)
    pav_evidence = pav or StageEvidence(StageStatus.NOT_REQUESTED)
    values = {
        "sequence_mean": _aggregate_value(sequence, "mean"),
        "sequence_max": _aggregate_value(sequence, "maximum"),
        "sequence_top_k_mean": _aggregate_value(sequence, "top_k_mean"),
        "sequence_count_over": _aggregate_value(sequence, "count_over"),
        "sequence_fraction_over": _aggregate_value(sequence, "fraction_over"),
        "sequence_longest_run_over": _aggregate_value(sequence, "longest_run_over"),
        "sequence_n_windows": _aggregate_value(sequence, "n_windows"),
        "sequence_n_eligible": _aggregate_value(sequence, "n_eligible"),
        "sequence_score_coverage": _sequence_score_coverage(
            sequence, prepared.scorable_bases
        ),
        "sequence_available": _stage_available(sequence),
        "sequence_length_bin_500_749": sequence_bins.get("500-749", 0),
        "sequence_length_bin_750_999": sequence_bins.get("750-999", 0),
        "sequence_length_bin_1000": sequence_bins.get("1000", 0),
        **local_features,
        **multiscale_features,
        "cds_mean": _aggregate_value(cds, "mean"),
        "cds_max": _aggregate_value(cds, "maximum"),
        "cds_top_k_mean": _aggregate_value(cds, "top_k_mean"),
        "cds_count_over": _aggregate_value(cds, "count_over"),
        "cds_fraction_over": _aggregate_value(cds, "fraction_over"),
        "cds_n_orfs": _aggregate_value(cds, "n_orfs"),
        "cds_n_eligible": _aggregate_value(cds, "n_eligible"),
        "cds_covered_coding_bases": _aggregate_value(cds, "covered_coding_bases"),
        "cds_coding_coverage": _aggregate_value(cds, "coding_coverage"),
        "cds_available": _stage_available(cds),
        "pav_mean": _aggregate_value(pav_evidence, "mean"),
        "pav_max": _aggregate_value(pav_evidence, "maximum"),
        "pav_n_windows": _aggregate_value(pav_evidence, "n_windows"),
        "pav_n_scored": _aggregate_value(pav_evidence, "n_scored"),
        "pav_scored_coverage": _aggregate_value(pav_evidence, "scored_coverage"),
        "pav_available": _stage_available(pav_evidence),
        "length": prepared.length,
        "scorable_coverage": prepared.scorable_coverage,
        "acgt_coverage": prepared.acgt_coverage,
        "ambiguous_fraction": 1.0 - prepared.acgt_coverage,
        "out_of_support": float(out_of_support),
    }
    transformed_mean = _aggregate_value(pav_evidence, "transformed_mean")
    if transformed_mean is not None:
        for index, value in enumerate(transformed_mean):
            values[f"pav_transformed_mean_{index}"] = value
    layer24_mean = _aggregate_value(pav_evidence, "layer24_mean")
    if layer24_mean is not None:
        for index, value in enumerate(layer24_mean):
            values[f"layer24_mean_{index}"] = value
    attention_entropy = _aggregate_value(pav_evidence, "attention_entropy")
    if attention_entropy is not None:
        for index, value in enumerate(attention_entropy):
            values[f"attention_entropy_{index}"] = value
    return values


@dataclass(frozen=True)
class _RouterProfile:
    kind: str
    feature_names: tuple[str, ...]
    estimator: Any
    scaler: Any
    imputations: Mapping[str, float]
    natural_max: float
    ai_min: float
    min_acgt_coverage: float
    min_sequence_score_coverage: float
    min_fragment_length: int
    allow_sequence_unavailable_routing: bool
    audit_rate: float
    audit_seed: str
    metadata: Mapping[str, Any]

    @classmethod
    def from_artifact(
        cls, artifact: Mapping[str, Any], kind: str, *, allow_uncertified: bool
    ) -> "_RouterProfile":
        validated = validate_artifact(artifact, expected_kind=kind)
        feature_names = tuple(validated["feature_names"])
        if len(set(feature_names)) != len(feature_names):
            raise _artifact_error("feature_names must not contain duplicates")
        unknown = sorted(name for name in set(feature_names) if not _supported_feature(name))
        if unknown:
            raise _artifact_error(f"unsupported feature_names: {', '.join(unknown)}")
        for prefix in ("sequence", "cds", "pav"):
            if any(name.startswith(f"{prefix}_") for name in feature_names) and (
                f"{prefix}_available" not in feature_names
            ):
                raise _artifact_error(
                    f"{prefix} aggregate features require {prefix}_available"
                )
        if kind == "router_cpu" and _uses_pav_observer_features(feature_names):
            raise _artifact_error("router_cpu must not declare pAV or observer features")
        if kind == "router_full" and "pav_available" not in feature_names:
            raise _artifact_error("router_full must declare pav_available")

        metadata = validated["metadata"]
        certification = metadata["certification_status"]
        certified = (
            certification if type(certification) is bool else certification.get("certified")
        )
        if not certified and not allow_uncertified:
            raise _artifact_error("uncertified router artifact requires allow_uncertified=True")

        thresholds = metadata.get("thresholds")
        if not isinstance(thresholds, Mapping):
            raise _artifact_error("metadata must include router thresholds")
        natural_max = _finite_number(thresholds.get("natural_max"), field="natural_max")
        ai_min = _finite_number(thresholds.get("ai_min"), field="ai_min")
        if not 0.0 <= natural_max < ai_min <= 1.0:
            raise _artifact_error("thresholds must satisfy 0 <= natural_max < ai_min <= 1")
        min_coverage = _finite_number(
            metadata.get("min_acgt_coverage", 1.0), field="min_acgt_coverage"
        )
        if not 0.0 <= min_coverage <= 1.0:
            raise _artifact_error("min_acgt_coverage must be between 0 and 1")
        if "min_sequence_score_coverage" not in metadata:
            raise _artifact_error(
                "metadata must include canonical min_sequence_score_coverage"
            )
        min_sequence_score_coverage = _finite_number(
            metadata["min_sequence_score_coverage"],
            field="min_sequence_score_coverage",
        )
        if not 0.0 <= min_sequence_score_coverage <= 1.0:
            raise _artifact_error(
                "min_sequence_score_coverage must be between 0 and 1"
            )
        min_fragment_length = metadata.get("min_fragment_length", 500)
        if (
            type(min_fragment_length) is not int
            or min_fragment_length < 1
        ):
            raise _artifact_error("min_fragment_length must be a positive integer")
        allow_sequence_unavailable_routing = metadata.get(
            "allow_sequence_unavailable_routing", False
        )
        if type(allow_sequence_unavailable_routing) is not bool:
            raise _artifact_error(
                "allow_sequence_unavailable_routing must be a boolean"
            )
        audit_rate = _finite_number(metadata.get("audit_rate", 0.0), field="audit_rate")
        if not 0.0 <= audit_rate <= 1.0:
            raise _artifact_error("audit_rate must be between 0 and 1")
        audit_seed = metadata.get("audit_seed", "gos-v4-audit")
        if not isinstance(audit_seed, str) or not audit_seed:
            raise _artifact_error("audit_seed must be a non-empty string")

        payload = validated["payload"]
        if not isinstance(payload, Mapping):
            raise _artifact_error("payload must be a mapping")
        if "model" in payload:
            raise _artifact_error(
                "payload model is an unsupported legacy alias; use estimator"
            )
        estimator = payload.get("estimator")
        scaler = payload.get("scaler")
        if not callable(getattr(estimator, "predict_proba", None)):
            raise _artifact_error("payload estimator must provide predict_proba")
        if not callable(getattr(scaler, "transform", None)):
            raise _artifact_error("payload scaler must provide transform")
        raw_imputations = payload.get("imputation_values")
        if not isinstance(raw_imputations, Mapping):
            raise _artifact_error("payload imputation_values must map feature names to values")
        missing = [name for name in feature_names if name not in raw_imputations]
        if missing:
            raise _artifact_error(f"imputation_values missing features: {', '.join(missing)}")
        imputations = {
            name: _finite_number(raw_imputations[name], field=f"imputation for {name!r}")
            for name in feature_names
        }
        return cls(
            kind,
            feature_names,
            estimator,
            scaler,
            imputations,
            natural_max,
            ai_min,
            min_coverage,
            min_sequence_score_coverage,
            min_fragment_length,
            allow_sequence_unavailable_routing,
            audit_rate,
            audit_seed,
            metadata,
        )

    def vector(self, values: Mapping[str, Any]) -> np.ndarray:
        row: list[float] = []
        for name in self.feature_names:
            value = values.get(name)
            if value is None:
                value = self.imputations[name]
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"router feature {name!r} must be finite after imputation")
            row.append(float(value))
        return np.asarray(row, dtype=float)

    def transform(self, vector: np.ndarray) -> np.ndarray:
        transformed = np.asarray(self.scaler.transform(vector.reshape(1, -1)), dtype=float)
        if transformed.shape != (1, len(self.feature_names)) or not np.all(
            np.isfinite(transformed)
        ):
            raise ValueError("router scaler returned invalid feature values")
        return transformed

    def score_transformed(self, transformed: np.ndarray) -> float:
        probabilities = np.asarray(self.estimator.predict_proba(transformed), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[0] != 1:
            raise ValueError("router estimator returned invalid probabilities")
        classes = getattr(self.estimator, "classes_", None)
        index = 1
        if classes is not None:
            matches = np.flatnonzero(np.asarray(classes) == 1)
            if len(matches) != 1:
                raise ValueError("router estimator must expose one AI class labelled 1")
            index = int(matches[0])
        if probabilities.shape[1] <= index:
            raise ValueError("router estimator is missing the AI probability column")
        score = float(probabilities[0, index])
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("router estimator returned an invalid AI probability")
        return score

    @property
    def can_route_short_fragment(self) -> bool:
        if self.allow_sequence_unavailable_routing:
            return True
        return (
            self.min_fragment_length < 500
            and self.min_sequence_score_coverage == 0.0
            and any(name.startswith("local_") for name in self.feature_names)
        )

    @property
    def certified(self) -> bool:
        certification = self.metadata["certification_status"]
        return (
            certification
            if type(certification) is bool
            else certification["certified"]
        )


class GOSV4Detector:
    """Compose v4 evidence stages into a calibrated tri-state decision."""

    def __init__(
        self,
        sequence_stage: SequenceStage,
        cds_stage: CDSStage,
        router_cpu: Mapping[str, Any],
        support_model: NaturalSupportModel,
        *,
        pav_stage: PAVStage | None = None,
        router_full: Mapping[str, Any] | None = None,
        preprocessing_policy: str = "split",
        min_fragment_length: int | None = None,
        allow_uncertified: bool = False,
    ) -> None:
        if type(allow_uncertified) is not bool:
            raise TypeError("allow_uncertified must be an exact boolean")
        if not callable(getattr(sequence_stage, "score", None)):
            raise TypeError("sequence_stage must provide score(prepared)")
        if not callable(getattr(cds_stage, "score", None)):
            raise TypeError("cds_stage must provide score(prepared)")
        if not callable(getattr(support_model, "is_out_of_support", None)):
            raise TypeError("support_model must provide is_out_of_support(features)")
        if pav_stage is not None and not callable(getattr(pav_stage, "score", None)):
            raise TypeError("pav_stage must provide score(prepared)")
        if router_full is not None and pav_stage is None:
            raise ValueError("router_full requires a pAV stage")
        if pav_stage is not None and router_full is None:
            raise ValueError("pav_stage requires router_full")
        if not isinstance(preprocessing_policy, str):
            raise TypeError("preprocessing_policy must be a string")
        if min_fragment_length is not None and (
            type(min_fragment_length) is not int or min_fragment_length < 1
        ):
            raise ValueError("min_fragment_length must be a positive integer")
        component_artifacts = {"router_cpu": router_cpu}
        if router_full is not None:
            component_artifacts["router_full"] = router_full
        for name, stage in (
            ("sequence", sequence_stage),
            ("cds", cds_stage),
            ("pav", pav_stage),
        ):
            artifact = getattr(stage, "_artifact", None)
            if artifact is not None:
                component_artifacts[name] = artifact
        validate_artifact_compatibility(component_artifacts)
        sequence_certified = _stage_certification(
            sequence_stage, "sequence", allow_uncertified=allow_uncertified
        )
        cds_certified = _stage_certification(
            cds_stage, "CDS", allow_uncertified=allow_uncertified
        )
        pav_certified = (
            _stage_certification(
                pav_stage, "pAV", allow_uncertified=allow_uncertified
            )
            if pav_stage is not None
            else True
        )
        self._sequence_stage = sequence_stage
        self._cds_stage = cds_stage
        self._cpu = _RouterProfile.from_artifact(
            router_cpu, "router_cpu", allow_uncertified=allow_uncertified
        )
        self._support_model = support_model
        self._pav_stage = pav_stage
        self._full = (
            _RouterProfile.from_artifact(
                router_full, "router_full", allow_uncertified=allow_uncertified
            )
            if router_full is not None
            else None
        )
        self._preprocessing_policy = preprocessing_policy
        self._min_fragment_length = (
            min_fragment_length
            if min_fragment_length is not None
            else min(
                self._cpu.min_fragment_length,
                self._full.min_fragment_length if self._full is not None else 500,
            )
        )
        self.certified = bool(
            sequence_certified
            and cds_certified
            and pav_certified
            and self._cpu.certified
            and (self._full is None or self._full.certified)
        )

    @staticmethod
    def _scores(
        prepared: PreparedSequence,
        sequence: StageEvidence,
        cds: StageEvidence,
        *,
        out_of_support: bool,
        sequence_score_coverage: float,
    ) -> dict[str, Any]:
        return {
            "scorable_coverage": prepared.scorable_coverage,
            "sequence_score_coverage": sequence_score_coverage,
            "out_of_support": out_of_support,
            "sequence_mean": _aggregate_value(sequence, "mean"),
            "sequence_max": _aggregate_value(sequence, "maximum"),
            "sequence_top_k_mean": _aggregate_value(sequence, "top_k_mean"),
            "sequence_count_over": _aggregate_value(sequence, "count_over"),
            "sequence_fraction_over": _aggregate_value(sequence, "fraction_over"),
            "sequence_longest_run_over": _aggregate_value(
                sequence, "longest_run_over"
            ),
            "cds_mean": _aggregate_value(cds, "mean"),
            "cds_max": _aggregate_value(cds, "maximum"),
            "cds_top_k_mean": _aggregate_value(cds, "top_k_mean"),
            "cds_count_over": _aggregate_value(cds, "count_over"),
            "cds_fraction_over": _aggregate_value(cds, "fraction_over"),
            "cds_coding_coverage": _aggregate_value(cds, "coding_coverage"),
        }

    def _audit_requested(self, seq: str, source_id: str | None) -> bool:
        if self._cpu.audit_rate <= 0.0:
            return False
        if self._cpu.audit_rate >= 1.0:
            return True
        identity = source_id or hashlib.sha256(seq.encode("utf-8")).hexdigest()
        digest = hashlib.sha256(
            f"{self._cpu.audit_seed}\0{identity}".encode("utf-8")
        ).digest()
        sample = int.from_bytes(digest[:8], "big") / float(1 << 64)
        return sample < self._cpu.audit_rate

    def scan(self, seq: str, source_id: str | None = None) -> ScanResult:
        if source_id is not None and (not isinstance(source_id, str) or not source_id):
            raise ValueError("source_id must be a non-empty string or None")
        if seq == "":
            return ScanResult.inconclusive(
                reason="insufficient_length", length=0, acgt_coverage=0.0
            )
        prepared = prepare_sequence(
            seq,
            policy=self._preprocessing_policy,
            min_fragment_length=self._min_fragment_length,
        )
        sequence = self._sequence_stage.score(prepared)
        cds = self._cds_stage.score(prepared)
        provisional = _feature_values(
            prepared, sequence, cds, None, out_of_support=False
        )
        cpu_vector = self._cpu.vector(provisional)
        cpu_transformed = self._cpu.transform(cpu_vector)
        out_of_support = bool(self._support_model.is_out_of_support(cpu_transformed[0]))
        values = _feature_values(
            prepared, sequence, cds, None, out_of_support=out_of_support
        )
        cpu_transformed = self._cpu.transform(self._cpu.vector(values))
        cpu_score = self._cpu.score_transformed(cpu_transformed)
        sequence_score_coverage = _sequence_score_coverage(
            sequence, prepared.scorable_bases
        )

        scores = self._scores(
            prepared,
            sequence,
            cds,
            out_of_support=out_of_support,
            sequence_score_coverage=sequence_score_coverage,
        )
        scores.update(
            {
                name: values.get(name)
                for name in LOCAL_COMPLEXITY_FEATURE_NAMES
                + MULTISCALE_SUMMARY_NAMES
            }
        )
        scores.update({"router": cpu_score, "router_cpu": cpu_score})
        thresholds = dict(self._cpu.metadata["thresholds"])
        thresholds["natural_max"] = self._cpu.natural_max
        thresholds["ai_min"] = self._cpu.ai_min
        thresholds["min_acgt_coverage"] = self._cpu.min_acgt_coverage
        effective_min_sequence_score_coverage = max(
            self._cpu.min_sequence_score_coverage,
            self._full.min_sequence_score_coverage if self._full is not None else 0.0,
        )
        thresholds["min_sequence_score_coverage"] = (
            effective_min_sequence_score_coverage
        )
        sequence_thresholds = getattr(self._sequence_stage, "_thresholds", None)
        local_thresholds = getattr(self._sequence_stage, "_local_thresholds", None)
        if isinstance(sequence_thresholds, Mapping):
            thresholds["sequence_window_by_length"] = dict(sequence_thresholds)
        if isinstance(local_thresholds, Mapping):
            thresholds["sequence_local_by_length"] = dict(local_thresholds)
        base_common = {
            "length": prepared.length,
            "acgt_coverage": prepared.acgt_coverage,
            "scores": scores,
            "thresholds": thresholds,
            "ignored_spans": prepared.ignored_spans,
            "windows": sequence.windows,
            "segments": sequence.segments,
            "orfs": cds.orfs,
            "stages": (sequence, cds),
            "warnings": tuple(sequence.warnings) + tuple(cds.warnings),
        }
        if prepared.scorable_coverage < self._cpu.min_acgt_coverage:
            return ScanResult.inconclusive(
                reason="insufficient_scorable_coverage", **base_common
            )
        route_without_sequence_stage = _can_route_without_sequence_stage(
            sequence, self._cpu
        )
        if (
            sequence.status is not StageStatus.OK
            and not route_without_sequence_stage
        ):
            return ScanResult.inconclusive(
                reason=sequence.reason or "sequence_unavailable", **base_common
            )

        abstained = self._cpu.natural_max < cpu_score < self._cpu.ai_min
        local_conflict = bool(sequence.segments) and cpu_score < self._cpu.ai_min
        audit_requested = self._audit_requested(seq, source_id)
        cds_natural_block = _cds_natural_block_reason(cds)
        insufficient_sequence_score_coverage = (
            sequence_score_coverage < effective_min_sequence_score_coverage
        )
        cpu_would_be_natural = cpu_score <= self._cpu.natural_max
        safety_rescue = cpu_would_be_natural and (
            cds_natural_block is not None or insufficient_sequence_score_coverage
        )
        if route_without_sequence_stage and cpu_score >= self._cpu.ai_min:
            return ScanResult.ai(reason="router_ai", **base_common)
        pav_required = (
            abstained
            or local_conflict
            or out_of_support
            or audit_requested
            or safety_rescue
        )
        if pav_required:
            if (
                self._full is not None
                and prepared.scorable_coverage < self._full.min_acgt_coverage
            ):
                thresholds["min_acgt_coverage"] = max(
                    self._cpu.min_acgt_coverage, self._full.min_acgt_coverage
                )
                return ScanResult.inconclusive(
                    reason="insufficient_scorable_coverage", **base_common
                )
            if self._pav_stage is None or self._full is None:
                reason = (
                    cds_natural_block
                    if cpu_would_be_natural and cds_natural_block is not None
                    else "insufficient_sequence_score_coverage"
                    if cpu_would_be_natural and insufficient_sequence_score_coverage
                    else "out_of_support"
                    if out_of_support
                    else "local_sequence_evidence_requires_pav"
                    if local_conflict
                    else "router_abstained"
                    if abstained
                    else "pav_unavailable"
                )
                return ScanResult.inconclusive(
                    reason=reason,
                    pav_triggered=True,
                    pav_status=StageStatus.UNAVAILABLE,
                    **base_common,
                )

            pav = self._pav_stage.score(prepared, source_id=source_id or "unknown")
            scores.update(
                {
                    "pav_mean": _aggregate_value(pav, "mean"),
                    "pav_max": _aggregate_value(pav, "maximum"),
                    "pav_n_windows": _aggregate_value(pav, "n_windows"),
                    "pav_n_scored": _aggregate_value(pav, "n_scored"),
                    "pav_scored_coverage": _aggregate_value(pav, "scored_coverage"),
                }
            )
            pav_common = {
                **base_common,
                "pav_triggered": True,
                "pav_status": pav.status,
                "stages": (sequence, cds, pav),
                "warnings": base_common["warnings"] + tuple(pav.warnings),
            }
            if pav.status is not StageStatus.OK:
                reason = "pav_failed" if pav.status is StageStatus.FAILED else "pav_unavailable"
                return ScanResult.inconclusive(reason=reason, **pav_common)

            full_values = _feature_values(
                prepared, sequence, cds, pav, out_of_support=out_of_support
            )
            full_score = self._full.score_transformed(
                self._full.transform(self._full.vector(full_values))
            )
            scores.update({"router": full_score, "router_full": full_score})
            thresholds.update(
                dict(self._full.metadata["thresholds"])
            )
            thresholds.update(
                {
                    "cpu_natural_max": self._cpu.natural_max,
                    "cpu_ai_min": self._cpu.ai_min,
                    "natural_max": self._full.natural_max,
                    "ai_min": self._full.ai_min,
                    "min_acgt_coverage": max(
                        self._cpu.min_acgt_coverage, self._full.min_acgt_coverage
                    ),
                    "min_sequence_score_coverage": (
                        effective_min_sequence_score_coverage
                    ),
                }
            )
            if full_score >= self._full.ai_min:
                return ScanResult.ai(reason="router_ai", **pav_common)
            if full_score <= self._full.natural_max:
                if route_without_sequence_stage:
                    return ScanResult.inconclusive(
                        reason=sequence.reason or "sequence_unavailable", **pav_common
                    )
                if cds_natural_block is not None:
                    return ScanResult.inconclusive(
                        reason=cds_natural_block, **pav_common
                    )
                if insufficient_sequence_score_coverage:
                    return ScanResult.inconclusive(
                        reason="insufficient_sequence_score_coverage", **pav_common
                    )
                return ScanResult.natural(reason="router_natural", **pav_common)
            return ScanResult.inconclusive(reason="router_abstained", **pav_common)

        if cpu_score <= self._cpu.natural_max:
            if route_without_sequence_stage:
                return ScanResult.inconclusive(
                    reason=sequence.reason or "sequence_unavailable", **base_common
                )
            if cds_natural_block is not None:
                return ScanResult.inconclusive(reason=cds_natural_block, **base_common)
            if insufficient_sequence_score_coverage:
                return ScanResult.inconclusive(
                    reason="insufficient_sequence_score_coverage", **base_common
                )
            return ScanResult.natural(reason="router_natural", **base_common)
        if cpu_score >= self._cpu.ai_min:
            return ScanResult.ai(reason="router_ai", **base_common)
        return ScanResult.inconclusive(reason="router_abstained", **base_common)

    def scan_sequence(self, seq: str, source_id: str | None = None) -> ScanResult:
        """Compatibility alias for :meth:`scan`."""

        return self.scan(seq, source_id=source_id)


__all__ = [
    "ATTENTION_ENTROPY_FEATURE_NAMES",
    "CPU_FEATURE_NAMES",
    "FULL_FEATURE_NAMES",
    "GOSV4Detector",
    "LAYER24_MEAN_FEATURE_NAMES",
    "PAV_FEATURE_NAMES",
    "STRATEGY_B_FEATURE_NAMES",
]
