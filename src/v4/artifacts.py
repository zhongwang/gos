"""Strict versioned joblib artifact contracts for GOS v4."""

from __future__ import annotations

import platform
from pathlib import Path
import re
from math import ceil
from typing import Any, Mapping

import joblib

from .calibration import rate_tolerance, wilson_interval


SCHEMA_VERSION = "gos-v4-artifact/1"
MODEL_VERSION = 4
_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "model_version",
        "artifact_kind",
        "feature_names",
        "metadata",
        "payload",
    }
)
_REQUIRED_METADATA_KEYS = frozenset(
    {
        "training_config",
        "source_split_manifest_hash",
        "dependency_versions",
        "certification_status",
    }
)


class ArtifactCompatibilityError(ValueError):
    """Raised when an artifact cannot safely be interpreted as GOS v4."""


def _compatibility_error(message: str) -> ArtifactCompatibilityError:
    return ArtifactCompatibilityError(f"GOS v4 artifact compatibility error: {message}")


def _validate_feature_names(feature_names: Any) -> list[str]:
    if not isinstance(feature_names, list) or not feature_names:
        raise _compatibility_error("feature_names must be a non-empty list")
    if any(not isinstance(name, str) or not name for name in feature_names):
        raise _compatibility_error("feature_names must contain non-empty strings")
    return feature_names


def _default_metadata() -> dict[str, Any]:
    """Return explicit defaults for artifacts created outside a training run.

    These defaults preserve the convenient Task 1 save API without implying
    that an ad-hoc artifact has been certified or trained from a recorded
    source split.
    """

    return {
        "training_config": {"status": "not_recorded"},
        "source_split_manifest_hash": "not_recorded",
        "dependency_versions": {
            "python": platform.python_version(),
            "joblib": str(getattr(joblib, "__version__", "unknown")),
        },
        "certification_status": {"certified": False},
    }


def _validate_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise _compatibility_error("metadata must be a dictionary")

    missing = sorted(_REQUIRED_METADATA_KEYS - set(metadata))
    if missing:
        raise _compatibility_error(f"metadata missing required fields: {', '.join(missing)}")

    training_config = metadata["training_config"]
    if not isinstance(training_config, dict) or not training_config:
        raise _compatibility_error("training_config must be a non-empty dictionary")

    manifest_hash = metadata["source_split_manifest_hash"]
    if not isinstance(manifest_hash, str) or not manifest_hash:
        raise _compatibility_error("source_split_manifest_hash must be a non-empty string")

    dependency_versions = metadata["dependency_versions"]
    if not isinstance(dependency_versions, dict) or not dependency_versions:
        raise _compatibility_error("dependency_versions must be a non-empty dictionary")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(version, str)
        or not version
        for name, version in dependency_versions.items()
    ):
        raise _compatibility_error(
            "dependency_versions must map non-empty dependency names to version strings"
        )

    certification_status = metadata["certification_status"]
    if type(certification_status) is bool:
        return metadata
    if (
        not isinstance(certification_status, dict)
        or type(certification_status.get("certified")) is not bool
    ):
        raise _compatibility_error(
            "certification_status must be a boolean or dictionary with boolean certified"
        )
    return metadata


def _certified(metadata: Mapping[str, Any]) -> bool:
    status = metadata["certification_status"]
    return status if type(status) is bool else status["certified"]


def _finite_probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _compatibility_error(f"{field} must be a finite probability")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise _compatibility_error(f"{field} must be a finite probability")
    return result


def _same_rate(left: float, right: float) -> bool:
    return abs(left - right) <= rate_tolerance(max(abs(left), abs(right)))


def _validate_duplicate_threshold(
    value: Any, canonical: float, *, field: str
) -> None:
    duplicate = _finite_probability(value, field)
    if duplicate != canonical:
        raise _compatibility_error(f"{field} does not exactly match calibration threshold")


def _validate_threshold_mapping_duplicate(
    value: Any, canonical: Mapping[str, float], *, field: str
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(canonical):
        raise _compatibility_error(
            f"{field} must match the calibrated length-bin thresholds"
        )
    for length_bin, threshold in canonical.items():
        _validate_duplicate_threshold(
            value[length_bin], threshold, field=f"{field}.{length_bin}"
        )


def _validate_calibration_record(
    record: Any,
    *,
    field: str,
    expected_sample_unit: str,
    expected_confidence_method: str | None = None,
    expected_quantile_method: str | None = None,
    natural: bool = False,
) -> float:
    if not isinstance(record, Mapping):
        raise _compatibility_error(f"{field} calibration must be a mapping")
    required = {
        "threshold",
        "sample_count",
        "record_count",
        "sample_unit",
        "confidence_interval",
        "confidence_method",
        "quantile_method",
        "certified",
        "required_sample_count",
    }
    target_field = "target_false_omission_rate" if natural else "target_fpr"
    achieved_field = "achieved_false_omission_rate" if natural else "achieved_fpr"
    required.update({target_field, achieved_field})
    if natural:
        required.update(
            {
                "natural_precision",
                "calibration_source_count",
                "natural_call_count",
                "false_omission_count",
                "natural_call_source_count",
            }
        )
    missing = sorted(required - set(record))
    if missing:
        raise _compatibility_error(
            f"{field} calibration missing required fields: {', '.join(missing)}"
        )
    threshold = _finite_probability(record["threshold"], f"{field}.threshold")
    target = _finite_probability(record[target_field], f"{field}.{target_field}")
    achieved = _finite_probability(record[achieved_field], f"{field}.{achieved_field}")
    if not 0.0 < target < 1.0:
        raise _compatibility_error(f"{field}.{target_field} must be in (0, 1)")
    if achieved > target + rate_tolerance(target):
        raise _compatibility_error(f"{field} certified calibration exceeds its target")
    for count_field in ("sample_count", "record_count", "required_sample_count"):
        count = record[count_field]
        if type(count) is not int or count < 1:
            raise _compatibility_error(f"{field}.{count_field} must be a positive integer")
    if record["sample_count"] < record["required_sample_count"]:
        raise _compatibility_error(f"{field} calibration sample count is below its minimum")
    expected_required = ceil(10.0 / target)
    if record["required_sample_count"] != expected_required:
        raise _compatibility_error(
            f"{field} calibration minimum does not match its target"
        )
    if record["record_count"] < record["sample_count"]:
        raise _compatibility_error(f"{field}.record_count cannot be below sample_count")
    for text_field in ("sample_unit", "confidence_method", "quantile_method"):
        if not isinstance(record[text_field], str) or not record[text_field]:
            raise _compatibility_error(f"{field}.{text_field} must be a non-empty string")
    if record["sample_unit"] != expected_sample_unit:
        raise _compatibility_error(
            f"{field}.sample_unit must be {expected_sample_unit!r}"
        )
    confidence_methods = {
        "wilson_95",
        "source_cluster_normal_95",
        "source_cluster_jackknife_95",
    }
    if record["confidence_method"] not in confidence_methods:
        raise _compatibility_error(f"{field}.confidence_method is unsupported")
    if (
        expected_confidence_method is not None
        and record["confidence_method"] != expected_confidence_method
    ):
        raise _compatibility_error(
            f"{field}.confidence_method must be {expected_confidence_method!r}"
        )
    quantile_methods = {
        "higher",
        "source_equal_weighted_higher",
        "largest_threshold_meeting_false_omission_constraint",
    }
    if record["quantile_method"] not in quantile_methods:
        raise _compatibility_error(f"{field}.quantile_method is unsupported")
    if (
        expected_quantile_method is not None
        and record["quantile_method"] != expected_quantile_method
    ):
        raise _compatibility_error(
            f"{field}.quantile_method must be {expected_quantile_method!r}"
        )
    interval = record["confidence_interval"]
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise _compatibility_error(f"{field}.confidence_interval must have two bounds")
    lower = _finite_probability(interval[0], f"{field}.confidence_interval[0]")
    upper = _finite_probability(interval[1], f"{field}.confidence_interval[1]")
    if (
        lower > achieved + rate_tolerance(achieved)
        or upper + rate_tolerance(achieved) < achieved
        or lower > upper
    ):
        raise _compatibility_error(f"{field}.confidence_interval must contain achieved rate")
    if record["certified"] is not True:
        raise _compatibility_error(f"{field}.certified must be true")
    if natural:
        precision = _finite_probability(
            record["natural_precision"], f"{field}.natural_precision"
        )
        if not _same_rate(precision, 1.0 - achieved):
            raise _compatibility_error(
                f"{field}.natural_precision must equal one minus achieved rate"
            )
        for count_field in (
            "calibration_source_count",
            "natural_call_count",
            "natural_call_source_count",
        ):
            count = record[count_field]
            if type(count) is not int or count < 1:
                raise _compatibility_error(
                    f"{field}.{count_field} must be a positive integer"
                )
        false_omission_count = record["false_omission_count"]
        if type(false_omission_count) is not int or false_omission_count < 0:
            raise _compatibility_error(
                f"{field}.false_omission_count must be a non-negative integer"
            )
        if record["natural_call_source_count"] != record["sample_count"]:
            raise _compatibility_error(
                f"{field}.sample_count must equal natural_call_source_count"
            )
        if false_omission_count > record["natural_call_count"]:
            raise _compatibility_error(
                f"{field}.false_omission_count cannot exceed natural_call_count"
            )
        if record["calibration_source_count"] < record["natural_call_source_count"]:
            raise _compatibility_error(
                f"{field}.calibration_source_count cannot be below natural calls"
            )
        if record["calibration_source_count"] > record["record_count"]:
            raise _compatibility_error(
                f"{field}.calibration_source_count cannot exceed record_count"
            )
        if record["natural_call_source_count"] > record["natural_call_count"]:
            raise _compatibility_error(
                f"{field}.natural_call_source_count cannot exceed natural_call_count"
            )
        if record["natural_call_count"] > record["record_count"]:
            raise _compatibility_error(
                f"{field}.natural_call_count cannot exceed record_count"
            )
        count_rate = false_omission_count / record["natural_call_count"]
        if not _same_rate(achieved, count_rate):
            raise _compatibility_error(
                f"{field}.{achieved_field} must equal "
                "false_omission_count / natural_call_count"
            )
        expected_interval = wilson_interval(
            false_omission_count, record["natural_call_count"]
        )
        if not _same_rate(lower, expected_interval[0]) or not _same_rate(
            upper, expected_interval[1]
        ):
            raise _compatibility_error(
                f"{field}.confidence_interval must equal the 95% Wilson interval"
            )
    return threshold


def _validate_certified_calibration(
    kind: str, metadata: Mapping[str, Any], payload: Any
) -> None:
    manifest_hash = metadata["source_split_manifest_hash"]
    if re.fullmatch(r"[0-9a-f]{64}", manifest_hash) is None:
        raise _compatibility_error(
            "certified artifact source-split manifest hash must be 64 lowercase hex characters"
        )
    thresholds = metadata.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise _compatibility_error("certified artifact requires threshold calibration metadata")
    calibration = thresholds.get("calibration")
    if kind == "sequence":
        window = thresholds.get("window")
        local = thresholds.get("local")
        if not isinstance(window, Mapping) or not window or not isinstance(local, Mapping) or not local:
            raise _compatibility_error(
                "certified sequence artifact requires per-length window and local thresholds"
            )
        if not isinstance(calibration, Mapping):
            raise _compatibility_error("certified sequence artifact requires calibration metadata")
        canonical_by_kind: dict[str, dict[str, float]] = {}
        for calibration_kind, declared in (("window", window), ("local", local)):
            records = calibration.get(calibration_kind)
            if not isinstance(records, Mapping) or set(records) != set(declared):
                raise _compatibility_error(
                    f"sequence {calibration_kind} calibration must match supported length bins"
                )
            for length_bin, record in records.items():
                expected_unit = (
                    "source_mean_window_score"
                    if calibration_kind == "window"
                    else "source_max_window_score"
                )
                calibrated_threshold = _validate_calibration_record(
                    record,
                    field=f"sequence.{calibration_kind}.{length_bin}",
                    expected_sample_unit=expected_unit,
                    expected_confidence_method="wilson_95",
                    expected_quantile_method="source_equal_weighted_higher",
                )
                _validate_duplicate_threshold(
                    declared[length_bin],
                    calibrated_threshold,
                    field=(
                        f"sequence.{calibration_kind}.{length_bin}.declared_threshold"
                    ),
                )
                canonical_by_kind.setdefault(calibration_kind, {})[
                    length_bin
                ] = calibrated_threshold
        for duplicate_field, calibration_kind in (
            ("length_bin_thresholds", "window"),
            ("local_thresholds", "local"),
        ):
            if duplicate_field in metadata:
                _validate_threshold_mapping_duplicate(
                    metadata[duplicate_field],
                    canonical_by_kind[calibration_kind],
                    field=f"sequence.{duplicate_field}",
                )
        for duplicate_field in ("local_threshold", "local_window_threshold"):
            if duplicate_field in metadata:
                for length_bin, calibrated_threshold in canonical_by_kind["local"].items():
                    _validate_duplicate_threshold(
                        metadata[duplicate_field],
                        calibrated_threshold,
                        field=f"sequence.{duplicate_field}.{length_bin}",
                    )
        return
    if kind in {"cds", "pav"}:
        expected_unit = (
            "source_max_orf_score"
            if kind == "cds"
            else "source_max_pav_window_score"
        )
        calibrated_threshold = _validate_calibration_record(
            calibration,
            field=kind,
            expected_sample_unit=expected_unit,
            expected_confidence_method="wilson_95",
            expected_quantile_method="source_equal_weighted_higher",
        )
        declared_key = "orf" if kind == "cds" else "window"
        _validate_duplicate_threshold(
            thresholds.get(declared_key),
            calibrated_threshold,
            field=f"{kind}.{declared_key}",
        )
        if kind == "cds":
            if "orf_threshold" in metadata:
                _validate_duplicate_threshold(
                    metadata["orf_threshold"],
                    calibrated_threshold,
                    field="cds.orf_threshold",
                )
            if "orf_threshold" in thresholds:
                _validate_duplicate_threshold(
                    thresholds["orf_threshold"],
                    calibrated_threshold,
                    field="cds.thresholds.orf_threshold",
                )
        else:
            if not isinstance(payload, Mapping):
                raise _compatibility_error("certified pav payload must be a mapping")
            if "window_threshold" in payload:
                _validate_duplicate_threshold(
                    payload["window_threshold"],
                    calibrated_threshold,
                    field="pav.payload.window_threshold",
                )
        return
    if kind in {"router_cpu", "router_full"}:
        if not isinstance(calibration, Mapping):
            raise _compatibility_error(f"certified {kind} requires router calibration metadata")
        ai_threshold = _validate_calibration_record(
            calibration.get("ai"),
            field=f"{kind}.ai",
            expected_sample_unit="source_blocked_full_sequence_record",
            expected_quantile_method="source_equal_weighted_higher",
        )
        natural_threshold = _validate_calibration_record(
            calibration.get("natural"),
            field=f"{kind}.natural",
            expected_sample_unit="source_blocked_full_sequence_record",
            expected_confidence_method="wilson_95",
            expected_quantile_method=(
                "largest_threshold_meeting_false_omission_constraint"
            ),
            natural=True,
        )
        _validate_duplicate_threshold(
            thresholds.get("ai_min"), ai_threshold, field=f"{kind}.ai_min"
        )
        _validate_duplicate_threshold(
            thresholds.get("natural_max"),
            natural_threshold,
            field=f"{kind}.natural_max",
        )


def validate_artifact(artifact: Any, expected_kind: str | None = None) -> dict[str, Any]:
    """Validate one already-loaded v4 artifact against the strict root contract."""

    if not isinstance(artifact, dict):
        raise _compatibility_error("artifact root must be a dictionary")

    keys = set(artifact)
    if keys != _ROOT_KEYS:
        missing = sorted(_ROOT_KEYS - keys)
        unexpected = sorted(keys - _ROOT_KEYS)
        details: list[str] = []
        if missing:
            details.append(f"missing root keys: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected root keys: {', '.join(unexpected)}")
        raise _compatibility_error("; ".join(details))

    if artifact["schema_version"] != SCHEMA_VERSION:
        raise _compatibility_error(
            f"unsupported schema version {artifact['schema_version']!r}; expected {SCHEMA_VERSION!r}"
        )
    if type(artifact["model_version"]) is not int or artifact["model_version"] != MODEL_VERSION:
        raise _compatibility_error(
            f"unsupported model version {artifact['model_version']!r}; expected {MODEL_VERSION}"
        )
    if not isinstance(artifact["artifact_kind"], str) or not artifact["artifact_kind"]:
        raise _compatibility_error("artifact_kind must be a non-empty string")
    _validate_feature_names(artifact["feature_names"])
    metadata = _validate_metadata(artifact["metadata"])
    if _certified(metadata):
        _validate_certified_calibration(
            artifact["artifact_kind"], metadata, artifact["payload"]
        )
    if expected_kind is not None and artifact["artifact_kind"] != expected_kind:
        raise _compatibility_error(
            f"expected {expected_kind} artifact, found {artifact['artifact_kind']}"
        )
    return artifact


def artifact_is_certified(artifact: Mapping[str, Any]) -> bool:
    """Return the exact certification claim from a validated v4 artifact."""

    certification = artifact["metadata"]["certification_status"]
    return certification if type(certification) is bool else certification["certified"]


def validate_artifact_compatibility(
    artifacts: Mapping[str, Mapping[str, Any]],
) -> None:
    """Fail closed unless checkpoint components share provenance and runtime metadata."""

    if not isinstance(artifacts, Mapping) or not artifacts:
        raise _compatibility_error("component artifacts must be a non-empty mapping")
    validated = {
        name: validate_artifact(artifact)
        for name, artifact in artifacts.items()
    }
    manifest_hashes = {
        artifact["metadata"]["source_split_manifest_hash"]
        for artifact in validated.values()
    }
    if len(manifest_hashes) != 1:
        raise _compatibility_error(
            "component source-split manifest hashes are not identical"
        )
    dependencies = {
        repr(sorted(artifact["metadata"]["dependency_versions"].items()))
        for artifact in validated.values()
    }
    if len(dependencies) != 1:
        raise _compatibility_error("component dependency metadata is incompatible")

    def common_training_config(artifact: Mapping[str, Any]) -> dict[str, Any]:
        config = dict(artifact["metadata"]["training_config"])
        for component_key in ("stage", "profile", "kind"):
            config.pop(component_key, None)
        return config

    configs = {
        repr(sorted(common_training_config(artifact).items()))
        for artifact in validated.values()
    }
    if len(configs) != 1:
        raise _compatibility_error("component training config metadata is incompatible")


def save_artifact(
    path: str | Path,
    kind: str,
    payload: Any,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and persist one self-contained, versioned joblib dictionary."""

    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    normalized_metadata = _default_metadata()
    normalized_metadata.update(dict(metadata))
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "artifact_kind": kind,
        "feature_names": list(normalized_metadata.get("feature_names", [])),
        "metadata": normalized_metadata,
        "payload": payload,
    }
    validate_artifact(artifact)
    joblib.dump(artifact, Path(path))
    return artifact


def load_artifact(path: str | Path, expected_kind: str) -> dict[str, Any]:
    """Load a joblib artifact only when it exactly matches the v4 contract."""

    if not isinstance(expected_kind, str) or not expected_kind:
        raise ValueError("expected_kind must be a non-empty string")
    try:
        artifact = joblib.load(Path(path))
    except ArtifactCompatibilityError:
        raise
    except Exception as error:
        raise _compatibility_error(f"could not load artifact: {error}") from error
    return validate_artifact(artifact, expected_kind=expected_kind)
