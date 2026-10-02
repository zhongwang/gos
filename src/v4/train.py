"""Source-disjoint training for strict GOS v4 artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from numbers import Real
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .artifacts import (
    MODEL_VERSION,
    SCHEMA_VERSION,
    artifact_is_certified,
    validate_artifact,
)
from .calibration import (
    CalibrationError,
    ManifestRecord,
    NaturalSupportModel,
    calibrate_ai_threshold,
    calibrate_natural_threshold,
    grouped_split,
)
from .cds import (
    _FEATURE_NAMES as CDS_STAGE_FEATURE_NAMES,
    _orf_features,
    find_orfs,
)
from .pav import ObserverConfig, limit_pav_candidate_windows
from .preprocessing import ScoringWindow, pav_candidate_windows, prepare_sequence, make_windows
from .pipeline import (
    CPU_FEATURE_NAMES,
    FULL_FEATURE_NAMES,
    STRATEGY_B_FEATURE_NAMES,
    _feature_values,
)
from .sequence import (
    _FEATURE_NAMES as SEQUENCE_STAGE_FEATURE_NAMES,
    _window_features,
)
from .types import Span, StageStatus


@dataclass(frozen=True)
class ManifestEntry:
    """One exact source identity and its source-level split assignment."""

    fasta: Path
    manifest_fasta: str
    label: int
    source_id: str
    role: str
    generator: str | None = None
    domain: str | None = None
    group: str | None = None
    mosaic_spans: str | None = None


def _required_text(row: dict[str, str | None], field: str, line_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest row {line_number} requires non-empty {field}")
    return value.strip()


def read_manifest(path: str | Path, *, split_seed: int = 7) -> tuple[ManifestEntry, ...]:
    """Read a TSV manifest, resolve FASTA paths, and assign whole-source roles."""

    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        for required in ("fasta", "label", "source_id"):
            if required not in fieldnames:
                raise ValueError(f"manifest requires {required} column")
        group_fields = tuple(
            field for field in ("group", "group_id", "split_group") if field in fieldnames
        )
        if "role" not in fieldnames and not group_fields:
            raise ValueError(
                "manifest requires role column or an explicit group/group_id/split_group column"
            )
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError("manifest must contain at least one source")

    parsed: list[dict[str, Any]] = []
    for line_number, row in enumerate(raw_rows, start=2):
        fasta_text = _required_text(row, "fasta", line_number)
        source_id = _required_text(row, "source_id", line_number)
        label_text = _required_text(row, "label", line_number)
        try:
            label = int(label_text)
        except ValueError as error:
            raise ValueError(f"manifest row {line_number} label must be 0 or 1") from error
        if label_text not in {"0", "1"} or label not in {0, 1}:
            raise ValueError(f"manifest row {line_number} label must be 0 or 1")
        role_text = (row.get("role") or "").strip() or None
        declared_group = next(
            (
                (row.get(field) or "").strip()
                for field in group_fields
                if (row.get(field) or "").strip()
            ),
            None,
        )
        if role_text is None and declared_group is None:
            raise ValueError(
                f"manifest row {line_number} requires role or a non-empty explicit group key"
            )
        group = declared_group or source_id
        fasta = Path(fasta_text).expanduser()
        if not fasta.is_absolute():
            fasta = manifest_path.parent / fasta
        parsed.append(
            {
                "fasta": fasta.resolve(),
                "manifest_fasta": fasta_text,
                "label": label,
                "source_id": source_id,
                "role": role_text,
                "generator": (row.get("generator") or "").strip() or None,
                "domain": (row.get("domain") or "").strip() or None,
                "group": group,
                "mosaic_spans": (row.get("mosaic_spans") or "").strip() or None,
            }
        )

    roles = {row["role"] for row in parsed}
    if roles == {None}:
        groups = sorted({row["group"] for row in parsed})
        if len(groups) < 3:
            raise ValueError("at least three explicit source groups are required for splitting")
        shuffled = np.random.default_rng(split_seed).permutation(groups).tolist()
        train_end = int(np.floor(0.6 * len(shuffled)))
        calibration_end = int(np.floor(0.8 * len(shuffled)))
        assigned = {
            group: (
                "train"
                if index < train_end
                else "calibration"
                if index < calibration_end
                else "test"
            )
            for index, group in enumerate(shuffled)
        }
        for row in parsed:
            row["role"] = assigned[row["group"]]
    elif None in roles:
        raise ValueError("manifest roles must be either explicit for every row or absent for every row")
    grouped_split(
        [
            ManifestRecord(
                row["manifest_fasta"], row["label"], row["source_id"], row["role"]
            )
            for row in parsed
        ],
        seed=split_seed,
    )

    source_roles: dict[str, str] = {}
    source_labels: dict[str, int] = {}
    group_roles: dict[str, str] = {}
    for row in parsed:
        role = row["role"]
        previous_role = source_roles.setdefault(row["source_id"], role)
        if previous_role != role:
            raise ValueError(
                f"source_id {row['source_id']!r} appears in both {previous_role!r} "
                f"and {role!r} roles"
            )
        previous_label = source_labels.setdefault(row["source_id"], row["label"])
        if previous_label != row["label"]:
            raise ValueError(f"source_id {row['source_id']!r} has conflicting labels")
        previous_group_role = group_roles.setdefault(row["group"], role)
        if previous_group_role != role:
            raise ValueError(f"source group {row['group']!r} crosses split roles")

    return tuple(ManifestEntry(**row) for row in parsed)


def _validate_source_rows(sources: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    materialized = tuple(sources)
    if not materialized:
        raise ValueError("source rows must not be empty")
    source_roles: dict[str, str] = {}
    source_labels: dict[str, int] = {}
    for source in materialized:
        if not isinstance(source, Mapping):
            raise TypeError("source rows must be mappings")
        source_id = source.get("source_id")
        role = source.get("role")
        label = source.get("label")
        sequence = source.get("sequence")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("every source row requires a non-empty source_id")
        if role not in {"train", "calibration", "test"}:
            raise ValueError("every source row requires a valid role")
        if type(label) is not int or label not in {0, 1}:
            raise ValueError("every source row label must be an exact integer in {0, 1}")
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("every source row requires a non-empty sequence")
        prior_role = source_roles.setdefault(source_id, role)
        if prior_role != role:
            raise ValueError(f"source_id {source_id!r} crosses source roles")
        prior_label = source_labels.setdefault(source_id, label)
        if prior_label != label:
            raise ValueError(f"source_id {source_id!r} has conflicting labels")
    return materialized


def _dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": __import__("sklearn").__version__,
        "joblib": joblib.__version__,
    }


def _coverage_fraction(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number, not a boolean")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return result


def _positive_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _source_manifest_hash(sources: Sequence[Mapping[str, Any]]) -> str:
    identity = [
        {
            "source_id": source["source_id"],
            "sequence_id": source.get("sequence_id"),
            "role": source["role"],
            "label": source["label"],
        }
        for source in sources
    ]
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _fit_stage_components(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    estimator: Any | None,
    scaler: Any | None,
) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    if features.ndim != 2 or features.shape[0] == 0 or not np.all(np.isfinite(features)):
        raise ValueError("training features must be a non-empty finite matrix")
    if labels.shape != (features.shape[0],) or set(labels.tolist()) != {0, 1}:
        raise ValueError("training feature rows must contain both natural and AI labels")
    clip_lower = np.quantile(features, 0.01, axis=0, method="linear")
    clip_upper = np.quantile(features, 0.99, axis=0, method="linear")
    clipped = np.clip(features, clip_lower, clip_upper)
    fitted_scaler = scaler if scaler is not None else StandardScaler()
    if not callable(getattr(fitted_scaler, "fit", None)) or not callable(
        getattr(fitted_scaler, "transform", None)
    ):
        raise TypeError("scaler must provide fit and transform")
    fitted_scaler.fit(clipped)
    transformed = np.asarray(fitted_scaler.transform(clipped), dtype=float)
    if transformed.shape != clipped.shape or not np.all(np.isfinite(transformed)):
        raise ValueError("scaler returned invalid training features")
    fitted_estimator = estimator if estimator is not None else LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=7
    )
    if not callable(getattr(fitted_estimator, "fit", None)) or not callable(
        getattr(fitted_estimator, "predict_proba", None)
    ):
        raise TypeError("estimator must provide fit and predict_proba")
    fitted_estimator.fit(transformed, labels)
    return fitted_estimator, fitted_scaler, clip_lower, clip_upper


def _calibration_row_scores(
    feature_rows: Sequence[tuple[str, int, np.ndarray]],
    *,
    estimator: Any,
    scaler: Any,
    clip_lower: np.ndarray,
    clip_upper: np.ndarray,
    label: int,
) -> np.ndarray:
    """Score held-out evidence at the unit where its threshold is applied."""

    selected = [row for row in feature_rows if row[1] == label]
    if not selected:
        kind = "natural" if label == 0 else "AI"
        raise ValueError(f"calibration role contains no scoreable {kind} evidence")
    matrix = np.asarray([row[2] for row in selected], dtype=float)
    transformed = np.asarray(
        scaler.transform(np.clip(matrix, clip_lower, clip_upper)), dtype=float
    )
    return _positive_scores(estimator, transformed)


def _scored_calibration_rows(
    feature_rows: Sequence[tuple[str, int, np.ndarray]],
    *,
    estimator: Any,
    scaler: Any,
    clip_lower: np.ndarray,
    clip_upper: np.ndarray,
    label: int,
) -> tuple[list[tuple[str, int, np.ndarray]], np.ndarray]:
    selected = [row for row in feature_rows if row[1] == label]
    scores = _calibration_row_scores(
        feature_rows,
        estimator=estimator,
        scaler=scaler,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
        label=label,
    )
    return selected, scores


def _source_summary_scores(
    rows: Sequence[tuple[str, int, np.ndarray]],
    scores: Sequence[float],
    *,
    reducer: str,
) -> np.ndarray:
    grouped: dict[str, list[float]] = {}
    for row, score in zip(rows, scores, strict=True):
        grouped.setdefault(row[0], []).append(float(score))
    if reducer == "mean":
        summary = [np.mean(values) for values in grouped.values()]
    elif reducer == "max":
        summary = [np.max(values) for values in grouped.values()]
    else:
        raise ValueError("reducer must be mean or max")
    return np.asarray(summary, dtype=float)


def _source_summary_ids(
    rows: Sequence[tuple[str, int, np.ndarray]],
) -> list[str]:
    """Return source IDs in the same first-seen order as summary scores."""

    return list(dict.fromkeys(row[0] for row in rows))


def _stage_metadata(
    sources: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    calibration: Any,
    training_config: Mapping[str, Any] | None,
    source_split_manifest_hash: str | None,
) -> dict[str, Any]:
    if isinstance(calibration, Mapping):
        flattened = [
            item
            for value in calibration.values()
            for item in (value.values() if isinstance(value, Mapping) else (value,))
        ]
        certification_status: dict[str, Any] = {
            "certified": bool(flattened and all(item.certified for item in flattened)),
            "length_bins": {
                name: (
                    {
                        calibration_kind: item.certification_status
                        for calibration_kind, item in value.items()
                    }
                    if isinstance(value, Mapping)
                    else value.certification_status
                )
                for name, value in calibration.items()
            },
        }
    else:
        certification_status = calibration.certification_status
    return {
        "training_config": dict(training_config or {"split": "source", "stage": kind}),
        "source_split_manifest_hash": source_split_manifest_hash
        or _source_manifest_hash(sources),
        "dependency_versions": _dependency_versions(),
        "certification_status": certification_status,
    }


def fit_sequence_stage(
    sources: Sequence[Mapping[str, Any]],
    *,
    estimator: Any | None = None,
    scaler: Any | None = None,
    target_fpr: float = 0.01,
    allow_uncertified: bool = False,
    training_config: Mapping[str, Any] | None = None,
    source_split_manifest_hash: str | None = None,
) -> dict[str, Any]:
    """Fit the source-window sequence classifier and held-out local threshold."""

    rows = _validate_source_rows(sources)
    train_features: list[np.ndarray] = []
    train_labels: list[int] = []
    calibration_features: list[tuple[str, int, str, np.ndarray]] = []
    for source in rows:
        if source["role"] not in {"train", "calibration"}:
            continue
        prepared = prepare_sequence(source["sequence"])
        windows = make_windows(prepared)
        for window in windows:
            features = _window_features(window)
            if source["role"] == "train":
                train_features.append(features)
                train_labels.append(source["label"])
            else:
                calibration_features.append(
                    (source["source_id"], source["label"], window.length_bin, features)
                )
    estimator, scaler, clip_lower, clip_upper = _fit_stage_components(
        np.asarray(train_features, dtype=float),
        np.asarray(train_labels, dtype=int),
        estimator=estimator,
        scaler=scaler,
    )
    calibrations: dict[str, dict[str, Any]] = {}
    natural_bins = sorted(
        {length_bin for _, label, length_bin, _ in calibration_features if label == 0}
    )
    for length_bin in natural_bins:
        bin_rows = [
            (source_id, label, features)
            for source_id, label, row_bin, features in calibration_features
            if row_bin == length_bin
        ]
        natural_rows, scores = _scored_calibration_rows(
            bin_rows,
            estimator=estimator,
            scaler=scaler,
            clip_lower=clip_lower,
            clip_upper=clip_upper,
            label=0,
        )
        summary_source_ids = _source_summary_ids(natural_rows)
        calibrations[length_bin] = {
            "window": calibrate_ai_threshold(
                _source_summary_scores(natural_rows, scores, reducer="mean"),
                target_fpr,
                allow_uncertified=allow_uncertified,
                source_ids=summary_source_ids,
                sample_unit="source_mean_window_score",
            ),
            "local": calibrate_ai_threshold(
                _source_summary_scores(natural_rows, scores, reducer="max"),
                target_fpr,
                allow_uncertified=allow_uncertified,
                source_ids=summary_source_ids,
                sample_unit="source_max_window_score",
            ),
        }
    if not calibrations:
        raise ValueError("calibration role contains no natural sequence windows")
    metadata = _stage_metadata(
        rows,
        kind="sequence",
        calibration=calibrations,
        training_config=training_config,
        source_split_manifest_hash=source_split_manifest_hash,
    )
    metadata.update(
        {
            "length_bin_thresholds": {
                length_bin: float(calibration["window"].threshold)
                for length_bin, calibration in calibrations.items()
            },
            "local_thresholds": {
                length_bin: float(calibration["local"].threshold)
                for length_bin, calibration in calibrations.items()
            },
            "thresholds": {
                "window": {
                    length_bin: float(calibration["window"].threshold)
                    for length_bin, calibration in calibrations.items()
                },
                "local": {
                    length_bin: float(calibration["local"].threshold)
                    for length_bin, calibration in calibrations.items()
                },
                "calibration": {
                    "window": {
                        length_bin: _calibration_dict(calibration["window"])
                        for length_bin, calibration in calibrations.items()
                    },
                    "local": {
                        length_bin: _calibration_dict(calibration["local"])
                        for length_bin, calibration in calibrations.items()
                    },
                },
            },
        }
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "artifact_kind": "sequence",
        "feature_names": list(SEQUENCE_STAGE_FEATURE_NAMES),
        "metadata": metadata,
        "payload": {
            "estimator": estimator,
            "scaler": scaler,
            "clip_lower": clip_lower,
            "clip_upper": clip_upper,
        },
    }
    return validate_artifact(artifact, expected_kind="sequence")


def _codon_reference_tables(
    natural_training_orfs: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    codon_counts: dict[str, int] = {}
    pair_counts: dict[str, int] = {}
    for sequence in natural_training_orfs:
        codons = [sequence[offset : offset + 3] for offset in range(0, len(sequence) - 3, 3)]
        for codon in codons:
            codon_counts[codon] = codon_counts.get(codon, 0) + 1
        for first, second in zip(codons, codons[1:]):
            pair = first + second
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    if not codon_counts:
        raise ValueError("natural train role contains no complete ORFs for CDS reference fitting")
    maximum = max(codon_counts.values())
    cai_weights = {codon: count / maximum for codon, count in codon_counts.items()}
    pair_total = sum(pair_counts.values())
    cps_table = {
        pair: float(np.log((count + 1.0) / (pair_total + len(pair_counts))))
        for pair, count in pair_counts.items()
    }
    return cai_weights, cps_table


def fit_cds_stage(
    sources: Sequence[Mapping[str, Any]],
    *,
    estimator: Any | None = None,
    scaler: Any | None = None,
    target_fpr: float = 0.01,
    allow_uncertified: bool = False,
    training_config: Mapping[str, Any] | None = None,
    source_split_manifest_hash: str | None = None,
) -> dict[str, Any]:
    """Fit the global-background ORF classifier and held-out ORF threshold."""

    rows = _validate_source_rows(sources)
    orfs_by_source: list[tuple[Mapping[str, Any], Any]] = []
    natural_training_orfs: list[str] = []
    for source in rows:
        if source["role"] not in {"train", "calibration"}:
            continue
        prepared = prepare_sequence(source["sequence"], min_fragment_length=1)
        for fragment in prepared.fragments:
            for orf in find_orfs(fragment.sequence):
                orfs_by_source.append((source, orf))
                if source["role"] == "train" and source["label"] == 0:
                    natural_training_orfs.append(orf.sequence)
    cai_weights, cps_table = _codon_reference_tables(natural_training_orfs)
    train_features: list[np.ndarray] = []
    train_labels: list[int] = []
    calibration_features: list[tuple[str, int, np.ndarray]] = []
    for source, orf in orfs_by_source:
        features = _orf_features(orf, cai_weights, cps_table)
        if source["role"] == "train":
            train_features.append(features)
            train_labels.append(source["label"])
        else:
            calibration_features.append((source["source_id"], source["label"], features))
    estimator, scaler, clip_lower, clip_upper = _fit_stage_components(
        np.asarray(train_features, dtype=float),
        np.asarray(train_labels, dtype=int),
        estimator=estimator,
        scaler=scaler,
    )
    natural_rows, natural_scores = _scored_calibration_rows(
        calibration_features,
        estimator=estimator,
        scaler=scaler,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
        label=0,
    )
    calibration = calibrate_ai_threshold(
        _source_summary_scores(natural_rows, natural_scores, reducer="max"),
        target_fpr,
        allow_uncertified=allow_uncertified,
        source_ids=_source_summary_ids(natural_rows),
        sample_unit="source_max_orf_score",
    )
    metadata = _stage_metadata(
        rows,
        kind="cds",
        calibration=calibration,
        training_config=training_config,
        source_split_manifest_hash=source_split_manifest_hash,
    )
    metadata.update(
        {
            "background": "global",
            "orf_threshold": float(calibration.threshold),
            "thresholds": {
                "orf": float(calibration.threshold),
                "calibration": _calibration_dict(calibration),
            },
        }
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "artifact_kind": "cds",
        "feature_names": list(CDS_STAGE_FEATURE_NAMES),
        "metadata": metadata,
        "payload": {
            "estimator": estimator,
            "scaler": scaler,
            "clip_lower": clip_lower,
            "clip_upper": clip_upper,
            "cai_weights": cai_weights,
            "cps_table": cps_table,
        },
    }
    return validate_artifact(artifact, expected_kind="cds")


def _row_value(row: Mapping[str, Any], name: str) -> Any:
    features = row.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("aggregate row features must be a mapping")
    return features.get(name)


def _validate_aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("aggregate rows must not be empty")
    source_roles: dict[str, str] = {}
    source_labels: dict[str, int] = {}
    for row in materialized:
        if not isinstance(row, Mapping):
            raise TypeError("aggregate rows must be mappings")
        source_id = row.get("source_id")
        role = row.get("role")
        label = row.get("label")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("every aggregate row requires a non-empty source_id")
        if role not in {"train", "calibration", "test"}:
            raise ValueError("every aggregate row requires a valid role")
        if type(label) is not int or label not in {0, 1}:
            raise ValueError("every aggregate row label must be an exact integer in {0, 1}")
        prior_role = source_roles.setdefault(source_id, role)
        if prior_role != role:
            raise ValueError(f"source_id {source_id!r} crosses aggregate roles")
        prior_label = source_labels.setdefault(source_id, label)
        if prior_label != label:
            raise ValueError(f"source_id {source_id!r} has conflicting labels")
        if not isinstance(row.get("features"), Mapping):
            raise ValueError("aggregate row features must be a mapping")
    return materialized


def _fit_matrix(
    rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> tuple[np.ndarray, dict[str, float]]:
    imputation_values: dict[str, float] = {}
    for name in feature_names:
        available: list[float] = []
        for row in rows:
            value = _row_value(row, name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(f"router feature {name!r} must be finite or None")
            available.append(float(value))
        imputation_values[name] = float(np.median(available)) if available else 0.0
    return _matrix(rows, feature_names, imputation_values), imputation_values


def _matrix(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    imputation_values: Mapping[str, float],
) -> np.ndarray:
    values: list[list[float]] = []
    for row in rows:
        current: list[float] = []
        for name in feature_names:
            value = _row_value(row, name)
            if value is None:
                value = imputation_values[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(f"router feature {name!r} must be finite after imputation")
            current.append(float(value))
        values.append(current)
    return np.asarray(values, dtype=float)


def _positive_scores(estimator: Any, transformed: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(transformed), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != transformed.shape[0]:
        raise ValueError("estimator returned invalid probabilities")
    classes = np.asarray(getattr(estimator, "classes_", [0, 1]))
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1 or probabilities.shape[1] <= int(matches[0]):
        raise ValueError("estimator must expose one AI class labelled 1")
    scores = probabilities[:, int(matches[0])]
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("estimator returned invalid AI probabilities")
    return scores


def _manifest_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    identity = [
        {"source_id": row["source_id"], "role": row["role"], "label": row["label"]}
        for row in rows
    ]
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _calibration_dict(calibration: Any) -> dict[str, Any]:
    output = dict(vars(calibration))
    interval = output.get("confidence_interval")
    if interval is not None:
        output["confidence_interval"] = list(interval)
    return output


def fit_router(
    aggregate_rows: Sequence[Mapping[str, Any]],
    *,
    kind: str = "router_cpu",
    feature_names: Sequence[str] | None = None,
    estimator: Any | None = None,
    scaler: Any | None = None,
    target_fpr: float = 0.01,
    target_fnr: float = 0.01,
    allow_uncertified: bool = False,
    min_acgt_coverage: float = 0.8,
    min_sequence_score_coverage: float = 1.0,
    min_fragment_length: int = 500,
    support_model: NaturalSupportModel | None = None,
    training_config: Mapping[str, Any] | None = None,
    source_split_manifest_hash: str | None = None,
) -> dict[str, Any]:
    """Fit one router on train rows and calibrate it on held-out source rows."""

    if kind not in {"router_cpu", "router_full"}:
        raise ValueError("kind must be router_cpu or router_full")
    names = tuple(
        feature_names
        if feature_names is not None
        else CPU_FEATURE_NAMES
        if kind == "router_cpu"
        else FULL_FEATURE_NAMES
    )
    if (
        not names
        or len(set(names)) != len(names)
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("feature_names must contain unique non-empty strings")
    if type(allow_uncertified) is not bool:
        raise TypeError("allow_uncertified must be an exact boolean")
    min_acgt_coverage = _coverage_fraction(
        min_acgt_coverage, field="min_acgt_coverage"
    )
    min_sequence_score_coverage = _coverage_fraction(
        min_sequence_score_coverage, field="min_sequence_score_coverage"
    )
    min_fragment_length = _positive_int(
        min_fragment_length, field="min_fragment_length"
    )
    rows = _validate_aggregate_rows(aggregate_rows)
    train_rows = [row for row in rows if row["role"] == "train"]
    calibration_rows = [row for row in rows if row["role"] == "calibration"]
    labels = np.asarray([row["label"] for row in train_rows], dtype=int)
    if len(train_rows) == 0 or set(labels.tolist()) != {0, 1}:
        raise ValueError("train role must contain both natural and AI sources")
    if not calibration_rows:
        raise ValueError("calibration role must not be empty")

    train_matrix, imputations = _fit_matrix(train_rows, names)
    fitted_scaler = scaler if scaler is not None else StandardScaler()
    if not callable(getattr(fitted_scaler, "fit", None)) or not callable(
        getattr(fitted_scaler, "transform", None)
    ):
        raise TypeError("scaler must provide fit and transform")
    fitted_scaler.fit(train_matrix)
    transformed_train = np.asarray(fitted_scaler.transform(train_matrix), dtype=float)
    if transformed_train.shape != train_matrix.shape or not np.all(np.isfinite(transformed_train)):
        raise ValueError("scaler returned invalid training features")

    if support_model is None:
        natural_train = transformed_train[labels == 0]
        if natural_train.shape[0] < 2:
            raise ValueError("train role requires at least two natural rows for support fitting")
        support_model = NaturalSupportModel.fit(natural_train)

        all_matrix = _matrix(rows, names, imputations)
        transformed_all = np.asarray(fitted_scaler.transform(all_matrix), dtype=float)
        if transformed_all.shape != all_matrix.shape or not np.all(
            np.isfinite(transformed_all)
        ):
            raise ValueError("scaler returned invalid support features")
        for row, transformed in zip(rows, transformed_all, strict=True):
            features = row["features"]
            if not isinstance(features, dict):
                raise TypeError("aggregate row features must be mutable dictionaries")
            features["out_of_support"] = float(
                support_model.is_out_of_support(transformed)
            )

        # The classifier sees the real support indicator, while the support
        # model itself was fit first with the provisional false indicator used
        # at inference for its support check.
        train_matrix = _matrix(train_rows, names, imputations)
        transformed_train = np.asarray(
            fitted_scaler.transform(train_matrix), dtype=float
        )
    elif not callable(getattr(support_model, "is_out_of_support", None)):
        raise TypeError("support_model must provide is_out_of_support(features)")

    fitted_estimator = estimator or LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=7
    )
    if not callable(getattr(fitted_estimator, "fit", None)) or not callable(
        getattr(fitted_estimator, "predict_proba", None)
    ):
        raise TypeError("estimator must provide fit and predict_proba")
    fitted_estimator.fit(transformed_train, labels)

    calibration_matrix = _matrix(calibration_rows, names, imputations)
    transformed_calibration = np.asarray(
        fitted_scaler.transform(calibration_matrix), dtype=float
    )
    calibration_scores = _positive_scores(fitted_estimator, transformed_calibration)
    calibration_labels = np.asarray([row["label"] for row in calibration_rows], dtype=int)
    calibration_source_ids = [row["source_id"] for row in calibration_rows]
    natural_mask = calibration_labels == 0
    if not np.any(natural_mask) or not np.any(calibration_labels == 1):
        raise ValueError("calibration role must contain both natural and AI records")
    ai_calibration = calibrate_ai_threshold(
        calibration_scores[natural_mask],
        target_fpr,
        allow_uncertified=allow_uncertified,
        source_ids=[
            source_id
            for source_id, is_natural in zip(
                calibration_source_ids, natural_mask, strict=True
            )
            if is_natural
        ],
        sample_unit="source_blocked_full_sequence_record",
    )
    natural_calibration = calibrate_natural_threshold(
        calibration_scores,
        calibration_labels,
        target_false_omission_rate=target_fnr,
        allow_uncertified=allow_uncertified,
        source_ids=calibration_source_ids,
        sample_unit="source_blocked_full_sequence_record",
        max_threshold=float(np.nextafter(ai_calibration.threshold, -np.inf)),
    )
    ai_min = float(ai_calibration.threshold)
    natural_max = float(natural_calibration.threshold)
    if not 0.0 <= natural_max < ai_min <= 1.0:
        raise CalibrationError("calibration cannot produce ordered natural and AI thresholds")

    certified = bool(ai_calibration.certified and natural_calibration.certified)
    calibration_metadata = {
        "ai": _calibration_dict(ai_calibration),
        "natural": _calibration_dict(natural_calibration),
    }
    metadata = {
        "training_config": dict(training_config or {"split": "source", "profile": kind}),
        "source_split_manifest_hash": source_split_manifest_hash or _manifest_hash(rows),
        "dependency_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit-learn": __import__("sklearn").__version__,
            "joblib": joblib.__version__,
        },
        "certification_status": {
            "certified": certified,
            "ai": ai_calibration.certification_status,
            "natural": natural_calibration.certification_status,
        },
        "thresholds": {
            "natural_max": natural_max,
            "ai_min": ai_min,
            "calibration": calibration_metadata,
        },
        "min_acgt_coverage": min_acgt_coverage,
        "min_sequence_score_coverage": min_sequence_score_coverage,
        "min_fragment_length": min_fragment_length,
        "allow_sequence_unavailable_routing": (
            min_fragment_length < 500 and min_sequence_score_coverage == 0.0
        ),
    }
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "artifact_kind": kind,
        "feature_names": list(names),
        "metadata": metadata,
        "payload": {
            "estimator": fitted_estimator,
            "scaler": fitted_scaler,
            "imputation_values": imputations,
            "support_model": support_model,
        },
    }
    return validate_artifact(artifact, expected_kind=kind)


def _read_fasta(entry: ManifestEntry) -> list[dict[str, Any]]:
    if not entry.fasta.is_file():
        raise ValueError(
            f"FASTA for source_id {entry.source_id!r} does not exist: "
            f"{entry.manifest_fasta}"
        )
    records: list[dict[str, Any]] = []
    sequence_id: str | None = None
    chunks: list[str] = []

    def append_record() -> None:
        if sequence_id is None:
            return
        sequence = "".join(chunks).replace(" ", "").upper()
        if not sequence:
            raise ValueError(
                f"FASTA {entry.manifest_fasta!r} contains empty record {sequence_id!r}"
            )
        records.append(
            {
                "source_id": entry.source_id,
                "sequence_id": sequence_id,
                "role": entry.role,
                "label": entry.label,
                "sequence": sequence,
                "metadata": {
                    "fasta": entry.manifest_fasta,
                    "generator": entry.generator,
                    "domain": entry.domain,
                    "group": entry.group,
                    "mosaic_spans": entry.mosaic_spans,
                },
            }
        )

    with entry.fasta.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                append_record()
                sequence_id = line[1:].split(maxsplit=1)[0]
                if not sequence_id:
                    raise ValueError(
                        f"FASTA {entry.manifest_fasta!r} has an empty header at line {line_number}"
                    )
                chunks = []
            else:
                if sequence_id is None:
                    raise ValueError(
                        f"FASTA {entry.manifest_fasta!r} must begin with a header"
                    )
                chunks.append(line)
    append_record()
    if not records:
        raise ValueError(f"FASTA {entry.manifest_fasta!r} contains no records")
    sequence_ids = [record["sequence_id"] for record in records]
    if len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError(f"FASTA {entry.manifest_fasta!r} contains duplicate record IDs")
    return records


def _manifest_entries_hash(entries: Sequence[ManifestEntry]) -> str:
    canonical = [
        {
            "fasta": entry.manifest_fasta,
            "label": entry.label,
            "source_id": entry.source_id,
            "role": entry.role,
            "generator": entry.generator,
            "domain": entry.domain,
            "group": entry.group,
            "mosaic_spans": entry.mosaic_spans,
        }
        for entry in entries
    ]
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _aggregate_records(
    sources: Sequence[Mapping[str, Any]],
    sequence_artifact: Mapping[str, Any],
    cds_artifact: Mapping[str, Any],
    *,
    pav_stage: Any | None = None,
    min_fragment_length: int = 500,
    allow_uncertified: bool = False,
) -> list[dict[str, Any]]:
    from .cds import CDSStage
    from .sequence import SequenceStage

    sequence_stage = SequenceStage(
        sequence_artifact, allow_uncertified=allow_uncertified
    )
    cds_stage = CDSStage(cds_artifact, allow_uncertified=allow_uncertified)
    min_fragment_length = _positive_int(
        min_fragment_length, field="min_fragment_length"
    )
    records: list[dict[str, Any]] = []
    for source in sources:
        prepared = prepare_sequence(
            source["sequence"],
            min_fragment_length=min_fragment_length,
        )
        sequence = sequence_stage.score(prepared)
        cds = cds_stage.score(prepared)
        pav = (
            pav_stage.score(prepared, source_id=source["source_id"])
            if pav_stage is not None
            else None
        )
        features = _feature_values(
            prepared, sequence, cds, pav, out_of_support=False
        )
        records.append(
            {
                "source_id": source["source_id"],
                "sequence_id": source["sequence_id"],
                "role": source["role"],
                "label": source["label"],
                "metadata": _json_value(source.get("metadata", {})),
                "stage_aggregates": {
                    "sequence": _json_value(sequence.aggregates),
                    "cds": _json_value(cds.aggregates),
                    "pav": _json_value(pav.aggregates) if pav is not None else None,
                },
                "availability": {
                    "sequence": sequence.status.value,
                    "cds": cds.status.value,
                    "pav": pav.status.value if pav is not None else "not_requested",
                },
                "features": _json_value(features),
            }
        )
    return records


def _test_metrics(
    sources: Sequence[Mapping[str, Any]], detector: Any
) -> dict[str, Any]:
    """Measure untouched test sources through the production tri-state router."""

    test_sources = [source for source in sources if source["role"] == "test"]
    if not test_sources:
        raise ValueError("test role must not be empty")
    if not callable(getattr(detector, "scan", None)):
        raise TypeError("detector must provide scan(sequence, source_id)")
    results = [
        detector.scan(source["sequence"], source_id=source["source_id"])
        for source in test_sources
    ]
    predictions = [result.label.value for result in results]
    binary_predictions = [result.prediction for result in results]
    labels = [source["label"] for source in test_sources]
    decided = [
        (label, prediction)
        for label, prediction in zip(labels, binary_predictions, strict=True)
        if prediction is not None
    ]
    correct = sum(label == prediction for label, prediction in decided)
    natural_count = sum(label == 0 for label in labels)
    ai_count = sum(label == 1 for label in labels)
    false_positives = sum(
        label == 0 and prediction == 1
        for label, prediction in zip(labels, binary_predictions, strict=True)
    )
    true_positives = sum(
        label == 1 and prediction == 1
        for label, prediction in zip(labels, binary_predictions, strict=True)
    )
    reasons: list[str | None] = [result.reason for result in results]
    reason_counts = {
        reason: reasons.count(reason)
        for reason in sorted({reason for reason in reasons if reason is not None})
    }
    pav_failures = sum(
        result.pav_status is StageStatus.FAILED for result in results
    )
    return {
        "role": "test",
        "source_ids": list(
            dict.fromkeys(source["source_id"] for source in test_sources)
        ),
        "n_records": len(test_sources),
        "n_sources": len({source["source_id"] for source in test_sources}),
        "abstention_rate": 1.0 - len(decided) / len(test_sources),
        "conditional_accuracy": correct / len(decided) if decided else None,
        "fpr": false_positives / natural_count if natural_count else None,
        "recall": true_positives / ai_count if ai_count else None,
        "scores": [result.scores.get("router") for result in results],
        "predictions": predictions,
        "binary_predictions": binary_predictions,
        "labels": labels,
        "reasons": reasons,
        "reason_counts": reason_counts,
        "pav_trigger_rate": sum(result.pav_triggered for result in results)
        / len(results),
        "pav_failure_rate": pav_failures / len(results),
    }


def _artifact_calibration(artifact: Mapping[str, Any]) -> Any:
    thresholds = artifact["metadata"].get("thresholds", {})
    return thresholds.get("calibration")


def _pav_training_windows(
    source: Mapping[str, Any],
    *,
    max_windows_per_record: int | None = None,
) -> tuple[ScoringWindow, ...]:
    """Use the exact same deterministic pAV candidates as inference."""

    return limit_pav_candidate_windows(
        pav_candidate_windows(prepare_sequence(source["sequence"])),
        max_windows_per_record,
    )


def _pav_observer_configuration(
    observer: Any, observer_id: str
) -> ObserverConfig:
    """Return exact typed provenance for a production or injected observer."""

    declared = getattr(observer, "observer_config", None)
    if not isinstance(declared, ObserverConfig):
        raise ValueError("pAV observer requires an ObserverConfig")
    fingerprint = getattr(observer, "observer_fingerprint", None)
    if not isinstance(fingerprint, str) or fingerprint != declared.fingerprint:
        raise ValueError("pAV observer fingerprint must match its ObserverConfig")
    if not isinstance(observer_id, str) or not observer_id:
        raise ValueError("pAV observer_id must be a non-empty string")
    return declared


def _fit_pav_stage(
    train_sources: Sequence[Mapping[str, Any]],
    calibration_sources: Sequence[Mapping[str, Any]],
    *,
    observer: Any,
    observer_id: str,
    observer_config: ObserverConfig,
    cache: Any,
    target_fpr: float,
    allow_uncertified: bool,
    training_config: Mapping[str, Any],
    source_split_manifest_hash: str,
    estimator: Any | None = None,
    scaler: Any | None = None,
    n_heads: int = 8,
    max_windows_per_record: int | None = None,
) -> dict[str, Any]:
    """Fit signed pAV heads from multi-offset train/calibration observations."""

    from .pav import (
        _observer_cache_identity,
        observer_features,
        pav_cache_key,
        select_signed_heads,
        transform_signed_heads,
    )
    if not isinstance(observer_config, ObserverConfig):
        raise TypeError("observer_config must be an ObserverConfig")
    cache_observer_id = _observer_cache_identity(observer_id, observer_config)

    def observed_rows(
        sources: Sequence[Mapping[str, Any]],
    ) -> list[tuple[str, int, np.ndarray]]:
        output: list[tuple[str, int, np.ndarray]] = []
        for source in sources:
            for window in _pav_training_windows(
                source,
                max_windows_per_record=max_windows_per_record,
            ):
                key = pav_cache_key(
                    source["source_id"],
                    window.span.start,
                    window.span.end,
                    window.sequence,
                    cache_observer_id,
                    "gos-v4-pav-cache/1",
                )
                values = cache.get(key)
                if values is None:
                    observed = observer_features(observer(window.sequence))
                    cache[key] = observed
                else:
                    observed = observer_features(values)
                output.append((source["source_id"], source["label"], observed.pav))
        return output

    training_raw = observed_rows(train_sources)
    calibration_raw = observed_rows(calibration_sources)
    if not training_raw or not calibration_raw:
        raise ValueError("pAV fitting requires scoreable train and calibration windows")
    raw_matrix = np.asarray([row[2] for row in training_raw], dtype=float)
    labels = np.asarray([row[1] for row in training_raw], dtype=int)
    selected_count = min(n_heads, raw_matrix.shape[1])
    selection = select_signed_heads(raw_matrix, labels, selected_count)
    transformed_train = transform_signed_heads(
        raw_matrix,
        selection.heads,
        selection.signs,
        selection.centers,
        selection.scales,
    )
    estimator, scaler, clip_lower, clip_upper = _fit_stage_components(
        transformed_train,
        labels,
        estimator=estimator,
        scaler=scaler,
    )
    transformed_calibration = transform_signed_heads(
        np.asarray([row[2] for row in calibration_raw], dtype=float),
        selection.heads,
        selection.signs,
        selection.centers,
        selection.scales,
    )
    calibration_features = [
        (source_id, label, features)
        for (source_id, label, _), features in zip(
            calibration_raw, transformed_calibration, strict=True
        )
    ]
    natural_rows, natural_scores = _scored_calibration_rows(
        calibration_features,
        estimator=estimator,
        scaler=scaler,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
        label=0,
    )
    calibration = calibrate_ai_threshold(
        _source_summary_scores(natural_rows, natural_scores, reducer="max"),
        target_fpr,
        allow_uncertified=allow_uncertified,
        source_ids=_source_summary_ids(natural_rows),
        sample_unit="source_max_pav_window_score",
    )
    metadata = {
        "training_config": dict(training_config),
        "source_split_manifest_hash": source_split_manifest_hash,
        "dependency_versions": _dependency_versions(),
        "certification_status": calibration.certification_status,
        "observer_id": observer_id,
        "observer_config": observer_config.to_dict(),
        "observer_fingerprint": observer_config.fingerprint,
        "normalization": observer_config.normalization,
        "min_scored_coverage": 1.0,
        "max_windows_per_record": max_windows_per_record,
        "thresholds": {
            "window": float(calibration.threshold),
            "calibration": _calibration_dict(calibration),
        },
    }
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "artifact_kind": "pav",
        "feature_names": [f"pav_head_{head}" for head in selection.heads],
        "metadata": metadata,
        "payload": {
            "estimator": estimator,
            "scaler": scaler,
            "clip_lower": clip_lower,
            "clip_upper": clip_upper,
            "selected_heads": selection.heads,
            "signs": selection.signs,
            "centers": selection.centers,
            "scales": selection.scales,
            "window_threshold": float(calibration.threshold),
        },
    }
    return validate_artifact(artifact, expected_kind="pav")


def train_from_manifest(
    manifest: str | Path,
    out_dir: str | Path,
    *,
    skip_pav: bool = False,
    allow_uncertified: bool = False,
    split_seed: int = 7,
    target_fpr: float = 0.01,
    target_fnr: float = 0.01,
    min_acgt_coverage: float = 0.8,
    min_sequence_score_coverage: float = 1.0,
    min_fragment_length: int = 250,
    sequence_estimator: Any | None = None,
    sequence_scaler: Any | None = None,
    cds_estimator: Any | None = None,
    cds_scaler: Any | None = None,
    router_estimator: Any | None = None,
    router_scaler: Any | None = None,
    router_full_estimator: Any | None = None,
    router_full_scaler: Any | None = None,
    router_full_feature_names: Sequence[str] | None = None,
    pav_trainer: Any | None = None,
    pav_observer: Any | None = None,
    pav_observer_id: str | None = None,
    pav_cache: Any | None = None,
    max_pav_windows_per_record: int | None = None,
) -> dict[str, Any]:
    """Train v4 artifacts from a source-disjoint manifest."""

    if type(skip_pav) is not bool or type(allow_uncertified) is not bool:
        raise TypeError("skip_pav and allow_uncertified must be exact booleans")
    min_acgt_coverage = _coverage_fraction(
        min_acgt_coverage, field="min_acgt_coverage"
    )
    min_sequence_score_coverage = _coverage_fraction(
        min_sequence_score_coverage, field="min_sequence_score_coverage"
    )
    min_fragment_length = _positive_int(
        min_fragment_length, field="min_fragment_length"
    )
    entries = read_manifest(manifest, split_seed=split_seed)
    source_labels = {
        entry.label for entry in entries if entry.role == "train"
    }
    if source_labels != {0, 1}:
        raise ValueError("train role must contain both natural and AI sources")
    sources = [source for entry in entries for source in _read_fasta(entry)]
    manifest_hash = _manifest_entries_hash(entries)
    training_config = {
        "split": "source",
        "split_seed": split_seed,
        "target_fpr": target_fpr,
        "target_fnr": target_fnr,
        "skip_pav": skip_pav,
        "max_pav_windows_per_record": max_pav_windows_per_record,
        "min_fragment_length": min_fragment_length,
    }

    sequence_artifact = fit_sequence_stage(
        sources,
        estimator=sequence_estimator,
        scaler=sequence_scaler,
        target_fpr=target_fpr,
        allow_uncertified=allow_uncertified,
        training_config={**training_config, "stage": "sequence"},
        source_split_manifest_hash=manifest_hash,
    )
    cds_artifact = fit_cds_stage(
        sources,
        estimator=cds_estimator,
        scaler=cds_scaler,
        target_fpr=target_fpr,
        allow_uncertified=allow_uncertified,
        training_config={**training_config, "stage": "cds"},
        source_split_manifest_hash=manifest_hash,
    )
    pav_artifact = None
    pav_stage = None
    if not skip_pav:
        if not callable(pav_observer):
            raise ValueError("pAV aggregation requires an injected pav_observer")
        if not isinstance(pav_observer_id, str) or not pav_observer_id:
            raise ValueError("pAV training requires a non-empty pav_observer_id")
        effective_cache = pav_cache if pav_cache is not None else {}
        if not hasattr(effective_cache, "get") or not hasattr(effective_cache, "__setitem__"):
            raise TypeError("pav_cache must be a mutable mapping")
        trainer = pav_trainer if callable(pav_trainer) else _fit_pav_stage
        observer_config = _pav_observer_configuration(
            pav_observer, pav_observer_id
        )
        pav_artifact = trainer(
            [source for source in sources if source["role"] == "train"],
            [source for source in sources if source["role"] == "calibration"],
            observer=pav_observer,
            observer_id=pav_observer_id,
            observer_config=observer_config,
            cache=effective_cache,
            target_fpr=target_fpr,
            allow_uncertified=allow_uncertified,
            training_config={**training_config, "stage": "pav"},
            source_split_manifest_hash=manifest_hash,
            max_windows_per_record=max_pav_windows_per_record,
        )
        pav_artifact = validate_artifact(pav_artifact, expected_kind="pav")
        if pav_artifact["metadata"]["source_split_manifest_hash"] != manifest_hash:
            raise ValueError("pAV artifact source-split manifest hash does not match training")
        from .pav import PAVStage

        pav_stage = PAVStage(
            pav_artifact,
            observer=pav_observer,
            observer_id=pav_observer_id,
            cache=effective_cache,
            allow_uncertified=allow_uncertified,
        )
    aggregate_rows = _aggregate_records(
        sources,
        sequence_artifact,
        cds_artifact,
        pav_stage=pav_stage,
        min_fragment_length=min_fragment_length,
        allow_uncertified=allow_uncertified,
    )
    router_cpu = fit_router(
        aggregate_rows,
        kind="router_cpu",
        feature_names=CPU_FEATURE_NAMES,
        estimator=router_estimator,
        scaler=router_scaler,
        target_fpr=target_fpr,
        target_fnr=target_fnr,
        allow_uncertified=allow_uncertified,
        min_acgt_coverage=min_acgt_coverage,
        min_sequence_score_coverage=min_sequence_score_coverage,
        min_fragment_length=min_fragment_length,
        training_config={**training_config, "profile": "cpu"},
        source_split_manifest_hash=manifest_hash,
    )
    router_full = None
    if pav_artifact is not None:
        router_full = fit_router(
            aggregate_rows,
            kind="router_full",
            feature_names=(
                FULL_FEATURE_NAMES
                if router_full_feature_names is None
                else router_full_feature_names
            ),
            estimator=router_full_estimator,
            scaler=router_full_scaler,
            target_fpr=target_fpr,
            target_fnr=target_fnr,
            allow_uncertified=allow_uncertified,
            min_acgt_coverage=min_acgt_coverage,
            min_sequence_score_coverage=min_sequence_score_coverage,
            min_fragment_length=min_fragment_length,
            support_model=router_cpu["payload"]["support_model"],
            training_config={**training_config, "profile": "full"},
            source_split_manifest_hash=manifest_hash,
        )

    from .cds import CDSStage
    from .pipeline import GOSV4Detector
    from .sequence import SequenceStage

    sequence_stage = SequenceStage(
        sequence_artifact, allow_uncertified=allow_uncertified
    )
    cds_stage = CDSStage(cds_artifact, allow_uncertified=allow_uncertified)
    cpu_detector = GOSV4Detector(
        sequence_stage,
        cds_stage,
        router_cpu,
        router_cpu["payload"]["support_model"],
        allow_uncertified=allow_uncertified,
    )
    metrics = _test_metrics(sources, cpu_detector)
    if router_full is not None:
        full_detector = GOSV4Detector(
            sequence_stage,
            cds_stage,
            router_cpu,
            router_cpu["payload"]["support_model"],
            pav_stage=pav_stage,
            router_full=router_full,
            allow_uncertified=allow_uncertified,
        )
        metrics["router_full"] = _test_metrics(sources, full_detector)

    component_artifacts = {
        "sequence": sequence_artifact,
        "cds": cds_artifact,
        "router_cpu": router_cpu,
    }
    if pav_artifact is not None and router_full is not None:
        component_artifacts.update(
            {"pav": pav_artifact, "router_full": router_full}
        )
    component_certification = {
        name: artifact_is_certified(artifact)
        for name, artifact in component_artifacts.items()
    }
    certified = bool(
        component_certification
        and all(component_certification.values())
    )
    calibration_report = {
        "schema_version": "gos-v4-calibration/1",
        "model_version": MODEL_VERSION,
        "source_split_manifest_hash": manifest_hash,
        "certified": certified,
        "component_certification": component_certification,
        "sequence": _artifact_calibration(sequence_artifact),
        "cds": _artifact_calibration(cds_artifact),
        "router_cpu": _artifact_calibration(router_cpu),
        "pav": _artifact_calibration(pav_artifact) if pav_artifact is not None else None,
        "router_full": _artifact_calibration(router_full) if router_full is not None else None,
    }

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(sequence_artifact, destination / "sequence.joblib")
    joblib.dump(cds_artifact, destination / "cds.joblib")
    joblib.dump(router_cpu, destination / "router_cpu.joblib")
    if pav_artifact is not None and router_full is not None:
        pav_directory = destination / "pav"
        pav_directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(pav_artifact, pav_directory / "model.joblib")
        with (pav_directory / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": pav_artifact["schema_version"],
                    "model_version": pav_artifact["model_version"],
                    "artifact_kind": pav_artifact["artifact_kind"],
                    "feature_names": pav_artifact["feature_names"],
                    "metadata": pav_artifact["metadata"],
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        joblib.dump(router_full, destination / "router_full.joblib")
    with (destination / "aggregates.jsonl").open("w", encoding="utf-8") as handle:
        for row in aggregate_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    with (destination / "calibration.json").open("w", encoding="utf-8") as handle:
        json.dump(calibration_report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (destination / "metrics_test.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {
        "sequence": destination / "sequence.joblib",
        "cds": destination / "cds.joblib",
        "router_cpu": destination / "router_cpu.joblib",
        "router_full": destination / "router_full.joblib" if router_full is not None else None,
        "pav": destination / "pav" / "model.joblib" if pav_artifact is not None else None,
        "calibration": destination / "calibration.json",
        "metrics_test": destination / "metrics_test.json",
        "aggregates": destination / "aggregates.jsonl",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train source-disjoint, versioned GOS v4 artifacts."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--skip-pav", action="store_true")
    parser.add_argument("--allow-uncertified", action="store_true")
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--target-fnr", type=float, default=0.01)
    parser.add_argument("--min-acgt-coverage", type=float, default=0.8)
    parser.add_argument("--min-sequence-score-coverage", type=float, default=1.0)
    parser.add_argument("--min-fragment-length", type=int, default=250)
    parser.add_argument(
        "--pav-cache",
        type=Path,
        help="Optional append-only JSONL cache for pAV observer vectors.",
    )
    parser.add_argument(
        "--max-pav-windows-per-record",
        type=int,
        help="Optional deterministic cap on pAV candidates per input record.",
    )
    parser.add_argument(
        "--router-full-profile",
        choices=("legacy", "early_layer24_attention_cpu_lowdim"),
        default="legacy",
        help="Full-router feature schema to train when pAV is enabled.",
    )
    return parser


def _production_pav_observer(
    observer_config: ObserverConfig | None = None,
    *,
    observer_id: str | None = None,
) -> Any:
    """Build a lightweight observer proxy that loads NT on its first window."""

    from .pav import (
        NucleotideTransformerObserver,
        _DEFAULT_MAX_LENGTH,
        _DEFAULT_MODEL_NAME,
        _DEFAULT_REVISION,
    )

    if observer_config is None:
        observer_config = ObserverConfig(
            model_name=_DEFAULT_MODEL_NAME,
            revision=_DEFAULT_REVISION,
            tokenizer_identity=_DEFAULT_MODEL_NAME,
            max_length=_DEFAULT_MAX_LENGTH,
            normalization="none",
        )
    if not isinstance(observer_config, ObserverConfig):
        raise TypeError("observer_config must be an ObserverConfig")
    if observer_id is not None and (
        not isinstance(observer_id, str) or not observer_id
    ):
        raise ValueError("observer_id must be a non-empty string")
    effective_observer_id = (
        observer_config.model_name if observer_id is None else observer_id
    )

    class LazyProductionObserver:
        def __init__(self) -> None:
            self._loaded: Any | None = None
            self.observer_config = observer_config
            self.observer_fingerprint = observer_config.fingerprint
            self.observer_id = effective_observer_id

        def __call__(self, sequence: str) -> Any:
            if self._loaded is None:
                self._loaded = NucleotideTransformerObserver.load(self.observer_config)
            return self._loaded(sequence)

    return LazyProductionObserver()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the manifest training CLI."""

    options = _build_parser().parse_args(argv)
    pav_observer = None if options.skip_pav else _production_pav_observer()
    pav_cache = None
    if pav_observer is not None and options.pav_cache is not None:
        from .pav import JsonlPAVCache

        pav_cache = JsonlPAVCache(options.pav_cache, log_every=50)
    try:
        train_from_manifest(
            options.manifest,
            options.out_dir,
            skip_pav=options.skip_pav,
            allow_uncertified=options.allow_uncertified,
            split_seed=options.split_seed,
            target_fpr=options.target_fpr,
            target_fnr=options.target_fnr,
            min_acgt_coverage=options.min_acgt_coverage,
            min_sequence_score_coverage=options.min_sequence_score_coverage,
            min_fragment_length=options.min_fragment_length,
            pav_observer=pav_observer,
            pav_observer_id=(
                pav_observer.observer_id if pav_observer is not None else None
            ),
            pav_cache=pav_cache,
            max_pav_windows_per_record=options.max_pav_windows_per_record,
            router_full_feature_names=(
                STRATEGY_B_FEATURE_NAMES
                if options.router_full_profile == "early_layer24_attention_cpu_lowdim"
                else None
            ),
        )
    finally:
        if pav_cache is not None:
            pav_cache.close()
    return 0


__all__ = [
    "ManifestEntry",
    "fit_cds_stage",
    "fit_router",
    "fit_sequence_stage",
    "main",
    "read_manifest",
    "train_from_manifest",
]


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess CLI tests
    raise SystemExit(main())
