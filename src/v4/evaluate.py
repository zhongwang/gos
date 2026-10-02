"""Checkpoint-only evaluation and window-score counterfactual summaries.

This module deliberately contains no fitting or calibration path.  It loads
the persisted CPU checkpoint once, evaluates only records explicitly assigned
to the manifest's ``test`` role, and writes a JSON audit report.
"""

from __future__ import annotations

import argparse
import json
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from .artifacts import (
    MODEL_VERSION,
    artifact_is_certified,
    load_artifact,
    validate_artifact_compatibility,
)
from .calibration import wilson_interval
from .cds import CDSStage
from .pav import (
    ObserverConfig,
    PAVStage,
    observer_config_from_artifact,
    observer_id_from_artifact,
)
from .pipeline import GOSV4Detector
from .sequence import SequenceStage, parse_window_thresholds
from .train import _manifest_entries_hash, _read_fasta, read_manifest
from .types import ScanResult, StageStatus


_REPORT_SCHEMA_VERSION = "gos-v4-evaluation/1"
_LOGIT_EPSILON = 1e-6


def _probability(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real probability")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return result


def _labels_and_predictions(
    labels: Iterable[Any], predictions: Iterable[Any]
) -> tuple[tuple[int, ...], tuple[int | None, ...]]:
    observed_labels = tuple(labels)
    observed_predictions = tuple(predictions)
    if len(observed_labels) != len(observed_predictions):
        raise ValueError("labels and predictions must have equal length")
    for label in observed_labels:
        if type(label) is not int or label not in {0, 1}:
            raise ValueError("labels must be exact integers in {0, 1}")
    for prediction in observed_predictions:
        if prediction is not None and (type(prediction) is not int or prediction not in {0, 1}):
            raise ValueError("predictions must be 0, 1, or None")
    return observed_labels, observed_predictions


def binary_metrics(labels: Iterable[Any], predictions: Iterable[Any]) -> dict[str, Any]:
    """Return test-set AI recall and natural false-positive-rate statistics.

    Abstentions count as neither a positive nor a negative call.  They remain
    in the recall denominator, which makes abstention separate from detection
    recall rather than silently removing hard AI rows.
    """

    observed_labels, observed_predictions = _labels_and_predictions(labels, predictions)
    natural_count = sum(label == 0 for label in observed_labels)
    ai_count = sum(label == 1 for label in observed_labels)
    false_positives = sum(
        label == 0 and prediction == 1
        for label, prediction in zip(observed_labels, observed_predictions, strict=True)
    )
    true_positives = sum(
        label == 1 and prediction == 1
        for label, prediction in zip(observed_labels, observed_predictions, strict=True)
    )
    fpr = false_positives / natural_count if natural_count else None
    fpr_ci = wilson_interval(false_positives, natural_count) if natural_count else None
    return {
        "n_natural": natural_count,
        "n_ai": ai_count,
        "false_positives": false_positives,
        "true_positives": true_positives,
        "fpr": fpr,
        "fpr_ci": fpr_ci,
        "fpr_wilson_interval": fpr_ci,
        "recall": true_positives / ai_count if ai_count else None,
    }


def abstention_metrics(labels: Iterable[Any], predictions: Iterable[Any]) -> dict[str, Any]:
    """Return abstention frequency and accuracy conditional on a decision."""

    observed_labels, observed_predictions = _labels_and_predictions(labels, predictions)
    decided = [
        (label, prediction)
        for label, prediction in zip(observed_labels, observed_predictions, strict=True)
        if prediction is not None
    ]
    record_count = len(observed_labels)
    decided_count = len(decided)
    return {
        "n_records": record_count,
        "n_decided": decided_count,
        "n_abstained": record_count - decided_count,
        "abstention_rate": (record_count - decided_count) / record_count if record_count else None,
        "conditional_accuracy": (
            sum(label == prediction for label, prediction in decided) / decided_count
            if decided_count
            else None
        ),
    }


def _counterfactual_thresholds(threshold: Any, scores: np.ndarray) -> tuple[np.ndarray, Any]:
    raw = np.asarray(threshold)
    if raw.ndim == 0:
        value = _probability(raw.item(), name="threshold")
        return np.full(scores.shape, value, dtype=float), value
    if raw.ndim != 1 or raw.shape != scores.shape:
        raise ValueError("threshold must be a scalar or one probability per window score")
    if np.issubdtype(raw.dtype, np.bool_) or (
        raw.dtype == object and any(isinstance(value, (bool, np.bool_)) for value in raw)
    ):
        raise TypeError("threshold must contain numeric probabilities, not booleans")
    try:
        values = raw.astype(float, copy=False)
    except (TypeError, ValueError) as error:
        raise TypeError("threshold must contain numeric probabilities") from error
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("threshold must contain finite probabilities in [0, 1]")
    return values, values.tolist()


def counterfactual_aggregates(window_scores: Any, threshold: Any) -> dict[str, Any]:
    """Aggregate one already-scored window array without rescoring windows.

    The logit sum clips only at a documented numerical boundary, so scores of
    exactly zero or one remain finite while all routing counterfactuals use the
    same stored window scores.
    """

    values = np.asarray(window_scores)
    if values.ndim != 1:
        raise ValueError("window_scores must be one-dimensional")
    if np.issubdtype(values.dtype, np.bool_):
        raise TypeError("window_scores must be numeric probabilities, not booleans")
    try:
        scores = values.astype(float, copy=False)
    except (TypeError, ValueError) as error:
        raise TypeError("window_scores must be numeric probabilities") from error
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("window_scores must be finite probabilities in [0, 1]")
    thresholds, reported_threshold = _counterfactual_thresholds(threshold, scores)
    if not scores.size:
        return {
            "n_windows": 0,
            "threshold": reported_threshold,
            "mean": None,
            "max": None,
            "top_k": None,
            "fraction_over": None,
            "calibrated_majority": None,
            "logit_sum": None,
        }
    fraction_over = float(np.mean(scores >= thresholds))
    clipped = np.clip(scores, _LOGIT_EPSILON, 1.0 - _LOGIT_EPSILON)
    return {
        "n_windows": int(scores.size),
        "threshold": reported_threshold,
        "mean": float(np.mean(scores)),
        "max": float(np.max(scores)),
        "top_k": float(np.mean(np.sort(scores)[-min(3, scores.size) :])),
        "fraction_over": fraction_over,
        "calibrated_majority": fraction_over,
        "logit_sum": float(np.sum(np.log(clipped / (1.0 - clipped)))),
    }


def _has_full_checkpoint(model_dir: Path) -> bool:
    pav_path = model_dir / "pav" / "model.joblib"
    full_path = model_dir / "router_full.joblib"
    if pav_path.is_file() != full_path.is_file():
        raise ValueError("full checkpoint requires both pav/model.joblib and router_full.joblib")
    return pav_path.is_file()


def _load_pav_observer_config(model_dir: Path) -> ObserverConfig:
    """Load the full checkpoint's declared observer provenance before proxy creation."""

    return _load_pav_observer_spec(model_dir)[0]


def _load_pav_observer_spec(model_dir: Path) -> tuple[ObserverConfig, str]:
    """Load the canonical value config and its separate artifact display identity."""

    artifact = load_artifact(model_dir / "pav" / "model.joblib", "pav")
    return (
        observer_config_from_artifact(artifact),
        observer_id_from_artifact(artifact),
    )


def _load_detector(
    model_dir: Path,
    *,
    pav_observer: Callable[[str], Any] | None = None,
    pav_observer_id: str | None = None,
    pav_cache: MutableMapping[str, Any] | None = None,
    allow_uncertified: bool = False,
) -> tuple[GOSV4Detector, dict[str, Any], str]:
    """Load saved CPU/full checkpoint artifacts once and compose its detector."""

    sequence_artifact = load_artifact(model_dir / "sequence.joblib", "sequence")
    cds_artifact = load_artifact(model_dir / "cds.joblib", "cds")
    router_artifact = load_artifact(model_dir / "router_cpu.joblib", "router_cpu")
    support_model = router_artifact["payload"].get("support_model")
    if support_model is None:
        raise ValueError("router_cpu artifact is missing its natural support model")
    artifacts: dict[str, Any] = {
        "sequence": sequence_artifact,
        "cds": cds_artifact,
        "router_cpu": router_artifact,
    }
    validate_artifact_compatibility(artifacts)
    if type(allow_uncertified) is not bool:
        raise TypeError("allow_uncertified must be an exact boolean")
    sequence_stage = SequenceStage(
        sequence_artifact, allow_uncertified=allow_uncertified
    )
    cds_stage = CDSStage(cds_artifact, allow_uncertified=allow_uncertified)
    if not _has_full_checkpoint(model_dir):
        return (
            GOSV4Detector(
                sequence_stage,
                cds_stage,
                router_artifact,
                support_model,
                allow_uncertified=allow_uncertified,
            ),
            artifacts,
            "cpu",
        )
    if not callable(pav_observer):
        raise ValueError("full checkpoint requires an injected pAV observer")
    if not isinstance(pav_observer_id, str) or not pav_observer_id:
        raise ValueError("full checkpoint requires a non-empty pAV observer_id")
    pav_artifact = load_artifact(model_dir / "pav" / "model.joblib", "pav")
    full_router = load_artifact(model_dir / "router_full.joblib", "router_full")
    pav_stage = PAVStage(
        pav_artifact,
        observer=pav_observer,
        observer_id=pav_observer_id,
        cache=pav_cache,
        allow_uncertified=allow_uncertified,
    )
    artifacts.update({"pav": pav_artifact, "router_full": full_router})
    validate_artifact_compatibility(artifacts)
    return (
        GOSV4Detector(
            sequence_stage,
            cds_stage,
            router_artifact,
            support_model,
            pav_stage=pav_stage,
            router_full=full_router,
            allow_uncertified=allow_uncertified,
        ),
        artifacts,
        "full",
    )


def _reverse_complement_preserving_unknowns(sequence: str) -> str:
    """Reverse-complement canonical bases while retaining ambiguous symbols."""

    return sequence.upper().translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _length_group(length: int) -> str:
    if length < 500:
        return "0-499"
    if length < 750:
        return "500-749"
    if length < 1000:
        return "750-999"
    if length < 2000:
        return "1000-1999"
    return "2000+"


def _scored_windows(result: ScanResult) -> tuple[Any, ...]:
    return tuple(
        window
        for window in result.windows
        if window.eligible and window.score is not None
    )


def _window_thresholds(sequence_artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    return parse_window_thresholds(sequence_artifact["metadata"])


def _counterfactual_evidence(
    result: ScanResult, sequence_artifact: Mapping[str, Any]
) -> dict[str, Any]:
    """Project stored result windows into counterfactual routing summaries."""

    scored_windows = _scored_windows(result)
    scores = np.asarray([window.score for window in scored_windows], dtype=float)
    thresholds = _window_thresholds(sequence_artifact)
    by_length_bin: dict[str, Any] = {}
    for length_bin in sorted({window.length_bin for window in result.windows if window.length_bin}):
        score_array = np.asarray(
            [
                window.score
                for window in result.windows
                if window.length_bin == length_bin and window.eligible and window.score is not None
            ],
            dtype=float,
        )
        threshold = thresholds.get(length_bin)
        if threshold is not None:
            by_length_bin[length_bin] = counterfactual_aggregates(score_array, threshold)
    threshold_vector: list[float] = []
    for window in scored_windows:
        threshold = thresholds.get(window.length_bin)
        if threshold is None:
            raise ValueError(
                f"scored window length bin {window.length_bin!r} lacks a saved threshold"
            )
        threshold_vector.append(_probability(threshold, name="saved window threshold"))
    return {
        "window_scores": scores.tolist(),
        "counterfactuals": counterfactual_aggregates(
            scores, np.asarray(threshold_vector, dtype=float)
        ),
        "by_length_bin": by_length_bin,
    }


def _rate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [row["label"] for row in rows]
    predictions = [row["prediction"] for row in rows]
    report = {**binary_metrics(labels, predictions), **abstention_metrics(labels, predictions)}
    total = len(rows)
    triggered = sum(bool(row["pav_triggered"]) for row in rows)
    failures = sum(row["pav_status"] == StageStatus.FAILED.value for row in rows)
    reasons = [row["reason"] for row in rows if row["reason"] is not None]
    report.update(
        {
            "reason_distribution": {reason: reasons.count(reason) for reason in sorted(set(reasons))},
            "reason_counts": {reason: reasons.count(reason) for reason in sorted(set(reasons))},
            "pav_trigger_rate": triggered / total if total else None,
            "pav_failure_rate": failures / total if total else None,
            "pav_failure_rate_given_trigger": failures / triggered if triggered else None,
        }
    )
    router_scores = [
        float(row["router_score"])
        for row in rows
        if isinstance(row.get("router_score"), Real)
        and np.isfinite(row["router_score"])
    ]
    report["median_router_score"] = (
        float(np.median(router_scores)) if router_scores else None
    )
    deltas = [row["reverse_complement_score_delta"] for row in rows if row["reverse_complement_score_delta"] is not None]
    report["reverse_complement_delta"] = {
        "n_compared": len(deltas),
        "mean_absolute": float(np.mean(np.abs(deltas))) if deltas else None,
        "maximum_absolute": float(np.max(np.abs(deltas))) if deltas else None,
        "mean_signed": float(np.mean(deltas)) if deltas else None,
    }
    paired_variances = []
    for row in rows:
        original = row.get("router_score")
        reverse = row.get("reverse_complement_router_score")
        if (
            isinstance(original, Real)
            and not isinstance(original, bool)
            and np.isfinite(original)
            and isinstance(reverse, Real)
            and not isinstance(reverse, bool)
            and np.isfinite(reverse)
        ):
            paired_variances.append(float(np.var([float(original), float(reverse)])))
    report["paired_orientation_score_variance"] = _variance_summary(
        paired_variances, unit="pair"
    )
    crop_variances = [
        float(row["crop_jitter_score_variance"])
        for row in rows
        if isinstance(row.get("crop_jitter_score_variance"), Real)
        and not isinstance(row.get("crop_jitter_score_variance"), bool)
        and np.isfinite(row["crop_jitter_score_variance"])
    ]
    report["crop_jitter_score_variance"] = _variance_summary(
        crop_variances, unit="record"
    )
    report["mosaic_localization"] = _mosaic_localization(rows)
    return report


def _variance_summary(values: Sequence[float], *, unit: str) -> dict[str, Any]:
    return {
        f"n_{unit}s": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "maximum": float(np.max(values)) if values else None,
    }


def _parse_spans(value: Any) -> tuple[tuple[int, int], ...] | None:
    if value is None or value == "":
        return None
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [part.split("-", 1) for part in value.split(",") if part]
    if isinstance(parsed, Mapping):
        parsed = parsed.get("spans")
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("mosaic_spans must be JSON spans or start-end pairs")
    spans: list[tuple[int, int]] = []
    for item in parsed:
        if isinstance(item, Mapping):
            start, end = item.get("start"), item.get("end")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = item
        else:
            raise ValueError("each mosaic span must contain start and end")
        try:
            start, end = int(start), int(end)
        except (TypeError, ValueError) as error:
            raise ValueError("mosaic span coordinates must be integers") from error
        if start < 0 or end <= start:
            raise ValueError("mosaic spans must be non-empty half-open intervals")
        spans.append((start, end))
    return tuple(spans)


def _merged_spans(spans: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _span_length(spans: Sequence[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merged_spans(spans))


def _intersection_length(
    first: Sequence[tuple[int, int]], second: Sequence[tuple[int, int]]
) -> int:
    left = _merged_spans(first)
    right = _merged_spans(second)
    total = 0
    i = j = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _mosaic_localization(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    annotated = 0
    true_positive = 0
    annotated_positive = 0
    predicted_positive = 0
    for row in rows:
        metadata = row.get("metadata")
        raw = metadata.get("mosaic_spans") if isinstance(metadata, Mapping) else None
        truth = _parse_spans(raw)
        if truth is None:
            continue
        annotated += 1
        predicted = _parse_spans(row.get("predicted_segments", [])) or ()
        true_positive += _intersection_length(truth, predicted)
        annotated_positive += _span_length(truth)
        predicted_positive += _span_length(predicted)
    if not annotated:
        return {
            "available": False,
            "reason": "mosaic_annotations_unavailable",
            "n_annotated_records": 0,
        }
    union = annotated_positive + predicted_positive - true_positive
    return {
        "available": True,
        "n_annotated_records": annotated,
        "true_positive_bases": true_positive,
        "annotated_positive_bases": annotated_positive,
        "predicted_positive_bases": predicted_positive,
        "base_precision": (
            true_positive / predicted_positive if predicted_positive else None
        ),
        "base_recall": (
            true_positive / annotated_positive if annotated_positive else None
        ),
        "base_iou": true_positive / union if union else None,
    }


def _crop_jitter_sequences(sequence: str) -> tuple[str, ...]:
    if len(sequence) < 3:
        return ()
    trim = min(20, max(2, len(sequence) // 20))
    crop_length = len(sequence) - trim
    offsets = sorted({0, trim // 2, trim})
    return tuple(sequence[offset : offset + crop_length] for offset in offsets)


def _metadata_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        "generator": {},
        "domain": {},
        "length": {},
    }
    for row in rows:
        metadata = row["metadata"]
        for field in ("generator", "domain"):
            value = metadata.get(field)
            if isinstance(value, str) and value:
                groups[field].setdefault(value, []).append(row)
        groups["length"].setdefault(row["length_group"], []).append(row)
    return {
        field: {name: _rate_metrics(group_rows) for name, group_rows in sorted(values.items())}
        for field, values in groups.items()
    }


def _json_dump(path: Path, report: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def evaluate_checkpoint(
    manifest: str | Path,
    model_dir: str | Path,
    *,
    pav_observer: Callable[[str], np.ndarray] | None = None,
    pav_observer_id: str | None = None,
    pav_cache: MutableMapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    write_report: bool = True,
    allow_uncertified: bool = False,
) -> dict[str, Any]:
    """Evaluate the saved detector exactly once over manifest test records.

    No estimator fitting, calibration, threshold selection, or non-test manifest
    scan occurs here.  A full checkpoint requires an injected pAV observer;
    otherwise it fails instead of silently taking the CPU-only route.  An
    explicit ``output_path`` is the only write destination, while
    ``write_report=False`` is a no-write mode.
    """

    model_path = Path(model_dir)
    entries = read_manifest(manifest)
    manifest_hash = _manifest_entries_hash(entries)
    test_entries = tuple(entry for entry in entries if entry.role == "test")
    if not test_entries:
        raise ValueError("manifest has no test-role entries")
    if type(write_report) is not bool:
        raise TypeError("write_report must be an exact boolean")
    if type(allow_uncertified) is not bool:
        raise TypeError("allow_uncertified must be an exact boolean")
    detector, artifacts, profile = _load_detector(
        model_path,
        pav_observer=pav_observer,
        pav_observer_id=pav_observer_id,
        pav_cache=pav_cache,
        allow_uncertified=allow_uncertified,
    )
    rows: list[dict[str, Any]] = []
    for entry in test_entries:
        for source in _read_fasta(entry):
            result = detector.scan(source["sequence"], source_id=source["source_id"])
            reverse = detector.scan(
                _reverse_complement_preserving_unknowns(source["sequence"]),
                source_id=source["source_id"],
            )
            score = result.scores.get("router")
            reverse_score = reverse.scores.get("router")
            score_delta = (
                float(score) - float(reverse_score)
                if isinstance(score, Real) and isinstance(reverse_score, Real)
                else None
            )
            crop_scores: list[float] = []
            for index, crop in enumerate(_crop_jitter_sequences(source["sequence"])):
                crop_result = detector.scan(
                    crop,
                    source_id=f"{source['source_id']}:crop-jitter-{index}",
                )
                crop_score = crop_result.scores.get("router")
                if isinstance(crop_score, Real) and np.isfinite(crop_score):
                    crop_scores.append(float(crop_score))
            crop_variance = float(np.var(crop_scores)) if len(crop_scores) >= 2 else None
            counterfactual_evidence = _counterfactual_evidence(
                result, artifacts["sequence"]
            )
            rows.append(
                {
                    "source_id": source["source_id"],
                    "sequence_id": source["sequence_id"],
                    "role": "test",
                    "label": source["label"],
                    "prediction": result.prediction,
                    "decision": result.label.value,
                    "router_profile": profile,
                    "router_used": "full" if "router_full" in result.scores else "cpu",
                    "reason": result.reason,
                    "router_score": score,
                    "length": result.length,
                    "length_group": _length_group(result.length),
                    "pav_triggered": result.pav_triggered,
                    "pav_status": result.pav_status.value,
                    "reverse_complement_router_score": reverse_score,
                    "reverse_complement_score_delta": score_delta,
                    "crop_jitter_router_scores": crop_scores,
                    "crop_jitter_score_variance": crop_variance,
                    "metadata": dict(source.get("metadata", {})),
                    "predicted_segments": result.to_dict()["segments"],
                    "counterfactual_evidence": counterfactual_evidence,
                    "counterfactuals": counterfactual_evidence["counterfactuals"],
                }
            )
    report: dict[str, Any] = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "role": "test",
        "manifest": {
            "source_split_manifest_hash": manifest_hash,
        },
        "checkpoint": {
            "profile": profile,
            "certified": detector.certified,
            "component_certification": {
                name: artifact_is_certified(artifact)
                for name, artifact in artifacts.items()
            },
            "source_split_manifest_hash": next(
                iter(artifacts.values())
            )["metadata"]["source_split_manifest_hash"],
        },
        "overall": _rate_metrics(rows),
        "groups": _metadata_groups(rows),
        "records": rows,
    }
    if write_report:
        _json_dump(
            Path(output_path) if output_path is not None else model_path / "evaluation_test.json",
            report,
        )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved GOS v4 CPU or full checkpoint on manifest test records."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional JSON destination (default: MODEL_DIR/evaluation_test.json).",
    )
    parser.add_argument(
        "--allow-uncertified",
        action="store_true",
        help="Permit exploratory checkpoints explicitly marked uncertified.",
    )
    parser.add_argument(
        "--pav-cache",
        type=Path,
        help="Optional append-only JSONL cache for pAV observer vectors.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run checkpoint-only evaluation and persist a JSON report."""

    options = _build_parser().parse_args(argv)
    pav_observer = None
    pav_observer_id = None
    pav_cache = None
    if _has_full_checkpoint(options.model_dir):
        from .train import _production_pav_observer

        observer_config, artifact_observer_id = _load_pav_observer_spec(
            options.model_dir
        )
        pav_observer = _production_pav_observer(
            observer_config,
            observer_id=artifact_observer_id,
        )
        pav_observer_id = pav_observer.observer_id
        if options.pav_cache is not None:
            from .pav import JsonlPAVCache

            pav_cache = JsonlPAVCache(options.pav_cache, log_every=50)
    try:
        evaluate_checkpoint(
            options.manifest,
            options.model_dir,
            pav_observer=pav_observer,
            pav_observer_id=pav_observer_id,
            pav_cache=pav_cache,
            output_path=options.out,
            allow_uncertified=options.allow_uncertified,
        )
    finally:
        if pav_cache is not None:
            pav_cache.close()
    return 0


__all__ = [
    "abstention_metrics",
    "binary_metrics",
    "counterfactual_aggregates",
    "evaluate_checkpoint",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess CLI tests
    raise SystemExit(main())
