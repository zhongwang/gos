"""Matched-row comparison against the historical v3 OOD benchmark pool."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
import re
from typing import Any, Mapping, MutableMapping, Sequence

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from .evaluate import _has_full_checkpoint, _load_detector, _load_pav_observer_spec
from .pav import JsonlPAVCache
from .train import _production_pav_observer
from .types import ScanResult, StageStatus


_SCHEMA_VERSION = "gos-v4-matched-pool/1"
_SAFE_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_V3_FEATURES_USED = (
    "gzip_ratio",
    "lzma_ratio",
    "kmer_periodicity",
    "stop_density",
    "mean_pav_top_k",
)


@dataclass(frozen=True)
class LegacyRow:
    """One sequence-bearing row from the legacy v3 CSV pool."""

    split: str
    row_index: int
    source: str
    label: int
    sequence: str


def read_legacy_csv(path: str | Path, *, split: str) -> list[LegacyRow]:
    """Read the legacy seq/label/source CSV without changing row order."""

    if not isinstance(split, str) or not split:
        raise ValueError("split must be a non-empty string")
    rows: list[LegacyRow] = []
    csv.field_size_limit(2**31 - 1)
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"seq", "label", "source"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("legacy CSV requires seq, label, and source columns")
        for row_index, row in enumerate(reader):
            sequence = row.get("seq")
            source = row.get("source")
            if not isinstance(sequence, str) or not sequence:
                raise ValueError(f"legacy row {row_index} has an empty seq")
            if not isinstance(source, str) or not source:
                raise ValueError(f"legacy row {row_index} has an empty source")
            try:
                label = int(row.get("label", ""))
            except ValueError as error:
                raise ValueError(f"legacy row {row_index} has a non-integer label") from error
            if label not in {0, 1}:
                raise ValueError(f"legacy row {row_index} label must be 0 or 1")
            rows.append(
                LegacyRow(
                    split=split,
                    row_index=row_index,
                    source=source,
                    label=label,
                    sequence=sequence,
                )
            )
    if not rows:
        raise ValueError("legacy CSV must contain at least one row")
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_number} must contain an object")
            rows.append(row)
    return rows


def _feature_index(row: Mapping[str, Any]) -> int:
    value = row.get("idx", row.get("index"))
    if type(value) is not int:
        raise ValueError("v3 feature rows require an integer idx or index")
    return value


def _validate_features(
    features: Sequence[Mapping[str, Any]],
    csv_rows: Sequence[LegacyRow],
    *,
    feature_path: Path,
) -> None:
    if len(features) != len(csv_rows):
        raise ValueError(f"{feature_path} row count does not match the recovered CSV")
    for expected, feature in zip(csv_rows, features, strict=True):
        if _feature_index(feature) != expected.row_index:
            raise ValueError(f"{feature_path} has a row-index mismatch")
        if feature.get("source") != expected.source:
            raise ValueError(f"{feature_path} has a source mismatch")
        if feature.get("label") != expected.label:
            raise ValueError(f"{feature_path} has a label mismatch")
        if feature.get("pav_all_heads") is None:
            raise ValueError(f"{feature_path} contains a failed pAV row")


def _rows_to_v3_matrix(
    features: Sequence[Mapping[str, Any]],
    selected_heads: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    labels: list[int] = []
    for feature in features:
        pav = np.asarray(feature["pav_all_heads"], dtype=float)
        if pav.ndim != 1 or not len(pav) or not np.all(np.isfinite(pav)):
            raise ValueError("pav_all_heads must be a finite one-dimensional vector")
        rows.append(
            [
                float(feature["gzip_ratio"]),
                float(feature["lzma_ratio"]),
                float(feature["kmer_periodicity"]),
                float(feature["stop_density"]),
                float(pav[list(selected_heads)].mean()),
            ]
        )
        labels.append(int(feature["label"]))
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(_V3_FEATURES_USED):
        raise ValueError("invalid v3 feature matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("v3 feature matrix contains non-finite values")
    return matrix, np.asarray(labels, dtype=int)


def _positive_scores(estimator: Any, matrix: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(matrix), dtype=float)
    classes = np.asarray(getattr(estimator, "classes_", [0, 1]))
    matches = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2 or probabilities.shape[0] != matrix.shape[0] or len(matches) != 1:
        raise ValueError("estimator did not return one AI score per row")
    scores = probabilities[:, int(matches[0])]
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("estimator returned invalid AI scores")
    return scores


def _score_v3_features(
    features: Sequence[Mapping[str, Any]],
    selected_heads: Sequence[int],
    *,
    model: Any,
    scaler: Any,
    clip_lo: np.ndarray,
    clip_hi: np.ndarray,
) -> np.ndarray:
    matrix, _labels = _rows_to_v3_matrix(features, selected_heads)
    clipped = np.clip(matrix, clip_lo, clip_hi)
    transformed = np.asarray(scaler.transform(clipped), dtype=float)
    if transformed.shape != matrix.shape or not np.all(np.isfinite(transformed)):
        raise ValueError("v3 scaler returned invalid features")
    return _positive_scores(model, transformed)


def _threshold_prediction(score: float, threshold: float) -> int:
    return int(score >= threshold)


def score_v3_run(
    run_dir: str | Path,
    *,
    ood_ai_rows: Sequence[LegacyRow],
    ood_natural_rows: Sequence[LegacyRow],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score recovered OOD rows with a freshly materialized v3 run directory."""

    run_path = Path(run_dir)
    with (run_path / "selected_heads.json").open("r", encoding="utf-8") as handle:
        selected_heads = tuple(int(head) for head in json.load(handle)["head_indices"])
    with (run_path / "results.json").open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    threshold = float(results["threshold_5pct_fpr"])
    model = joblib.load(run_path / "model.pkl")
    scaler_payload = joblib.load(run_path / "scaler.pkl")
    scaler = scaler_payload["scaler"]
    clip_lo = np.asarray(scaler_payload["clip_lo"], dtype=float)
    clip_hi = np.asarray(scaler_payload["clip_hi"], dtype=float)
    rows_by_split = {
        "ood_ai": tuple(ood_ai_rows),
        "ood_natural": tuple(ood_natural_rows),
    }
    output: list[dict[str, Any]] = []
    for split, csv_rows in rows_by_split.items():
        features_path = run_path / f"features_{split}.jsonl"
        features = _read_jsonl(features_path)
        _validate_features(features, csv_rows, feature_path=features_path)
        scores = _score_v3_features(
            features,
            selected_heads,
            model=model,
            scaler=scaler,
            clip_lo=clip_lo,
            clip_hi=clip_hi,
        )
        for row, score in zip(csv_rows, scores, strict=True):
            score_float = float(score)
            output.append(
                {
                    "split": split,
                    "row_index": row.row_index,
                    "source": row.source,
                    "label": row.label,
                    "score": score_float,
                    "prediction": _threshold_prediction(score_float, threshold),
                }
            )
    return output, {"threshold_5pct_fpr": threshold, "selected_heads": list(selected_heads)}


def _status_value(value: StageStatus | str) -> str:
    return value.value if isinstance(value, StageStatus) else str(value)


def _v4_output_row(row: LegacyRow, result: ScanResult) -> dict[str, Any]:
    scores = result.scores
    router_used = "full" if "router_full" in scores else "cpu"
    return {
        "split": row.split,
        "row_index": row.row_index,
        "source": row.source,
        "label": row.label,
        "decision": result.label.value,
        "prediction": result.prediction,
        "reason": result.reason,
        "router_used": router_used,
        "score": scores.get("router"),
        "router_cpu_score": scores.get("router_cpu"),
        "router_full_score": scores.get("router_full"),
        "pav_triggered": result.pav_triggered,
        "pav_status": _status_value(result.pav_status),
    }


def score_v4_rows(detector: Any, rows: Sequence[LegacyRow]) -> list[dict[str, Any]]:
    """Scan exact legacy rows and return compact records without raw sequences."""

    if not callable(getattr(detector, "scan", None)):
        raise TypeError("detector must expose scan(sequence, source_id)")
    output: list[dict[str, Any]] = []
    for row in rows:
        result = detector.scan(row.sequence, source_id=f"{row.split}:{row.row_index}")
        output.append(_v4_output_row(row, result))
    return output


def _finite_scores(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    scores: list[float] = []
    for row in rows:
        score = row.get("score")
        if isinstance(score, Real) and not isinstance(score, bool) and np.isfinite(score):
            scores.append(float(score))
    return scores


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return OOD natural FPR and per-AI-source AUROC/recall summaries."""

    materialized = list(rows)
    natural_rows = [row for row in materialized if row["label"] == 0]
    ai_rows = [row for row in materialized if row["label"] == 1]
    natural_scores = _finite_scores(natural_rows)
    natural_labels = [row["prediction"] for row in natural_rows]
    ai_labels = [row["prediction"] for row in ai_rows]
    per_ai: dict[str, Any] = {}
    aurocs: list[float] = []
    recalls: list[float] = []
    for source in sorted({str(row["source"]) for row in ai_rows}):
        source_rows = [row for row in ai_rows if row["source"] == source]
        source_scores = _finite_scores(source_rows)
        y_true = np.concatenate(
            [
                np.ones(len(source_scores), dtype=int),
                np.zeros(len(natural_scores), dtype=int),
            ]
        )
        y_score = np.asarray(source_scores + natural_scores, dtype=float)
        auroc = float(roc_auc_score(y_true, y_score))
        predictions = [row["prediction"] for row in source_rows]
        recall = sum(prediction == 1 for prediction in predictions) / len(source_rows)
        aurocs.append(auroc)
        recalls.append(recall)
        per_ai[source] = {
            "n": len(source_rows),
            "auroc": auroc,
            "recall": recall,
            "abstention_rate": sum(prediction is None for prediction in predictions)
            / len(source_rows),
        }
    per_natural: dict[str, Any] = {}
    for source in sorted({str(row["source"]) for row in natural_rows}):
        source_rows = [row for row in natural_rows if row["source"] == source]
        predictions = [row["prediction"] for row in source_rows]
        per_natural[source] = {
            "n": len(source_rows),
            "fpr": sum(prediction == 1 for prediction in predictions) / len(source_rows),
            "abstention_rate": sum(prediction is None for prediction in predictions)
            / len(source_rows),
        }
    all_predictions = natural_labels + ai_labels
    return {
        "n_ood_ai": len(ai_rows),
        "n_ood_natural": len(natural_rows),
        "macro_auroc": float(np.mean(aurocs)),
        "macro_recall": float(np.mean(recalls)),
        "ood_natural_fpr": sum(prediction == 1 for prediction in natural_labels)
        / len(natural_labels),
        "abstention_rate": sum(prediction is None for prediction in all_predictions)
        / len(all_predictions),
        "per_ai_source": per_ai,
        "per_natural_source": per_natural,
    }


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _validate_profile_name(name: str) -> str:
    if not isinstance(name, str) or not _SAFE_PROFILE_NAME.fullmatch(name):
        raise ValueError("profile_name must be a safe non-empty filename stem")
    return name


def _validate_score_rows(
    profile: str,
    score_rows: Sequence[Mapping[str, Any]],
    expected_rows: Sequence[LegacyRow],
) -> None:
    if len(score_rows) != len(expected_rows):
        raise ValueError(f"{profile} score count does not match the matched CSV rows")
    for score, expected in zip(score_rows, expected_rows, strict=True):
        if (
            score.get("split") != expected.split
            or score.get("row_index") != expected.row_index
            or score.get("source") != expected.source
            or score.get("label") != expected.label
        ):
            raise ValueError(f"{profile} score rows do not match the matched CSV order")


def _load_v4_detector(
    model_dir: Path,
    *,
    pav_cache: MutableMapping[str, Any] | None,
    allow_uncertified: bool,
) -> Any:
    pav_observer = None
    pav_observer_id = None
    if _has_full_checkpoint(model_dir):
        observer_config, artifact_observer_id = _load_pav_observer_spec(model_dir)
        pav_observer = _production_pav_observer(
            observer_config,
            observer_id=artifact_observer_id,
        )
        pav_observer_id = pav_observer.observer_id
    detector, _artifacts, _profile = _load_detector(
        model_dir,
        pav_observer=pav_observer,
        pav_observer_id=pav_observer_id,
        pav_cache=pav_cache,
        allow_uncertified=allow_uncertified,
    )
    return detector


def _profile_label(profile: str) -> str:
    labels = {
        "v3": "v3",
        "v4_cpu": "v4 CPU",
        "v4_full": "v4 full",
        "v4_full_fpr05": "v4 full fpr05",
    }
    return labels.get(profile, profile.replace("_", " "))


def write_comparison_markdown(path: str | Path, comparison: Mapping[str, Any]) -> None:
    summaries = comparison["summaries"]
    profiles = list(summaries)
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write("# Matched v3-pool comparison\n\n")
        handle.write(f"Rows: `{comparison['n_rows']}` exact legacy OOD rows.\n\n")
        handle.write(
            "| Profile | Macro AUROC | Macro recall | OOD-natural FPR | Abstention |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: |\n")
        for name in profiles:
            summary = summaries[name]
            handle.write(
                f"| {_profile_label(name)} | {summary['macro_auroc']:.6f} | "
                f"{summary['macro_recall']:.6f} | "
                f"{summary['ood_natural_fpr']:.6f} | "
                f"{summary['abstention_rate']:.6f} |\n"
            )
        handle.write("\n## Per-source AI recall\n\n")
        handle.write(
            "| Source | " + " | ".join(_profile_label(name) for name in profiles) + " |\n"
        )
        handle.write("| --- | " + " | ".join("---:" for _ in profiles) + " |\n")
        sources = sorted(
            {
                source
                for summary in summaries.values()
                for source in summary["per_ai_source"]
            }
        )
        for source in sources:
            recalls = " | ".join(
                f"{summaries[name]['per_ai_source'][source]['recall']:.6f}"
                for name in profiles
            )
            handle.write(f"| {source} | {recalls} |\n")


def score_additional_v4_profile(
    *,
    profile_name: str,
    model_dir: str | Path,
    ood_ai_csv: str | Path,
    ood_natural_csv: str | Path,
    baseline_score_dir: str | Path,
    out_dir: str | Path,
    pav_cache_path: str | Path | None = None,
    allow_uncertified: bool = False,
) -> dict[str, Any]:
    """Score one new v4 profile without overwriting baseline score rows."""

    safe_profile_name = _validate_profile_name(profile_name)
    ood_ai_rows = read_legacy_csv(ood_ai_csv, split="ood_ai")
    ood_natural_rows = read_legacy_csv(ood_natural_csv, split="ood_natural")
    all_rows = [*ood_ai_rows, *ood_natural_rows]
    baseline_dir = Path(baseline_score_dir)
    profile_rows = {
        "v3": _read_jsonl(baseline_dir / "v3_scores.jsonl"),
        "v4_cpu": _read_jsonl(baseline_dir / "v4_cpu_scores.jsonl"),
        "v4_full": _read_jsonl(baseline_dir / "v4_full_scores.jsonl"),
    }
    fpr05_path = baseline_dir / "v4_full_fpr05_scores.jsonl"
    if fpr05_path.is_file():
        profile_rows["v4_full_fpr05"] = _read_jsonl(fpr05_path)
    if safe_profile_name in profile_rows:
        raise ValueError(f"profile_name {safe_profile_name!r} already exists")
    for profile, rows in profile_rows.items():
        _validate_score_rows(profile, rows, all_rows)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_context = (
        JsonlPAVCache(pav_cache_path, log_every=50)
        if pav_cache_path is not None
        else None
    )
    try:
        detector = _load_v4_detector(
            Path(model_dir),
            pav_cache=cache_context,
            allow_uncertified=allow_uncertified,
        )
        new_rows = score_v4_rows(detector, all_rows)
    finally:
        if cache_context is not None:
            cache_context.close()

    profile_rows[safe_profile_name] = new_rows
    _validate_score_rows(safe_profile_name, new_rows, all_rows)
    write_jsonl(output_dir / f"{safe_profile_name}_scores.jsonl", new_rows)

    summaries = {profile: summarize_rows(rows) for profile, rows in profile_rows.items()}
    comparison = {
        "schema_version": _SCHEMA_VERSION,
        "n_rows": len(all_rows),
        "n_ood_ai": len(ood_ai_rows),
        "n_ood_natural": len(ood_natural_rows),
        "baselines": {
            profile: str(baseline_dir / f"{profile}_scores.jsonl")
            for profile in profile_rows
            if profile != safe_profile_name
        },
        safe_profile_name: {"model_dir": str(Path(model_dir))},
        "summaries": summaries,
    }
    comparison_path = output_dir / f"comparison_matched_pool_{safe_profile_name}.json"
    with comparison_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_comparison_markdown(
        output_dir / f"comparison_matched_pool_{safe_profile_name}.md",
        comparison,
    )
    return comparison


def run_matched_pool(
    *,
    v3_run_dir: str | Path,
    ood_ai_csv: str | Path,
    ood_natural_csv: str | Path,
    v4_cpu_model_dir: str | Path,
    v4_full_model_dir: str | Path,
    out_dir: str | Path,
    pav_cache_path: str | Path | None = None,
    allow_uncertified: bool = False,
) -> dict[str, Any]:
    """Run all matched-pool scoring profiles and write comparison artifacts."""

    ood_ai_rows = read_legacy_csv(ood_ai_csv, split="ood_ai")
    ood_natural_rows = read_legacy_csv(ood_natural_csv, split="ood_natural")
    all_rows = [*ood_ai_rows, *ood_natural_rows]
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    v3_rows, v3_metadata = score_v3_run(
        v3_run_dir,
        ood_ai_rows=ood_ai_rows,
        ood_natural_rows=ood_natural_rows,
    )
    cpu_detector = _load_v4_detector(
        Path(v4_cpu_model_dir),
        pav_cache=None,
        allow_uncertified=allow_uncertified,
    )
    cache_context = (
        JsonlPAVCache(pav_cache_path, log_every=50)
        if pav_cache_path is not None
        else None
    )
    try:
        full_detector = _load_v4_detector(
            Path(v4_full_model_dir),
            pav_cache=cache_context,
            allow_uncertified=allow_uncertified,
        )
        cpu_rows = score_v4_rows(cpu_detector, all_rows)
        full_rows = score_v4_rows(full_detector, all_rows)
    finally:
        if cache_context is not None:
            cache_context.close()

    profile_rows = {
        "v3": v3_rows,
        "v4_cpu": cpu_rows,
        "v4_full": full_rows,
    }
    for profile, rows in profile_rows.items():
        write_jsonl(output_dir / f"{profile}_scores.jsonl", rows)

    summaries = {profile: summarize_rows(rows) for profile, rows in profile_rows.items()}
    comparison = {
        "schema_version": _SCHEMA_VERSION,
        "n_rows": len(all_rows),
        "n_ood_ai": len(ood_ai_rows),
        "n_ood_natural": len(ood_natural_rows),
        "v3": {
            **v3_metadata,
            "run_dir": str(Path(v3_run_dir)),
        },
        "v4_cpu": {"model_dir": str(Path(v4_cpu_model_dir))},
        "v4_full": {"model_dir": str(Path(v4_full_model_dir))},
        "summaries": summaries,
        "deltas": {
            "v4_full_minus_v3_macro_auroc": summaries["v4_full"]["macro_auroc"]
            - summaries["v3"]["macro_auroc"],
            "v4_full_minus_v3_macro_recall": summaries["v4_full"]["macro_recall"]
            - summaries["v3"]["macro_recall"],
            "v4_full_minus_v4_cpu_macro_recall": summaries["v4_full"]["macro_recall"]
            - summaries["v4_cpu"]["macro_recall"],
        },
    }
    comparison_path = output_dir / "comparison_matched_pool.json"
    with comparison_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_comparison_markdown(output_dir / "comparison_matched_pool.md", comparison)
    return comparison


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare v3 and v4 on the exact recovered v3 OOD rows."
    )
    parser.add_argument("--v3-run-dir", required=True, type=Path)
    parser.add_argument("--ood-ai-csv", required=True, type=Path)
    parser.add_argument("--ood-natural-csv", required=True, type=Path)
    parser.add_argument("--v4-cpu-model-dir", required=True, type=Path)
    parser.add_argument("--v4-full-model-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--pav-cache", type=Path)
    parser.add_argument(
        "--allow-uncertified",
        action="store_true",
        help="Permit exploratory checkpoints explicitly marked uncertified.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = _build_parser().parse_args(argv)
    run_matched_pool(
        v3_run_dir=options.v3_run_dir,
        ood_ai_csv=options.ood_ai_csv,
        ood_natural_csv=options.ood_natural_csv,
        v4_cpu_model_dir=options.v4_cpu_model_dir,
        v4_full_model_dir=options.v4_full_model_dir,
        out_dir=options.out_dir,
        pav_cache_path=options.pav_cache,
        allow_uncertified=options.allow_uncertified,
    )
    return 0


__all__ = [
    "LegacyRow",
    "read_legacy_csv",
    "run_matched_pool",
    "score_additional_v4_profile",
    "score_v3_run",
    "score_v4_rows",
    "summarize_rows",
    "write_jsonl",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
