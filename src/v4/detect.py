"""FASTA detection CLI with complete JSON and compact TSV projections."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO

from .evaluate import _has_full_checkpoint, _load_detector, _load_pav_observer_spec
from .train import _production_pav_observer


_TSV_FIELDS = (
    "id",
    "length",
    "label",
    "prediction",
    "router_score",
    "sequence_mean",
    "sequence_max",
    "cds_max",
    "pav_status",
    "acgt_coverage",
    "reason",
    "segments",
)


def read_fasta(paths: Iterable[str | Path]) -> list[dict[str, str]]:
    """Read non-empty, uniquely identified FASTA records from input paths."""

    records: list[dict[str, str]] = []
    identifiers: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"input FASTA does not exist: {path}")
        identifier: str | None = None
        chunks: list[str] = []

        def append_record() -> None:
            nonlocal identifier, chunks
            if identifier is None:
                return
            sequence = "".join(chunks).replace(" ", "").upper()
            if not sequence:
                raise ValueError(f"FASTA {path} contains empty record {identifier!r}")
            if identifier in identifiers:
                raise ValueError(f"duplicate FASTA record id {identifier!r}")
            identifiers.add(identifier)
            records.append({"id": identifier, "sequence": sequence})

        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    append_record()
                    identifier = line[1:].split(maxsplit=1)[0]
                    if not identifier:
                        raise ValueError(f"FASTA {path} has an empty header at line {line_number}")
                    chunks = []
                else:
                    if identifier is None:
                        raise ValueError(f"FASTA {path} must begin with a header")
                    chunks.append(line)
        append_record()
    if not records:
        raise ValueError("input FASTA files contain no records")
    return records


def _value(value: Any) -> Any:
    return "NA" if value is None else value


def _segments(segments: Any) -> str:
    if not isinstance(segments, list):
        return ""
    projected: list[str] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        start, end = segment.get("start"), segment.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        strength = segment.get("strength")
        if strength is None:
            strength = segment.get("calibrated_strength", segment.get("score", "NA"))
        projected.append(f"{start}-{end}:{strength}")
    return ",".join(projected)


def tsv_row(record_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Project a complete result dictionary into the documented TSV schema."""

    scores = result.get("scores")
    if not isinstance(scores, Mapping):
        scores = {}
    return {
        "id": record_id,
        "length": _value(result.get("length")),
        "label": _value(result.get("label")),
        "prediction": _value(result.get("prediction")),
        "router_score": _value(scores.get("router")),
        "sequence_mean": _value(scores.get("sequence_mean")),
        "sequence_max": _value(scores.get("sequence_max")),
        "cds_max": _value(scores.get("cds_max")),
        "pav_status": _value(result.get("pav_status")),
        "acgt_coverage": _value(result.get("acgt_coverage")),
        "reason": _value(result.get("reason")),
        "segments": _segments(result.get("segments")),
    }


def _write_json(records: Sequence[Mapping[str, Any]], handle: TextIO) -> None:
    json.dump(list(records), handle, indent=2, sort_keys=True)
    handle.write("\n")


def _write_tsv(records: Sequence[Mapping[str, Any]], handle: TextIO) -> None:
    writer = csv.DictWriter(handle, fieldnames=_TSV_FIELDS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan FASTA records with a saved GOS v4 checkpoint."
    )
    parser.add_argument("--input", required=True, nargs="+", type=Path, help="One or more FASTA files.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    parser.add_argument("--out", type=Path, help="Write results to this file instead of stdout.")
    parser.add_argument(
        "--allow-uncertified",
        action="store_true",
        help="Permit exploratory checkpoints explicitly marked uncertified.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Scan inputs once and keep machine-readable output free of summaries."""

    options = _build_parser().parse_args(argv)
    pav_observer = None
    pav_observer_id = None
    if _has_full_checkpoint(options.model_dir):
        observer_config, artifact_observer_id = _load_pav_observer_spec(
            options.model_dir
        )
        pav_observer = _production_pav_observer(
            observer_config,
            observer_id=artifact_observer_id,
        )
        pav_observer_id = pav_observer.observer_id
    detector, _artifacts, profile = _load_detector(
        options.model_dir,
        pav_observer=pav_observer,
        pav_observer_id=pav_observer_id,
        allow_uncertified=options.allow_uncertified,
    )
    rows: list[dict[str, Any]] = []
    for source in read_fasta(options.input):
        result = detector.scan(source["sequence"], source_id=source["id"])
        rows.append({"id": source["id"], **result.to_dict()})

    output_rows: Sequence[Mapping[str, Any]]
    if options.format == "json":
        output_rows = rows
        write = _write_json
    else:
        output_rows = [tsv_row(row["id"], row) for row in rows]
        write = _write_tsv
    if options.out is None:
        write(output_rows, sys.stdout)
    else:
        with options.out.open("w", encoding="utf-8", newline="") as handle:
            write(output_rows, handle)
    labels = {label: sum(row["label"] == label for row in rows) for label in ("AI", "natural", "inconclusive")}
    print(
        "scanned "
        f"{len(rows)} records with {profile} checkpoint "
        f"(AI={labels['AI']}, natural={labels['natural']}, inconclusive={labels['inconclusive']})",
        file=sys.stderr,
    )
    return 0


__all__ = ["main", "read_fasta", "tsv_row"]


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess CLI tests
    raise SystemExit(main())
