"""Scan FASTA records with the shipped distilled student."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys

from .detector import GOSDetector


def read_fasta(path):
    """Read opaque FASTA strings, removing formatting whitespace and uppercasing."""
    seen = set()
    header, pieces = None, []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield _record(header, pieces)
                header, pieces = line[1:].strip(), []
                if not header:
                    raise ValueError(f"empty FASTA header at line {line_number}")
                record_id = header.split()[0]
                if record_id in seen:
                    raise ValueError(f"duplicate FASTA ID: {record_id}")
                seen.add(record_id)
            else:
                if header is None:
                    raise ValueError(f"sequence before FASTA header at line {line_number}")
                pieces.append("".join(line.split()).upper())
    if header is None:
        raise ValueError("FASTA contains no records")
    yield _record(header, pieces)


def _record(header, pieces):
    record_id = header.split()[0]
    sequence = "".join(pieces)
    if not sequence:
        raise ValueError(f"empty FASTA record: {record_id}")
    labels = re.findall(r"\btrue_label=([^\s|\]]+)", header)
    if labels and (len(labels) != 1 or labels[0] not in {"0", "1"}):
        raise ValueError(f"invalid true_label for {record_id}; expected 0 or 1")
    return record_id, sequence, int(labels[0]) if labels else None


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="FASTA file")
    parser.add_argument("--model", required=True, type=Path, help="local GenomeOcean-100M v1.2 snapshot")
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parents[2] / "models/gos_detector/gos_detector_distilled_genomeocean_100m.pt")
    parser.add_argument("--decision-head", type=Path, help="defaults to decision_head.json beside checkpoint")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--batch-size", type=positive_int, default=16)
    args = parser.parse_args(argv)
    try:
        # Prevent accidental replacement of an input or model asset.
        protected = [args.input, args.checkpoint, args.decision_head or args.checkpoint.with_name("decision_head.json")]
        output = args.output.resolve()
        if any(output == path.resolve() for path in protected) or output.is_relative_to(args.model.resolve()):
            raise ValueError("output must not overwrite input or model assets")
        detector = GOSDetector(model=args.model, checkpoint=args.checkpoint,
                               decision_head=args.decision_head, device=args.device, dtype=args.dtype)
        records, pending = [], []

        def scan_pending():
            try:
                predictions = detector.scan_batch([item[1] for item in pending])
            except ValueError as error:
                raise ValueError(f"batch {[item[0] for item in pending]}: {error}") from error
            for (record_id, _, true_label), prediction in zip(pending, predictions):
                records.append({"id": record_id, "true_label": true_label, **prediction})
            pending.clear()

        for record in read_fasta(args.input):
            pending.append(record)
            if len(pending) == args.batch_size:
                scan_pending()
        if pending:
            scan_pending()
        payload = {
            "schema_version": "gos-scan/1",
            "observer": "GenomeOcean-100M v1.2 distilled student",
            "checkpoint": args.checkpoint.name,
            "threshold_logit": detector.decision.threshold_logit,
            "calibration": detector.decision.calibration,
            "runtime": {"device": str(detector.device), "dtype": args.dtype,
                        "batch_size": args.batch_size, "python": platform.python_version(),
                        **{name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")}},
            "records": records,
        }
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"GOS: {error}\n")
    print(f"Scanned {len(records)} records -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
