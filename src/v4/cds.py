"""Six-frame CDS evidence with coordinate-preserving ORF records."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import ArtifactCompatibilityError, artifact_is_certified, validate_artifact
from .preprocessing import PreparedSequence
from .types import ORFEvidence, Span, StageEvidence, StageStatus


_START_CODONS = frozenset({"ATG"})
_STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})
_COMPLEMENT = str.maketrans("ACGT", "TGCA")
_FEATURE_NAMES = (
    "len",
    "gc1",
    "gc2",
    "gc3",
    "entropy",
    "shadow_stops",
    "fourier_3",
    "cai",
    "cpb",
)
_CODON_TABLE = {
    "ATA": "I", "ATC": "I", "ATT": "I", "ATG": "M", "ACA": "T", "ACC": "T",
    "ACG": "T", "ACT": "T", "AAC": "N", "AAT": "N", "AAA": "K", "AAG": "K",
    "AGC": "S", "AGT": "S", "AGA": "R", "AGG": "R", "CTA": "L", "CTC": "L",
    "CTG": "L", "CTT": "L", "CCA": "P", "CCC": "P", "CCG": "P", "CCT": "P",
    "CAC": "H", "CAT": "H", "CAA": "Q", "CAG": "Q", "CGA": "R", "CGC": "R",
    "CGG": "R", "CGT": "R", "GTA": "V", "GTC": "V", "GTG": "V", "GTT": "V",
    "GCA": "A", "GCC": "A", "GCG": "A", "GCT": "A", "GAC": "D", "GAT": "D",
    "GAA": "E", "GAG": "E", "GGA": "G", "GGC": "G", "GGG": "G", "GGT": "G",
    "TCA": "S", "TCC": "S", "TCG": "S", "TCT": "S", "TTC": "F", "TTT": "F",
    "TTA": "L", "TTG": "L", "TAC": "Y", "TAT": "Y", "TGC": "C", "TGT": "C",
    "TGG": "W",
}


@dataclass(frozen=True)
class ORFRecord:
    """One complete ORF, with its interval in the original query."""

    strand: str
    frame: int
    span: Span
    sequence: str

    def __post_init__(self) -> None:
        if self.strand not in {"+", "-"}:
            raise ValueError("ORF strand must be '+' or '-'")
        if self.frame not in {0, 1, 2}:
            raise ValueError("ORF frame must be 0, 1, or 2")
        if len(self.sequence) != self.span.length:
            raise ValueError("ORF sequence length must match its span")


def reverse_complement(seq: str) -> str:
    """Return the uppercase reverse complement of one ACGT sequence."""

    if not isinstance(seq, str):
        raise TypeError("seq must be a string")
    sequence = seq.upper()
    if any(base not in "ACGT" for base in sequence):
        raise ValueError("seq must contain only A, C, G, and T")
    return sequence.translate(_COMPLEMENT)[::-1]


def _orfs_on_strand(sequence: str, strand: str, min_length: int, original_length: int) -> list[ORFRecord]:
    records: list[ORFRecord] = []
    for frame in range(3):
        offset = frame
        while offset <= len(sequence) - 3:
            if sequence[offset : offset + 3] not in _START_CODONS:
                offset += 3
                continue
            stop = offset + 3
            while stop <= len(sequence) - 3 and sequence[stop : stop + 3] not in _STOP_CODONS:
                stop += 3
            if stop <= len(sequence) - 3:
                end = stop + 3
                orf_sequence = sequence[offset:end]
                if len(orf_sequence) >= min_length:
                    if strand == "+":
                        span = Span(offset, end)
                    else:
                        span = Span(original_length - end, original_length - offset)
                    records.append(ORFRecord(strand, frame, span, orf_sequence))
                offset = end
            else:
                offset += 3
    return records


def find_orfs(seq: str, min_length: int = 300) -> tuple[ORFRecord, ...]:
    """Find complete start-to-stop ORFs across all six reading frames."""

    if not isinstance(seq, str):
        raise TypeError("seq must be a string")
    if not isinstance(min_length, int) or isinstance(min_length, bool) or min_length < 1:
        raise ValueError("min_length must be a positive integer")
    sequence = seq.upper()
    if any(base not in "ACGT" for base in sequence):
        raise ValueError("seq must contain only A, C, G, and T")
    reverse = reverse_complement(sequence)
    records = _orfs_on_strand(sequence, "+", min_length, len(sequence))
    records.extend(_orfs_on_strand(reverse, "-", min_length, len(sequence)))
    return tuple(sorted(records, key=lambda record: (record.span.start, record.span.end, record.strand, record.frame)))


@dataclass(frozen=True)
class CDSAggregates:
    """Aggregate CDS evidence, keeping sparse ORF hits visible."""

    n_eligible: int
    n_orfs: int
    available: bool
    mean: float | None
    maximum: float | None
    top_k_mean: float | None
    count_over: int | None
    fraction_over: float | None
    covered_coding_bases: int
    coding_coverage: float | None


def _artifact_error(message: str) -> ArtifactCompatibilityError:
    return ArtifactCompatibilityError(f"GOS v4 CDS artifact compatibility error: {message}")


def _number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise _artifact_error(f"{field} must be a finite number")
    return float(value)


def _validate_number_table(value: Any, *, field: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise _artifact_error(f"{field} must be a mapping")
    result: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise _artifact_error(f"{field} keys must be non-empty strings")
        result[key] = _number(item, field=f"{field}[{key!r}]")
    return result


def _clip_vector(payload: Mapping[str, Any], primary: str) -> np.ndarray:
    raw = payload.get(primary)
    try:
        vector = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as error:
        raise _artifact_error(f"{primary} must be a nine-value numeric vector") from error
    if vector.shape != (len(_FEATURE_NAMES),) or not np.all(np.isfinite(vector)):
        raise _artifact_error(f"{primary} must be a nine-value finite numeric vector")
    return vector


def _orf_features(orf: ORFRecord, cai_weights: Mapping[str, float], cps_table: Mapping[str, float]) -> np.ndarray:
    """Port the v3 codon feature formulas without importing legacy modules."""

    sequence = orf.sequence
    n_bases = len(sequence)
    codons = [sequence[offset : offset + 3] for offset in range(0, n_bases - 3, 3)]
    if not codons:
        raise ValueError("ORF must contain at least one non-stop codon")
    gc1 = sum(codon[0] in "GC" for codon in codons) / len(codons)
    gc2 = sum(codon[1] in "GC" for codon in codons) / len(codons)
    gc3 = sum(codon[2] in "GC" for codon in codons) / len(codons)

    stops_f1 = 0
    stops_f2 = 0
    for offset in range(1, n_bases - 3):
        if sequence[offset : offset + 3] in _STOP_CODONS:
            if offset % 3 == 1:
                stops_f1 += 1
            elif offset % 3 == 2:
                stops_f2 += 1
    shadow_stops = (stops_f1 + stops_f2) / (n_bases / 1000)

    counts = Counter(codons)
    proportions = np.asarray(list(counts.values()), dtype=float) / len(codons)
    entropy = float(-np.sum(proportions * np.log2(proportions)))

    encoded = np.zeros((4, n_bases), dtype=float)
    for offset, base in enumerate(sequence):
        encoded["ACGT".index(base), offset] = 1.0
    period_index = n_bases // 3
    period_power = 0.0
    total_power = 0.0
    for channel in encoded:
        power = np.abs(np.fft.fft(channel)) ** 2
        period_power += power[period_index]
        total_power += np.sum(power[1:])
    fourier_3 = float(period_power / total_power) if total_power else 0.0

    cai_values = [cai_weights.get(codon, 0.01) for codon in codons if codon in _CODON_TABLE]
    cai = float(np.exp(np.mean(np.log(cai_values)))) if cai_values else 0.0
    cps_values = [
        cps_table.get(codons[index] + codons[index + 1], -5.0)
        for index in range(len(codons) - 1)
    ]
    cpb = float(np.mean(cps_values)) if cps_values else 0.0
    return np.array(
        [n_bases, gc1, gc2, gc3, entropy, shadow_stops, fourier_3, cai, cpb], dtype=float
    )


def _covered_bases(orfs: Sequence[ORFEvidence]) -> int:
    intervals = sorted((orf.span for orf in orfs), key=lambda span: (span.start, span.end))
    if not intervals:
        return 0
    total = 0
    start, end = intervals[0].start, intervals[0].end
    for span in intervals[1:]:
        if span.start <= end:
            end = max(end, span.end)
        else:
            total += end - start
            start, end = span.start, span.end
    return total + end - start


def _aggregate_orfs(
    orfs: Sequence[ORFEvidence], *, threshold: float, scorable_bases: int
) -> CDSAggregates:
    eligible = [orf for orf in orfs if orf.eligible and orf.score is not None]
    covered = _covered_bases(orfs)
    coverage = covered / scorable_bases if scorable_bases else None
    if not eligible:
        return CDSAggregates(0, len(orfs), False, None, None, None, None, None, covered, coverage)
    scores = np.asarray([orf.score for orf in eligible], dtype=float)
    over = scores >= threshold
    top_k = np.partition(scores, len(scores) - min(3, len(scores)))[-min(3, len(scores)) :]
    return CDSAggregates(
        n_eligible=len(eligible),
        n_orfs=len(orfs),
        available=True,
        mean=float(scores.mean()),
        maximum=float(scores.max()),
        top_k_mean=float(top_k.mean()),
        count_over=int(over.sum()),
        fraction_over=float(over.mean()),
        covered_coding_bases=covered,
        coding_coverage=coverage,
    )


def score_known_orfs(scores: Sequence[float], threshold: float) -> StageEvidence:
    """Construct aggregate evidence from already-calibrated ORF scores for tests/tools."""

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a one-dimensional finite numeric sequence")
    orfs = tuple(
        ORFEvidence(Span(index * 3, index * 3 + 3), "+", 0, float(score), score >= threshold)
        for index, score in enumerate(values)
    )
    aggregates = _aggregate_orfs(orfs, threshold=threshold, scorable_bases=len(orfs) * 3)
    if not aggregates.available:
        return StageEvidence(StageStatus.UNAVAILABLE, "no_orfs", aggregates=aggregates)
    return StageEvidence(StageStatus.OK, aggregates=aggregates, orfs=orfs, n_scored=len(orfs))


class CDSStage:
    """Score all prepared-fragment ORFs with a strict global-background artifact."""

    def __init__(
        self, artifact: Mapping[str, Any], *, allow_uncertified: bool = False
    ):
        if type(allow_uncertified) is not bool:
            raise TypeError("allow_uncertified must be an exact boolean")
        artifact = validate_artifact(artifact, expected_kind="cds")
        self.certified = artifact_is_certified(artifact)
        if not self.certified and not allow_uncertified:
            raise _artifact_error(
                "uncertified CDS artifact requires allow_uncertified=True"
            )
        if artifact.get("feature_names") != list(_FEATURE_NAMES):
            raise _artifact_error(
                "feature_names must be len, gc1, gc2, gc3, entropy, shadow_stops, fourier_3, cai, cpb in order"
            )
        metadata = artifact.get("metadata")
        payload = artifact.get("payload")
        if not isinstance(metadata, Mapping) or not isinstance(payload, Mapping):
            raise _artifact_error("metadata and payload must be mappings")
        self._artifact = artifact
        if metadata.get("background") != "global":
            raise _artifact_error("metadata background must be 'global'")
        thresholds = metadata.get("thresholds")
        if not isinstance(thresholds, Mapping):
            raise _artifact_error("metadata must include ORF threshold metadata")
        calibration = thresholds.get("calibration")
        if not isinstance(calibration, Mapping):
            raise _artifact_error("metadata must include ORF calibration metadata")
        self._threshold = _number(
            calibration.get("threshold"), field="calibrated ORF threshold"
        )
        self._estimator = payload.get("estimator")
        self._scaler = payload.get("scaler")
        if not callable(getattr(self._estimator, "predict_proba", None)):
            raise _artifact_error("payload estimator must provide predict_proba")
        if not callable(getattr(self._scaler, "transform", None)):
            raise _artifact_error("payload scaler must provide transform")
        self._clip_lower = _clip_vector(payload, "clip_lower")
        self._clip_upper = _clip_vector(payload, "clip_upper")
        if np.any(self._clip_lower > self._clip_upper):
            raise _artifact_error("clip_lower must not exceed clip_upper")
        self._cai_weights = _validate_number_table(payload.get("cai_weights"), field="cai_weights")
        self._cps_table = _validate_number_table(payload.get("cps_table"), field="cps_table")

    def _score_orf(self, orf: ORFRecord) -> float:
        features = np.clip(
            _orf_features(orf, self._cai_weights, self._cps_table), self._clip_lower, self._clip_upper
        ).reshape(1, -1)
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
    def _prepared_orfs(prepared: PreparedSequence) -> tuple[ORFRecord, ...]:
        records: list[ORFRecord] = []
        for fragment in prepared.fragments:
            for orf in find_orfs(fragment.sequence):
                span = Span(fragment.span.start + orf.span.start, fragment.span.start + orf.span.end)
                frame = (
                    span.start % 3
                    if orf.strand == "+"
                    else (prepared.length - span.end) % 3
                )
                records.append(ORFRecord(orf.strand, frame, span, orf.sequence))
        return tuple(sorted(records, key=lambda record: (record.span.start, record.span.end, record.strand, record.frame)))

    def score(self, prepared: PreparedSequence) -> StageEvidence:
        if not isinstance(prepared, PreparedSequence):
            raise TypeError("prepared must be a PreparedSequence")
        records = self._prepared_orfs(prepared)
        if not records:
            aggregates = _aggregate_orfs((), threshold=self._threshold, scorable_bases=prepared.scorable_bases)
            return StageEvidence(StageStatus.UNAVAILABLE, "no_orfs", aggregates=aggregates)
        failures = 0
        evidence: list[ORFEvidence] = []
        for record in records:
            try:
                score = self._score_orf(record)
            except Exception:
                failures += 1
                evidence.append(
                    ORFEvidence(record.span, record.strand, record.frame, None, eligible=False, sequence=record.sequence)
                )
                continue
            evidence.append(
                ORFEvidence(
                    record.span,
                    record.strand,
                    record.frame,
                    score,
                    score >= self._threshold,
                    sequence=record.sequence,
                )
            )
        aggregates = _aggregate_orfs(evidence, threshold=self._threshold, scorable_bases=prepared.scorable_bases)
        if not aggregates.available:
            return StageEvidence(
                StageStatus.FAILED,
                "no_orf_scores",
                aggregates=aggregates,
                orfs=tuple(evidence),
                n_failed=failures,
            )
        return StageEvidence(
            StageStatus.OK,
            aggregates=aggregates,
            orfs=tuple(evidence),
            n_scored=aggregates.n_eligible,
            n_failed=failures,
            warnings=("some_orf_scores_failed",) if failures else (),
        )
