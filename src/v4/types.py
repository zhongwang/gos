"""JSON-safe public evidence and result records for GOS v4."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Mapping
import math


RESULT_SCHEMA_VERSION = "gos-v4-result/1"


class DecisionLabel(str, Enum):
    """The public tri-state decision labels."""

    AI = "AI"
    NATURAL = "natural"
    INCONCLUSIVE = "inconclusive"


class StageStatus(str, Enum):
    """Availability status for an evidence-producing stage."""

    OK = "ok"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    NOT_REQUESTED = "not_requested"


def _json_value(value: Any) -> Any:
    """Recursively convert contract records to JSON-native values.

    The public result deliberately never serializes a field named ``sequence``.
    ORF sequence is useful internally, but returning it would needlessly expose
    the caller's raw nucleotide data in a result artifact.
    """

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Span):
        return value.to_dict()
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_value(value.to_dict())
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
            if str(key) != "sequence"
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            converted = tolist()
        except (TypeError, ValueError):
            converted = value
        if converted is not value:
            return _json_value(converted)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_value(item())
        except ValueError:
            # Array-like values are not part of the public contract; leave a
            # non-scalar object for the caller's JSON encoder to reject.
            pass
    return value


@dataclass(frozen=True)
class Span:
    """A zero-based, half-open interval in the original query."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError("span coordinates must be integers")
        if self.start < 0 or self.end < 0:
            raise ValueError("span coordinates must be non-negative")
        if self.end < self.start:
            raise ValueError("span end must be greater than or equal to start")

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class WindowEvidence:
    """A calibrated sequence-window score with original-query coordinates."""

    span: Span
    score: float | None
    over_threshold: bool = False
    eligible: bool = True
    length_bin: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.span.to_dict(),
            "score": self.score,
            "over_threshold": self.over_threshold,
            "eligible": self.eligible,
            "length_bin": self.length_bin,
        }


@dataclass(frozen=True)
class SegmentEvidence:
    """Merged local evidence with an auditable score and threshold excess."""

    span: Span
    score: float
    calibrated_strength: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(self.score)
            or not 0.0 <= self.score <= 1.0
        ):
            raise ValueError("segment score must be a finite probability")
        if (
            isinstance(self.calibrated_strength, bool)
            or not isinstance(self.calibrated_strength, (int, float))
            or not math.isfinite(self.calibrated_strength)
            or self.calibrated_strength < 0.0
        ):
            raise ValueError("calibrated_strength must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.span.to_dict(),
            "score": float(self.score),
            "calibrated_strength": float(self.calibrated_strength),
        }


@dataclass(frozen=True)
class ORFEvidence:
    """An ORF score; ``sequence`` is internal-only and is not serialized."""

    span: Span
    strand: str
    frame: int
    score: float | None
    over_threshold: bool = False
    eligible: bool = True
    sequence: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.strand not in {"+", "-"}:
            raise ValueError("ORF strand must be '+' or '-'")
        if not isinstance(self.frame, int) or self.frame not in {0, 1, 2}:
            raise ValueError("ORF frame must be 0, 1, or 2")

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.span.to_dict(),
            "strand": self.strand,
            "frame": self.frame,
            "score": self.score,
            "over_threshold": self.over_threshold,
            "eligible": self.eligible,
        }


@dataclass(frozen=True)
class StageEvidence:
    """Evidence from one scoring stage without a forced provenance decision."""

    status: StageStatus
    reason: str | None = None
    aggregates: Any = None
    windows: tuple[WindowEvidence, ...] = ()
    segments: tuple[Any, ...] = ()
    orfs: tuple[ORFEvidence, ...] = ()
    n_scored: int = 0
    n_failed: int = 0
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, StageStatus):
            object.__setattr__(self, "status", StageStatus(self.status))

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "status": self.status,
                "reason": self.reason,
                "aggregates": self.aggregates,
                "windows": self.windows,
                "segments": self.segments,
                "orfs": self.orfs,
                "n_scored": self.n_scored,
                "n_failed": self.n_failed,
                "warnings": self.warnings,
            }
        )


@dataclass(frozen=True)
class ScanResult:
    """The complete public GOS v4 scan result.

    ``prediction`` is deliberately nullable: only AI and natural decisions map
    to binary values.  An inconclusive result must remain distinct until a
    caller explicitly chooses a collapse policy.
    """

    label: DecisionLabel
    prediction: int | None
    reason: str | None
    length: int
    acgt_coverage: float
    pav_triggered: bool = False
    pav_status: StageStatus = StageStatus.NOT_REQUESTED
    scores: Mapping[str, Any] = field(default_factory=dict)
    thresholds: Mapping[str, Any] = field(default_factory=dict)
    ignored_spans: tuple[Any, ...] = ()
    windows: tuple[WindowEvidence, ...] = ()
    segments: tuple[Any, ...] = ()
    orfs: tuple[ORFEvidence, ...] = ()
    stages: tuple[StageEvidence, ...] = ()
    warnings: tuple[str, ...] = ()
    schema_version: str = field(default=RESULT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.label, DecisionLabel):
            object.__setattr__(self, "label", DecisionLabel(self.label))
        if not isinstance(self.pav_status, StageStatus):
            object.__setattr__(self, "pav_status", StageStatus(self.pav_status))
        if self.length < 0:
            raise ValueError("length must be non-negative")
        if not 0.0 <= self.acgt_coverage <= 1.0:
            raise ValueError("acgt_coverage must be between 0 and 1")

        if self.prediction is not None and type(self.prediction) is not int:
            raise ValueError("prediction must be an exact integer 0 or 1, or None")

        expected_prediction = {
            DecisionLabel.AI: 1,
            DecisionLabel.NATURAL: 0,
            DecisionLabel.INCONCLUSIVE: None,
        }[self.label]
        if self.prediction != expected_prediction:
            raise ValueError("prediction must match the tri-state decision label")

    @classmethod
    def ai(cls, *, length: int, acgt_coverage: float, **kwargs: Any) -> "ScanResult":
        return cls(
            label=DecisionLabel.AI,
            prediction=1,
            reason=kwargs.pop("reason", None),
            length=length,
            acgt_coverage=acgt_coverage,
            **kwargs,
        )

    @classmethod
    def natural(cls, *, length: int, acgt_coverage: float, **kwargs: Any) -> "ScanResult":
        return cls(
            label=DecisionLabel.NATURAL,
            prediction=0,
            reason=kwargs.pop("reason", None),
            length=length,
            acgt_coverage=acgt_coverage,
            **kwargs,
        )

    @classmethod
    def inconclusive(
        cls, *, reason: str, length: int, acgt_coverage: float, **kwargs: Any
    ) -> "ScanResult":
        return cls(
            label=DecisionLabel.INCONCLUSIVE,
            prediction=None,
            reason=reason,
            length=length,
            acgt_coverage=acgt_coverage,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "schema_version": self.schema_version,
                "label": self.label,
                "prediction": self.prediction,
                "reason": self.reason,
                "length": self.length,
                "acgt_coverage": self.acgt_coverage,
                "pav_triggered": self.pav_triggered,
                "pav_status": self.pav_status,
                "scores": self.scores,
                "thresholds": self.thresholds,
                "ignored_spans": self.ignored_spans,
                "windows": self.windows,
                "segments": self.segments,
                "orfs": self.orfs,
                "stages": self.stages,
                "warnings": self.warnings,
            }
        )
