"""Public contracts for the GOS v4 detector framework."""

from .artifacts import ArtifactCompatibilityError, load_artifact, save_artifact, validate_artifact
from .pipeline import GOSV4Detector
from .types import (
    DecisionLabel,
    ORFEvidence,
    ScanResult,
    SegmentEvidence,
    Span,
    StageEvidence,
    StageStatus,
    WindowEvidence,
)

__all__ = [
    "ArtifactCompatibilityError",
    "DecisionLabel",
    "GOSV4Detector",
    "ORFEvidence",
    "ScanResult",
    "SegmentEvidence",
    "Span",
    "StageEvidence",
    "StageStatus",
    "WindowEvidence",
    "load_artifact",
    "save_artifact",
    "validate_artifact",
]
