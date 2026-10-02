"""GOS: distilled GenomeOcean observer and production CPU decision head."""
from .decision import DecisionHead
from .detector import GOSDetector
from .student import GenomeOceanStudent

__all__ = ["DecisionHead", "GOSDetector", "GenomeOceanStudent"]
