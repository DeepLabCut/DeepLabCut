from .base import EXTERNAL_DETECTORS, BaseExternalDetector, DetectionResult

# Import all external detectors here to populate the registry
from .mock import MockExternalDetector

__all__ = [
    "BaseExternalDetector",
    "EXTERNAL_DETECTORS",
    "DetectionResult",
]
