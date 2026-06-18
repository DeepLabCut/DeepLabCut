from .base import EXTERNAL_DETECTORS, BaseExternalDetector, DetectionResult, PrecomputedDetectorRunner

# Import all external detectors here to populate the registry
from .mock import MockExternalDetector

__all__ = [
    "BaseExternalDetector",
    "EXTERNAL_DETECTORS",
    "DetectionResult",
    "PrecomputedDetectorRunner",
]
