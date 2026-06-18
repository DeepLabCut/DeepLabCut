# deeplabcut/pose_estimation_pytorch/models/detectors/external/models/owl_v2/__init__.py

from deeplabcut.pose_estimation_pytorch.models.detectors.external.models.owl_v2.config import (
    OWLv2DetectorConfig,
)
from deeplabcut.pose_estimation_pytorch.models.detectors.external.models.owl_v2.model import (
    OWLv2DetectorModel,
)

__all__ = [
    "OWLv2DetectorConfig",
    "OWLv2DetectorModel",
]
