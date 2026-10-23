#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests for the PyTorch video analysis API."""

import io
import pickle
from pathlib import Path
from unittest.mock import Mock

import pytest

from deeplabcut.core.config import ProjectConfig
from deeplabcut.pose_estimation_pytorch.apis.videos import _generate_metadata
from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig


class _NoGlobalsUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"pickle references {module}.{name}")


def test_video_metadata_pickles_without_custom_types(tmp_path: Path) -> None:
    """``_meta.pickle`` must load on any OS and without DeepLabCut (#3562).

    A pickled ``PosixPath`` cannot be instantiated on Windows (and
    ``WindowsPath`` not on Linux), so the metadata must hold plain data only.
    """
    project_config = ProjectConfig(
        project_path=tmp_path,
        bodyparts=["nose"],
        individuals=["mouse"],
        multianimalproject=False,
    )
    pose_config = PoseConfig.build(
        project_config,
        tmp_path / "pytorch_config.yaml",
        top_down=False,
        net_type="resnet_50",
    )
    video = Mock(dimensions=(128, 128), fps=30.0)
    video.get_n_frames.return_value = 100

    metadata = _generate_metadata(
        cfg=project_config,
        pytorch_config=pose_config,
        dlc_scorer="DLC_Resnet50_testOct23shuffle1_snapshot_1",
        train_fraction=0.95,
        batch_size=1,
        cropping=None,
        runtime=(0.0, 1.0),
        video=video,
    )

    try:
        _NoGlobalsUnpickler(io.BytesIO(pickle.dumps(metadata, pickle.HIGHEST_PROTOCOL))).load()
    except pickle.UnpicklingError as err:
        pytest.fail(str(err))
