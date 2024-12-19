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
import pickle
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import deeplabcut.pose_estimation_pytorch.runners as runners


def test_load_snapshot_weights_only_error(tmpdir_factory):
    snapshot_dir = Path(tmpdir_factory.mktemp("snapshot-dir"))
    snapshot_path = snapshot_dir / "snapshot.pt"
    torch.save(dict(content=np.zeros(10)), str(snapshot_path))

    with pytest.raises(pickle.UnpicklingError):
        runners.Runner.load_snapshot(
            snapshot_path, device="cpu", model=Mock(), weights_only=True
        )
