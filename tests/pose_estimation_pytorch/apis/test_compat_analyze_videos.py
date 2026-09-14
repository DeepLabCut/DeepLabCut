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
import pytest

import deeplabcut.pose_estimation_pytorch.apis as pytorch_apis
from deeplabcut.compat import Engine, analyze_videos


@pytest.mark.parametrize("overwrite", [True, False])
def test_analyze_videos_forwards_overwrite(monkeypatch, overwrite):
    """``overwrite`` must reach the PyTorch API rather than being pinned to False.

    Regression for #3513
    """
    captured = {}
    monkeypatch.setattr(pytorch_apis, "analyze_videos", lambda config, **kwargs: captured.update(kwargs))

    analyze_videos("config.yaml", ["video.mp4"], engine=Engine.PYTORCH, overwrite=overwrite)

    assert captured["overwrite"] is overwrite
