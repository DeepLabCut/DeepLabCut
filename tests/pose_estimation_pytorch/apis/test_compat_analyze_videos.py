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

    Regression for #3513: a hard-coded ``overwrite=False`` plus ``overwrite`` in
    ``**torch_kwargs`` used to raise ``TypeError: got multiple values for keyword
    argument 'overwrite'``. Using an explicit ``overwrite`` parameter on the mock
    would still raise if the wrapper double-passed the flag.
    """
    captured = {}

    def fake_analyze_videos(config, *, overwrite=False, **kwargs):
        captured["overwrite"] = overwrite
        captured["kwargs"] = kwargs
        return "mock-scorer"

    monkeypatch.setattr(pytorch_apis, "analyze_videos", fake_analyze_videos)

    result = analyze_videos(
        "config.yaml",
        ["video.mp4"],
        engine=Engine.PYTORCH,
        overwrite=overwrite,
    )

    assert result == "mock-scorer"
    assert captured["overwrite"] is overwrite
    assert "overwrite" not in captured["kwargs"]
