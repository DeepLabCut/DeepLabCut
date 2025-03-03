import numpy as np
import pandas as pd
import pytest

from deeplabcut.pose_estimation_pytorch.apis.tracklets import build_tracklets


@pytest.mark.parametrize(
    "assemblies_data, inference_cfg, joints, scorer, num_frames, unique_bodyparts",
    [
        (
            # assemblies_data
            {
                "single": {
                    0: np.array([[1, 2, 0.9]]),
                    1: np.array([[1, 3, 0.7]]),
                    2: np.array([[0, 1, 0.9]]),
                },
                0: [
                    np.array([[10, 20, 0.9, -1], [30, 40, 0.8, -1]]),
                    np.array([[13, 23, 0.9, -1], [33, 43, 0.8, -1]]),
                ],
                1: [
                    np.array([[9, 19, 0.9, -1], [29, 41, 0.8, -1]]),
                    np.array([[15, 21, 0.9, -1], [35, 45, 0.8, -1]]),
                ],
                2: [
                    np.array([[13, 23, 0.9, -1], [33, 43, 0.8, -1]]),
                    np.array([[10, 20, 0.9, -1], [30, 40, 0.8, -1]]),
                ],
            },
            # inference_cfg
            {"max_age": 3, "min_hits": 1, "topktoretain": 1, "pcutoff": 0.5},
            # joints
            ["nose", "ear"],
            # scorer
            "DLC",
            # num_frames
            3,
            # unique_bodyparts
            ["led"],
        ),
        (
            # assemblies_data
            {
                0: [
                    np.array([[10, 20, 0.9, -1], [30, 40, 0.8, -1]]),
                    np.array([[13, 23, 0.9, -1], [33, 43, 0.8, -1]]),
                ],
                1: [
                    np.array([[9, 19, 0.9, -1], [29, 41, 0.8, -1]]),
                    np.array([[15, 21, 0.9, -1], [35, 45, 0.8, -1]]),
                ],
                2: [
                    np.array([[13, 23, 0.9, -1], [33, 43, 0.8, -1]]),
                    np.array([[10, 20, 0.9, -1], [30, 40, 0.8, -1]]),
                ],
            },
            # inference_cfg
            {"max_age": 3, "min_hits": 1, "topktoretain": 1, "pcutoff": 0.5},
            # joints
            ["nose", "ear"],
            # scorer
            "DLC",
            # num_frames
            3,
            # unique_bodyparts
            None,
        ),
    ],
)
def test_build_tracklets(
    assemblies_data: dict,
    inference_cfg: dict,
    joints: list,
    scorer: str,
    num_frames: int,
    unique_bodyparts: list,
):
    # Run the function
    tracklets = build_tracklets(
        assemblies_data=assemblies_data,
        track_method="box",
        inference_cfg=inference_cfg,
        joints=joints,
        scorer=scorer,
        num_frames=num_frames,
        unique_bodyparts=unique_bodyparts,
        identity_only=False,
    )

    # # Assertions
    assert "header" in tracklets
    assert isinstance(tracklets["header"], pd.MultiIndex)
    if unique_bodyparts:
        assert "single" in tracklets
    else:
        assert not "single" in tracklets

    assert isinstance(tracklets, dict)
