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
"""Tests for the bottom-up pytorch runner."""

from pathlib import Path
from typing import Any

import pytest

from deeplabcut.pose_estimation_pytorch.config import make_pytorch_pose_config
from deeplabcut.pose_estimation_pytorch.models import PoseModel
from deeplabcut.pose_estimation_pytorch.runners.train import build_training_runner
from deeplabcut.pose_estimation_pytorch.task import Task

SINGLE_ANIMAL_NETS = ["resnet_50"]
MULTI_ANIMAL_NETS = ["dekr_w18"]
NETS = [(n, False) for n in SINGLE_ANIMAL_NETS] + [(n, True) for n in MULTI_ANIMAL_NETS]


def print_dict(data: dict, indent: int = 0):
    for k, v in data.items():
        if isinstance(v, dict):
            print_dict(v, indent=indent + 2)
        else:
            print(f"{indent * ' '}{k}: {v}")


# @pytest.mark.skip(reason="This test is outdated and needs to be updated to reflect changes in the codebase.")


@pytest.mark.parametrize("net_type, multianimal", NETS)
def test_build_bottom_up_runner(
    net_type: str,
    multianimal: bool,
    tmp_path: Path,
) -> None:
    project_cfg: dict[str, Any] = {
        "multianimalproject": multianimal,
        "project_path": str(tmp_path),
    }
    if multianimal:
        project_cfg["bodyparts"] = "MULTI!"
        project_cfg["multianimalbodyparts"] = ["head", "shoulder", "knee", "toe"]
        project_cfg["uniquebodyparts"] = []
        project_cfg["individuals"] = ["tom", "jerry"]
    else:
        project_cfg["bodyparts"] = ["head", "shoulder", "knee", "toe"]
        project_cfg["uniquebodyparts"] = []
        project_cfg["individuals"] = ["tom"]

    root_path = Path(__file__).parent.parent
    template_path = (root_path / "other/test_configs/pytorch_config.yaml").resolve()
    assert template_path.is_file(), f"Template config not found at {template_path}"

    pytorch_cfg = make_pytorch_pose_config(project_cfg, str(template_path), net_type)
    pose_model = PoseModel.build(pytorch_cfg["model"])

    # NOTE: @C-Achard 2026-03-18 This file was not named with test_* as a prefix,
    #  so it never ran in CI. A lot of imports are outdated and non-existent
    # FIX: replace RUNNERS registry with build_training_runner and remove unused imports
    runner = build_training_runner(
        runner_config=pytorch_cfg["runner"],
        model_folder=tmp_path,
        task=Task.BOTTOM_UP,
        model=pose_model,
        device=pytorch_cfg["device"],
        logger=None,
    )
    assert runner is not None
