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
import torch

from deeplabcut.pose_estimation_pytorch.config import make_pytorch_pose_config
from deeplabcut.pose_estimation_pytorch.models import LOSS_AGGREGATORS, PREDICTORS, PoseModel
from deeplabcut.pose_estimation_pytorch.models.criterions.aggregators import WeightedLossAggregator
from deeplabcut.pose_estimation_pytorch.runners.schedulers import LRListScheduler
from deeplabcut.pose_estimation_pytorch.runners.train import build_training_runner
from deeplabcut.utils import auxiliaryfunctions

SINGLE_ANIMAL_NETS = ["resnet_50"]
MULTI_ANIMAL_NETS = ["dekr_w18"]
NETS = [(n, False) for n in SINGLE_ANIMAL_NETS] + [(n, True) for n in MULTI_ANIMAL_NETS]


def print_dict(data: dict, indent: int = 0):
    for k, v in data.items():
        if isinstance(v, dict):
            print_dict(v, indent=indent + 2)
        else:
            print(f"{indent * ' '}{k}: {v}")


@pytest.mark.parametrize("net_type, multianimal", NETS)
def test_build_bottom_up_runner(
    net_type: str,
    multianimal: bool,
) -> None:
    project_cfg: dict[str, Any] = {"multianimalproject": multianimal}
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
    template_path = root_path / "other/test_configs/pytorch_config.yaml"
    template_path.resolve()
    assert template_path.is_file(), f"Template config not found at {template_path}"
    auxiliaryfunctions.read_plainconfig(str(template_path))
    pytorch_cfg = make_pytorch_pose_config(project_cfg, str(template_path), net_type)
    print_dict(pytorch_cfg)

    pose_model = PoseModel.build(pytorch_cfg["model"])

    head_criterions = []
    for head_cfg in pytorch_cfg["model"]["heads"]:
        crit_cfg = head_cfg["criterion"]
        criterion_weight = crit_cfg.get("weight", 1)
        criterion = LOSS_AGGREGATORS.build({k: v for k, v in crit_cfg.items() if k != "weight"})
        head_criterions.append((criterion_weight, criterion))
    criterion = WeightedLossAggregator(head_criterions)

    get_optimizer = getattr(torch.optim, pytorch_cfg["optimizer"]["type"])
    optimizer = get_optimizer(params=pose_model.parameters(), **pytorch_cfg["optimizer"]["params"])

    PREDICTORS.build(dict(pytorch_cfg["model"]["predictor"]))

    if pytorch_cfg.get("scheduler"):
        if pytorch_cfg["scheduler"]["type"] == "LRListScheduler":
            _scheduler = LRListScheduler
        else:
            _scheduler = getattr(torch.optim.lr_scheduler, pytorch_cfg["scheduler"]["type"])
        _scheduler(optimizer=optimizer, **pytorch_cfg["scheduler"]["params"])
    else:
        pass

    # NOTE: @C-Achard 2026-03-18 This file was not named with test_* as a prefix,
    #  so it never ran in CI. A lot of imports are outdated and non-existent
    # FIX: replace RUNNERS registry with build_training_runner and remove unused imports
    runner_config = {
        **pytorch_cfg["solver"],
        "optimizer": pytorch_cfg["optimizer"],
        "scheduler": pytorch_cfg.get("scheduler"),
    }
    _ = build_training_runner(
        runner_config=runner_config,
        model_folder=Path("."),
        task=pose_model.task,
        model=pose_model,
        device=pytorch_cfg["device"],
        logger=None,
    )
