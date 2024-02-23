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
""" Tests for the bottom-up pytorch runner """
from pathlib import Path
from typing import Dict, Any

import pytest
import torch
from deeplabcut.pose_estimation_pytorch.config import make_pytorch_pose_config

from deeplabcut.pose_estimation_pytorch.models import PoseModel, LOSSES, PREDICTORS
from deeplabcut.pose_estimation_pytorch.models.criterion import WeightedAggregateLoss
from deeplabcut.pose_estimation_pytorch.runners import RUNNERS
from deeplabcut.pose_estimation_pytorch.runners.schedulers import LRListScheduler
from deeplabcut.utils import auxiliaryfunctions


SINGLE_ANIMAL_NETS = ["resnet_50"]
MULTI_ANIMAL_NETS = ["dekr_w18"]
NETS = [(n, False) for n in SINGLE_ANIMAL_NETS] + [(n, True) for n in MULTI_ANIMAL_NETS]


def print_dict(data: Dict, indent: int = 0):
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
    project_cfg: Dict[str, Any] = {"multianimalproject": multianimal}
    if multianimal:
        project_cfg["bodyparts"] = "MULTI!"
        project_cfg["multianimalbodyparts"] = ["head", "shoulder", "knee", "toe"]
        project_cfg["uniquebodyparts"] = []
        project_cfg["individuals"] = ["tom", "jerry"]
    else:
        project_cfg["bodyparts"] = ["head", "shoulder", "knee", "toe"]
        project_cfg["uniquebodyparts"] = []
        project_cfg["individuals"] = ["tom"]

    root_path = Path(auxiliaryfunctions.get_deeplabcut_path())
    template_path = root_path / "pose_estimation_pytorch" / "apis" / "pytorch_config.yaml"
    template = auxiliaryfunctions.read_plainconfig(str(template_path))
    pytorch_cfg = make_pytorch_pose_config(project_cfg, str(template_path), net_type)
    print_dict(pytorch_cfg)

    pose_model = PoseModel.build(pytorch_cfg["model"])

    head_criterions = []
    for head_cfg in pytorch_cfg["model"]["heads"]:
        crit_cfg = head_cfg["criterion"]
        criterion_weight = crit_cfg.get("weight", 1)
        criterion = LOSSES.build({k: v for k, v in crit_cfg.items() if k != "weight"})
        head_criterions.append((criterion_weight, criterion))
    criterion = WeightedAggregateLoss(head_criterions)

    get_optimizer = getattr(torch.optim, pytorch_cfg["optimizer"]["type"])
    optimizer = get_optimizer(
        params=pose_model.parameters(), **pytorch_cfg["optimizer"]["params"]
    )

    predictor = PREDICTORS.build(dict(pytorch_cfg["model"]["predictor"]))

    if pytorch_cfg.get("scheduler"):
        if pytorch_cfg["scheduler"]["type"] == "LRListScheduler":
            _scheduler = LRListScheduler
        else:
            _scheduler = getattr(
                torch.optim.lr_scheduler, pytorch_cfg["scheduler"]["type"]
            )
        scheduler = _scheduler(
            optimizer=optimizer, **pytorch_cfg["scheduler"]["params"]
        )
    else:
        scheduler = None

    logger = None
    runner = RUNNERS.build(
        dict(
            **pytorch_cfg["solver"],
            model=pose_model,
            criterion=criterion,
            optimizer=optimizer,
            predictor=predictor,
            cfg=pytorch_cfg,
            device=pytorch_cfg["device"],
            scheduler=scheduler,
            logger=logger,
        )
    )
