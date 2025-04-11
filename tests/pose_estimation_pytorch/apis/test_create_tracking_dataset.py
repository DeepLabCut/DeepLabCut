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
"""Tests method to create the tracking dataset in PyTorch"""
from pathlib import Path

import torch

import deeplabcut.pose_estimation_pytorch as dlc_torch
import deeplabcut.pose_estimation_pytorch.apis.tracking_dataset as tracking_dataset
import deeplabcut.pose_estimation_pytorch.models as models


class MockLoader(dlc_torch.Loader):
    """Mock loader for data"""

    def __init__(self, tmp_folder: Path, bodyparts: list[str] | None = None):
        if bodyparts is None:
            bodyparts = ["nose", "left_eye", "right_eye", "tail_base"]
        self.bodyparts = bodyparts

        model_config_path = tmp_folder / "pytorch_config.yaml"
        dlc_torch.config.make_pytorch_pose_config(
            project_config=dlc_torch.config.make_basic_project_config(
                dataset_path=str(tmp_folder),
                bodyparts=self.bodyparts,
                max_individuals=3,
            ),
            pose_config_path=tmp_folder / "pytorch_config.yaml",
            net_type="resnet_50",
            save=True,
        )
        super().__init__(
            str(tmp_folder),
            str(tmp_folder / "labeled-data"),
            model_config_path,
        )

    def load_data(self, mode: str = "train") -> dict[str, list[dict]]:
        return {
            "annotations": [],
            "categories": [],
            "images": [],
        }

    def get_dataset_parameters(self) -> dlc_torch.PoseDatasetParameters:
        return dlc_torch.PoseDatasetParameters(
            bodyparts=self.bodyparts,
            unique_bpts=[],
            individuals=self.model_cfg["metadata"]["individuals"],
        )


def test_build_feature_extraction_runner(tmp_path_factory):
    tmp_folder = Path(tmp_path_factory.mktemp("tmp-project"))

    loader = MockLoader(tmp_folder=tmp_folder)
    model = models.PoseModel.build(loader.model_cfg["model"])
    snapshot_path = loader.model_folder / "snapshot.pt"
    torch.save(dict(model=model.state_dict()), snapshot_path)
    _ = tracking_dataset.build_feature_extraction_runner(
        loader=loader,
        snapshot_path=snapshot_path,
        device="cpu",
        batch_size=1,
    )



