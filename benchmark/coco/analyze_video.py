"""Run video analysis"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from deeplabcut.pose_estimation_pytorch.apis.analyze_videos import (
    create_df_from_prediction,
    video_inference,
    VideoIterator,
)
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.utils.make_labeled_video import _create_labeled_video


def main(
    video_path: str | Path,
    model_config: str,
    snapshot_path: str,
    detector_path: str | None,
    num_animals: int = 1,
):
    video_path = Path(video_path)
    model_cfg = read_config_as_dict(model_config)
    pose_task = Task(model_cfg["method"])
    pose_runner, detector_runner = get_inference_runners(
        model_config=model_cfg,
        snapshot_path=snapshot_path,
        max_individuals=num_animals,
        num_bodyparts=len(model_cfg["metadata"]["bodyparts"]),
        num_unique_bodyparts=len(model_cfg["metadata"]["unique_bodyparts"]),
        with_identity=model_cfg["metadata"].get("with_identity", False),
        transform=None,
        detector_path=detector_path,
        detector_transform=None,
    )
    predictions, video_metadata = video_inference(
        video_path,
        task=pose_task,
        pose_runner=pose_runner,
        detector_runner=detector_runner,
        with_identity=False,
        return_video_metadata=True,
    )

    pred_bodyparts = np.stack([p["bodyparts"][..., :3] for p in predictions])
    pred_unique_bodyparts = None
    bbox = (0, video_metadata["resolution"][0], 0, video_metadata["resolution"][1])

    cfg = copy.deepcopy(model_cfg)
    cfg["individuals"] = [f"individual_{i}" for i in range(num_animals)]
    cfg["bodyparts"] = cfg["metadata"]["bodyparts"]
    cfg["uniquebodyparts"] = []
    cfg["multianimalbodyparts"] = cfg["metadata"]["bodyparts"]

    dlc_scorer = ""
    if detector_path is not None:
        dlc_scorer += Path(detector_path).stem
    dlc_scorer += Path(snapshot_path).stem

    output_prefix = f"{video_path.stem}_{dlc_scorer}"
    output_path = video_path.parent
    output_h5 = output_path / (output_prefix + ".h5")
    _ = create_df_from_prediction(
        pred_bodyparts=pred_bodyparts,
        pred_unique_bodyparts=pred_unique_bodyparts,
        dlc_scorer=dlc_scorer,
        cfg=cfg,
        output_path=output_path,
        output_prefix=output_prefix,
    )
    _create_labeled_video(
        str(video_path),
        str(output_h5),
        pcutoff=0.6,
        fps=video_metadata["fps"],
        bbox=bbox,
        output_path=str(output_path / f"{output_prefix}_labeled.mp4"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("model_config_path")
    parser.add_argument("snapshot_path")
    parser.add_argument("--detector_path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_animals", type=int, default=1)
    args = parser.parse_args()
    main(
        video_path=args.video_path,
        model_config=args.model_config_path,
        snapshot_path=args.snapshot_path,
        detector_path=args.detector_path,
        num_animals=args.num_animals,
    )
