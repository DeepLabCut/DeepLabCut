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
import os
import pickle
import time
from pathlib import Path
from typing import Optional, Union

import torch

from deeplabcut.pose_estimation_pytorch.apis.analyze_videos import (
    create_df_from_prediction,
    video_inference,
)
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
    _get_config_model_paths,
    _update_config,
    raise_warning_if_called_directly,
    select_device,
)
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.utils.auxiliaryfunctions import get_deeplabcut_path, read_config
from deeplabcut.utils.make_labeled_video import _create_labeled_video


def construct_bodypart_names(max_individuals, bodyparts):
    multianimalbodyparts = []
    for i in range(max_individuals):
        for bodypart in bodyparts:
            multianimalbodyparts.append(f"{bodypart}_{i}")
    return multianimalbodyparts


def _video_inference_superanimal(
    video_paths: Union[str, list],
    project_name: str,
    model_name: str,
    max_individuals: int,
    pcutoff: float,
    device: Optional[str] = None,
    dest_folder: Optional[str] = None,
) -> dict:
    """

    Perform inference on a video using a superanimal model from the model zoo specified by `superanimal_name`.
    During inference, the video is analyzed using the specified model and the results are saved in the specified
    destination folder. The predictions are saved in the form of a .h5 file. The video with the predictions is saved
    in the form of a .mp4 file.

    WARNING: This function is an internal utility function and should not be
    called directly. It is designed to be used by deeplabcut.modelzoo.api.video_inference.py

    Args:
        video_paths: Path to the video to be analyzed or list of paths to videos to be analyzed

        project_name: Name of the superanimal project (e.g. superanimal_quadruped)

        model_name: Name of the model (e.g. hrnetw32)

        max_individuals: Maximum number of individuals in the video

        pcutoff: Cutoff for cutting off the predicted keypoints with probability lower than pcutoff

        device: The device on which to perform the operation.
            If not specified, the device is automatically determined by the
            `select_device` function. Defaults to None, which triggers
            automatic device selection.
        dest_folder: Destination folder for the results. If not specified, the
            results are saved in the same folder as the video. Defaults to None.

    Returns:
        results: Dictionary with the result pd.DataFrame for each video

    Raises:
        Warning: If the function is called directly.
    """

    raise_warning_if_called_directly()
    (
        model_config,
        project_config,
        pose_model_path,
        detector_model_path,
    ) = _get_config_model_paths(project_name, model_name)
    if device is None:
        device = select_device()

    config = {**project_config, **model_config}
    config = _update_config(config, max_individuals, device)

    pose_runner, detector_runner = get_inference_runners(
        config,
        snapshot_path=pose_model_path,
        max_individuals=max_individuals,
        num_bodyparts=len(config["bodyparts"]),
        num_unique_bodyparts=0,
        detector_path=detector_model_path
    )
    pose_task = Task(config.get("method", "BU"))
    results = {}

    if isinstance(video_paths, str):
        video_paths = [video_paths]

    if dest_folder is None:
        dest_folder = Path(video_paths[0]).parent

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for video_path in video_paths:
        print(f"Processing video {video_path}")

        prediction, video_metadata = video_inference(
            video_path,
            task=pose_task,
            pose_runner=pose_runner,
            detector_runner=detector_runner,
            return_video_metadata=True,
        )
        bbox = (0, video_metadata["resolution"][0], 0, video_metadata["resolution"][1])
        print(f"Saving results to {dest_folder}")
        config["uniquebodyparts"] = []
        config["multianimalbodyparts"] = config["bodyparts"]

        dlc_scorer = f"{project_name}_{model_name}"
        output_prefix = f"{Path(video_path).stem}_{dlc_scorer}"
        output_path = Path(dest_folder)
        df = create_df_from_prediction(
            prediction,
            dlc_scorer,
            config,
            output_path,
            output_prefix,
        )

        results[video_path] = df

        output_h5 = Path(output_path) / f"{output_prefix}.h5"
        output_video = output_path / f"{output_prefix}_labeled.mp4"
        _create_labeled_video(
            video_path,
            output_h5,
            pcutoff=pcutoff,
            fps=video_metadata["fps"],
            bbox=bbox,
            output_path=str(output_video),
        )

        print(f"Video with predictions was saved as {output_path}")

    return results
