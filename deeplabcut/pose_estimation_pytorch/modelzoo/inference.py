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
import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from deeplabcut.modelzoo.utils import get_superanimal_colormaps
from deeplabcut.pose_estimation_pytorch.apis.analyze_videos import (
    create_df_from_prediction,
    video_inference,
)
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
    get_config_model_paths,
    raise_warning_if_called_directly,
    select_device,
    update_config,
)
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.utils.make_labeled_video import _create_labeled_video
from deeplabcut.utils.auxiliaryfunctions import read_config


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        return json.JSONEncoder.default(self, obj)


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
    customized_pose_checkpoint: Optional[str] = None,
    customized_detector_checkpoint: Optional[str] = None,
    customized_model_config: Optional[str] = None,
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

        customized_pose_checkpoint: A customized checkpoint to replace the default
            SuperAnimal pose checkpoint

        customized_detector_checkpoint: A customized checkpoint to replace the default
            SuperAnimal detector checkpoint

    Returns:
        results: Dictionary with the result pd.DataFrame for each video

    Raises:
        Warning: If the function is called directly.
    """
    raise_warning_if_called_directly()

    if device is None:
        device = select_device()

    pose_model_path = None
    detector_path = None
    if customized_model_config is None:
        (
            model_config,
            project_config,
            pose_model_path,
            detector_path,
        ) = get_config_model_paths(project_name, model_name)

        config = {**project_config, **model_config}
        config = update_config(config, max_individuals, device)
    else:
        config = read_config(customized_model_config)
        config["bodyparts"] = config["metadata"]["bodyparts"]

        if customized_pose_checkpoint is None:
            raise ValueError(
                "When specifying a `customized_model_config`, you must also specify "
                "the `customized_pose_checkpoint` that goes with it."
            )

    if customized_pose_checkpoint is not None:
        pose_model_path = customized_pose_checkpoint
    if customized_detector_checkpoint is not None:
        detector_path = customized_detector_checkpoint

    individuals = [f"animal{i}" for i in range(max_individuals)]
    config["individuals"] = individuals

    pose_runner, detector_runner = get_inference_runners(
        config,
        snapshot_path=pose_model_path,
        max_individuals=max_individuals,
        num_bodyparts=len(config["bodyparts"]),
        num_unique_bodyparts=0,
        detector_path=detector_path,
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

        dlc_scorer = f"{project_name}_{model_name}"

        output_prefix = f"{Path(video_path).stem}_{dlc_scorer}"
        output_path = Path(dest_folder)
        output_h5 = Path(output_path) / f"{output_prefix}.h5"

        # if there are no customized checkpoints passed, it's before adaptation
        if (
            customized_pose_checkpoint is None
            and customized_detector_checkpoint is None
        ):
            output_json = str(output_h5).replace(".h5", "_before_adapt.json")
        else:
            output_json = str(output_h5).replace(".h5", "_after_adapt.json")
        # also output json file so it's easier for video adaptation to handle

        predictions, video_metadata = video_inference(
            video_path,
            task=pose_task,
            pose_runner=pose_runner,
            detector_runner=detector_runner,
            return_video_metadata=True,
        )
        pred_bodyparts = np.stack([p["bodyparts"][..., :3] for p in predictions])
        pred_unique_bodyparts = None

        bbox = (0, video_metadata["resolution"][0], 0, video_metadata["resolution"][1])
        print(f"Saving results to {dest_folder}")
        config["uniquebodyparts"] = []
        config["multianimalbodyparts"] = config["bodyparts"]

        df = create_df_from_prediction(
            pred_bodyparts=pred_bodyparts,
            pred_unique_bodyparts=pred_unique_bodyparts,
            dlc_scorer=dlc_scorer,
            cfg=config,
            output_path=output_path,
            output_prefix=output_prefix,
        )

        results[video_path] = df

        with open(output_json, "w") as f:
            json.dump(predictions, f, cls=NumpyEncoder)

        if (
            customized_pose_checkpoint is not None
            and customized_detector_checkpoint is not None
        ):
            # FIXME: customized pose and customized detector passed does not mean it's adapted anymore
            output_video = output_path / f"{output_prefix}_labeled_after_adapt.mp4"
        else:
            output_video = output_path / f"{output_prefix}_labeled.mp4"

        superanimal_colormaps = get_superanimal_colormaps()
        colormap = superanimal_colormaps[project_name]

        _create_labeled_video(
            video_path,
            output_h5,
            pcutoff=pcutoff,
            fps=video_metadata["fps"],
            bbox=bbox,
            cmap=colormap,
            output_path=str(output_video),
        )

        print(f"Video with predictions was saved as {output_path}")

    return results
