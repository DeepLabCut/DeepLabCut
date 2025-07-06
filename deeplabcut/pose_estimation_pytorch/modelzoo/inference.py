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

from deeplabcut.modelzoo.utils import get_super_animal_scorer, get_superanimal_colormaps
from deeplabcut.pose_estimation_pytorch.apis.videos import (
    create_df_from_prediction,
    video_inference,
    VideoIterator,
)
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
    raise_warning_if_called_directly,
)
from deeplabcut.utils.make_labeled_video import create_video


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
    superanimal_name: str,
    model_cfg: dict,
    model_snapshot_path: str | Path,
    detector_snapshot_path: str | Path | None,
    max_individuals: int,
    pcutoff: float,
    batch_size: int = 1,
    detector_batch_size: int = 1,
    cropping: list[int] | None = None,
    dest_folder: Optional[str] = None,
    output_suffix: str = "",
    plot_bboxes: bool = True,
    bboxes_pcutoff: float = 0.9,
) -> dict:
    """
    Perform inference on a video using a superanimal model from the model zoo specified by `superanimal_name`.
    During inference, the video is analyzed using the specified model and the results are saved in the specified
    destination folder. The predictions are saved in the form of a .h5 file. The video with the predictions is saved
    in the form of a .mp4 file.

    WARNING: This function is an internal utility function and should not be
    called directly. It is designed to be used by deeplabcut.modelzoo.api.video_inference.py

    Args:
        video_paths: Path to the video to be analyzed or list of paths to videos to be
            analyzed
        superanimal_name: Name of the SuperAnimal project (e.g. superanimal_quadruped)
        model_cfg: The name of the pose model architecture to use for inference.
        model_snapshot_path: The path to the pose model snapshot to use for inference.
        detector_snapshot_path: The path to the detector snapshot to use for inference.
        max_individuals: Maximum number of individuals in the video
        pcutoff: Cutoff for cutting off the predicted keypoints with probability lower
            than pcutoff
        batch_size: The batch size to use for video inference.
        cropping: List of cropping coordinates as [x1, x2, y1, y2]. Note that the same
            cropping parameters will then be used for all videos. If different video
            crops are desired, run ``video_inference_superanimal`` on individual videos
            with the corresponding cropping coordinates.
        detector_batch_size: The batch size to use for the detector for video inference.
        dest_folder: Destination folder for the results. If not specified, the
            results are saved in the same folder as the video. Defaults to None.
        output_suffix: The suffix to add to output file names (e.g. _before_adapt)
        plot_bboxes: Whether to plot bounding boxes in the output video
        bboxes_pcutoff: Confidence threshold for bounding box plotting

    Returns:
        results: Dictionary with the result pd.DataFrame for each video

    Raises:
        Warning: If the function is called directly.
    """
    raise_warning_if_called_directly()
    pose_runner, detector_runner = get_inference_runners(
        model_config=model_cfg,
        snapshot_path=model_snapshot_path,
        max_individuals=max_individuals,
        num_bodyparts=len(model_cfg["metadata"]["bodyparts"]),
        num_unique_bodyparts=0,
        batch_size=batch_size,
        detector_batch_size=detector_batch_size,
        detector_path=detector_snapshot_path,
    )
    results = {}

    if isinstance(video_paths, str):
        video_paths = [video_paths]

    if dest_folder is None:
        dest_folder = Path(video_paths[0]).parent

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for video_path in video_paths:
        print(f"Processing video {video_path}")

        dlc_scorer = get_super_animal_scorer(
            superanimal_name, model_snapshot_path, detector_snapshot_path
        )

        output_prefix = f"{Path(video_path).stem}_{dlc_scorer}"
        output_path = Path(dest_folder)
        output_h5 = Path(output_path) / f"{output_prefix}.h5"

        output_json = output_h5.with_suffix(".json")
        if len(output_suffix) > 0:
            output_json = output_json.with_stem(output_h5.stem + output_suffix)

        video = VideoIterator(video_path, cropping=cropping)
        predictions = video_inference(
            video,
            pose_runner=pose_runner,
            detector_runner=detector_runner,
        )

        bbox_keys_in_predictions = {"bboxes", "bbox_scores"}
        bboxes_list = [
            {key: value for key, value in p.items() if key in bbox_keys_in_predictions}
            for i, p in enumerate(predictions)
        ]

        bbox = cropping
        if cropping is None:
            vid_w, vid_h = video.dimensions
            bbox = (0, vid_w, 0, vid_h)

        print(f"Saving results to {dest_folder}")
        df = create_df_from_prediction(
            predictions=predictions,
            dlc_scorer=dlc_scorer,
            multi_animal=True,
            model_cfg=model_cfg,
            output_path=output_path,
            output_prefix=output_prefix,
        )

        results[video_path] = df
        with open(output_json, "w") as f:
            json.dump(predictions, f, cls=NumpyEncoder)

        output_video = output_path / f"{output_prefix}_labeled.mp4"
        if len(output_suffix) > 0:
            output_video = output_video.with_stem(output_video.stem + output_suffix)

        superanimal_colormaps = get_superanimal_colormaps()
        colormap = superanimal_colormaps[superanimal_name]

        create_video(
            video_path,
            output_h5,
            pcutoff=pcutoff,
            fps=video.fps,
            bbox=bbox,
            cmap=colormap,
            output_path=str(output_video),
            plot_bboxes=plot_bboxes,
            bboxes_list=bboxes_list,
            bboxes_pcutoff=bboxes_pcutoff,
        )
        print(f"Video with predictions was saved as {output_path}")

    return results
