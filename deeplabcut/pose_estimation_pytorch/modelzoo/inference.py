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
from pathlib import Path

import numpy as np
import torch

from deeplabcut.core.config import read_config_as_dict
from deeplabcut.modelzoo.weight_initialization import build_weight_init
from deeplabcut.modelzoo.utils import get_super_animal_scorer, get_superanimal_colormaps
from deeplabcut.pose_estimation_pytorch.apis.videos import (
    create_df_from_prediction,
    video_inference,
    VideoIterator,
)
from deeplabcut.pose_estimation_pytorch.apis.utils import (
    get_inference_runners,
    get_pose_inference_runner,
    get_filtered_coco_detector_inference_runner,
)
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
    raise_warning_if_called_directly,
    load_super_animal_config,
    update_config,
)
from deeplabcut.pose_estimation_pytorch.runners import InferenceRunner
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
    video_paths: str | list,
    superanimal_name: str,
    model_cfg: dict,
    model_snapshot_path: str | Path,
    detector_snapshot_path: str | Path | None,
    max_individuals: int,
    pcutoff: float,
    batch_size: int = 1,
    detector_batch_size: int = 1,
    cropping: list[int] | None = None,
    dest_folder: str | Path | None = None,
    output_suffix: str = "",
    plot_bboxes: bool = True,
    bboxes_pcutoff: float = 0.9,
    create_labeled_video: bool = True,
    torchvision_detector_name: str | None = None,
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
        create_labeled_video (bool):
            Specifies if a labeled video needs to be created, True by default.
        torchvision_detector_name: If using a filtered torchvision detector, the torchvision model name

    Returns:
        results: Dictionary with the result pd.DataFrame for each video

    Raises:
        Warning: If the function is called directly.
    """
    raise_warning_if_called_directly()

    if superanimal_name == "superanimal_humanbody":
        if not torchvision_detector_name:
            torchvision_detector_name = "fasterrcnn_mobilenet_v3_large_fpn"
        COCO_PERSON = 1  # COCO class ID for person
        detector_runner = get_filtered_coco_detector_inference_runner(
            model_name=torchvision_detector_name,
            category_id=COCO_PERSON,
            batch_size=detector_batch_size,
            max_individuals=max_individuals,
            model_config=model_cfg,
        )
        pose_runner = get_pose_inference_runner(
            model_cfg,
            snapshot_path=model_snapshot_path,
            batch_size=batch_size,
            max_individuals=max_individuals,
        )
    else:
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

    dest_folder = ( 
        Path(video_paths[0]).parent if dest_folder is None
        else Path(dest_folder)
    )
    dest_folder.mkdir(parents=True, exist_ok=True)
    if create_labeled_video:
        superanimal_colormaps = get_superanimal_colormaps()
        colormap = superanimal_colormaps[superanimal_name]

    for video_path in video_paths:
        print(f"Processing video {video_path}")

        dlc_scorer = get_super_animal_scorer(
            superanimal_name,
            model_snapshot_path,
            detector_snapshot_path,
            torchvision_detector_name,
        )

        output_prefix = f"{Path(video_path).stem}_{dlc_scorer}"
        output_h5 = dest_folder / f"{output_prefix}.h5"

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
            output_path=dest_folder,
            output_prefix=output_prefix,
        )

        results[video_path] = df
        with open(output_json, "w") as f:
            json.dump(predictions, f, cls=NumpyEncoder)

        if create_labeled_video:
            output_video = dest_folder / f"{output_prefix}_labeled.mp4"
            if len(output_suffix) > 0:
                output_video = output_video.with_stem(output_video.stem + output_suffix)

            create_video(
                video_path,
                output_h5,
                pcutoff=pcutoff,
                fps=video.fps,
                bbox=bbox,
                cmap=colormap,
                output_path=output_video,
                plot_bboxes=plot_bboxes,
                bboxes_list=bboxes_list,
                bboxes_pcutoff=bboxes_pcutoff,
            )
            print(f"Video with predictions was saved as {output_video}")

    return results


def create_superanimal_inference_runners(
    superanimal_name: str,
    model_name: str,
    detector_name: str | None = None,
    max_individuals: int = 10,
    batch_size: int = 1,
    detector_batch_size: int = 1,
    device: str | None = "auto",
    customized_model_config: str | Path | dict | None = None,
    customized_pose_checkpoint: str | Path | None = None,
    customized_detector_checkpoint: str | Path | None = None,
) -> tuple[InferenceRunner, InferenceRunner, dict]:
    """Create SuperAnimal inference runners for in-memory batched inference.

    This helper is intended for Model Zoo inference pipelines that run directly on
    arrays. It prepares pose/detector runners and returns them with the resolved
    model config.

    Args:
        superanimal_name: Name of the SuperAnimal dataset, e.g.
            ``"superanimal_quadruped"``.
        model_name: Pose model architecture name, e.g. ``"hrnet_w32"``.
        detector_name: Detector architecture name for top-down inference, e.g.
            ``"fasterrcnn_resnet50_fpn_v2"``.
        max_individuals: Maximum number of individuals to keep per frame.
        batch_size: Batch size for pose inference.
        detector_batch_size: Batch size for detector inference.
        device: Device for inference. If ``"auto"`` or ``None``, resolves to CUDA
            when available, else CPU.
        customized_model_config: Optional path or dict for a custom model config.
            If not provided, uses the default SuperAnimal config.
        customized_pose_checkpoint: Optional custom pose checkpoint path.
        customized_detector_checkpoint: Optional custom detector checkpoint path.

    Returns:
        tuple: ``(pose_runner, detector_runner, model_cfg)`` where:
            - ``pose_runner`` is the pose inference runner
            - ``detector_runner`` is the detector inference runner
            - ``model_cfg`` is the resolved model configuration dict

    Example:
        >>> from pathlib import Path
        >>> import numpy as np
        >>> from PIL import Image
        >>> from deeplabcut.pose_estimation_pytorch.modelzoo.inference import (
        ...     create_superanimal_inference_runners,
        ... )
        >>>
        >>> img_paths = [
        ...     "/path/to/images/frame_0000.png",
        ...     "/path/to/images/frame_0001.png",
        ...     "/path/to/images/frame_0002.png",
        ... ]
        >>> images = [np.asarray(Image.open(Path(p)).convert("RGB")) for p in img_paths]
        >>>
        >>> pose_runner, det_runner, model_cfg = create_superanimal_inference_runners(
        ...     superanimal_name="superanimal_quadruped",
        ...     model_name="hrnet_w32",
        ...     detector_name="fasterrcnn_resnet50_fpn_v2",
        ...     max_individuals=10,
        ...     batch_size=1,
        ...     detector_batch_size=1,
        ... )
        >>>
        >>> det_preds = det_runner.inference(images)
        >>> pose_inputs = list(zip(images, det_preds))
        >>> pose_preds = pose_runner.inference(pose_inputs)
        >>> print(len(det_preds), len(pose_preds))
    """
    if model_name.startswith("fmpose3d"):
        raise NotImplementedError(
            "FMPose3D is not supported in this helper. Use the FMPose3D inference API."
        )

    if superanimal_name == "superanimal_humanbody":
        raise NotImplementedError(
            "superanimal_humanbody is currently not supported by this helper because "
            "it relies on modelzoo.build_weight_init, which does not support this dataset."
        )

    if detector_name is None:
        raise ValueError(
            "Please provide `detector_name` for SuperAnimal top-down inference setup."
        )

    if device in (None, "auto"):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if customized_model_config is not None:
        if isinstance(customized_model_config, (str, Path)):
            model_cfg = read_config_as_dict(customized_model_config)
        else:
            model_cfg = customized_model_config.copy()
    else:
        model_cfg = load_super_animal_config(
            super_animal=superanimal_name,
            model_name=model_name,
            detector_name=detector_name,
        )

    model_cfg = update_config(model_cfg, max_individuals=max_individuals, device=device)
    weight_init = build_weight_init(
        cfg=model_cfg,
        super_animal=superanimal_name,
        model_name=model_name,
        detector_name=detector_name,
        with_decoder=False,
        memory_replay=False,
        customized_pose_checkpoint=customized_pose_checkpoint,
        customized_detector_checkpoint=customized_detector_checkpoint,
    )

    pose_runner, detector_runner = get_inference_runners(
        model_config=model_cfg,
        snapshot_path=weight_init.snapshot_path,
        max_individuals=max_individuals,
        num_bodyparts=len(model_cfg["metadata"]["bodyparts"]),
        num_unique_bodyparts=len(model_cfg["metadata"]["unique_bodyparts"]),
        batch_size=batch_size,
        detector_batch_size=detector_batch_size,
        detector_path=weight_init.detector_snapshot_path,
    )
    return pose_runner, detector_runner, model_cfg
