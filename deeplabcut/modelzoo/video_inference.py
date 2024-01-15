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
import warnings
from typing import Optional, Union

from dlclibrary.dlcmodelzoo.modelzoo_download import download_huggingface_model

from deeplabcut.modelzoo.utils import parse_project_model_name
from deeplabcut.utils.auxiliaryfunctions import get_deeplabcut_path


def video_inference_superanimal(
    videos: Union[str, list],
    superanimal_name: str,
    scale_list: list = [],
    videotype: str = "mp4",
    dest_folder: Optional[str] = None,
    video_adapt: bool = False,
    plot_trajectories: bool = False,
    pcutoff: float = 0.1,
    adapt_iterations: int = 1000,
    pseudo_threshold: float = 0.1,
    max_individuals: int = 10,
    device: Optional[str] = None,
):
    """
    This function performs inference on videos using a superanimal model. The model is downloaded from hugginface to the `modelzoo/checkpoints` folder.

    Args:

        videos (str or list): The path to the video or a list of paths to videos.
        superanimal_name (str): The name of the superanimal model.
                                The name should be in the format: {project_name}_{modelname}.
                                For example: `superanimal_topviewmouse_dlcrnet` or `superanimal_quadruped_hrnet`.
        scale_list (list): A list of different resolutions for the spatial pyramid. Used only for bottom up models.

        videotype (str): Checks for the extension of the video in case the input to the video is a directory.
                         Only videos with this extension are analyzed. The default is ``.mp4``.
        dest_folder (str): The path to the folder where the results should be saved.

        video_adapt (bool): Whether to perform video adaptation. The default is False.
                            Current we do not support video adaptation in pytorch models.
        plot_trajectories (bool): Whether to plot the trajectories. The default is False.

        pcutoff (float): The p-value cutoff for the confidence of the prediction. The default is 0.1.

        adapt_iterations (int): Number of iterations for adaptation training. Empirically 1000 is sufficient.

        pseudo_threshold (float): The pseudo-label threshold for the confidence of the prediction. The default is 0.1.

        max_individuals (int): The maximum number of individuals in the video. The default is 30. Used only for top down models.

        device (str): The device to use for inference. The default is None (CPU). Used only for pytorch models.

    Raises:
        NotImplementedError: If the model is not found in the modelzoo.
        Warning: If the superanimal_name will be deprecated in the future.

    """
    project_name, model_name = parse_project_model_name(superanimal_name)

    dlc_root_path = get_deeplabcut_path()
    modelzoo_path = os.path.join(dlc_root_path, "modelzoo")
    available_architectures = json.load(
        open(os.path.join(modelzoo_path, "models_to_framework.json"), "r")
    )
    framework = available_architectures[model_name]
    print(f"Using {framework} for model {model_name}")

    weight_folder = os.path.join(modelzoo_path, "checkpoints")

    redownload = False
    if framework == "pytorch":
        pose_model_name = f"{project_name}_{model_name}.pth"
        detector_name = f"{project_name}_fasterrcnn.pt"
        rename_mapping = {
            "pose_model.pth": pose_model_name,
            "detector.pt": detector_name,
        }
        pose_model_path = os.path.join(weight_folder, pose_model_name)
        detector_model_path = os.path.join(weight_folder, detector_name)
        if not (
            os.path.exists(pose_model_path) and os.path.exists(detector_model_path)
        ):
            redownload = True
    elif framework == "tensorflow":
        weight_folder = os.path.join(weight_folder, f"{project_name}_{model_name}")
        redownload = not os.path.isdir(weight_folder)
        rename_mapping = {}

    if redownload:
        download_huggingface_model(
            superanimal_name, target_dir=weight_folder, rename_mapping=rename_mapping
        )

    if framework == "tensorflow":
        from deeplabcut.pose_estimation_tensorflow.modelzoo.api.superanimal_inference import (
            _video_inference_superanimal,
        )

        if isinstance(videos, str):
            videos = [videos]
        _video_inference_superanimal(
            videos,
            project_name,
            model_name,
            scale_list,
            videotype,
            video_adapt,
            plot_trajectories,
            pcutoff,
            adapt_iterations,
            pseudo_threshold,
        )
    elif framework == "pytorch":
        from deeplabcut.pose_estimation_pytorch.modelzoo.inference import (
            _video_inference_superanimal,
        )

        if video_adapt:
            warnings.warn(f"Video adaptation is not yet implemented for HRNet models.")

        _video_inference_superanimal(
            videos,
            project_name,
            model_name,
            max_individuals,
            pcutoff,
            device,
            dest_folder,
        )
