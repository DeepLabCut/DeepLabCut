#
# DeepLabCut Toolbox (deeplabcut.org)
# (c) A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Frozen TensorFlow-specific SuperAnimal video inference.

NOT actively maintained. Exists only as a reference for legacy TF support.
Delete this module when ``video_inference_superanimal`` drops TF support.
"""

from __future__ import annotations

from dlclibrary.dlcmodelzoo.modelzoo_download import download_huggingface_model

from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
    get_snapshot_folder_path,
)


def video_inference_superanimal_tf(
    videos,
    superanimal_name,
    model_name,
    scale_list,
    video_extensions,
    video_adapt,
    plot_trajectories,
    pcutoff,
    adapt_iterations,
    pseudo_threshold,
    *,
    create_labeled_video,
    **kwargs,
):
    """TensorFlow path for ``video_inference_superanimal`` (model ``"dlcrnet"``).

    Downloads the HuggingFace snapshot if missing, then delegates to the
    TensorFlow inference implementation.
    """
    from deeplabcut.pose_estimation_tensorflow.modelzoo.api.superanimal_inference import (
        _video_inference_superanimal,
    )

    weight_folder = get_snapshot_folder_path() / f"{superanimal_name}_{model_name}"
    if not weight_folder.exists():
        download_huggingface_model(
            superanimal_name,
            target_dir=str(weight_folder),
            rename_mapping=None,
        )

    videos_list = [videos] if isinstance(videos, str) else videos
    _video_inference_superanimal(
        videos_list,
        superanimal_name,
        model_name,
        scale_list,
        video_extensions,
        video_adapt,
        plot_trajectories,
        pcutoff,
        adapt_iterations,
        pseudo_threshold,
        create_labeled_video=create_labeled_video,
    )
