#
# DeepLabCut Toolbox (deeplabcut.org)
# (c) A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Public API for DeepLabCut modelzoo inference functions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from deeplabcut.api._tf_routing import with_tensorflow_fallback
from deeplabcut.core.deprecation import renamed_parameter
from deeplabcut.modelzoo.video_inference import (
    video_inference_superanimal as _impl,
)


@with_tensorflow_fallback(
    when=lambda *a, **kw: kw.get("model_name") == "dlcrnet",
    tensorflow_module="deeplabcut.tensorflow_compat.superanimal_inference",
    tensorflow_name="video_inference_superanimal_tf",
)
@renamed_parameter(old="videotype", new="video_extensions", since="3.0.0")
def video_inference_superanimal(
    videos: str | list,
    superanimal_name: str,
    model_name: str,
    detector_name: str | None = None,
    scale_list: list | None = None,
    video_extensions: str | Sequence[str] | None = None,
    dest_folder: str | Path | None = None,
    cropping: list[int] | None = None,
    video_adapt: bool = False,
    plot_trajectories: bool = False,
    batch_size: int = 1,
    detector_batch_size: int = 1,
    pcutoff: float = 0.1,
    adapt_iterations: int = 1000,
    pseudo_threshold: float = 0.1,
    bbox_threshold: float = 0.9,
    detector_epochs: int = 4,
    pose_epochs: int = 4,
    max_individuals: int = 10,
    video_adapt_batch_size: int = 8,
    device: str | None = "auto",
    customized_pose_checkpoint: str | None = None,
    customized_detector_checkpoint: str | None = None,
    customized_model_config: str | None = None,
    plot_bboxes: bool = True,
    create_labeled_video: bool = True,
    fmpose_return_3d: bool = False,
):
    """This function performs inference on videos using a pretrained SuperAnimal model.

    IMPORTANT: Note that since we have both TensorFlow and PyTorch Engines, we will
    route the engine based on the model you select:

        * dlcrnet -> TensorFlow
        * all others -> PyTorch

    Args:
        videos (str or list): The path to the video or a list of paths to videos.
        superanimal_name (str): The name of the SuperAnimal dataset for which to load a
            pre-trained model.
        model_name (str): The model architecture to use for inference.
        detector_name (str): For top-down models (only available with the PyTorch
            framework), the type of object detector to use for inference.
        scale_list (list): A list of different resolutions for the spatial pyramid. Used
            only for bottom up models.
        video_extensions (str | Sequence[str] | None, optional):
            Checks for the extension of the video in case the input to the
            video is a directory. Only videos with this extension are analyzed. The
            default is ``.mp4``.
        dest_folder (str | Path | None, optional): The path to the folder where the
            results should be saved.
        cropping (list or None, optional): Only for SuperAnimal models running with the
            PyTorch engine. List of cropping coordinates as [x1, x2, y1, y2]. Note that
            the same cropping parameters will then be used for all videos. If different
            video crops are desired, run ``video_inference_superanimal`` on individual
            videos with the corresponding cropping coordinates. Defaults to None.
        video_adapt (bool): Whether to perform video adaptation. The default is False.
            You only need to perform it on one video because the adaptation generalizes
            to all videos that are similar.
        plot_trajectories (bool): Whether to plot the trajectories. The default is
            False.
        batch_size (int): The batch size to use for video inference. Only for PyTorch
            models.
        detector_batch_size (int): The batch size to use for the detector during video
            inference. Only for PyTorch.
        pcutoff (float): The p-value cutoff for the confidence of the prediction. The
            default is 0.1.
        adapt_iterations (int): Number of iterations for adaptation training.
            Empirically 1000 is sufficient.
        bbox_threshold (float): The pseudo-label threshold for the confidence of the
            detector. The default is 0.9.
        detector_epochs (int): Used in the PyTorch engine. The number of epochs for
            training the detector. The default is 4.
        pose_epochs (int): Used in the PyTorch engine. The number of epochs for training
            the pose estimator. The default is 4.
        pseudo_threshold (float): The pseudo-label threshold for the confidence of the
            prediction. The default is 0.1.
        max_individuals (int): The maximum number of individuals in the video. The
            default is 30. Used only for top down models.
        video_adapt_batch_size (int): The batch size to use for video adaptation.
        device (str): The device to use for inference. The default is None (CPU). Used
            only for PyTorch models.
        customized_pose_checkpoint (str): Used in the PyTorch engine. If specified, it
            replaces the default pose checkpoint.
        customized_detector_checkpoint (str): Used in the PyTorch engine. If specified,
            it replaces the default detector checkpoint.
        customized_model_config (str): Used for loading customized model config. Only
            supported in Pytorch.
        plot_bboxes (bool): If using Top-Down approach, whether to plot the detector's
            bounding boxes. The default is True.
        create_labeled_video (bool): Specifies if a labeled video needs to be created,
            True by default.
        fmpose_return_3d (bool): Only used when ``model_name`` starts with
            ``"fmpose3d"``. If True, include in-memory 3D poses in the return payload
            (per video: ``{"df_2d": ..., "df_3d": ...}``). If False (default), keep the
            legacy return payload with only the 2D DataFrame per video.

    Raises:
        NotImplementedError: If the model is not found in the modelzoo.

    Warns:
        UserWarning: If an fmpose3d ``model_name`` is used with a mismatched
        ``superanimal_name``.

    Note:
        (Model Explanation) SuperAnimal-Quadruped:
        ``superanimal_quadruped`` models aim to work across a large range of quadruped
        animals, from horses, dogs, sheep, rodents, to elephants. The camera perspective
        is orthogonal to the animal ("side view"), and most of the data includes the
        animals face (thus the front and side of the animal). You will note we have
        several variants that differ in speed vs. performance, so please do test them
        out on your data to see which is best suited for your application. Also note we
        have a "video adaptation" feature, which lets you adapt your data to the model
        in a self-supervised way. No labeling needed!

        All model snapshots are automatically downloaded to modelzoo/checkpoints when
        used.

        - PLEASE SEE THE FULL DATASHEET: https://zenodo.org/records/10619173
        - MORE DETAILS ON THE MODELS (detector, pose estimators):
            https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped
        - We provide several models:
            - ``hrnet_w32`` (Top-Down pose estimation model, PyTorch engine)
                An ``hrnet_w32`` is a top-down model that is paired with a detector.
                That means it takes a cropped image from an object detector and predicts
                the keypoints. When selecting this variant, a ``detector_name`` must be
                set with one of the provided object detectors.
            - ``dlcrnet`` (TensorFlow engine)
                This is a bottom-up model that predicts all keypoints then groups them
                into individuals. This can be faster, but more error prone.
        - We provide one object detector (only for the PyTorch engine):
            - ``fasterrcnn_resnet50_fpn_v2``
                This is a FasterRCNN model with a ResNet backbone, see
                https://pytorch.org/vision/stable/models/faster_rcnn.html

        (Model Explanation) SuperAnimal-TopViewMouse:
        ``superanimal_topviewmouse`` aims to work across lab mice in different lab settings
        from a top-view perspective; this is very popular in many behavioral assays in
        freely moving mice.

        All model snapshots are automatically downloaded to modelzoo/checkpoints when
        used.

        - `PLEASE SEE THE FULL DATASHEET HERE <https://zenodo.org/records/10618947>`_
        - `MORE DETAILS ON THE MODELS (detector, pose estimators) <https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse>`_
        - We provide several models:
            - ``hrnet_w32`` (Top-Down pose estimation model, PyTorch engine)
                An ``hrnet_w32`` is a top-down model that is paired with a detector.
                That means it takes a cropped image from an object detector and predicts
                the keypoints. When selecting this variant, a ``detector_name`` must be
                set with one of the provided object detectors.
            - ``dlcrnet`` (TensorFlow engine)
                This is a bottom-up model that predicts all keypoints then groups them
                into individuals. This can be faster, but more error prone.
        - We provide one object detector (only for the PyTorch engine):
            - ``fasterrcnn_resnet50_fpn_v2``
                This is a FasterRCNN model with a ResNet backbone, see
                https://pytorch.org/vision/stable/models/faster_rcnn.html

        (Model Explanation) SuperAnimal-Bird:
        ``superanimal_superbird`` model aims to work on various bird species. It was
        developed during the 2024 DLC AI Residency Program. More info can be
        `found here <https://deeplabcut.medium.com/deeplabcut-ai-residency-2024-recap-working-with-the-superanimal-bird-model-and-dlc-3-0-live-e55807ca2c7c>`_

        (Model Explanation) SuperAnimal-HumanBody:
        ``superanimal_humanbody`` models aim to work across human body pose estimation
        from various camera perspectives and environments. The models are designed to
        handle different human poses, activities, and lighting conditions commonly
        found in human motion analysis, sports analysis, and behavioral studies.

        All model snapshots are automatically downloaded to modelzoo/checkpoints when
        used.

        - We provide:
            - ``rtmpose_x`` (Top-Down pose estimation model, PyTorch engine)
                An ``rtmpose_x`` is a top-down model that is paired with a detector.
                That means it takes a cropped image from an object detector and predicts
                the keypoints. When selecting this variant, a ``detector_name`` must be
                set with one of the provided object detectors. This model uses 17 body
                parts in the COCO body7 format.
        - The following object detectors can be used:
            - ``fasterrcnn_mobilenet_v3_large_fpn`` (default)
                This is a FasterRCNN model with a MobileNet backbone
            - ``fasterrcnn_resnet50_fpn``
            - ``fasterrcnn_resnet50_fpn_v2``
            For more info, see https://pytorch.org/vision/stable/models/faster_rcnn.html

        Tips:
        * max_individuals: make sure you correctly give the number of individuals. Our
            inference api will only give up to max_individuals number of predictions.
        * pseudo_threshold: the higher you set, the more aggressively you filter low
            confidence predictions during video adaptation.
        * bbox_threshold: the higher you set, the more aggressively you filter low
            confidence bounding boxes during video adaptation. Different from our paper,
            we now add video adaptation to the object detector as well.
        * detector_epochs and pose_epochs do not need to be too high as video adaptation
            does not require too much training. However, you can make them higher if you
            see a substantial gain in the training logs.
        * scale_list: it's recommended to leave this as empty list []. Empirically
            [200, 300, 400] works well. We needed to do this as bottom-up models in
            TensorFlow are sensitive to the scales of the image. If you find your
            predictions not good without scale_list or it's too hard to find the right
            scale_list, you can try to use the PyTorch engine.

    Examples:
        Using the module import path:

            import deeplabcut.modelzoo.video_inference.video_inference_superanimal as video_inference_superanimal
            video_inference_superanimal(
                videos=["/mnt/md0/shaokai/DLCdev/3mice_video1_short.mp4"],
                superanimal_name="superanimal_topviewmouse",
                model_name="hrnet_w32",
                detector_name="fasterrcnn_resnet50_fpn_v2",
                video_adapt=True,
                max_individuals=3,
                pseudo_threshold=0.1,
                bbox_threshold=0.9,
                detector_epochs=4,
                pose_epochs=4,
            )
        Using ``scale_list``:

            from deeplabcut.modelzoo.video_inference import video_inference_superanimal
            videos = ["/path/to/my/video.mp4"]
            superanimal_name = "superanimal_topviewmouse"
            video_extensions = "mp4"
            scale_list = [200, 300, 400]
            video_inference_superanimal(
                videos,
                superanimal_name,
                model_name="hrnet_w32",
                detector_name="fasterrcnn_resnet50_fpn_v2",
                scale_list=scale_list,
                video_extensions=video_extensions,
                video_adapt=True,
            )
    """
    return _impl(
        videos=videos,
        superanimal_name=superanimal_name,
        model_name=model_name,
        detector_name=detector_name,
        scale_list=scale_list,
        video_extensions=video_extensions,
        dest_folder=dest_folder,
        cropping=cropping,
        video_adapt=video_adapt,
        plot_trajectories=plot_trajectories,
        batch_size=batch_size,
        detector_batch_size=detector_batch_size,
        pcutoff=pcutoff,
        adapt_iterations=adapt_iterations,
        pseudo_threshold=pseudo_threshold,
        bbox_threshold=bbox_threshold,
        detector_epochs=detector_epochs,
        pose_epochs=pose_epochs,
        max_individuals=max_individuals,
        video_adapt_batch_size=video_adapt_batch_size,
        device=device,
        customized_pose_checkpoint=customized_pose_checkpoint,
        customized_detector_checkpoint=customized_detector_checkpoint,
        customized_model_config=customized_model_config,
        plot_bboxes=plot_bboxes,
        create_labeled_video=create_labeled_video,
        fmpose_return_3d=fmpose_return_3d,
    )
