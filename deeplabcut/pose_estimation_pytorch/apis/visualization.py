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
"""Methods to help with visualization of model outputs"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.collections as collections
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import deeplabcut.core.visualization as visualization
import deeplabcut.pose_estimation_pytorch.apis.utils as utils
import deeplabcut.pose_estimation_pytorch.data as data
import deeplabcut.pose_estimation_pytorch.data.preprocessor as preprocessor
import deeplabcut.pose_estimation_pytorch.models as models
from deeplabcut.core.config import read_config_as_dict
from deeplabcut.core.engine import Engine
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.utils import auxiliaryfunctions


def create_labeled_images(
    predictions: dict[str, dict[str, np.ndarray | np.ndarray]],
    out_folder: str | Path,
    pcutoff: float = 0.6,
    bboxes_pcutoff: float = 0.6,
    mode: str = "bodypart",
    cmap: str | colors.Colormap = "rainbow",
    dot_size: int = 12,
    alpha_value: float = 0.7,
    skeleton: list[tuple[int, int]] | None = None,
    skeleton_color: str = "k",
    close_figure_after_save: bool = True,
):
    """Plots model predictions on images.

    Args:
        predictions: The predictions to plot. A dictionary mapping image paths to
            the predictions made by the model on that image. The predictions should
            contain a "bodyparts" key, mapping to an array of shape (max_individuals,
            num_bodyparts, 3) containing predicted bodyparts. If there are any unique
            bodyparts predicted, then it should also contain a "unique_bodyparts" key,
            mapping to an array of shape (1, num_bodyparts, 3) containing the predicted
            unique bodyparts.
        out_folder: The folder where model predictions should be saved.
        pcutoff: The p-cutoff score above which predicted bodyparts are displayed with
            a "⋅" marker, and below which they are displayed with a "X" marker.
        bboxes_pcutoff: The bounding box cutoff score, below which predicted bounding
            boxes are shown with a dashed line.
        mode: One of "bodypart", "individual". Whether to color predictions by
            bodypart or individual.
        cmap: The colormap to use to plot predictions.
        dot_size: The size of the bodypart prediction markers.
        alpha_value: The transparency value of the bodypart prediction markers.
        skeleton: If skeletons should be plotted, the list of bodyparts that constitute
            the skeletons.
        skeleton_color: The color with which to plot the skeleton, if one is given.
        close_figure_after_save: Whether to close figures after saving the labeled
            images to disk.
    """
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True)

    color_by_individual = mode == "individual"
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)

    for image_path, image_predictions in predictions.items():
        # Load frame
        frame = Image.open(str(image_path))

        # get pose predictions
        pred = image_predictions["bodyparts"]
        total_idv, total_bodyparts = pred.shape[:2]
        unique_pred = None
        if "unique_bodyparts" in image_predictions:
            unique_pred = image_predictions["unique_bodyparts"][0]
            total_idv += 1
            total_bodyparts += len(unique_pred)

        # create plot
        fig, ax = plt.subplots()
        ax.imshow(frame)

        # plot bodyparts
        for idx, pose in enumerate(pred):
            xy, scores = pose[:, :2], pose[:, 2]
            mask = scores > pcutoff
            if np.sum(pose) < 0 or np.sum(mask) <= 0:
                continue

            bones = []
            if skeleton is not None:
                for idx_1, idx_2 in skeleton:
                    if scores[idx_1] > pcutoff and scores[idx_2] > pcutoff:
                        bones.append(xy[[idx_1, idx_2]])

            kwargs = dict(s=dot_size)
            if color_by_individual:
                kwargs["c"] = cmap(idx / total_idv)
            else:
                c = np.linspace(0, 1, total_bodyparts)[:len(pose)][mask]
                kwargs["c"] = c
                kwargs["cmap"] = cmap

            xy = xy[mask]
            ax.scatter(xy[:, 0], xy[:, 1], **kwargs)
            if len(bones) > 0:
                ax.add_collection(
                    collections.LineCollection(
                        bones, colors=skeleton_color, alpha=alpha_value
                    )
                )

        # plot unique bodyparts
        if unique_pred is not None:
            xy, scores = unique_pred[:, :2], unique_pred[:, 2]
            mask = scores > pcutoff
            if np.sum(mask) <= 0:
                continue

            kwargs = dict(s=dot_size)
            if color_by_individual:
                kwargs["c"] = cmap(1)
            else:
                c = np.linspace(0, 1, total_bodyparts)
                kwargs["c"] = c[-len(unique_pred):][mask]
                kwargs["cmap"] = cmap

            xy = xy[mask]
            ax.scatter(xy[:, 0], xy[:, 1], **kwargs)

        # plot bounding boxes
        if "bboxes" in image_predictions:
            bboxes = image_predictions["bboxes"]
            bbox_scores = image_predictions["bbox_scores"]
            for idx, (bbox, score) in enumerate(zip(bboxes, bbox_scores)):
                if score <= bboxes_pcutoff:
                    continue

                xmin, ymin, w, h = bbox
                rect = plt.Rectangle(
                    (xmin, ymin), w, h, fill=False, edgecolor="green", linewidth=2
                )
                ax.add_patch(rect)

        # save predictions
        output_path = out_folder / f"predictions_{Path(image_path).stem}.png"
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        fig.savefig(output_path)

        if close_figure_after_save:
            plt.close(fig)

    if close_figure_after_save:
        plt.close()


@torch.no_grad()
def extract_model_outputs(
    images: list[str] | list[Path],
    model: models.PoseModel,
    pre_processor: preprocessor.Preprocessor,
    device: str = "auto",
    context: list[dict[str, np.ndarray]] | None = None,
) -> list[dict[str, np.ndarray]]:
    """Obtains the outputs for a model for a list of images

    Args:
        images: List of image paths for which to get model outputs.
        model: The model for which to get model outputs.
        pre_processor: The pre-processor used to prepare the images before giving them
            to the model.
        device: The device on which to run inference.
        context: The context for each image to give to the pre-processor. For top-down
            models, this context should contain the bounding boxes to use for each
            image. This should be in a format:
                [
                    {"bboxes": array of shape (num_bboxes, 4)},  # image 1 bboxes,
                    {"bboxes": array of shape (num_bboxes, 4)},  # image 2 bboxes,
                    ...,
                    {"bboxes": array of shape (num_bboxes, 4)},  # image N bboxes,
                ]

    Returns:
        A list containing a dict for each input image, in the format:
        {
            inputs: a numpy array containing the inputs given to the model for the image
            context: the context given alongside the image
            outputs: a dict containing the model outputs
        }
    """
    if context is not None and len(context) != len(images):
        raise ValueError(
            "When passing context along with the images (e.g. bounding boxes for "
            "top-down models), there should be the same number of elements in the "
            f"context as the number of images. Received {len(images)} images but "
            f"{len(context)} contexts."
        )

    model = model.to(device)
    model = model.eval()

    model_data = []
    for idx, image in enumerate(images):
        image_context = {}
        if context is not None:
            image_context = context[idx]

        inputs, image_context = pre_processor(image, image_context)
        output = model(inputs.to(device))

        for head, head_cfg in model.cfg["heads"].items():
            if (
                head_cfg["predictor"].get("apply_sigmoid", False)
                or head_cfg["predictor"]["type"] == "PartAffinityFieldPredictor"
            ):
                if "heatmap" in output[head]:
                    output[head]["heatmap"] = F.sigmoid(output[head]["heatmap"])

        output = {
            head: {name: output.cpu().numpy() for name, output in head_outputs.items()}
            for head, head_outputs in output.items()
        }
        model_data.append(
            dict(inputs=inputs.cpu().numpy(), context=context, outputs=output)
        )

    return model_data


def extract_maps(
    config,
    shuffle: int = 0,
    trainingsetindex: int | str = 0,
    device: str | None = None,
    rescale: bool = False,
    indices: list[int] | None = None,
    extract_paf: bool = True,
    modelprefix: str | None = "",
    snapshot_index: int | str | None = None,
    detector_snapshot_index: int | str | None = None,
) -> dict:
    """
    Extracts the different maps output by DeepLabCut models, such as scoremaps, location
    refinement fields and part-affinity fields.

    Args:
        config: Full path of the config.yaml file as a string.
        shuffle: Index of the shuffle for which to extract maps
        trainingset_index: Integer specifying which TrainingsetFraction to use. This
            variable can also be set to "all".
        rescale: Evaluate the model at the 'global_scale' variable (as set in the
            test/pose_config.yaml file for a particular project). Every image will be
            resized according to that scale and prediction will be compared to the
            resized ground truth. The error will be reported in pixels at rescaled to
            the *original* size. Example:
                For a [200, 200] pixel image evaluated at ``global_scale=0.5``,
                predictions are calculated on [100, 100] pixel images, compared to
                ``0.5*ground truth`` and this error is then multiplied by 2!. The
                evaluation images are also shown for the original size!
        indices: Optionally, you can only obtain maps for a subset of images in your
            dataset. The indices given here are the indices of the images for which
            maps will be extracted.
        modelprefix: Directory containing the deeplabcut models to use when evaluating
            the network. By default, the models are assumed to exist in the project
            folder.
        snapshot_index: Index (starting at 0) of the snapshot we want to extract maps
            with. To evaluate the last one, use -1. To extract maps for all snapshots,
            use "all".
        detector_snapshot_index: Only for TD models. If defined, uses the detector with
            the given index for pose estimation. To extract maps for all detector
            snapshots, use "all".

    Returns:
        a dict indexed by (trainingset_fraction, snapshot_index, image_index). For each
        key, the item contains a tuple of:
            (img, scmap, locref, paf, bpt_names, paf_graph, img_name, is_train)

    Examples
    --------
        If you want to extract the data for image 0 and 103 (of the training set) for
        model trained with shuffle 0.

        >>> deeplabcut.extract_maps(config, 0, indices=[0, 103])
    """
    cfg = read_config_as_dict(config)

    trainset_indices = [trainingsetindex]
    if trainingsetindex == "all":
        trainset_indices = [i for i in range(len(cfg["TrainingFraction"]))]
    if snapshot_index is None:
        snapshot_index = cfg["snapshotindex"]
    if detector_snapshot_index is None:
        detector_snapshot_index = cfg["detector_snapshotindex"]

    extracted_maps = {}
    for trainset_index in trainset_indices:
        loader = data.DLCLoader(
            config=config,
            shuffle=shuffle,
            trainset_index=trainset_index,
            modelprefix=modelprefix,
        )
        extracted_maps[loader.train_fraction] = {}

        # (img, scmap, locref, paf, bpt_names, paf_graph, img_name, is_train)
        metadata = loader.model_cfg["metadata"]
        bpt_names = metadata["bodyparts"] + metadata["unique_bodyparts"]
        paf_graph = []
        bpt_head_cfg = loader.model_cfg["model"]["heads"]["bodypart"]
        if bpt_head_cfg["type"] == "DLCRNetHead":
            paf_graph = bpt_head_cfg.get("predictor", {}).get("graph")
            paf_indices = bpt_head_cfg.get("predictor", {}).get("edges_to_keep")
            if paf_indices is not None:
                paf_graph = [paf_graph[i] for i in paf_indices]

        if device is not None:
            loader.model_cfg["device"] = device
        loader.model_cfg["device"] = utils.resolve_device(loader.model_cfg)
        device = loader.model_cfg["device"]

        if snapshot_index is None:
            snapshot_index = -1
        snapshots = utils.get_model_snapshots(
            snapshot_index, loader.model_folder, loader.pose_task
        )

        image_paths = loader.df.index
        if indices is not None:
            image_paths = [image_paths[idx] for idx in indices]
        if len(image_paths) > 0 and isinstance(image_paths[0], tuple):
            image_paths = [Path(*img_path) for img_path in image_paths]

        image_paths = [
            (loader.project_path / img_path).resolve() for img_path in image_paths
        ]

        context = _get_context(image_paths, loader, detector_snapshot_index, device)
        train_idx = set(loader.split["train"])
        for snapshot in snapshots:
            snapshot_id = snapshot.path.stem
            extracted_maps[loader.train_fraction][snapshot_id] = {}
            runner = utils.get_pose_inference_runner(
                model_config=loader.model_cfg,
                snapshot_path=snapshot.path,
            )
            results = extract_model_outputs(
                image_paths,
                runner.model,
                runner.preprocessor,
                runner.device,
                context=context,
            )
            for idx, result in enumerate(results):
                image_idx = idx
                if indices is not None:
                    image_idx = indices[idx]

                # key can be just image_idx, or (image_idx, bbox_idx) for TD models
                keys, images, outputs = _collect_model_outputs(
                    loader.pose_task, result, image_idx
                )
                for key, image, output in zip(keys, images, outputs):
                    parsed = _parse_model_outputs(
                        image,
                        output,
                        strides={
                            k: runner.model.get_stride(k)
                            for k in runner.model.heads.keys()
                        },
                        denormalize_image=True,
                    )
                    img_name = image_paths[idx].stem
                    if isinstance(key, tuple):
                        bbox_id = key[1]
                        img_name += f"_bbox{bbox_id:03d}"

                    is_train = image_idx in train_idx
                    extracted_maps[loader.train_fraction][snapshot_id][key] = (
                        *parsed,
                        None,
                        bpt_names,
                        paf_graph,
                        img_name,
                        is_train,
                    )

    # img, scmap, locref, paf, peaks, bpt_names, paf_graph, img_name, is_train
    return extracted_maps


def extract_save_all_maps(
    config: str | Path,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    comparison_bodyparts: str | list[str] = "all",
    extract_paf: bool = True,
    all_paf_in_one: bool = True,
    device: str | None = None,
    rescale: bool = False,
    indices: list[int] | None = None,
    modelprefix: str | None = "",
    snapshot_index: int | str | None = None,
    detector_snapshot_index: int | str | None = None,
    dest_folder: str | Path | None = None,
):
    """
    Extracts the scoremap, location refinement field and part affinity field prediction
    of the model. The maps will be rescaled to the size of the input image and stored
    in the corresponding model folder in /evaluation-results-pytorch.

    Args:
        config: Full path of the config.yaml file as a string.
        shuffle: Index of the shuffle for which to extract maps
        trainingset_index: Integer specifying which TrainingsetFraction to use. This
            variable can also be set to "all".
        comparison_bodyparts: The average error will be computed for those body parts
            only (Has to be a subset of the body parts).
        extract_paf: Extract part affinity fields by default. Note that turning it off
            will make the function much faster.
        all_paf_in_one: By default, all part affinity fields are displayed on a single
            frame. If false, individual fields are shown on separate frames.
        indices: Optionally, you can only obtain maps for a subset of images in your
            dataset. The indices given here are the indices of the images for which
            maps will be extracted.
        modelprefix: Directory containing the deeplabcut models to use when evaluating
            the network. By default, the models are assumed to exist in the project
            folder.
        snapshot_index: Index (starting at 0) of the snapshot we want to extract maps
            with. To evaluate the last one, use -1. To extract maps for all snapshots,
            use "all".
        detector_snapshot_index: Only for TD models. If defined, uses the detector with
            the given index for pose estimation. To extract maps for all detector
            snapshots, use "all".

    Examples
    --------
    Calculated maps for images 0, 1 and 33.
        >>> deeplabcut.extract_save_all_maps(
        >>>     "/analysis/project/reaching-task/config.yaml",
        >>>     shuffle=1,
        >>>     indices=[0, 1, 33]
        >>> )

    """
    cfg = read_config_as_dict(config)
    maps = extract_maps(
        config,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        device=device,
        rescale=rescale,
        indices=indices,
        snapshot_index=snapshot_index,
        detector_snapshot_index=detector_snapshot_index,
        modelprefix=modelprefix,
    )
    bpts_to_plot = auxiliaryfunctions.intersection_of_body_parts_and_ones_given_by_user(
        cfg, comparison_bodyparts
    )

    print("Saving plots...")
    for frac, values in maps.items():
        dest_folder = _get_maps_folder(cfg, frac, shuffle, modelprefix, dest_folder)
        dest_folder.mkdir(exist_ok=True)
        for snap, maps in values.items():
            for image_idx, image_maps in tqdm(maps.items()):
                (
                    image,
                    scmap,
                    locref,
                    paf,
                    peaks,
                    bpt_names,
                    paf_graph,
                    image_path,
                    training_image,
                ) = image_maps

                if not extract_paf:
                    paf = []

                label = "train" if training_image else "test"
                img_w, img_h = image.shape[1], image.shape[0]
                scmap = _prepare_maps_for_plotting(scmap, (img_w, img_h))
                if scmap is None:
                    raise ValueError("Cannot plot heatmaps - none output by the model")

                locref = _prepare_maps_for_plotting(locref, (img_w, img_h))
                if locref is not None:
                    locref = locref.reshape((img_h, img_w, -1, 2))
                paf = _prepare_maps_for_plotting(paf, (img_w, img_h))

                visualization.generate_model_output_plots(
                    output_folder=dest_folder,
                    image_name=Path(image_path).stem,
                    bodypart_names=bpt_names,
                    bodyparts_to_plot=bpts_to_plot,
                    image=image,
                    scmap=scmap,
                    locref=locref,
                    paf=paf,
                    paf_graph=paf_graph,
                    paf_all_in_one=all_paf_in_one,
                    paf_colormap=cfg["colormap"],
                    output_suffix=f"{label}_{shuffle}_{frac}_{snap}",
                )


def _get_context(
    image_paths: list[Path],
    loader: data.Loader,
    detector_snapshot_index: int | str | None,
    device: str,
) -> list[dict] | None:
    """Gets the context for top-down pose estimation models"""
    if loader.pose_task != Task.TOP_DOWN:
        return None

    det_snapshots = []
    if detector_snapshot_index is not None:
        det_snapshots = utils.get_model_snapshots(
            detector_snapshot_index, loader.model_folder, Task.DETECT
        )

    if detector_snapshot_index is None or len(det_snapshots) == 0:
        if detector_snapshot_index is None:
            print("No ``detector_snapshot_index`` given.")
        else:
            print(f"No detector snapshots found in {loader.model_folder}")
        print("Using GT bboxes to extract maps for this top-down model")

        bboxes_train = loader.ground_truth_bboxes(mode="train")
        bboxes_test = loader.ground_truth_bboxes(mode="test")
        bboxes = {**bboxes_train, **bboxes_test}
        return [
            dict(bboxes=bboxes[str(img_path)]["bboxes"]) for img_path in image_paths
        ]

    detector_runner = utils.get_detector_inference_runner(
        model_config=loader.model_cfg,
        snapshot_path=det_snapshots[-1].path,
        device=device,
    )
    return detector_runner.inference(image_paths)


def _collect_model_outputs(
    task: Task,
    result: dict,
    image_idx: int,
) -> tuple[list, list, list]:
    """Collects the model outputs into data that can be processed.

    Args:
        task: Whether the model is a bottom-up or top-down model.
        result: A result output by ``extract_model_outputs``.
        image_idx: The index of the image

    Returns: keys, images, outputs
        keys: The key for each image to plot.
        images: The images to plot for this input image (a single image for bottom-up
            models, and the number of bounding boxes for top-down models).
        outputs: The model outputs for each image.
    """
    if task == Task.TOP_DOWN:
        keys, images, outputs = [], [], []

        # parse each input individually
        num_bboxes = len(result["inputs"])
        for bbox_idx in range(num_bboxes):
            keys.append((image_idx, bbox_idx))
            images.append(result["inputs"][bbox_idx])
            outputs.append(
                {
                    head: {k: v[bbox_idx] for k, v in head_outputs.items()}
                    for head, head_outputs in result["outputs"].items()
                }
            )
        return keys, images, outputs

    # remove batch dimension
    return (
        [image_idx],
        [result["inputs"][0]],
        [
            {
                head: {k: v[0] for k, v in head_outputs.items()}
                for head, head_outputs in result["outputs"].items()
            }
        ],
    )


def _parse_model_outputs(
    image: np.ndarray,
    outputs: dict[str, dict[str, np.ndarray]],
    strides: dict[str, int],
    denormalize_image: bool = True,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Parses the model outputs into a format that can easily be plotted.

    Args:
        image: The image used to obtain the outputs.
        outputs: The model outputs.
        strides: The total stride for each model head.
        denormalize_image: Whether the image was normalized and should be de-normalized.

    Returns: (img, scmap, locref, paf)
        img: The (de-normalized) image used as input.
        scmap: The score maps output by the model.
        locref: The locref fields output by the model.
        paf: The part-affinity fields output by the model.
    """
    image = image.transpose((1, 2, 0))
    if denormalize_image:
        image = image * np.array([0.229, 0.224, 0.225])
        image = image + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)

    heatmaps = [h for h in outputs["bodypart"].get("heatmap", [])]
    locrefs = [m * strides["bodypart"] for m in outputs["bodypart"].get("locref", [])]
    paf = [p for p in outputs["bodypart"].get("paf", [])]

    if "unique_bodypart" in outputs:
        heatmaps += [h for h in outputs["unique_bodypart"].get("heatmap", [])]
        locrefs += [
            strides["unique_bodypart"] * m
            for m in outputs["unique_bodypart"].get("locref", [])
        ]

    return image, heatmaps, locrefs, paf


def _prepare_maps_for_plotting(
    maps: list[np.ndarray], image_size: tuple[int, int]
) -> np.ndarray | None:
    """Resizes all maps to the image size and concatenates them into a single array.

    Args:
        maps: The maps that will be shown on the image.
        image_size: The (width, height) of the input image.

    Returns:
        The resized maps, or None if the list of maps was empty.
    """
    if len(maps) == 0:
        return None

    img_w, img_h = image_size
    return np.stack(
        [
            cv2.resize(map_, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            for map_ in maps
        ],
        axis=-1,
    )


def _get_maps_folder(
    cfg: dict,
    train_frac: float,
    shuffle: int,
    model_prefix: str | None,
    dest_folder: str | Path | None,
) -> Path:
    """Gets the destination folder for output maps"""
    if dest_folder is None:
        project_path = Path(cfg["project_path"])
        eval_folder = auxiliaryfunctions.get_evaluation_folder(
            trainFraction=train_frac,
            shuffle=shuffle,
            cfg=cfg,
            engine=Engine.PYTORCH,
            modelprefix=model_prefix,
        )
        dest_folder = project_path / eval_folder / "maps"

    return Path(dest_folder)
