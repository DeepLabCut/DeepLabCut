#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Union

import albumentations as A
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from skimage.util import img_as_ubyte
import cv2

import deeplabcut.pose_estimation_pytorch as dlc
from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal, VideoReader
from deeplabcut.pose_estimation_pytorch.models.model import PoseModel
from deeplabcut.pose_estimation_pytorch.models.detectors import (
    DETECTORS,
    BaseDetector,
)
from deeplabcut.pose_estimation_pytorch.models.predictors import (
    PREDICTORS,
    BasePredictor,
)
from deeplabcut.pose_estimation_pytorch.apis.utils import (
    build_pose_model,
    read_yaml,
    get_model_snapshots,
    get_detector_snapshots,
    videos_in_folder,
)
from deeplabcut.pose_estimation_pytorch.apis.inference_utils import (
    get_predictions_bottom_up,
    get_predictions_top_down,
)


def video_inference(
    model: PoseModel,
    predictor: BasePredictor,
    video_path: Path,
    batch_size: int = 1,
    device: Optional[str] = None,
    transform: Optional[A.Compose] = None,
    colormode: Optional[str] = "RGB",
    method: Optional[str] = "bu",
    detector: Optional[BaseDetector] = None,
    top_down_predictor: Optional[BasePredictor] = None,
    max_num_animals: Optional[int] = 1,
    num_keypoints: Optional[int] = 1,
    frames_resized: Optional[bool] = False,
) -> List[np.ndarray]:
    """
    Runs inference on all frames of a video

    Args:
        model: the model with which to run inference
        predictor: the predictor to use alongside the model
        video_path: the path to the video onto which inference should be run
        batch_size: the batch size with which to run inference
        device: the torch device to use to run inference. Dynamic selection if None
        transform: the image augmentation transform to use on the video frames, if any
        colormode: RGB or BGR
        method: 'td' (Top Down) or 'bu' (Bottom Up)
        detector: Detector for top down approach
        top_down_predictor: Makes predictions from the cropped keypoints coordinates and
                            the detected bbox
        max_num_animals: max number of animals
        num_keypoints: number of keypoints
        frames_resized: Whether the frame are resized for inference or not

    Returns:
        for each frame in the video, a numpy array containing the output of the
        predictor for the frame
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set the model to eval mode and put it on the device
    model.eval()
    model.to(device)

    print(f"Loading {video_path}")
    video_reader = VideoReader(str(video_path))
    n_frames = video_reader.get_n_frames()
    vid_w, vid_h = video_reader.dimensions
    print(
        f"Video metadata: \n"
        f"  n_frames:   {n_frames}\n"
        f"  fps:        {video_reader.fps}\n"
        f"  resolution: w={vid_w}, h={vid_h}\n"
    )

    pbar = tqdm(total=n_frames, file=sys.stdout)
    predictions = []
    frame = video_reader.read_frame()
    original_size = frame.shape
    transformed_size = original_size
    if transform:
        # Apply transformation once only to see the shape after transformation
        transformed_size = transform(image=frame)["image"].shape

    batch_ind = 0  # Index of the current img in batch
    batch_frames = np.empty((batch_size, transformed_size[0], transformed_size[1], 3))

    with torch.no_grad():
        while frame is not None:
            if frame.dtype != np.uint8:
                frame = img_as_ubyte(frame)

            if colormode == "BGR":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if transform:
                frame = transform(image=frame)["image"]

            batch_frames[batch_ind] = frame
            if batch_ind == batch_size - 1:
                batch = torch.tensor(
                    batch_frames, device=device, dtype=torch.float
                ).permute(0, 3, 1, 2)
                if method.lower() == "td":
                    batched_predictions = get_predictions_top_down(
                        detector=detector,
                        top_down_predictor=top_down_predictor,
                        model=model,
                        pose_predictor=predictor,
                        images=batch,
                        max_num_animals=max_num_animals,
                        num_keypoints=num_keypoints,
                        device=device,
                    )
                elif method.lower() == "bu":
                    batched_predictions = get_predictions_bottom_up(
                        model=model,
                        predictor=predictor,
                        images=batch,
                    )
                else:
                    raise ValueError(
                        "Method must be either 'bu' (Bottom Up) or 'td' (Top Down)."
                    )
                for frame_pred in batched_predictions:
                    if frames_resized:
                        resizing_factor = (original_size[0] / transformed_size[0]), (
                            original_size[1] / transformed_size[1]
                        )
                        frame_pred[:, :, 0] = (
                            frame_pred[:, :, 0] * resizing_factor[1]
                            + resizing_factor[1] / 2
                        )
                        frame_pred[:, :, 1] = (
                            frame_pred[:, :, 1] * resizing_factor[0]
                            + resizing_factor[0] / 2
                        )
                    predictions.append(frame_pred)

            frame = video_reader.read_frame()
            batch_ind += 1
            batch_ind = batch_ind % batch_size
            pbar.update(1)

    return predictions


def analyze_videos(
    config_path: str,
    data_path: Union[str, List[str]],
    output_folder: Optional[str] = None,
    video_type: Optional[str] = None,
    dataset_index: int = 0,
    shuffle: int = 1,
    snapshot_index: Optional[int] = None,
    model_prefix: str = "",
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    transform: Optional[A.Compose] = None,
    inv_transform: Optional[A.Compose] = None,
    overwrite: bool = False,
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Makes pose estimation predictions based on a trained model
    TODO: finish doc

    Args:
        config_path:
        data_path:
        output_folder:
        video_type:
        dataset_index:
        shuffle:
        snapshot_index:
        model_prefix:
        batch_size:
        device:
        transform:
        inv_transform:
        overwrite:

    Returns:

    """
    # Create the output folder
    _create_output_folder(output_folder)

    # Load the project configuration
    project = dlc.DLCProject(
        shuffle=shuffle,
        proj_root=str(Path(config_path).parent),
    )
    project.convert2dict(mode="test")
    project_path = Path(project.cfg["project_path"])
    train_fraction = project.cfg["TrainingFraction"][dataset_index]
    model_folder = project_path / auxiliaryfunctions.get_model_folder(
        train_fraction,
        shuffle,
        project.cfg,
        modelprefix=model_prefix,
    )
    model_path = _get_model_path(model_folder, snapshot_index, project.cfg)
    model_epochs = int(model_path.stem.split("-")[-1])
    dlc_scorer, dlc_scorer_legacy = auxiliaryfunctions.get_scorer_name(
        project.cfg,
        shuffle,
        train_fraction,
        trainingsiterations=model_epochs,
        modelprefix=model_prefix,
    )
    # Get general project parameters
    max_num_animals = len(project.cfg.get("individuals", ["single"]))
    num_keypoints = len(auxiliaryfunctions.get_bodyparts(project.cfg))

    # Read the inference configuration, load the model
    pytorch_config_path = model_folder / "train" / "pytorch_config.yaml"
    pytorch_config = read_yaml(pytorch_config_path)
    pose_cfg_path = model_folder / "test" / "pose_cfg.yaml"
    pose_cfg = auxiliaryfunctions.read_config(pose_cfg_path)
    method = pytorch_config.get("method", "bu")

    # Get model parameters
    # TODO: Should we get the batch size from the inference pose_cfg? Or have an
    #  inference pytorch_cfg?
    if batch_size is None:
        batch_size = pytorch_config.get("batch_size", 1)
    pose_cfg["batch_size"] = batch_size
    individuals = project.cfg.get("individuals", ["single"])

    # Get data processing parameters
    # if images are resized for inference,
    # need to take that into account to go back to original space
    frames_resized_with_transform = pytorch_config["data"].get("resize", False)

    # Load model, predictor
    model = build_pose_model(pytorch_config["model"], pose_cfg)
    model.load_state_dict(torch.load(model_path))
    predictor: BasePredictor = PREDICTORS.build(dict(pytorch_config["predictor"]))
    detector: BaseDetector = None
    top_down_predictor: BasePredictor = None
    if method.lower() == "td":
        detector_path = _get_detector_path(model_folder, snapshot_index, project.cfg)
        detector = DETECTORS.build(dict(pytorch_config["detector"]["detector_model"]))
        detector.load_state_dict(torch.load(detector_path))

        top_down_predictor = PREDICTORS.build(
            {"type": "TopDownPredictor", "format_bbox": "xyxy"}
        )
    # Reading video and init variables
    videos = videos_in_folder(data_path, video_type)
    results = []
    for video in videos:
        if output_folder is None:
            output_path = video.parent
        else:
            output_path = Path(output_folder)

        output_prefix = video.stem + dlc_scorer
        output_h5 = output_path / f"{output_prefix}.h5"
        output_pkl = output_path / f"{output_prefix}_full.pickle"
        if not overwrite and output_pkl.exists():
            print(f"Video already analyzed at {output_pkl}!")
        else:
            runtime = [time.time()]
            predictions = video_inference(
                model=model,
                predictor=predictor,
                video_path=video,
                batch_size=batch_size,
                device=device,
                transform=transform,
                colormode=pytorch_config.get("colormode", "RGB"),
                method=method,
                detector=detector,
                top_down_predictor=top_down_predictor,
                max_num_animals=max_num_animals,
                num_keypoints=num_keypoints,
                frames_resized=frames_resized_with_transform,
            )
            runtime.append(time.time())

            print(f"Inference is done for {video}! Saving results...")
            metadata = _generate_metadata(
                config=project.cfg,
                pose_config=pose_cfg,
                pytorch_pose_config=pytorch_config,
                dlc_scorer=dlc_scorer,
                train_fraction=train_fraction,
                batch_size=batch_size,
                runtime=(runtime[0], runtime[1]),
                video=VideoReader(str(video)),
            )

            if len(individuals) > 1:
                print("Extracting ", len(individuals), "instances per bodypart")
                xyz_labs_orig = ["x", "y", "likelihood"]
                suffix = [str(s + 1) for s in range(len(individuals))]
                suffix[0] = ""  # first has empty suffix for backwards compatibility
                xyz_labs = [x + s for s in suffix for x in xyz_labs_orig]
            else:
                xyz_labs = ["x", "y", "likelihood"]

            results_df_index = pd.MultiIndex.from_product(
                [[dlc_scorer], pose_cfg["all_joints_names"], xyz_labs],
                names=["scorer", "bodyparts", "coords"],
            )
            df = pd.DataFrame(
                np.array(predictions).reshape((len(predictions), -1)),
                columns=results_df_index,
                index=range(len(predictions)),
            )
            df.to_hdf(
                str(output_h5),
                "df_with_missing",
                format="table",
                mode="w",
            )
            results.append((str(video), df))
            output_data = _generate_output_data(pose_cfg, predictions)
            _ = auxfun_multianimal.SaveFullMultiAnimalData(
                output_data, metadata, str(output_h5)
            )

    return results


def _create_output_folder(output_folder: Optional[Path]) -> None:
    if output_folder is not None:
        output_folder = Path(output_folder)
        if not output_folder.exists():
            print(f"Creating the output folder {output_folder}")
            output_folder.mkdir(parents=True)

        assert Path(
            output_folder
        ).is_dir(), f"Output folder must be a directory: you passed '{output_folder}'"


def _generate_metadata(
    config: dict,
    pose_config: dict,
    pytorch_pose_config: dict,
    dlc_scorer: str,
    train_fraction: int,
    batch_size: int,
    runtime: Tuple[float, float],
    video: VideoReader,
) -> dict:
    w, h = video.dimensions
    cropping = config.get("cropping", False)
    if cropping:
        cropping_parameters = [
            config["x1"],
            config["x2"],
            config["y1"],
            config["y2"],
        ]
    else:
        cropping_parameters = [0, w, 0, h]

    metadata = {
        "start": runtime[0],
        "stop": runtime[1],
        "run_duration": runtime[1] - runtime[0],
        "Scorer": dlc_scorer,
        "DLC-model-config file": pose_config,
        "DLC-model-pytorch-config file": pytorch_pose_config,
        "fps": video.fps,
        "batch_size": batch_size,
        "frame_dimensions": (w, h),
        "nframes": video.get_n_frames(),
        "iteration (active-learning)": config["iteration"],
        "training set fraction": train_fraction,
        "cropping": cropping,
        "cropping_parameters": cropping_parameters,
    }
    return {"data": metadata}


def _get_model_path(model_folder: Path, snapshot_index: int, config: dict) -> Path:
    trained_models = get_model_snapshots(model_folder / "train")

    if snapshot_index is None:
        snapshot_index = config["snapshotindex"]

    if snapshot_index == "all":
        print(
            "snapshotindex is set to 'all' in the config.yaml file. Running video "
            "analysis with all snapshots is very costly! Use the function "
            "'evaluate_network' to choose the best the snapshot. For now, changing "
            "snapshot index to -1. To evaluate another snapshot, you can change the "
            "value in the config file or call `analyze_videos` with your desired "
            "snapshot index."
        )
        snapshot_index = -1

    assert isinstance(
        snapshot_index, int
    ), f"snapshotindex must be an integer but was '{snapshot_index}'"
    return trained_models[snapshot_index]


def _get_detector_path(model_folder: Path, snapshot_index: int, config: dict) -> Path:
    trained_models = get_detector_snapshots(model_folder / "train")

    if snapshot_index is None:
        snapshot_index = config["snapshotindex"]

    if snapshot_index == "all":
        print(
            "snapshotindex is set to 'all' in the config.yaml file. Running video "
            "analysis with all snapshots is very costly! Use the function "
            "'evaluate_network' to choose the best the snapshot. For now, changing "
            "snapshot index to -1. To evaluate another snapshot, you can change the "
            "value in the config file or call `analyze_videos` with your desired "
            "snapshot index."
        )
        snapshot_index = -1

    assert isinstance(
        snapshot_index, int
    ), f"snapshotindex must be an integer but was '{snapshot_index}'"
    return trained_models[snapshot_index]


def _generate_output_data(
    pose_config: dict,
    predictions: List[np.ndarray],
) -> dict:
    output = {
        "metadata": {
            "nms radius": pose_config.get("nmsradius"),
            "minimal confidence": pose_config.get("minconfidence"),
            "sigma": pose_config.get("sigma", 1),
            "PAFgraph": pose_config.get("partaffinityfield_graph"),
            "PAFinds": pose_config.get(
                "paf_best",
                np.arange(len(pose_config.get("partaffinityfield_graph", []))),
            ),
            "all_joints": [[i] for i in range(len(pose_config["all_joints"]))],
            "all_joints_names": [
                pose_config["all_joints_names"][i]
                for i in range(len(pose_config["all_joints"]))
            ],
            "nframes": len(predictions),
        }
    }

    str_width = int(np.ceil(np.log10(len(predictions))))
    for frame_num, frame_predictions in enumerate(predictions):
        key = "frame" + str(frame_num).zfill(str_width)
        output[key] = frame_predictions.squeeze()

        # TODO: Do we want to keep the same format as in the TensorFlow version?
        #  On the one hand, it's "more" backwards compatible.
        #  On the other, might as well simplify the code. These files should only be loaded
        #    by the PyTorch version, and only predictions made by PyTorch models should be
        #    loaded using them
        # p_bodypart_indv = np.transpose(frame_predictions.squeeze(), axes=[1, 0, 2])
        # coords = [
        #     bodypart_predictions[:, :2] for bodypart_predictions in p_bodypart_indv
        # ]
        # scores = [
        #     bodypart_predictions[:, 2:] for bodypart_predictions in p_bodypart_indv
        # ]
        # output[key] = {
        #     "coordinates": (coords,),
        #     "confidence": scores,
        #     "costs": None,
        # }

    return output
