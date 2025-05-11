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
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from tqdm import tqdm

import deeplabcut.utils.auxiliaryfunctions as auxiliaryfunctions
import deeplabcut.utils.auxfun_multianimal as auxfun_multianimal
from deeplabcut.core import trackingutils
from deeplabcut.core.engine import Engine
from deeplabcut.core.inferenceutils import Assembly
from deeplabcut.pose_estimation_pytorch.apis.utils import (
    get_scorer_name,
    list_videos_in_folder,
)


def convert_detections2tracklets(
    config: str,
    videos: Union[str, List[str]],
    videotype: Optional[str] = None,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    overwrite: bool = False,
    destfolder: Optional[str] = None,
    ignore_bodyparts: Optional[List[str]] = None,
    inferencecfg: Optional[dict] = None,
    modelprefix="",
    greedy: bool = False,  # TODO(niels): implement greedy assembly during video analysis
    calibrate: bool = False,  # TODO(niels): implement assembly calibration during video analysis
    window_size: int = 0,  # TODO(niels): implement window size selection for assembly during video analysis
    identity_only=False,
    track_method="",
):
    """TODO: Documentation, clean & remove code duplication (with analyze video)"""
    cfg = auxiliaryfunctions.read_config(config)
    inference_cfg = inferencecfg
    track_method = auxfun_multianimal.get_track_method(cfg, track_method=track_method)

    if len(cfg["multianimalbodyparts"]) == 1 and track_method != "box":
        warnings.warn("Switching to `box` tracker for single point tracking...")
        track_method = "box"
        cfg["default_track_method"] = track_method
        auxiliaryfunctions.write_config(config, cfg)

    train_fraction = cfg["TrainingFraction"][trainingsetindex]
    start_path = os.getcwd()  # record cwd to return to this directory in the end

    # TODO: add cropping as in video analysis!
    # if cropping is not None:
    #    cfg['cropping']=True
    #    cfg['x1'],cfg['x2'],cfg['y1'],cfg['y2']=cropping
    #    print("Overwriting cropping parameters:", cropping)
    #    print("These are used for all videos, but won't be save to the cfg file.")

    rel_model_dir = auxiliaryfunctions.get_model_folder(
        train_fraction,
        shuffle,
        cfg,
        modelprefix=modelprefix,
        engine=Engine.PYTORCH,
    )
    model_dir = Path(cfg["project_path"]) / rel_model_dir
    path_test_config = model_dir / "test" / "pose_cfg.yaml"
    dlc_cfg = auxiliaryfunctions.read_plainconfig(str(path_test_config))

    if "multi-animal" not in dlc_cfg["dataset_type"]:
        raise ValueError("This function is only required for multianimal projects!")

    if track_method == "ctd":
        raise ValueError(
            "CTD tracking occurs directly during video analysis. No need to call "
            "`convert_detections2tracklets` with `track_method=='ctd'`."
        )

    if inference_cfg is None:
        inference_cfg = auxfun_multianimal.read_inferencecfg(
            model_dir / "test" / "inference_cfg.yaml", cfg
        )
    auxfun_multianimal.check_inferencecfg_sanity(cfg, inference_cfg)

    if len(cfg["multianimalbodyparts"]) == 1 and track_method != "box":
        warnings.warn("Switching to `box` tracker for single point tracking...")
        track_method = "box"
        # Also ensure `boundingboxslack` is greater than zero, otherwise overlap
        # between trackers cannot be evaluated, resulting in empty tracklets.
        inference_cfg["boundingboxslack"] = max(inference_cfg["boundingboxslack"], 40)

    dlc_scorer = get_scorer_name(
        cfg,
        shuffle,
        train_fraction,
        snapshot_index=None,
        detector_index=None,
        modelprefix=modelprefix,
    )

    videos = list_videos_in_folder(videos, videotype)
    if len(videos) == 0:
        print(f"No videos were found in {videos}")
        return

    for video in videos:
        print("Processing... ", video)
        if destfolder is None:
            output_path = video.parent
        else:
            output_path = Path(destfolder)
            output_path.mkdir(exist_ok=True, parents=True)

        video_name = video.stem

        data_prefix = video_name + dlc_scorer
        data_filename = output_path / (data_prefix + ".h5")
        print(f"Loading From {data_filename}")
        data, metadata = auxfun_multianimal.LoadFullMultiAnimalData(str(data_filename))
        if track_method == "ellipse":
            method = "el"
        elif track_method == "box":
            method = "bx"
        else:
            method = "sk"

        track_filename = output_path / (data_prefix + f"_{method}.pickle")
        if not overwrite and track_filename.exists():
            # TODO: check if metadata are identical (same parameters!)
            print(f"Tracklets already computed at {track_filename}")
            print("Set overwrite = True to overwrite.")
        else:
            assemblies_path = data_filename.with_stem(
                data_filename.stem + "_assemblies"
            ).with_suffix(".pickle")
            if not assemblies_path.exists():
                raise FileNotFoundError(
                    f"Could not find the assembles file {assemblies_path}. You're "
                    f"converting detections to tracklets using PyTorch, which "
                    "means the assemblies file must be created by the model when "
                    "analyzing the video!"
                )
            assemblies_data = auxiliaryfunctions.read_pickle(assemblies_path)

            tracklets = build_tracklets(
                assemblies_data=assemblies_data,
                track_method=track_method,
                inference_cfg=inference_cfg,
                joints=data["metadata"]["all_joints_names"],
                scorer=metadata["data"]["Scorer"],
                num_frames=data["metadata"]["nframes"],
                ignore_bodyparts=ignore_bodyparts,
                unique_bodyparts=cfg["uniquebodyparts"],
                identity_only=identity_only
            )

            with open(track_filename, "wb") as f:
                pickle.dump(tracklets, f, pickle.HIGHEST_PROTOCOL)

    os.chdir(str(start_path))
    print(
        "The tracklets were created (i.e., under the hood "
        "deeplabcut.convert_detections2tracklets was run). Now you can "
        "'refine_tracklets' in the GUI, or run 'deeplabcut.stitch_tracklets'."
    )


def build_tracklets(
    assemblies_data: dict,
    track_method: str,
    inference_cfg: dict,
    joints: list[str],
    scorer: str,
    num_frames: int,
    ignore_bodyparts: list[str]|None = None,
    unique_bodyparts: list|None = None,
    identity_only: bool = False
) -> dict :

    if track_method == "box":
        mot_tracker = trackingutils.SORTBox(
            inference_cfg["max_age"],
            inference_cfg["min_hits"],
            inference_cfg.get("iou_threshold", 0.3),
        )
    elif track_method == "skeleton":
        mot_tracker = trackingutils.SORTSkeleton(
            len(joints),
            inference_cfg["max_age"],
            inference_cfg["min_hits"],
            inference_cfg.get("oks_threshold", 0.5),
        )
    else:
        mot_tracker = trackingutils.SORTEllipse(
            inference_cfg.get("max_age", 1),
            inference_cfg.get("min_hits", 1),
            inference_cfg.get("iou_threshold", 0.6),
        )

    tracklets = {}

    df_index = _create_tracklets_header(joints, scorer)
    tracklets["header"] = df_index

    # Initialize storage of the 'single' individual track
    if unique_bodyparts:
        tracklets["single"] = {}
        _single = {}
        for index in range(num_frames):
            single_detection = assemblies_data["single"].get(index)
            if single_detection is None:
                continue
            _single[index] = np.asarray(single_detection)
        tracklets["single"].update(_single)

    pcutoff = inference_cfg.get("pcutoff")
    if inference_cfg["topktoretain"] == 1:
        tracklets[0] = {}
        for index in tqdm(range(num_frames)):
            assemblies = assemblies_data.get(index)
            if assemblies is None or len(assemblies) == 0:
                continue

            assembly = np.asarray(assemblies[0].data)
            assembly[assembly[..., 2] < pcutoff] = np.nan
            tracklets[0][index] = assembly
    else:
        multi_bpts = list(set(joints).difference(unique_bodyparts or []))
        keep = set(multi_bpts).difference(ignore_bodyparts or [])
        keep_inds = sorted(multi_bpts.index(bpt) for bpt in keep)
        for index in tqdm(range(num_frames)):
            assemblies = assemblies_data.get(index)
            if assemblies is None or len(assemblies) == 0:
                continue

            animals = np.stack([a for a in assemblies])
            animals[np.any(animals[..., :3] < 0, axis=-1), :2] = np.nan
            animals[animals[..., 2] < pcutoff, :2] = np.nan
            animal_mask = ~np.all(np.isnan(animals[:, :, :2]), axis=(1, 2))
            if ~np.any(animal_mask):
                continue
            animals = animals[animal_mask]

            if identity_only:
                # Optimal identity assignment based on soft voting
                mat = np.zeros((len(animals), inference_cfg["topktoretain"]))
                for row, animal_pose in enumerate(animals):
                    animal_pose = animal_pose[
                        ~np.isnan(animal_pose).any(axis=1)
                    ]
                    unique_ids, idx = np.unique(
                        animal_pose[:, 3], return_inverse=True
                    )
                    total_scores = np.bincount(idx, weights=animal_pose[:, 2])
                    softmax_id_scores = softmax(total_scores)
                    for pred_id, softmax_score in zip(
                            unique_ids.astype(int), softmax_id_scores
                    ):
                        mat[row, pred_id] = softmax_score

                inds = linear_sum_assignment(mat, maximize=True)
                trackers = np.c_[inds][:, ::-1]
            else:
                if track_method == "box":
                    xy = trackingutils.calc_bboxes_from_keypoints(
                        animals[:, keep_inds], inference_cfg["boundingboxslack"]
                    )  # TODO: get cropping parameters and utilize!
                else:
                    xy = animals[:, keep_inds, :2]
                trackers = mot_tracker.track(xy)

            strwidth = int(np.ceil(np.log10(num_frames)))
            imname = "frame" + str(index).zfill(strwidth)
            trackingutils.fill_tracklets(tracklets, trackers, animals, imname)

    return tracklets


def _create_tracklets_header(joints, dlc_scorer):
    bodypart_labels = [bpt for bpt in joints for _ in range(3)]
    scorers = len(bodypart_labels) * [dlc_scorer]
    xyl_value = int(len(bodypart_labels) / 3) * ["x", "y", "likelihood"]
    return pd.MultiIndex.from_arrays(
        np.vstack([scorers, bodypart_labels, xyl_value]),
        names=["scorer", "bodyparts", "coords"],
    )


def _conv_predictions_to_assemblies(
    image_names: List[str], predictions: Dict[str, np.ndarray]
) -> Dict[int, List[Assembly]]:
    """
    Converts predictions to an assemblies dictionary
    predictions shape (num_animals, num_keypoints, 2 or 3)
    """
    assemblies = {}
    if len(predictions) == 0:
        return assemblies

    for image_index, image_name in enumerate(image_names):
        frame_predictions = predictions.get(image_name)
        if frame_predictions is not None:
            num_kpts, num_animals, pred_shape = frame_predictions.shape
            kpt_lst = []
            for i in range(num_animals):
                animal_prediction = frame_predictions[:, i, :]
                ass_prediction = np.ones((num_kpts, 4), dtype=frame_predictions.dtype)
                ass_prediction[:, 3] = -ass_prediction[:, 3]
                ass_prediction[:, :pred_shape] = animal_prediction.copy()
                ass = Assembly.from_array(ass_prediction)
                if len(ass) > 0:
                    kpt_lst.append(ass)

            assemblies[image_index] = kpt_lst

    return assemblies
