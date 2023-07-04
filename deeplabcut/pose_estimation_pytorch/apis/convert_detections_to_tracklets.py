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
import pickle
import re
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from deeplabcut.pose_estimation_tensorflow.lib.inferenceutils import Assembly
from scipy.optimize import linear_sum_assignment

from tqdm import tqdm

from deeplabcut import auxiliaryfunctions
from deeplabcut.pose_estimation_tensorflow import load_config
from deeplabcut.pose_estimation_tensorflow.lib import trackingutils
from deeplabcut.utils import auxfun_multianimal
from deeplabcut.pose_estimation_pytorch.apis.utils import (
    get_model_snapshots,
    videos_in_folder,
)


def convert_detections2tracklets(
    config_path: str,
    data_path: Union[str, List[str]],
    output_folder: Optional[str] = None,
    video_type: Optional[str] = None,
    shuffle: int = 1,
    dataset_index: int = 0,
    overwrite: bool = False,
    ignore_bodyparts=None,
    inference_cfg: Optional[str] = None,
    modelprefix="",
    greedy=False,
    calibrate=False,
    window_size=0,
    identity_only=False,
    track_method="",
):
    """TODO: Documentation, clean & remove code duplication (with analyze video)"""
    cfg = auxiliaryfunctions.read_config(config_path)
    track_method = auxfun_multianimal.get_track_method(cfg, track_method=track_method)

    if len(cfg["multianimalbodyparts"]) == 1 and track_method != "box":
        warnings.warn("Switching to `box` tracker for single point tracking...")
        track_method = "box"
        cfg["default_track_method"] = track_method
        auxiliaryfunctions.write_config(config_path, cfg)

    train_fraction = cfg["TrainingFraction"][dataset_index]
    start_path = os.getcwd()  # record cwd to return to this directory in the end

    # TODO: add cropping as in video analysis!
    # if cropping is not None:
    #    cfg['cropping']=True
    #    cfg['x1'],cfg['x2'],cfg['y1'],cfg['y2']=cropping
    #    print("Overwriting cropping parameters:", cropping)
    #    print("These are used for all videos, but won't be save to the cfg file.")

    rel_model_dir = auxiliaryfunctions.get_model_folder(
        train_fraction, shuffle, cfg, modelprefix=modelprefix
    )
    model_dir = Path(cfg["project_path"]) / rel_model_dir
    path_test_config = model_dir / "test" / "pose_cfg.yaml"
    dlc_cfg = load_config(str(path_test_config))

    if "multi-animal" not in dlc_cfg["dataset_type"]:
        raise ValueError("This function is only required for multianimal projects!")

    if inference_cfg is None:
        path_inference_config = model_dir / "test" / "inference_cfg.yaml"
        inference_cfg = auxfun_multianimal.read_inferencecfg(path_inference_config, cfg)
    else:
        auxfun_multianimal.check_inferencecfg_sanity(cfg, inference_cfg)

    if len(cfg["multianimalbodyparts"]) == 1 and track_method != "box":
        warnings.warn("Switching to `box` tracker for single point tracking...")
        track_method = "box"
        # Also ensure `boundingboxslack` is greater than zero, otherwise overlap
        # between trackers cannot be evaluated, resulting in empty tracklets.
        inference_cfg["boundingboxslack"] = max(inference_cfg["boundingboxslack"], 40)

    # Check which snapshots are available and sort them by # iterations
    snapshots = get_model_snapshots(model_dir / "train")
    assert (
        len(snapshots) > 0
    ), f"No snapshots were found in the model directory {model_dir / 'train'}"
    snapshot_index = cfg["snapshotindex"]
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

    snapshot = snapshots[snapshot_index]
    print(f"Using snapshot {snapshot} for model {model_dir}")
    dlc_cfg["init_weights"] = str(snapshot)
    num_epochs = int(snapshot.stem.split("-")[-1])
    dlc_scorer, dlc_scorer_legacy = auxiliaryfunctions.get_scorer_name(
        cfg,
        shuffle,
        train_fraction,
        trainingsiterations=num_epochs,
        modelprefix=modelprefix,
    )

    # TODO: deal with lists of strings
    videos = videos_in_folder(data_path, video_type)
    if len(videos) == 0:
        print(f"No videos were found in {data_path}")
        return

    for video in videos:
        print("Processing... ", video)
        if output_folder is None:
            output_path = video.parent
        else:
            output_path = Path(output_folder)
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
            dlc_scorer = metadata["data"]["Scorer"]
            joints = data["metadata"]["all_joints_names"]
            n_joints = len(joints)

            # TODO: adjust this for multi + unique bodyparts!
            # this is only for multianimal parts and uniquebodyparts as one (not one
            # uniquebodyparts guy tracked etc.)
            bodypartlabels = [bpt for bpt in joints for _ in range(3)]
            scorers = len(bodypartlabels) * [dlc_scorer]
            xyl_value = int(len(bodypartlabels) / 3) * ["x", "y", "likelihood"]
            df_index = pd.MultiIndex.from_arrays(
                np.vstack([scorers, bodypartlabels, xyl_value]),
                names=["scorer", "bodyparts", "coords"],
            )
            image_names = [fn for fn in data if fn != "metadata"]

            if track_method == "box":
                mot_tracker = trackingutils.SORTBox(
                    inference_cfg["max_age"],
                    inference_cfg["min_hits"],
                    inference_cfg.get("oks_threshold", 0.3),
                )
            elif track_method == "skeleton":
                mot_tracker = trackingutils.SORTSkeleton(
                    n_joints,
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
            multi_bpts = cfg["multianimalbodyparts"]

            ass_unique = {}
            ass_assemblies = _conv_predictions_to_assemblies(image_names, data)
            ass_data = dict()
            for ind, assemblies in ass_assemblies.items():
                ass_data[ind] = [ass.data for ass in assemblies]
            if ass_unique:
                ass_data["single"] = ass_unique
            with open(
                data_filename.parent / (data_filename.stem + "_assemblies.pickle"), "wb"
            ) as file:
                pickle.dump(ass_data, file, pickle.HIGHEST_PROTOCOL)

            # Initialize storage of the 'single' individual track
            if cfg["uniquebodyparts"]:
                tracklets["single"] = {}
                _single = {}
                for index, image_name in enumerate(image_names):
                    single_detection = ass_unique.get(index)
                    if single_detection is None:
                        continue
                    imindex = int(re.findall(r"\d+", image_name)[0])
                    _single[imindex] = single_detection
                tracklets["single"].update(_single)

            if inference_cfg["topktoretain"] == 1:
                tracklets[0] = {}
                for index, image_name in tqdm(enumerate(image_names)):
                    assemblies = ass_assemblies.get(index)
                    if assemblies is None:
                        continue
                    tracklets[0][image_name] = assemblies[0].data
            else:
                keep = set(multi_bpts).difference(ignore_bodyparts or [])
                keep_inds = sorted(multi_bpts.index(bpt) for bpt in keep)
                for index, image_name in tqdm(enumerate(image_names)):
                    assemblies = ass_assemblies.get(index)
                    if assemblies is None:
                        continue

                    animals = np.stack([ass.data for ass in assemblies])
                    if not identity_only:
                        if track_method == "box":
                            xy = trackingutils.calc_bboxes_from_keypoints(
                                animals[:, keep_inds],
                                inference_cfg["boundingboxslack"],
                            )  # TODO: get cropping parameters and utilize!
                        else:
                            xy = animals[:, keep_inds, :2]
                        trackers = mot_tracker.track(xy)
                    else:
                        # Optimal identity assignment based on soft voting
                        mat = np.zeros((len(assemblies), inference_cfg["topktoretain"]))
                        for nrow, assembly in enumerate(assemblies):
                            for k, v in assembly.soft_identity.items():
                                mat[nrow, k] = v
                        inds = linear_sum_assignment(mat, maximize=True)
                        trackers = np.c_[inds][:, ::-1]
                    trackingutils.fill_tracklets(
                        tracklets, trackers, animals, image_name
                    )

            tracklets["header"] = df_index
            with open(track_filename, "wb") as f:
                pickle.dump(tracklets, f, pickle.HIGHEST_PROTOCOL)

    os.chdir(str(start_path))

    print(
        "The tracklets were created (i.e., under the hood "
        "deeplabcut.convert_detections2tracklets was run). Now you can "
        "'refine_tracklets' in the GUI, or run 'deeplabcut.stitch_tracklets'."
    )


def _conv_predictions_to_assemblies(
    image_names: List[str],
    predictions: Dict[str, np.ndarray],
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
