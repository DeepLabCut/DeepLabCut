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
import glob
import os
import os.path
import pickle
import time
import warnings
from pathlib import Path

import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
from dlclibrary.dlcmodelzoo.modelzoo_download import (
    download_huggingface_model,
    MODELOPTIONS,
)
from skimage.util import img_as_ubyte
from tqdm import tqdm

from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.core import predict as single_predict
from deeplabcut.pose_estimation_tensorflow.core import predict_multianimal as predict
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.auxfun_videos import VideoWriter

warnings.simplefilter("ignore", category=RuntimeWarning)


def get_multi_scale_frames(frame, scale_list):
    augs = []
    shapes = []
    for scale in scale_list:
        aug = iaa.Resize({"width": "keep-aspect-ratio", "height": scale})
        augs.append(aug)

    frames = []
    for i in range(len(scale_list)):
        resized_frame = augs[i](image=frame)
        frames.append(resized_frame)
        shapes.append(frames[-1].shape)

    return frames, shapes


def _project_pred_to_original_size(pred, old_shape, new_shape):
    old_h, old_w, _ = old_shape
    new_h, new_w, _ = new_shape
    ratio_h, ratio_w = old_h / new_h, old_w / new_w

    coordinate = pred["coordinates"][0]
    confidence = pred["confidence"]
    for kpt_id, coord_list in enumerate(coordinate):
        if len(coord_list) == 0:
            continue
        confidence_list = confidence[kpt_id]
        max_idx = np.argmax(confidence_list)
        # ratio_h and ratio_w should match though in reality it does not match exactly
        max_pred = coordinate[kpt_id][max_idx] * ratio_h

        # only keep the max

        confidence[kpt_id] = confidence_list[max_idx]
        coordinate[kpt_id] = max_pred
    return pred


def _average_multiple_scale_preds(
    preds,
    scale_list,
    num_kpts,
    cos_dist_threshold=0.997,
    confidence_threshold=0.1,
):
    if len(scale_list) < 2:
        return preds[0]

    xyp = np.zeros((len(scale_list), num_kpts, 3))
    for scale_id, pred in enumerate(preds):
        # empty prediction if pred is not a dict
        if not isinstance(pred, dict):
            xyp[scale_id] = np.nan
            continue
        coordinates = pred["coordinates"][0]
        confidence = pred["confidence"]
        for i, (coords, conf) in enumerate(zip(coordinates, confidence)):
            if not np.any(coords):
                continue
            xyp[scale_id, i, :2] = coords
            xyp[scale_id, i, 2] = conf
    xy = xyp[..., :2]

    # Compute cosine similarity
    mean_vec = np.nanmedian(xy, axis=0)
    dist_ = np.einsum("ijk,jk->ij", xy, mean_vec)
    n = np.linalg.norm(xy, axis=2) * np.linalg.norm(mean_vec, axis=1)
    dist = np.nan_to_num(dist_ / n)

    mask = np.logical_or(
        xyp[..., 2] < confidence_threshold,
        dist < cos_dist_threshold,
    )
    xyp[mask] = np.nan
    coords = np.nanmedian(xyp[..., :2], axis=0)
    conf = np.nanmedian(xyp[..., 2], axis=0)
    dict_ = {
        "coordinates": [list(coords[:, None])],
        "confidence": list(conf[:, None].astype(np.float32)),
    }
    return dict_


def _video_inference(
    test_cfg,
    sess,
    inputs,
    outputs,
    cap,
    nframes,
    batchsize,
    scale_list=[],
):
    strwidth = int(np.ceil(np.log10(nframes)))  # width for strings
    batch_ind = 0  # keeps track of which image within a batch should be written to

    nx, ny = cap.dimensions

    pbar = tqdm(total=nframes)
    counter = 0
    inds = []
    print("scale list", scale_list)
    PredicteData = {}
    # len(frames) -> (n_scale,)
    # frames[0].shape - > (batchsize, h, w, 3)
    multi_scale_batched_frames = None
    frame_shapes = None

    num_kpts = len(test_cfg["all_joints_names"])

    while cap.video.isOpened():
        # no crop needed
        _frame = cap.read_frame()
        if _frame is not None:
            frame = img_as_ubyte(_frame)

            old_shape = frame.shape
            frames, frame_shapes = get_multi_scale_frames(frame, scale_list)

            if multi_scale_batched_frames is None:
                multi_scale_batched_frames = [
                    np.empty(
                        (batchsize, frame.shape[0], frame.shape[1], 3), dtype="ubyte"
                    )
                    for frame in frames
                ]

            for scale_id, frame in enumerate(frames):
                multi_scale_batched_frames[scale_id][batch_ind] = frame
            inds.append(counter)
            if batch_ind == batchsize - 1:
                preds = []
                for scale_id, batched_frames in enumerate(multi_scale_batched_frames):
                    # batch full, start true inferencing
                    D = predict.predict_batched_peaks_and_costs(
                        test_cfg, batched_frames, sess, inputs, outputs
                    )
                    preds.append(D)
                    # only do this when animal is detected
                ind_start = inds[0]
                for i in range(batchsize):
                    ind = ind_start + i
                    PredicteData["frame" + str(ind).zfill(strwidth)] = []

                    for scale_id in range(len(scale_list)):
                        if i >= len(preds[scale_id]):
                            pred = []
                        else:
                            pred = preds[scale_id][i]
                        if pred != []:
                            pred = _project_pred_to_original_size(
                                pred, old_shape, frame_shapes[scale_id]
                            )

                        PredicteData["frame" + str(ind).zfill(strwidth)].append(pred)

                batch_ind = 0
                inds.clear()
            else:
                batch_ind += 1
        elif counter >= nframes:
            # in case we reach the end of the video
            if batch_ind > 0:
                preds = []
                for scale_id, batched_frames in enumerate(multi_scale_batched_frames):
                    D = predict.predict_batched_peaks_and_costs(
                        test_cfg,
                        batched_frames,
                        sess,
                        inputs,
                        outputs,
                    )

                    preds.append(D)

                ind_start = inds[0]
                for i in range(batchsize):
                    ind = ind_start + i
                    if ind >= nframes:
                        break
                    PredicteData["frame" + str(ind).zfill(strwidth)] = []
                    for scale_id in range(len(scale_list)):
                        if i >= len(preds[scale_id]):
                            pred = []
                        else:
                            pred = preds[scale_id][i]
                        if pred != []:
                            pred = _project_pred_to_original_size(
                                pred, old_shape, frame_shapes[scale_id]
                            )
                        PredicteData["frame" + str(ind).zfill(strwidth)].append(pred)

            break

        counter += 1
        pbar.update(1)

    cap.close()
    pbar.close()

    for k, v in PredicteData.items():
        if v != []:
            PredicteData[k] = _average_multiple_scale_preds(v, scale_list, num_kpts)

    PredicteData["metadata"] = {
        "nms radius": test_cfg.get("nmsradius", None),
        "minimal confidence": test_cfg.get("minconfidence", None),
        "sigma": test_cfg.get("sigma", 1),
        "PAFgraph": test_cfg.get("partaffinityfield_graph", None),
        "PAFinds": test_cfg.get(
            "paf_best", np.arange(len(test_cfg["partaffinityfield_graph"]))
        ),
        "all_joints": [[i] for i in range(len(test_cfg["all_joints"]))],
        "all_joints_names": [
            test_cfg["all_joints_names"][i] for i in range(len(test_cfg["all_joints"]))
        ],
        "nframes": nframes,
    }

    return PredicteData, nframes


def video_inference(
    videos,
    project_name,
    model_name,
    scale_list=[],
    videotype="avi",
    destfolder=None,
    batchsize=1,
    robust_nframes=False,
    allow_growth=False,
    init_weights="",
    customized_test_config="",
):
    dlc_root_path = auxiliaryfunctions.get_deeplabcut_path()

    if customized_test_config == "":
        project_cfg = load_config(
            os.path.join(
                dlc_root_path,
                "modelzoo",
                "project_configs",
                f"{project_name}.yaml",
            )
        )
        model_cfg = load_config(
            os.path.join(
                dlc_root_path,
                "modelzoo",
                "model_configs",
                f"{model_name}.yaml",
            )
        )
        test_cfg = {**project_cfg, **model_cfg}
        test_cfg["all_joints"] = [i for i in range(len(test_cfg["bobyparts"]))]
        test_cfg["all_joints_names"] = test_cfg["bobyparts"]
        num_joints = len(test_cfg["all_joints"])
        test_cfg["num_joints"] = num_joints
        test_cfg["num_limbs"] = int((num_joints * (num_joints - 1)) // 2)

    else:
        test_cfg = customized_test_config

    # add a temp folder for checkpoint
    weight_folder = str(
        Path(dlc_root_path)
        / "modelzoo"
        / "checkpoints"
        / f"{project_name}_{model_name}"
    )
    snapshots = glob.glob(os.path.join(weight_folder, "snapshot-*.index"))
    test_cfg["partaffinityfield_graph"] = []
    test_cfg["partaffinityfield_predict"] = False

    if init_weights != "":
        test_cfg["init_weights"] = init_weights
    else:
        if len(snapshots) == 0:
            raise FileNotFoundError(
                f"Did not find any super animal snapshots in {weight_folder}"
            )

        init_weights = os.path.abspath(snapshots[0]).replace(".index", "")
        test_cfg["init_weights"] = init_weights

    test_cfg["num_outputs"] = 1
    test_cfg["batch_size"] = batchsize

    sess, inputs, outputs = single_predict.setup_pose_prediction(
        test_cfg, allow_growth=allow_growth
    )
    DLCscorer = "DLC_" + Path(test_cfg["init_weights"]).stem
    videos = auxiliaryfunctions.get_list_of_videos(videos, videotype)

    datafiles = []
    for video in videos:
        vname = Path(video).stem

        videofolder = str(Path(video).parents[0])
        if destfolder is None:
            destfolder = videofolder
            auxiliaryfunctions.attempt_to_make_folder(destfolder)

        dataname = os.path.join(destfolder, vname + DLCscorer + ".h5")
        datafiles.append(dataname)

        if os.path.isfile(dataname):
            print("Video already analyzed!", dataname)
        else:
            print("Loading ", video)
            vid = VideoWriter(video)
            if len(scale_list) == 0:
                # spatial pyramid can still be useful for reducing jittering and quantization error
                scale_list = [vid.height - 50, vid.height, vid.height + 50]
            if robust_nframes:
                nframes = vid.get_n_frames(robust=True)
                duration = vid.calc_duration(robust=True)
                fps = nframes / duration
            else:
                nframes = len(vid)
                duration = vid.calc_duration(robust=False)
                fps = vid.fps

            nx, ny = vid.dimensions
            print(
                "Duration of video [s]: ",
                round(duration, 2),
                ", recorded with ",
                round(fps, 2),
                "fps!",
            )
            print(
                "Overall # of frames: ",
                nframes,
                " found with (before cropping) frame dimensions: ",
                nx,
                ny,
            )
            start = time.time()

            print("Starting to extract posture")

            # extra data
            PredicteData, nframes = _video_inference(
                test_cfg,
                sess,
                inputs,
                outputs,
                vid,
                nframes,
                int(test_cfg["batch_size"]),
                scale_list=scale_list,
            )

            stop = time.time()

            coords = [0, nx, 0, ny]

            dictionary = {
                "start": start,
                "stop": stop,
                "run_duration": stop - start,
                "Scorer": DLCscorer,
                "DLC-model-config file": test_cfg,
                "fps": fps,
                "batch_size": test_cfg["batch_size"],
                "frame_dimensions": (ny, nx),
                "nframes": nframes,
                "iteration (active-learning)": 0,
                "cropping": False,
                "training set fraction": 70,
                "cropping_parameters": coords,
            }
            metadata = {"data": dictionary}
            print("Saving results in %s..." % (destfolder))

            metadata_path = dataname.split(".h5")[0] + "_meta.pickle"

            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)

            xyz_labs = ["x", "y", "likelihood"]
            scorer = DLCscorer
            keypoint_names = test_cfg["all_joints_names"]
            product = [[scorer], keypoint_names, xyz_labs]
            names = ["scorer", "bodyparts", "coords"]
            columnindex = pd.MultiIndex.from_product(product, names=names)
            imagenames = [k for k in PredicteData.keys() if k != "metadata"]

            data = np.full((len(imagenames), len(columnindex)), np.nan)
            for i, imagename in enumerate(imagenames):
                dict_ = PredicteData[imagename]
                if dict_ == [] or dict_ == [[]]:
                    data[i, 2::3] = 0
                else:
                    keypoints = dict_["coordinates"][0]
                    confidence = dict_["confidence"]
                    temp = np.full((len(keypoints), 3), np.nan)
                    for n, (xy, c) in enumerate(zip(keypoints, confidence)):
                        if xy.size and c.size:
                            temp[n, :2] = xy
                            temp[n, 2] = c
                    data[i] = temp.flatten()
            df = pd.DataFrame(data, columns=columnindex, index=imagenames)
            df.to_hdf(dataname, key="df_with_missing")

    return init_weights, datafiles


def _video_inference_superanimal(
    videos,
    project_name,
    model_name,
    scale_list=[],
    videotype=".mp4",
    video_adapt=False,
    plot_trajectories=True,
    pcutoff=0.1,
    adapt_iterations=1000,
    pseudo_threshold=0.1,
):
    """
    WARNING: This function is an internal utility function and should not be
    called directly. It is designed to be used by deeplabcut.modelzoo.api.video_inference.py

    Makes prediction based on a super animal model. Note right now we only support single animal video inference

    The index of the trained network is specified by parameters in the config file (in particular the variable 'snapshotindex')

    Output: The labels are stored as MultiIndex Pandas Array, which contains the name of the network, body part name, (x, y) label position \n
            in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient Hierarchical Data Format (HDF) \n
            in the same directory, where the video is stored.

    Parameters
    ----------
    videos: list
        A list of strings containing the full paths to videos for analysis or a path to the directory, where all the videos with same extension are stored.

    superanimal_name: str
        The name of the superanimal model. We currently only support "superanimal_quadruped" and "superanimal_topviewmouse"
    scale_list: list
        A list of int containing the target height of the multi scale test time augmentation. By default it uses the original size. Users are advised to try a wide range of scale list when the super model does not give reasonable results

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed. The default is ``.avi``

    video_adapt: bool, optional
        Set True if you want to apply video adaptation to make the resulted video less jittering and better. However, adaptation training takes more time than usual video inference

    plot_trajectories: bool, optional (default=True)
        By default, plot the trajectories of various body parts across the video.

    pcutoff: float, optional
        Keypoints confidence that are under pcutoff will not be shown in the resulted video

    adapt_iterations: int, optional:
        Number of iterations for adaptation training

    pseudo_threshold: float, default 0.1
        Video adaptation only uses predictions that are above pseudo_threshold

    Given a list of scales for spatial pyramid, i.e. [600, 700]

    scale_list = range(600,800,100)

    superanimal_name = 'superanimal_topviewmouse'
    videotype = 'mp4'
    scale_list = [200, 300, 400]
    deeplabcut.video_inference_superanimal(
         video,
         superanimal_name,
         videotype = '.avi',
         scale_list = scale_list,
    )
    >>>
    """
    from deeplabcut.pose_estimation_tensorflow.modelzoo.api import (
        SpatiotemporalAdaptation,
    )

    superanimal_name = project_name + "_" + model_name
    for video in videos:
        modelfolder = Path(video).parent / f"{Path(video).stem}_video_adaptation"
        modelfolder.mkdir(exist_ok=True, parents=True)

        adapter = SpatiotemporalAdaptation(
            video,
            superanimal_name,
            modelfolder=str(modelfolder),
            videotype=video.split(".")[-1],
            scale_list=scale_list,
        )
        if not video_adapt:
            adapter.before_adapt_inference(
                make_video=True, pcutoff=pcutoff, plot_trajectories=plot_trajectories
            )
        else:
            adapter.before_adapt_inference(make_video=False)
            adapter.adaptation_training(
                adapt_iterations=adapt_iterations,
                pseudo_threshold=pseudo_threshold,
            )
            adapter.after_adapt_inference(
                pcutoff=pcutoff,
                plot_trajectories=plot_trajectories,
            )
