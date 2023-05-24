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

import os
import pickle
import shelve
import time
from pathlib import Path

import numpy as np
from skimage.color import rgba2rgb
from skimage.util import img_as_ubyte
from tqdm import tqdm

from deeplabcut.pose_estimation_tensorflow.core import predict_multianimal as predict
from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal
from deeplabcut.utils.auxfun_videos import VideoWriter
import pickle


def extract_bpt_feature_from_video(
    video,
    DLCscorer,
    trainFraction,
    cfg,
    dlc_cfg,
    sess,
    inputs,
    outputs,
    extra_dict,
    destfolder=None,
    robust_nframes=False,
):
    print("Starting to analyze % ", video)
    vname = Path(video).stem
    videofolder = str(Path(video).parents[0])
    if destfolder is None:
        destfolder = videofolder
    auxiliaryfunctions.attempttomakefolder(destfolder)
    dataname = os.path.join(destfolder, vname + DLCscorer + ".h5")

    assemble_filename = dataname.split(".h5")[0] + "_assemblies.pickle"

    feature_dict = shelve.open(
        dataname.split(".h5")[0] + "_bpt_features.pickle",
        protocol=pickle.DEFAULT_PROTOCOL,
    )

    with open(assemble_filename, "rb") as f:
        assemblies = pickle.load(f)
        print("Loading ", video)
        vid = VideoWriter(video)
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
        if int(dlc_cfg["batch_size"]) > 1:
            # for multi animal, seems only this is used
            PredicteData, nframes = GetPoseandCostsF_from_assemblies(
                cfg,
                dlc_cfg,
                sess,
                inputs,
                outputs,
                vid,
                nframes,
                int(dlc_cfg["batch_size"]),
                assemblies,
                feature_dict,
                extra_dict,
            )
        else:
            raise NotImplementedError(
                "Not implemented yet, please raise an GitHub issue if you need this."
            )


def AnalyzeMultiAnimalVideo(
    video,
    DLCscorer,
    trainFraction,
    cfg,
    dlc_cfg,
    sess,
    inputs,
    outputs,
    destfolder=None,
    robust_nframes=False,
    use_shelve=False,
):
    """Helper function for analyzing a video with multiple individuals"""

    print("Starting to analyze % ", video)
    vname = Path(video).stem
    videofolder = str(Path(video).parents[0])
    if destfolder is None:
        destfolder = videofolder
    auxiliaryfunctions.attempttomakefolder(destfolder)
    dataname = os.path.join(destfolder, vname + DLCscorer + ".h5")

    if os.path.isfile(dataname.split(".h5")[0] + "_full.pickle"):
        print("Video already analyzed!", dataname)
    else:
        print("Loading ", video)
        vid = VideoWriter(video)
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

        print(
            "Starting to extract posture from the video(s) with batchsize:",
            dlc_cfg["batch_size"],
        )
        if use_shelve:
            shelf_path = dataname.split(".h5")[0] + "_full.pickle"
        else:
            shelf_path = ""
        if int(dlc_cfg["batch_size"]) > 1:
            PredicteData, nframes = GetPoseandCostsF(
                cfg,
                dlc_cfg,
                sess,
                inputs,
                outputs,
                vid,
                nframes,
                int(dlc_cfg["batch_size"]),
                shelf_path,
            )
        else:
            PredicteData, nframes = GetPoseandCostsS(
                cfg,
                dlc_cfg,
                sess,
                inputs,
                outputs,
                vid,
                nframes,
                shelf_path,
            )

        stop = time.time()

        if cfg["cropping"] == True:
            coords = [cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"]]
        else:
            coords = [0, nx, 0, ny]

        dictionary = {
            "start": start,
            "stop": stop,
            "run_duration": stop - start,
            "Scorer": DLCscorer,
            "DLC-model-config file": dlc_cfg,
            "fps": fps,
            "batch_size": dlc_cfg["batch_size"],
            "frame_dimensions": (ny, nx),
            "nframes": nframes,
            "iteration (active-learning)": cfg["iteration"],
            "training set fraction": trainFraction,
            "cropping": cfg["cropping"],
            "cropping_parameters": coords,
        }
        metadata = {"data": dictionary}
        print("Video Analyzed. Saving results in %s..." % (destfolder))

        if use_shelve:
            metadata_path = dataname.split(".h5")[0] + "_meta.pickle"
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)
        else:
            _ = auxfun_multianimal.SaveFullMultiAnimalData(
                PredicteData, metadata, dataname
            )


def _get_features_dict(raw_coords, features, stride):
    from deeplabcut.pose_tracking_pytorch import (
        load_features_from_coord,
        convert_coord_from_img_space_to_feature_space,
    )

    coords_img_space = np.array(
        [coord[:, :2] for coord in raw_coords]
    )  # only first two columns are useful

    coords_feature_space = convert_coord_from_img_space_to_feature_space(
        coords_img_space,
        stride,
    )

    bpt_features = load_features_from_coord(
        features.astype(np.float16), coords_feature_space
    )
    return {"features": bpt_features, "coordinates": coords_img_space}


def GetPoseandCostsF_from_assemblies(
    cfg,
    dlc_cfg,
    sess,
    inputs,
    outputs,
    cap,
    nframes,
    batchsize,
    assemblies,
    feature_dict,
    extra_dict,
):

    """Batchwise prediction of pose"""
    strwidth = int(np.ceil(np.log10(nframes)))  # width for strings
    batch_ind = 0  # keeps track of which image within a batch should be written to
    batch_num = 0  # keeps track of which batch you are at
    if cfg["cropping"]:
        cap.set_bbox(cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"])
    nx, ny = cap.dimensions

    frames = np.empty(
        (batchsize, ny, nx, 3), dtype="ubyte"
    )  # this keeps all frames in a batch
    pbar = tqdm(total=nframes)
    counter = 0
    inds = []

    PredicteData = {}

    while cap.video.isOpened():
        frame = cap.read_frame(crop=cfg["cropping"])
        key = "frame" + str(counter).zfill(strwidth)
        if frame is not None:
            # Avoid overwriting data already on the shelf
            if key in feature_dict:
                continue

            frame = img_as_ubyte(frame)
            if frame.shape[-1] == 4:
                frame = rgba2rgb(frame)
            frames[batch_ind] = frame
            inds.append(counter)

            if batch_ind == batchsize - 1:

                preds = predict.predict_batched_peaks_and_costs(
                    dlc_cfg, frames, sess, inputs, outputs, extra_dict=extra_dict
                )
                if not preds:
                    continue

                D, features = preds
                for i, (ind, data) in enumerate(zip(inds, D)):
                    PredicteData["frame" + str(ind).zfill(strwidth)] = data
                    raw_coords = assemblies.get(ind)
                    if raw_coords is None:
                        continue
                    fname = "frame" + str(ind).zfill(strwidth)
                    feature_dict[fname] = _get_features_dict(
                        raw_coords,
                        features[i],
                        dlc_cfg["stride"],
                    )

                batch_ind = 0
                inds.clear()
                batch_num += 1
            else:
                batch_ind += 1
        elif counter >= nframes:
            if batch_ind > 0:

                preds = predict.predict_batched_peaks_and_costs(
                    dlc_cfg, frames, sess, inputs, outputs, extra_dict=extra_dict
                )
                if not preds:
                    continue

                D, features = preds
                for i, (ind, data) in enumerate(zip(inds, D)):
                    PredicteData["frame" + str(ind).zfill(strwidth)] = data
                    raw_coords = assemblies.get(ind)
                    if raw_coords is None:
                        continue
                    fname = "frame" + str(ind).zfill(strwidth)
                    feature_dict[fname] = _get_features_dict(
                        raw_coords,
                        features[i],
                        dlc_cfg["stride"],
                    )

            break
        counter += 1
        pbar.update(1)

    cap.close()
    pbar.close()
    feature_dict.close()
    PredicteData["metadata"] = {
        "nms radius": dlc_cfg["nmsradius"],
        "minimal confidence": dlc_cfg["minconfidence"],
        "sigma": dlc_cfg.get("sigma", 1),
        "PAFgraph": dlc_cfg["partaffinityfield_graph"],
        "PAFinds": dlc_cfg.get(
            "paf_best", np.arange(len(dlc_cfg["partaffinityfield_graph"]))
        ),
        "all_joints": [[i] for i in range(len(dlc_cfg["all_joints"]))],
        "all_joints_names": [
            dlc_cfg["all_joints_names"][i] for i in range(len(dlc_cfg["all_joints"]))
        ],
        "nframes": nframes,
    }
    return PredicteData, nframes


def GetPoseandCostsF(
    cfg,
    dlc_cfg,
    sess,
    inputs,
    outputs,
    cap,
    nframes,
    batchsize,
    shelf_path,
):
    """Batchwise prediction of pose"""
    strwidth = int(np.ceil(np.log10(nframes)))  # width for strings
    batch_ind = 0  # keeps track of which image within a batch should be written to
    batch_num = 0  # keeps track of which batch you are at
    if cfg["cropping"]:
        cap.set_bbox(cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"])
    nx, ny = cap.dimensions

    frames = np.empty(
        (batchsize, ny, nx, 3), dtype="ubyte"
    )  # this keeps all frames in a batch
    pbar = tqdm(total=nframes)
    counter = 0
    inds = []

    if shelf_path:
        db = shelve.open(
            shelf_path,
            protocol=pickle.DEFAULT_PROTOCOL,
        )
    else:
        db = dict()
    db["metadata"] = {
        "nms radius": dlc_cfg["nmsradius"],
        "minimal confidence": dlc_cfg["minconfidence"],
        "sigma": dlc_cfg.get("sigma", 1),
        "PAFgraph": dlc_cfg["partaffinityfield_graph"],
        "PAFinds": dlc_cfg.get(
            "paf_best", np.arange(len(dlc_cfg["partaffinityfield_graph"]))
        ),
        "all_joints": [[i] for i in range(len(dlc_cfg["all_joints"]))],
        "all_joints_names": [
            dlc_cfg["all_joints_names"][i] for i in range(len(dlc_cfg["all_joints"]))
        ],
        "nframes": nframes,
    }
    while cap.video.isOpened():
        frame = cap.read_frame(crop=cfg["cropping"])
        key = "frame" + str(counter).zfill(strwidth)
        if frame is not None:
            # Avoid overwriting data already on the shelf
            if isinstance(db, shelve.Shelf) and key in db:
                continue
            frame = img_as_ubyte(frame)
            if frame.shape[-1] == 4:
                frame = rgba2rgb(frame)
            frames[batch_ind] = frame
            inds.append(counter)
            if batch_ind == batchsize - 1:
                D = predict.predict_batched_peaks_and_costs(
                    dlc_cfg,
                    frames,
                    sess,
                    inputs,
                    outputs,
                )
                for ind, data in zip(inds, D):
                    db["frame" + str(ind).zfill(strwidth)] = data
                del D
                batch_ind = 0
                inds.clear()
                batch_num += 1
            else:
                batch_ind += 1
        elif counter >= nframes:
            if batch_ind > 0:
                D = predict.predict_batched_peaks_and_costs(
                    dlc_cfg,
                    frames,
                    sess,
                    inputs,
                    outputs,
                )
                for ind, data in zip(inds, D):
                    db["frame" + str(ind).zfill(strwidth)] = data
                del D
            break
        counter += 1
        pbar.update(1)

    cap.close()
    pbar.close()
    try:
        db.close()
    except AttributeError:
        pass
    return db, nframes


def GetPoseandCostsS(cfg, dlc_cfg, sess, inputs, outputs, cap, nframes, shelf_path):
    """Non batch wise pose estimation for video cap."""
    strwidth = int(np.ceil(np.log10(nframes)))  # width for strings
    if cfg["cropping"]:
        cap.set_bbox(cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"])

    if shelf_path:
        db = shelve.open(
            shelf_path,
            protocol=pickle.DEFAULT_PROTOCOL,
        )
    else:
        db = dict()
    db["metadata"] = {
        "nms radius": dlc_cfg["nmsradius"],
        "minimal confidence": dlc_cfg["minconfidence"],
        "sigma": dlc_cfg.get("sigma", 1),
        "PAFgraph": dlc_cfg["partaffinityfield_graph"],
        "PAFinds": dlc_cfg.get(
            "paf_best", np.arange(len(dlc_cfg["partaffinityfield_graph"]))
        ),
        "all_joints": [[i] for i in range(len(dlc_cfg["all_joints"]))],
        "all_joints_names": [
            dlc_cfg["all_joints_names"][i] for i in range(len(dlc_cfg["all_joints"]))
        ],
        "nframes": nframes,
    }
    pbar = tqdm(total=nframes)
    counter = 0
    while cap.video.isOpened():
        frame = cap.read_frame(crop=cfg["cropping"])
        key = "frame" + str(counter).zfill(strwidth)
        if frame is not None:
            # Avoid overwriting data already on the shelf
            if isinstance(db, shelve.Shelf) and key in db:
                continue
            frame = img_as_ubyte(frame)
            if frame.shape[-1] == 4:
                frame = rgba2rgb(frame)
            dets = predict.predict_batched_peaks_and_costs(
                dlc_cfg,
                np.expand_dims(frame, axis=0),
                sess,
                inputs,
                outputs,
            )
            db[key] = dets[0]
            del dets
        elif counter >= nframes:
            break
        counter += 1
        pbar.update(1)

    pbar.close()
    try:
        db.close()
    except AttributeError:
        pass
    return db, nframes
