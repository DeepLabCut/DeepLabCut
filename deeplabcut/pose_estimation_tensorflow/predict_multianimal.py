"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
import os
import os.path
import time
from pathlib import Path

import numpy as np
from skimage.util import img_as_ubyte
from tqdm import tqdm

from deeplabcut.pose_estimation_tensorflow.core import predict_multianimal as predict
from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal
from deeplabcut.utils.auxfun_videos import VideoWriter


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
):
    """ Helper function for analyzing a video with multiple individuals """

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

        print("Starting to extract posture")
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
            )
        else:
            PredicteData, nframes = GetPoseandCostsS(
                cfg, dlc_cfg, sess, inputs, outputs, vid, nframes,
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
        print("Saving results in %s..." % (destfolder))

        _ = auxfun_multianimal.SaveFullMultiAnimalData(PredicteData, metadata, dataname)


def GetPoseandCostsF(
    cfg, dlc_cfg, sess, inputs, outputs, cap, nframes, batchsize,
):
    """ Batchwise prediction of pose """
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
        if frame is not None:
            frames[batch_ind] = img_as_ubyte(frame)
            inds.append(counter)
            if batch_ind == batchsize - 1:
                D = predict.predict_batched_peaks_and_costs(
                    dlc_cfg, frames, sess, inputs, outputs,
                )
                for ind, data in zip(inds, D):
                    PredicteData["frame" + str(ind).zfill(strwidth)] = data
                batch_ind = 0
                inds.clear()
                batch_num += 1
            else:
                batch_ind += 1
        elif counter >= nframes:
            if batch_ind > 0:
                D = predict.predict_batched_peaks_and_costs(
                    dlc_cfg, frames, sess, inputs, outputs,
                )
                for ind, data in zip(inds, D):
                    PredicteData["frame" + str(ind).zfill(strwidth)] = data
            break
        counter += 1
        pbar.update(1)

    cap.close()
    pbar.close()
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


def GetPoseandCostsS(cfg, dlc_cfg, sess, inputs, outputs, cap, nframes):
    """ Non batch wise pose estimation for video cap."""
    strwidth = int(np.ceil(np.log10(nframes)))  # width for strings
    if cfg["cropping"]:
        cap.set_bbox(cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"])

    PredicteData = {}  # np.zeros((nframes, 3 * len(dlc_cfg['all_joints_names'])))
    pbar = tqdm(total=nframes)
    counter = 0
    while cap.video.isOpened():
        frame = cap.read_frame(crop=cfg["cropping"])
        if frame is not None:
            frame = img_as_ubyte(frame)
            dets = predict.predict_batched_peaks_and_costs(
                dlc_cfg, np.expand_dims(frame, axis=0), sess, inputs, outputs,
            )
            PredicteData["frame" + str(counter).zfill(strwidth)] = dets[0]
        elif counter >= nframes:
            break
        counter += 1
        pbar.update(1)

    pbar.close()
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
