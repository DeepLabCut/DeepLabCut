"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

####################################################
# Dependencies
####################################################

import argparse
import os
import os.path
import pickle
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from skimage.util import img_as_ubyte
from tqdm import tqdm

from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.core import predict
from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils, trackingutils
from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal


####################################################
# Loading data, and defining model folder
####################################################


def analyze_videos(
    config,
    videos,
    videotype="avi",
    shuffle=1,
    trainingsetindex=0,
    gputouse=None,
    save_as_csv=False,
    destfolder=None,
    batchsize=None,
    cropping=None,
    TFGPUinference=True,
    dynamic=(False, 0.5, 10),
    modelprefix="",
    robust_nframes=False,
    allow_growth=False
):
    """
    Makes prediction based on a trained network. The index of the trained network is specified by parameters in the config file (in particular the variable 'snapshotindex')

    Output: The labels are stored as MultiIndex Pandas Array, which contains the name of the network, body part name, (x, y) label position \n
            in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient Hierarchical Data Format (HDF) \n
            in the same directory, where the video is stored. However, if the flag save_as_csv is set to True, the data can also be exported in \n
            comma-separated values format (.csv), which in turn can be imported in many programs, such as MATLAB, R, Prism, etc.

    Parameters
    ----------
    config: string
        Full path of the config.yaml file as a string.

    videos: list
        A list of strings containing the full paths to videos for analysis or a path to the directory, where all the videos with same extension are stored.

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed. The default is ``.avi``

    shuffle: int, optional
        An integer specifying the shuffle index of the training dataset used for training the network. The default is 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU put None.
    See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    save_as_csv: bool, optional
        Saves the predictions in a .csv file. The default is ``False``; if provided it must be either ``True`` or ``False``

    destfolder: string, optional
        Specifies the destination folder for analysis data (default is the path of the video). Note that for subsequent analysis this
        folder also needs to be passed.

    batchsize: int, default from pose_cfg.yaml
        Change batch size for inference; if given overwrites value in pose_cfg.yaml

    cropping: list, optional (default=None)
        List of cropping coordinates as [x1, x2, y1, y2].
        Note that the same cropping parameters will then be used for all videos.
        If different video crops are desired, run 'analyze_videos' on individual videos
        with the corresponding cropping coordinates.

    TFGPUinference: bool, default: True
        Perform inference on GPU with TensorFlow code. Introduced in "Pretraining boosts out-of-domain robustness for pose estimation" by
        Alexander Mathis, Mert Yüksekgönül, Byron Rogers, Matthias Bethge, Mackenzie W. Mathis Source: https://arxiv.org/abs/1909.11229

    dynamic: triple containing (state, detectiontreshold, margin)
        If the state is true, then dynamic cropping will be performed. That means that if an object is detected (i.e. any body part > detectiontreshold),
        then object boundaries are computed according to the smallest/largest x position and smallest/largest y position of all body parts. This  window is
        expanded by the margin and from then on only the posture within this crop is analyzed (until the object is lost, i.e. <detectiontreshold). The
        current position is utilized for updating the crop window for the next frame (this is why the margin is important and should be set large
        enough given the movement of the animal).

    robust_nframes: bool, optional (default=False)
        Evaluate a video's number of frames in a robust manner.
        This option is slower (as the whole video is read frame-by-frame),
        but does not rely on metadata, hence its robustness against file corruption.

    allow_growth: bool, default false.
        For some smaller GPUs the memory issues happen. If true, the memory allocator does not pre-allocate the entire specified
        GPU memory region, instead starting small and growing as needed. See issue: https://forum.image.sc/t/how-to-stop-running-out-of-vram/30551/2

    Examples
    --------

    Windows example for analyzing 1 video
    >>> deeplabcut.analyze_videos('C:\\myproject\\reaching-task\\config.yaml',['C:\\yourusername\\rig-95\\Videos\\reachingvideo1.avi'])
    --------

    If you want to analyze only 1 video
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/reachingvideo1.avi'])
    --------

    If you want to analyze all videos of type avi in a folder:
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos'],videotype='.avi')
    --------

    If you want to analyze multiple videos
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/reachingvideo1.avi','/analysis/project/videos/reachingvideo2.avi'])
    --------

    If you want to analyze multiple videos with shuffle = 2
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/reachingvideo1.avi','/analysis/project/videos/reachingvideo2.avi'],shuffle=2)

    --------
    If you want to analyze multiple videos with shuffle = 2 and save results as an additional csv file too
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/reachingvideo1.avi','/analysis/project/videos/reachingvideo2.avi'],shuffle=2,save_as_csv=True)
    --------

    """
    if "TF_CUDNN_USE_AUTOTUNE" in os.environ:
        del os.environ["TF_CUDNN_USE_AUTOTUNE"]  # was potentially set during training

    if gputouse is not None:  # gpu selection
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gputouse)

    tf.compat.v1.reset_default_graph()
    start_path = os.getcwd()  # record cwd to return to this directory in the end

    cfg = auxiliaryfunctions.read_config(config)
    trainFraction = cfg["TrainingFraction"][trainingsetindex]

    if cropping is not None:
        cfg["cropping"] = True
        cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"] = cropping
        print("Overwriting cropping parameters:", cropping)
        print("These are used for all videos, but won't be save to the cfg file.")

    modelfolder = os.path.join(
        cfg["project_path"],
        str(
            auxiliaryfunctions.GetModelFolder(
                trainFraction, shuffle, cfg, modelprefix=modelprefix
            )
        ),
    )
    path_test_config = Path(modelfolder) / "test" / "pose_cfg.yaml"
    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError(
            "It seems the model for shuffle %s and trainFraction %s does not exist."
            % (shuffle, trainFraction)
        )

    # Check which snapshots are available and sort them by # iterations
    try:
        Snapshots = np.array(
            [
                fn.split(".")[0]
                for fn in os.listdir(os.path.join(modelfolder, "train"))
                if "index" in fn
            ]
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."
            % (shuffle, shuffle)
        )

    if cfg["snapshotindex"] == "all":
        print(
            "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!"
        )
        snapshotindex = -1
    else:
        snapshotindex = cfg["snapshotindex"]

    increasing_indices = np.argsort([int(m.split("-")[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]

    print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)

    ##################################################
    # Load and setup CNN part detector
    ##################################################

    # Check if data already was generated:
    dlc_cfg["init_weights"] = os.path.join(
        modelfolder, "train", Snapshots[snapshotindex]
    )
    trainingsiterations = (dlc_cfg["init_weights"].split(os.sep)[-1]).split("-")[-1]
    # Update number of output and batchsize
    dlc_cfg["num_outputs"] = cfg.get("num_outputs", dlc_cfg.get("num_outputs", 1))

    if batchsize == None:
        # update batchsize (based on parameters in config.yaml)
        dlc_cfg["batch_size"] = cfg["batch_size"]
    else:
        dlc_cfg["batch_size"] = batchsize
        cfg["batch_size"] = batchsize

    if "multi-animal" in dlc_cfg["dataset_type"]:
        dynamic = (False, 0.5, 10)  # setting dynamic mode to false
        TFGPUinference = False

    if dynamic[0]:  # state=true
        # (state,detectiontreshold,margin)=dynamic
        print("Starting analysis in dynamic cropping mode with parameters:", dynamic)
        dlc_cfg["num_outputs"] = 1
        TFGPUinference = False
        dlc_cfg["batch_size"] = 1
        print(
            "Switching batchsize to 1, num_outputs (per animal) to 1 and TFGPUinference to False (all these features are not supported in this mode)."
        )

    # Name for scorer:
    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.GetScorerName(
        cfg,
        shuffle,
        trainFraction,
        trainingsiterations=trainingsiterations,
        modelprefix=modelprefix,
    )
    if dlc_cfg["num_outputs"] > 1:
        if TFGPUinference:
            print(
                "Switching to numpy-based keypoint extraction code, as multiple point extraction is not supported by TF code currently."
            )
            TFGPUinference = False
        print("Extracting ", dlc_cfg["num_outputs"], "instances per bodypart")
        xyz_labs_orig = ["x", "y", "likelihood"]
        suffix = [str(s + 1) for s in range(dlc_cfg["num_outputs"])]
        suffix[0] = ""  # first one has empty suffix for backwards compatibility
        xyz_labs = [x + s for s in suffix for x in xyz_labs_orig]
    else:
        xyz_labs = ["x", "y", "likelihood"]

    if TFGPUinference:
        sess, inputs, outputs = predict.setup_GPUpose_prediction(dlc_cfg,allow_growth=allow_growth)
    else:
        sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg,allow_growth=allow_growth)

    pdindex = pd.MultiIndex.from_product(
        [[DLCscorer], dlc_cfg["all_joints_names"], xyz_labs],
        names=["scorer", "bodyparts", "coords"],
    )

    ##################################################
    # Looping over videos
    ##################################################
    Videos = auxiliaryfunctions.Getlistofvideos(videos, videotype)
    if len(Videos) > 0:
        if "multi-animal" in dlc_cfg["dataset_type"]:
            from deeplabcut.pose_estimation_tensorflow.predict_multianimal import (
                AnalyzeMultiAnimalVideo,
            )
            for video in Videos:
                AnalyzeMultiAnimalVideo(
                    video,
                    DLCscorer,
                    trainFraction,
                    cfg,
                    dlc_cfg,
                    sess,
                    inputs,
                    outputs,
                    destfolder,
                    robust_nframes=robust_nframes,
                )
        else:
            for video in Videos:
                DLCscorer = AnalyzeVideo(
                    video,
                    DLCscorer,
                    DLCscorerlegacy,
                    trainFraction,
                    cfg,
                    dlc_cfg,
                    sess,
                    inputs,
                    outputs,
                    pdindex,
                    save_as_csv,
                    destfolder,
                    TFGPUinference,
                    dynamic,
                )

        os.chdir(str(start_path))
        if "multi-animal" in dlc_cfg["dataset_type"]:
            print(
                "The videos are analyzed. Time to assemble animals and track 'em... \n Call 'create_video_with_all_detections' to check multi-animal detection quality before tracking."
            )
            print(
                "If the tracking is not satisfactory for some videos, consider expanding the training set. You can use the function 'extract_outlier_frames' to extract a few representative outlier frames."
            )
        else:
            print(
                "The videos are analyzed. Now your research can truly start! \n You can create labeled videos with 'create_labeled_video'"
            )
            print(
                "If the tracking is not satisfactory for some videos, consider expanding the training set. You can use the function 'extract_outlier_frames' to extract a few representative outlier frames."
            )
        return DLCscorer  # note: this is either DLCscorer or DLCscorerlegacy depending on what was used!
    else:
        print("No video(s) were found. Please check your paths and/or 'video_type'.")
        return DLCscorer


def checkcropping(cfg, cap):
    print(
        "Cropping based on the x1 = %s x2 = %s y1 = %s y2 = %s. You can adjust the cropping coordinates in the config.yaml file."
        % (cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"])
    )
    nx = cfg["x2"] - cfg["x1"]
    ny = cfg["y2"] - cfg["y1"]
    if nx > 0 and ny > 0:
        pass
    else:
        raise Exception("Please check the order of cropping parameter!")
    if (
        cfg["x1"] >= 0
        and cfg["x2"] < int(cap.get(3) + 1)
        and cfg["y1"] >= 0
        and cfg["y2"] < int(cap.get(4) + 1)
    ):
        pass  # good cropping box
    else:
        raise Exception("Please check the boundary of cropping!")
    return int(ny), int(nx)


def GetPoseF(cfg, dlc_cfg, sess, inputs, outputs, cap, nframes, batchsize):
    """ Batchwise prediction of pose """
    PredictedData = np.zeros(
        (nframes, dlc_cfg["num_outputs"] * 3 * len(dlc_cfg["all_joints_names"]))
    )
    batch_ind = 0  # keeps track of which image within a batch should be written to
    batch_num = 0  # keeps track of which batch you are at
    ny, nx = int(cap.get(4)), int(cap.get(3))
    if cfg["cropping"]:
        ny, nx = checkcropping(cfg, cap)

    frames = np.empty(
        (batchsize, ny, nx, 3), dtype="ubyte"
    )  # this keeps all frames in a batch
    pbar = tqdm(total=nframes)
    counter = 0
    step = max(10, int(nframes / 100))
    inds = []
    while cap.isOpened():
        if counter % step == 0:
            pbar.update(step)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if cfg["cropping"]:
                frames[batch_ind] = img_as_ubyte(
                    frame[cfg["y1"] : cfg["y2"], cfg["x1"] : cfg["x2"]]
                )
            else:
                frames[batch_ind] = img_as_ubyte(frame)
            inds.append(counter)
            if batch_ind == batchsize - 1:
                pose = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs)
                PredictedData[inds] = pose
                batch_ind = 0
                inds.clear()
                batch_num += 1
            else:
                batch_ind += 1
        elif counter >= nframes:
            if batch_ind > 0:
                pose = predict.getposeNP(
                    frames, dlc_cfg, sess, inputs, outputs
                )  # process the whole batch (some frames might be from previous batch!)
                PredictedData[inds[:batch_ind]] = pose[:batch_ind]
            break
        counter += 1

    pbar.close()
    return PredictedData, nframes


def GetPoseS(cfg, dlc_cfg, sess, inputs, outputs, cap, nframes):
    """ Non batch wise pose estimation for video cap."""
    if cfg["cropping"]:
        ny, nx = checkcropping(cfg, cap)

    PredictedData = np.zeros(
        (nframes, dlc_cfg["num_outputs"] * 3 * len(dlc_cfg["all_joints_names"]))
    )
    pbar = tqdm(total=nframes)
    counter = 0
    step = max(10, int(nframes / 100))
    while cap.isOpened():
        if counter % step == 0:
            pbar.update(step)

        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if cfg["cropping"]:
                frame = img_as_ubyte(
                    frame[cfg["y1"] : cfg["y2"], cfg["x1"] : cfg["x2"]]
                )
            else:
                frame = img_as_ubyte(frame)
            pose = predict.getpose(frame, dlc_cfg, sess, inputs, outputs)
            PredictedData[
                counter, :
            ] = (
                pose.flatten()
            )  # NOTE: thereby cfg['all_joints_names'] should be same order as bodyparts!
        elif counter >= nframes:
            break
        counter += 1

    pbar.close()
    return PredictedData, nframes


def GetPoseS_GTF(cfg, dlc_cfg, sess, inputs, outputs, cap, nframes):
    """ Non batch wise pose estimation for video cap."""
    if cfg["cropping"]:
        ny, nx = checkcropping(cfg, cap)

    pose_tensor = predict.extract_GPUprediction(
        outputs, dlc_cfg
    )  # extract_output_tensor(outputs, dlc_cfg)
    PredictedData = np.zeros((nframes, 3 * len(dlc_cfg["all_joints_names"])))
    pbar = tqdm(total=nframes)
    counter = 0
    step = max(10, int(nframes / 100))
    while cap.isOpened():
        if counter % step == 0:
            pbar.update(step)

        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if cfg["cropping"]:
                frame = img_as_ubyte(
                    frame[cfg["y1"] : cfg["y2"], cfg["x1"] : cfg["x2"]]
                )
            else:
                frame = img_as_ubyte(frame)

            pose = sess.run(
                pose_tensor,
                feed_dict={inputs: np.expand_dims(frame, axis=0).astype(float)},
            )
            pose[:, [0, 1, 2]] = pose[:, [1, 0, 2]]
            # pose = predict.getpose(frame, dlc_cfg, sess, inputs, outputs)
            PredictedData[
                counter, :
            ] = (
                pose.flatten()
            )  # NOTE: thereby cfg['all_joints_names'] should be same order as bodyparts!
        elif counter >= nframes:
            break
        counter += 1

    pbar.close()
    return PredictedData, nframes


def GetPoseF_GTF(cfg, dlc_cfg, sess, inputs, outputs, cap, nframes, batchsize):
    """ Batchwise prediction of pose """
    PredictedData = np.zeros((nframes, 3 * len(dlc_cfg["all_joints_names"])))
    batch_ind = 0  # keeps track of which image within a batch should be written to
    batch_num = 0  # keeps track of which batch you are at
    ny, nx = int(cap.get(4)), int(cap.get(3))
    if cfg["cropping"]:
        ny, nx = checkcropping(cfg, cap)

    pose_tensor = predict.extract_GPUprediction(
        outputs, dlc_cfg
    )  # extract_output_tensor(outputs, dlc_cfg)
    frames = np.empty(
        (batchsize, ny, nx, 3), dtype="ubyte"
    )  # this keeps all frames in a batch
    pbar = tqdm(total=nframes)
    counter = 0
    step = max(10, int(nframes / 100))
    inds = []
    while cap.isOpened():
        if counter % step == 0:
            pbar.update(step)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if cfg["cropping"]:
                frames[batch_ind] = img_as_ubyte(
                    frame[cfg["y1"] : cfg["y2"], cfg["x1"] : cfg["x2"]]
                )
            else:
                frames[batch_ind] = img_as_ubyte(frame)
            inds.append(counter)
            if batch_ind == batchsize - 1:
                # pose = predict.getposeNP(frames,dlc_cfg, sess, inputs, outputs)
                pose = sess.run(pose_tensor, feed_dict={inputs: frames})
                pose[:, [0, 1, 2]] = pose[
                    :, [1, 0, 2]
                ]  # change order to have x,y,confidence
                pose = np.reshape(
                    pose, (batchsize, -1)
                )  # bring into batchsize times x,y,conf etc.
                PredictedData[inds] = pose
                batch_ind = 0
                inds.clear()
                batch_num += 1
            else:
                batch_ind += 1
        elif counter >= nframes:
            if batch_ind > 0:
                # pose = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs) #process the whole batch (some frames might be from previous batch!)
                pose = sess.run(pose_tensor, feed_dict={inputs: frames})
                pose[:, [0, 1, 2]] = pose[:, [1, 0, 2]]
                pose = np.reshape(pose, (batchsize, -1))
                PredictedData[inds[:batch_ind]] = pose[:batch_ind]
            break
        counter += 1

    pbar.close()
    return PredictedData, nframes


def getboundingbox(x, y, nx, ny, margin):
    x1 = max([0, int(np.amin(x)) - margin])
    x2 = min([nx, int(np.amax(x)) + margin])
    y1 = max([0, int(np.amin(y)) - margin])
    y2 = min([ny, int(np.amax(y)) + margin])
    return x1, x2, y1, y2


def GetPoseDynamic(
    cfg, dlc_cfg, sess, inputs, outputs, cap, nframes, detectiontreshold, margin
):
    """ Non batch wise pose estimation for video cap by dynamically cropping around previously detected parts."""
    if cfg["cropping"]:
        ny, nx = checkcropping(cfg, cap)
    else:
        ny, nx = (int(cap.get(4)), int(cap.get(3)))
    x1, x2, y1, y2 = 0, nx, 0, ny
    detected = False
    # TODO: perform detection on resized image (For speed)

    PredictedData = np.zeros((nframes, 3 * len(dlc_cfg["all_joints_names"])))
    pbar = tqdm(total=nframes)
    counter = 0
    step = max(10, int(nframes / 100))
    while cap.isOpened():
        if counter % step == 0:
            pbar.update(step)

        ret, frame = cap.read()
        if ret:
            # print(counter,x1,x2,y1,y2,detected)
            originalframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if cfg["cropping"]:
                frame = img_as_ubyte(
                    originalframe[cfg["y1"] : cfg["y2"], cfg["x1"] : cfg["x2"]]
                )[y1:y2, x1:x2]
            else:
                frame = img_as_ubyte(originalframe[y1:y2, x1:x2])

            pose = predict.getpose(frame, dlc_cfg, sess, inputs, outputs).flatten()
            detection = np.any(pose[2::3] > detectiontreshold)  # is anything detected?
            if detection:
                pose[0::3], pose[1::3] = (
                    pose[0::3] + x1,
                    pose[1::3] + y1,
                )  # offset according to last bounding box
                x1, x2, y1, y2 = getboundingbox(
                    pose[0::3], pose[1::3], nx, ny, margin
                )  # coordinates for next iteration
                if not detected:
                    detected = True  # object detected
            else:
                if (
                    detected and (x1 + y1 + y2 - ny + x2 - nx) != 0
                ):  # was detected in last frame and dyn. cropping was performed >> but object lost in cropped variant >> re-run on full frame!
                    # print("looking again, lost!")
                    if cfg["cropping"]:
                        frame = img_as_ubyte(
                            originalframe[cfg["y1"] : cfg["y2"], cfg["x1"] : cfg["x2"]]
                        )
                    else:
                        frame = img_as_ubyte(originalframe)
                    pose = predict.getpose(
                        frame, dlc_cfg, sess, inputs, outputs
                    ).flatten()  # no offset is necessary

                x0, y0 = x1, y1
                x1, x2, y1, y2 = 0, nx, 0, ny
                detected = False

            PredictedData[counter, :] = pose
        elif counter >= nframes:
            break
        counter += 1

    pbar.close()
    return PredictedData, nframes


def AnalyzeVideo(
    video,
    DLCscorer,
    DLCscorerlegacy,
    trainFraction,
    cfg,
    dlc_cfg,
    sess,
    inputs,
    outputs,
    pdindex,
    save_as_csv,
    destfolder=None,
    TFGPUinference=True,
    dynamic=(False, 0.5, 10),
):
    """ Helper function for analyzing a video. """
    print("Starting to analyze % ", video)

    if destfolder is None:
        destfolder = str(Path(video).parents[0])
    auxiliaryfunctions.attempttomakefolder(destfolder)
    vname = Path(video).stem
    try:
        _ = auxiliaryfunctions.load_analyzed_data(destfolder, vname, DLCscorer)
    except FileNotFoundError:
        print("Loading ", video)
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise IOError(
                "Video could not be opened. Please check that the the file integrity."
            )
        fps = cap.get(
            5
        )  # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
        nframes = int(cap.get(7))
        duration = nframes * 1.0 / fps
        size = (int(cap.get(4)), int(cap.get(3)))

        ny, nx = size
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

        dynamic_analysis_state, detectiontreshold, margin = dynamic
        start = time.time()
        print("Starting to extract posture")
        if dynamic_analysis_state:
            PredictedData, nframes = GetPoseDynamic(
                cfg,
                dlc_cfg,
                sess,
                inputs,
                outputs,
                cap,
                nframes,
                detectiontreshold,
                margin,
            )
            # GetPoseF_GTF(cfg,dlc_cfg, sess, inputs, outputs,cap,nframes,int(dlc_cfg["batch_size"]))
        else:
            if int(dlc_cfg["batch_size"]) > 1:
                if TFGPUinference:
                    PredictedData, nframes = GetPoseF_GTF(
                        cfg,
                        dlc_cfg,
                        sess,
                        inputs,
                        outputs,
                        cap,
                        nframes,
                        int(dlc_cfg["batch_size"]),
                    )
                else:
                    PredictedData, nframes = GetPoseF(
                        cfg,
                        dlc_cfg,
                        sess,
                        inputs,
                        outputs,
                        cap,
                        nframes,
                        int(dlc_cfg["batch_size"]),
                    )
            else:
                if TFGPUinference:
                    PredictedData, nframes = GetPoseS_GTF(
                        cfg, dlc_cfg, sess, inputs, outputs, cap, nframes
                    )
                else:
                    PredictedData, nframes = GetPoseS(
                        cfg, dlc_cfg, sess, inputs, outputs, cap, nframes
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
            "cropping_parameters": coords
            # "gpu_info": device_lib.list_local_devices()
        }
        metadata = {"data": dictionary}

        print(f"Saving results in {destfolder}...")
        dataname = os.path.join(destfolder, vname + DLCscorer + ".h5")
        auxiliaryfunctions.SaveData(
            PredictedData[:nframes, :],
            metadata,
            dataname,
            pdindex,
            range(nframes),
            save_as_csv,
        )
    finally:
        return DLCscorer


def GetPosesofFrames(
    cfg, dlc_cfg, sess, inputs, outputs, directory, framelist, nframes, batchsize, rgb
):
    """ Batchwise prediction of pose for frame list in directory"""
    # from skimage.io import imread
    from deeplabcut.utils.auxfun_videos import imread

    print("Starting to extract posture")
    if rgb:
        im = imread(os.path.join(directory, framelist[0]), mode="RGB")
    else:
        im = imread(os.path.join(directory, framelist[0]))

    ny, nx, nc = np.shape(im)
    print(
        "Overall # of frames: ",
        nframes,
        " found with (before cropping) frame dimensions: ",
        nx,
        ny,
    )

    PredictedData = np.zeros(
        (nframes, dlc_cfg["num_outputs"] * 3 * len(dlc_cfg["all_joints_names"]))
    )
    batch_ind = 0  # keeps track of which image within a batch should be written to
    batch_num = 0  # keeps track of which batch you are at
    if cfg["cropping"]:
        print(
            "Cropping based on the x1 = %s x2 = %s y1 = %s y2 = %s. You can adjust the cropping coordinates in the config.yaml file."
            % (cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"])
        )
        nx, ny = cfg["x2"] - cfg["x1"], cfg["y2"] - cfg["y1"]
        if nx > 0 and ny > 0:
            pass
        else:
            raise Exception("Please check the order of cropping parameter!")
        if (
            cfg["x1"] >= 0
            and cfg["x2"] < int(np.shape(im)[1])
            and cfg["y1"] >= 0
            and cfg["y2"] < int(np.shape(im)[0])
        ):
            pass  # good cropping box
        else:
            raise Exception("Please check the boundary of cropping!")

    pbar = tqdm(total=nframes)
    counter = 0
    step = max(10, int(nframes / 100))

    if batchsize == 1:
        for counter, framename in enumerate(framelist):
            # frame=imread(os.path.join(directory,framename),mode='RGB')
            if rgb:
                im = imread(os.path.join(directory, framename), mode="RGB")
            else:
                im = imread(os.path.join(directory, framename))

            if counter % step == 0:
                pbar.update(step)

            if cfg["cropping"]:
                frame = img_as_ubyte(
                    im[cfg["y1"] : cfg["y2"], cfg["x1"] : cfg["x2"], :]
                )
            else:
                frame = img_as_ubyte(im)

            pose = predict.getpose(frame, dlc_cfg, sess, inputs, outputs)
            PredictedData[counter, :] = pose.flatten()
    else:
        frames = np.empty(
            (batchsize, ny, nx, 3), dtype="ubyte"
        )  # this keeps all the frames of a batch
        for counter, framename in enumerate(framelist):
            if rgb:
                im = imread(os.path.join(directory, framename), mode="RGB")
            else:
                im = imread(os.path.join(directory, framename))

            if counter % step == 0:
                pbar.update(step)

            if cfg["cropping"]:
                frames[batch_ind] = img_as_ubyte(
                    im[cfg["y1"] : cfg["y2"], cfg["x1"] : cfg["x2"], :]
                )
            else:
                frames[batch_ind] = img_as_ubyte(im)

            if batch_ind == batchsize - 1:
                pose = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs)
                PredictedData[
                    batch_num * batchsize : (batch_num + 1) * batchsize, :
                ] = pose
                batch_ind = 0
                batch_num += 1
            else:
                batch_ind += 1

        if (
            batch_ind > 0
        ):  # take care of the last frames (the batch that might have been processed)
            pose = predict.getposeNP(
                frames, dlc_cfg, sess, inputs, outputs
            )  # process the whole batch (some frames might be from previous batch!)
            PredictedData[
                batch_num * batchsize : batch_num * batchsize + batch_ind, :
            ] = pose[:batch_ind, :]

    pbar.close()
    return PredictedData, nframes, nx, ny


def analyze_time_lapse_frames(
    config,
    directory,
    frametype=".png",
    shuffle=1,
    trainingsetindex=0,
    gputouse=None,
    save_as_csv=False,
    rgb=True,
    modelprefix="",
):
    """
    Analyzed all images (of type = frametype) in a folder and stores the output in one file.

    You can crop the frames (before analysis), by changing 'cropping'=True and setting 'x1','x2','y1','y2' in the config file.

    Output: The labels are stored as MultiIndex Pandas Array, which contains the name of the network, body part name, (x, y) label position \n
            in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient Hierarchical Data Format (HDF) \n
            in the same directory, where the video is stored. However, if the flag save_as_csv is set to True, the data can also be exported in \n
            comma-separated values format (.csv), which in turn can be imported in many programs, such as MATLAB, R, Prism, etc.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    directory: string
        Full path to directory containing the frames that shall be analyzed

    frametype: string, optional
        Checks for the file extension of the frames. Only images with this extension are analyzed. The default is ``.png``

    shuffle: int, optional
        An integer specifying the shuffle index of the training dataset used for training the network. The default is 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU put None.
    See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    save_as_csv: bool, optional
        Saves the predictions in a .csv file. The default is ``False``; if provided it must be either ``True`` or ``False``

    rbg: bool, optional.
        Whether to load image as rgb; Note e.g. some tiffs do not alow that option in imread, then just set this to false.

    Examples
    --------
    If you want to analyze all frames in /analysis/project/timelapseexperiment1
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml','/analysis/project/timelapseexperiment1')
    --------

    If you want to analyze all frames in /analysis/project/timelapseexperiment1
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml','/analysis/project/timelapseexperiment1')
    --------

    Note: for test purposes one can extract all frames from a video with ffmeg, e.g. ffmpeg -i testvideo.avi thumb%04d.png
    """
    if "TF_CUDNN_USE_AUTOTUNE" in os.environ:
        del os.environ["TF_CUDNN_USE_AUTOTUNE"]  # was potentially set during training

    if gputouse is not None:  # gpu selection
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gputouse)

    tf.compat.v1.reset_default_graph()
    start_path = os.getcwd()  # record cwd to return to this directory in the end

    cfg = auxiliaryfunctions.read_config(config)
    trainFraction = cfg["TrainingFraction"][trainingsetindex]
    modelfolder = os.path.join(
        cfg["project_path"],
        str(
            auxiliaryfunctions.GetModelFolder(
                trainFraction, shuffle, cfg, modelprefix=modelprefix
            )
        ),
    )
    path_test_config = Path(modelfolder) / "test" / "pose_cfg.yaml"
    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError(
            "It seems the model for shuffle %s and trainFraction %s does not exist."
            % (shuffle, trainFraction)
        )
    # Check which snapshots are available and sort them by # iterations
    try:
        Snapshots = np.array(
            [
                fn.split(".")[0]
                for fn in os.listdir(os.path.join(modelfolder, "train"))
                if "index" in fn
            ]
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."
            % (shuffle, shuffle)
        )

    if cfg["snapshotindex"] == "all":
        print(
            "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!"
        )
        snapshotindex = -1
    else:
        snapshotindex = cfg["snapshotindex"]

    increasing_indices = np.argsort([int(m.split("-")[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]

    print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)

    ##################################################
    # Load and setup CNN part detector
    ##################################################

    # Check if data already was generated:
    dlc_cfg["init_weights"] = os.path.join(
        modelfolder, "train", Snapshots[snapshotindex]
    )
    trainingsiterations = (dlc_cfg["init_weights"].split(os.sep)[-1]).split("-")[-1]

    # update batchsize (based on parameters in config.yaml)
    dlc_cfg["batch_size"] = cfg["batch_size"]

    # Name for scorer:
    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.GetScorerName(
        cfg,
        shuffle,
        trainFraction,
        trainingsiterations=trainingsiterations,
        modelprefix=modelprefix,
    )
    sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)

    # update number of outputs and adjust pandas indices
    dlc_cfg["num_outputs"] = cfg.get("num_outputs", 1)

    xyz_labs_orig = ["x", "y", "likelihood"]
    suffix = [str(s + 1) for s in range(dlc_cfg["num_outputs"])]
    suffix[0] = ""  # first one has empty suffix for backwards compatibility
    xyz_labs = [x + s for s in suffix for x in xyz_labs_orig]

    pdindex = pd.MultiIndex.from_product(
        [[DLCscorer], dlc_cfg["all_joints_names"], xyz_labs],
        names=["scorer", "bodyparts", "coords"],
    )

    if gputouse is not None:  # gpu selectinon
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gputouse)

    ##################################################
    # Loading the images
    ##################################################
    # checks if input is a directory
    if os.path.isdir(directory) == True:
        """
        Analyzes all the frames in the directory.
        """
        print("Analyzing all frames in the directory: ", directory)
        os.chdir(directory)
        framelist = np.sort([fn for fn in os.listdir(os.curdir) if (frametype in fn)])
        vname = Path(directory).stem
        notanalyzed, dataname, DLCscorer = auxiliaryfunctions.CheckifNotAnalyzed(
            directory, vname, DLCscorer, DLCscorerlegacy, flag="framestack"
        )
        if notanalyzed:
            nframes = len(framelist)
            if nframes > 0:
                start = time.time()

                PredictedData, nframes, nx, ny = GetPosesofFrames(
                    cfg,
                    dlc_cfg,
                    sess,
                    inputs,
                    outputs,
                    directory,
                    framelist,
                    nframes,
                    dlc_cfg["batch_size"],
                    rgb,
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
                    "config file": dlc_cfg,
                    "batch_size": dlc_cfg["batch_size"],
                    "num_outputs": dlc_cfg["num_outputs"],
                    "frame_dimensions": (ny, nx),
                    "nframes": nframes,
                    "cropping": cfg["cropping"],
                    "cropping_parameters": coords,
                }
                metadata = {"data": dictionary}

                print("Saving results in %s..." % (directory))

                auxiliaryfunctions.SaveData(
                    PredictedData[:nframes, :],
                    metadata,
                    dataname,
                    pdindex,
                    framelist,
                    save_as_csv,
                )
                print("The folder was analyzed. Now your research can truly start!")
                print(
                    "If the tracking is not satisfactory for some frome, consider expanding the training set."
                )
            else:
                print(
                    "No frames were found. Consider changing the path or the frametype."
                )

    os.chdir(str(start_path))


def _convert_detections_to_tracklets(
    cfg,
    inference_cfg,
    data,
    metadata,
    output_path,
    track_method="ellipse",
    greedy=False,
    calibrate=False,
):
    joints = data["metadata"]["all_joints_names"]
    partaffinityfield_graph = data["metadata"]["PAFgraph"]
    paf_inds = data["metadata"]["PAFinds"]
    paf_graph = [partaffinityfield_graph[l] for l in paf_inds]

    if track_method == "box":
        mot_tracker = trackingutils.Sort(inference_cfg)
    elif track_method == "skeleton":
        mot_tracker = trackingutils.SORT(
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

    ass = inferenceutils.Assembler(
        data,
        max_n_individuals=inference_cfg["topktoretain"],
        n_multibodyparts=len(cfg["multianimalbodyparts"]),
        graph=paf_graph,
        paf_inds=list(paf_inds),
        greedy=greedy,
        pcutoff=inference_cfg.get("pcutoff", 0.1),
        min_affinity=inference_cfg.get("pafthreshold", 0.05),
    )
    if calibrate:
        trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(cfg)
        train_data_file = os.path.join(
            cfg["project_path"],
            str(trainingsetfolder),
            "CollectedData_" + cfg["scorer"] + ".h5",
        )
        ass.calibrate(train_data_file)
    ass.assemble()

    output_path, _ = os.path.splitext(output_path)
    output_path += ".pickle"
    ass.to_pickle(output_path.replace(".pickle", "_assemblies.pickle"))

    if cfg["uniquebodyparts"]:
        tracklets["single"] = {}
        tracklets["single"].update(ass.unique)

    for i, imname in tqdm(enumerate(ass.metadata["imnames"])):
        assemblies = ass.assemblies.get(i)
        if assemblies is None:
            continue
        animals = np.stack([ass.data[:, :3] for ass in assemblies])
        if track_method == "box":
            bboxes = trackingutils.calc_bboxes_from_keypoints(
                animals, inference_cfg.get("boundingboxslack", 0)
            )  # TODO: get cropping parameters and utilize!
            trackers = mot_tracker.update(bboxes)
        else:
            xy = animals[..., :2]
            trackers = mot_tracker.track(xy)
        trackingutils.fill_tracklets(tracklets, trackers, animals, imname)

    bodypartlabels = [joint for joint in joints for _ in range(3)]
    numentries = len(bodypartlabels)
    scorers = numentries * [metadata["data"]["Scorer"]]
    xylvalue = len(bodypartlabels) // 3 * ["x", "y", "likelihood"]
    pdindex = pd.MultiIndex.from_arrays(
        np.vstack([scorers, bodypartlabels, xylvalue]),
        names=["scorer", "bodyparts", "coords"],
    )
    tracklets["header"] = pdindex
    with open(output_path, "wb") as f:
        pickle.dump(tracklets, f, pickle.HIGHEST_PROTOCOL)


def convert_detections2tracklets(
    config,
    videos,
    videotype="avi",
    shuffle=1,
    trainingsetindex=0,
    overwrite=False,
    destfolder=None,
    ignore_bodyparts=None,
    inferencecfg=None,
    modelprefix="",
    track_method="ellipse",
    greedy=False,
    calibrate=False,
    window_size=0,
    identity_only=False,
):
    """
    This should be called at the end of deeplabcut.analyze_videos for multianimal projects!

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    videos : list
        A list of strings containing the full paths to videos for analysis or a path to the directory, where all the videos with same extension are stored.

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed. The default is ``.avi``

    shuffle: int, optional
        An integer specifying the shuffle index of the training dataset used for training the network. The default is 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    overwrite: bool, optional.
        Overwrite tracks file i.e. recompute tracks from full detections and overwrite.

    destfolder: string, optional
        Specifies the destination folder for analysis data (default is the path of the video). Note that for subsequent analysis this
        folder also needs to be passed.

    track_method: str, optional
        Method used to track animals, either 'box', 'skeleton', or 'ellipse'.
        By default, a constant velocity Kalman filter is used to track
        covariance error ellipses fitted to an individual's body parts.

    ignore_bodyparts: optional
        List of body part names that should be ignored during tracking (advanced).
        By default, all the body parts are used.

    inferencecfg: Default is None.
        Configuaration file for inference (assembly of individuals). Ideally
        should be optained from cross validation (during evaluation). By default
        the parameters are loaded from inference_cfg.yaml, but these get_level_values
        can be overwritten.

    calibrate: bool, optional (default=False)
        If True, use training data to calibrate the animal assembly procedure.
        This improves its robustness to wrong body part links,
        but requires very little missing data.

    window_size: int, optional (default=0)
        Recurrent connections in the past `window_size` frames are
        prioritized during assembly. By default, no temporal coherence cost
        is added, and assembly is driven mainly by part affinity costs.

    identity_only: bool, optional (default=False)
        If True and animal identity was learned by the model,
        assembly and tracking rely exclusively on identity prediction.

    Examples
    --------
    If you want to convert detections to tracklets:
    >>> deeplabcut.convert_detections2tracklets('/analysis/project/reaching-task/config.yaml',[]'/analysis/project/video1.mp4'], videotype='.mp4')
    --------

    """
    if track_method not in ("box", "skeleton", "ellipse"):
        raise ValueError(
            "Invalid tracking method. Only `box`, `skeleton` and `ellipse` are currently supported."
        )

    cfg = auxiliaryfunctions.read_config(config)
    trainFraction = cfg["TrainingFraction"][trainingsetindex]
    start_path = os.getcwd()  # record cwd to return to this directory in the end

    # TODO: addd cropping as in video analysis!
    # if cropping is not None:
    #    cfg['cropping']=True
    #    cfg['x1'],cfg['x2'],cfg['y1'],cfg['y2']=cropping
    #    print("Overwriting cropping parameters:", cropping)
    #    print("These are used for all videos, but won't be save to the cfg file.")

    modelfolder = os.path.join(
        cfg["project_path"],
        str(
            auxiliaryfunctions.GetModelFolder(
                trainFraction, shuffle, cfg, modelprefix=modelprefix
            )
        ),
    )
    path_test_config = Path(modelfolder) / "test" / "pose_cfg.yaml"
    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError(
            "It seems the model for shuffle %s and trainFraction %s does not exist."
            % (shuffle, trainFraction)
        )

    if "multi-animal" not in dlc_cfg["dataset_type"]:
        raise ValueError("This function is only required for multianimal projects!")

    path_inference_config = Path(modelfolder) / "test" / "inference_cfg.yaml"
    if inferencecfg is None:  # then load or initialize
        inferencecfg = auxfun_multianimal.read_inferencecfg(path_inference_config, cfg)
    else:
        auxfun_multianimal.check_inferencecfg_sanity(cfg, inferencecfg)

    # Check which snapshots are available and sort them by # iterations
    try:
        Snapshots = np.array(
            [
                fn.split(".")[0]
                for fn in os.listdir(os.path.join(modelfolder, "train"))
                if "index" in fn
            ]
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."
            % (shuffle, shuffle)
        )

    if cfg["snapshotindex"] == "all":
        print(
            "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!"
        )
        snapshotindex = -1
    else:
        snapshotindex = cfg["snapshotindex"]

    increasing_indices = np.argsort([int(m.split("-")[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]
    print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)
    dlc_cfg["init_weights"] = os.path.join(
        modelfolder, "train", Snapshots[snapshotindex]
    )
    trainingsiterations = (dlc_cfg["init_weights"].split(os.sep)[-1]).split("-")[-1]

    # Name for scorer:
    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.GetScorerName(
        cfg,
        shuffle,
        trainFraction,
        trainingsiterations=trainingsiterations,
        modelprefix=modelprefix,
    )

    ##################################################
    # Looping over videos
    ##################################################
    Videos = auxiliaryfunctions.Getlistofvideos(videos, videotype)
    if len(Videos) > 0:
        for video in Videos:
            print("Processing... ", video)
            videofolder = str(Path(video).parents[0])
            if destfolder is None:
                destfolder = videofolder
            auxiliaryfunctions.attempttomakefolder(destfolder)
            vname = Path(video).stem
            dataname = os.path.join(destfolder, vname + DLCscorer + ".h5")
            data, metadata = auxfun_multianimal.LoadFullMultiAnimalData(dataname)
            if track_method == "ellipse":
                method = "el"
            elif track_method == "box":
                method = "bx"
            else:
                method = "sk"
            trackname = dataname.split(".h5")[0] + f"_{method}.pickle"
            # NOTE: If dataname line above is changed then line below is obsolete?
            # trackname = trackname.replace(videofolder, destfolder)
            if (
                os.path.isfile(trackname) and not overwrite
            ):  # TODO: check if metadata are identical (same parameters!)
                print("Tracklets already computed", trackname)
                print("Set overwrite = True to overwrite.")
            else:
                print("Analyzing", dataname)
                DLCscorer = metadata["data"]["Scorer"]
                all_jointnames = data["metadata"]["all_joints_names"]

                numjoints = len(all_jointnames)

                # TODO: adjust this for multi + unique bodyparts!
                # this is only for multianimal parts and uniquebodyparts as one (not one uniquebodyparts guy tracked etc. )
                bodypartlabels = [bpt for i, bpt in enumerate(all_jointnames) for _ in range(3)]
                scorers = len(bodypartlabels) * [DLCscorer]
                xylvalue = int(len(bodypartlabels) / 3) * ["x", "y", "likelihood"]
                pdindex = pd.MultiIndex.from_arrays(
                    np.vstack([scorers, bodypartlabels, xylvalue]),
                    names=["scorer", "bodyparts", "coords"],
                )

                imnames = [fn for fn in data if fn != "metadata"]

                if track_method == "box":
                    mot_tracker = trackingutils.Sort(inferencecfg)
                elif track_method == "skeleton":
                    mot_tracker = trackingutils.SORT(
                        numjoints,
                        inferencecfg["max_age"],
                        inferencecfg["min_hits"],
                        inferencecfg.get("oks_threshold", 0.5),
                    )
                else:
                    mot_tracker = trackingutils.SORTEllipse(
                        inferencecfg.get("max_age", 1),
                        inferencecfg.get("min_hits", 1),
                        inferencecfg.get("iou_threshold", 0.6),
                    )
                tracklets = {}
                multi_bpts = cfg["multianimalbodyparts"]
                ass = inferenceutils.Assembler(
                    data,
                    max_n_individuals=inferencecfg["topktoretain"],
                    n_multibodyparts=len(multi_bpts),
                    greedy=greedy,
                    pcutoff=inferencecfg.get("pcutoff", 0.1),
                    min_affinity=inferencecfg.get("pafthreshold", 0.05),
                    window_size=window_size,
                    identity_only=identity_only,
                )
                if calibrate:
                    trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(cfg)
                    train_data_file = os.path.join(
                        cfg["project_path"],
                        str(trainingsetfolder),
                        "CollectedData_" + cfg["scorer"] + ".h5",
                    )
                    ass.calibrate(train_data_file)
                ass.assemble()
                ass.to_pickle(dataname.split(".h5")[0] + "_assemblies.pickle")

                if cfg[
                    "uniquebodyparts"
                ]:  # Initialize storage of the 'single' individual track
                    tracklets["single"] = {}
                    _single = {}
                    for index, imname in enumerate(imnames):
                        single_detection = ass.unique.get(index)
                        if single_detection is None:
                            continue
                        imindex = int(re.findall(r"\d+", imname)[0])
                        _single[imindex] = single_detection
                    tracklets["single"].update(_single)

                keep = set(multi_bpts).difference(ignore_bodyparts or [])
                keep_inds = sorted(multi_bpts.index(bpt) for bpt in keep)
                for index, imname in tqdm(enumerate(imnames)):
                    assemblies = ass.assemblies.get(index)
                    if assemblies is None:
                        continue
                    animals = np.stack([ass.data for ass in assemblies])
                    if not identity_only:
                        if track_method == "box":
                            bboxes = trackingutils.calc_bboxes_from_keypoints(
                                animals[:, keep_inds], inferencecfg["boundingboxslack"], offset=0
                            )  # TODO: get cropping parameters and utilize!
                            trackers = mot_tracker.update(bboxes)
                        else:
                            xy = animals[:, keep_inds, :2]
                            trackers = mot_tracker.track(xy)
                    else:
                        # Optimal identity assignment based on soft voting
                        mat = np.zeros((len(assemblies), inferencecfg["topktoretain"]))
                        for nrow, assembly in enumerate(assemblies):
                            for k, v in assembly.soft_identity.items():
                                mat[nrow, k] = v
                        inds = linear_sum_assignment(mat, maximize=True)
                        trackers = np.c_[inds][:, ::-1]
                    trackingutils.fill_tracklets(tracklets, trackers, animals, imname)

                tracklets["header"] = pdindex
                with open(trackname, "wb") as f:
                    pickle.dump(tracklets, f, pickle.HIGHEST_PROTOCOL)

        os.chdir(str(start_path))

        print("The tracklets were created. Now you can 'refine_tracklets'.")
        # print("If the tracking is not satisfactory for some videos, consider expanding the training set. You can use the function 'extract_outlier_frames' to extract any outlier frames!")
    else:
        print("No video(s) found. Please check your path!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("config")
    cli_args = parser.parse_args()
