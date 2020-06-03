"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

# TODO: Add support for alternate multi output format, as create_labeled_video still fails on old format.

####################################################
# Dependencies
####################################################
import argparse
import os
import os.path
import time
from pathlib import Path

from deeplabcut.pose_estimation_tensorflow.nnet import predict, processing

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.util import img_as_ubyte
from tqdm import tqdm

from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.utils import auxiliaryfunctions


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
    c_engine=False,
    robust_nframes=False,
    num_outputs=None,
    multi_output_format="default",
    predictor=None,
    predictor_settings=None
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

    c_engine: bool, optional (default=False)
        If True, uses C code to detect 2D local maxima for multianimal inference.
        Pure-Python functions are used by default, which, although slower, do not require the user
        to install Cython and compile external code.

    robust_nframes: bool, optional (default=False)
        Evaluate a video's number of frames in a robust manner.
        This option is slower (as the whole video is read frame-by-frame),
        but does not rely on metadata, hence its robustness against file corruption.

    num_outputs: int, default: from config.yaml, or 1 if not set in config.yaml.
        Allows the user to set the number of predictions for bodypart, overriding the option in the config file.

    multi_output_format: Determines the multi output format used. "default" uses the default format, while
        "separate-bodyparts" separates the multi output predictions such that each is its own body part. Defaults to
        "default", and passing any rouge values sets it to "defualt". Does nothing if num_outputs = 1

    predictor: The prediction algorithm to use on the probability outputs of the deeplabcut neural net. Defaults to
        "argmax". The options available depends on the currently available Predictor plugins in the
        predictors folder. To list all currently available plugins, run the 'deeplabcut.list_predictor_plugins' method.

    predictor_settings: Optional dictionary of strings to any. This will specify what settings a predictor should use,
        completely ignoring any settings specified in the config.yaml. Default value is None, which
        tells this method to use the settings specified in the config.yaml.

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

    tf.reset_default_graph()
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
    old_num_outputs = dlc_cfg["num_outputs"]
    dlc_cfg["num_outputs"] = int(num_outputs) if((num_outputs is not None) and (num_outputs >= 1)) else dlc_cfg["num_outputs"]

    if batchsize == None:
        # update batchsize (based on parameters in config.yaml)
        dlc_cfg["batch_size"] = cfg["batch_size"]
    else:
        dlc_cfg["batch_size"] = batchsize
        cfg["batch_size"] = batchsize

    if "multi-animal" in dlc_cfg["dataset_type"]:
        dynamic = (False, 0.5, 10)  # setting dynamic mode to false
        TFGPUinference = False
        predictor = None

    if(predictor is not None):
        # If predictor plugin was selected, disable dynamic mode and GPU predictions.
        predictor_cls = processing.get_predictor(predictor)
        TFGPUinference = False
        dynamic = (False, 0.5, 10)
        print(f"Predictor '{predictor}' selected, disabling GPU predictions and dynamic cropping as both of these are not supported.")
    else:
        predictor_cls = processing.get_predictor("argmax")

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

    # sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)
    if TFGPUinference:
        sess, inputs, outputs = predict.setup_GPUpose_prediction(dlc_cfg)
    else:
        sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)

    pdindex = GetPandasHeader(dlc_cfg["all_joints_names"], dlc_cfg["num_outputs"], multi_output_format, DLCscorer)

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
                    pdindex,
                    save_as_csv,
                    destfolder,
                    c_engine=c_engine,
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
                    predictor_cls,
                    predictor_settings
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

        dlc_cfg["num_outputs"] = old_num_outputs

        return DLCscorer  # note: this is either DLCscorer or DLCscorerlegacy depending on what was used!
    else:
        dlc_cfg["num_outputs"] = old_num_outputs

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


def GetPandasHeader(body_parts, num_outputs, out_format, dlc_scorer):
    """
    Creates the pandas data header for the passed body parts and number of outputs.

    body_parts: The list of body part names. List of strings.
    num_outputs: The number of outputs per body part, and integer.
    out_format: The output format, either 'separate-bodyparts' or 'default'.
    dlc_scorer: A string, being the name of the DLC Scorer for this DLC instance.

    Returns: A pandas MultiIndex, being the header entries for the DLC output data.
    """
    # Set this up differently depending on the format...
    if(out_format == "separate-bodyparts" and num_outputs > 1):
        # Format which allocates new bodyparts for each prediction by simply adding "__number" to the end of the part's
        # name.
        print("Outputting predictions as separate body parts...")
        suffixes = [f"__{i + 1}" for i in range(num_outputs)]
        suffixes[0] = ""
        all_joints = [bp + s for bp in body_parts for s in suffixes]
        return pd.MultiIndex.from_product([[dlc_scorer], all_joints, ['x', 'y', 'likelihood']],
                                             names=['scorer', 'bodyparts', 'coords'])
    else:
        # The original multi output format, multiple predictions stored under each body part
        suffixes = [str(i + 1) for i in range(num_outputs)]
        suffixes[0] = ""
        sub_headers = [state + s for s in suffixes for state in ['x', 'y', 'likelihood']]
        return pd.MultiIndex.from_product([[dlc_scorer], body_parts, sub_headers],
                                             names=['scorer', 'bodyparts', 'coords'])


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

            if batch_ind == batchsize - 1:
                pose = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs)
                PredictedData[
                    batch_num * batchsize : (batch_num + 1) * batchsize, :
                ] = pose
                batch_ind = 0
                batch_num += 1
            else:
                batch_ind += 1
        else:
            nframes = counter
            print("Detected frames: ", nframes)
            if batch_ind > 0:
                pose = predict.getposeNP(
                    frames, dlc_cfg, sess, inputs, outputs
                )  # process the whole batch (some frames might be from previous batch!)
                PredictedData[
                    batch_num * batchsize : batch_num * batchsize + batch_ind, :
                ] = pose[:batch_ind, :]
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
        else:
            nframes = counter
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
        else:
            nframes = counter
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

            if batch_ind == batchsize - 1:
                # pose = predict.getposeNP(frames,dlc_cfg, sess, inputs, outputs)
                pose = sess.run(pose_tensor, feed_dict={inputs: frames})
                pose[:, [0, 1, 2]] = pose[
                    :, [1, 0, 2]
                ]  # change order to have x,y,confidence
                pose = np.reshape(
                    pose, (batchsize, -1)
                )  # bring into batchsize times x,y,conf etc.
                PredictedData[
                    batch_num * batchsize : (batch_num + 1) * batchsize, :
                ] = pose

                batch_ind = 0
                batch_num += 1
            else:
                batch_ind += 1
        else:
            nframes = counter
            print("Detected frames: ", nframes)
            if batch_ind > 0:
                # pose = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs) #process the whole batch (some frames might be from previous batch!)
                pose = sess.run(pose_tensor, feed_dict={inputs: frames})
                pose[:, [0, 1, 2]] = pose[:, [1, 0, 2]]
                pose = np.reshape(pose, (batchsize, -1))
                PredictedData[
                    batch_num * batchsize : batch_num * batchsize + batch_ind, :
                ] = pose[:batch_ind, :]

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
        else:
            nframes = counter
            break
        counter += 1

    pbar.close()
    return PredictedData, nframes


# Utility method used by AnalyzeVideo, gets the settings for the given predictor plugin
def GetPredictorSettings(cfg, predictor_cls, usr_passed_settings = None):
    """ Get the predictor settings from deeplabcut config and return a dictionary for plugin to use... """
    # Grab setting blueprints for predictor plugins(list of tuples of name, desc, default val)....
    setting_info = predictor_cls.get_settings()
    name = predictor_cls.get_name()

    if(setting_info is None):
        return None

    # Pull out the name and default values into a dictionary...
    setting_info = {name: def_val for (name, desc, def_val) in setting_info}


    if(usr_passed_settings is None):
        # If the dlc config contains a category predictors, and predictors contains a category named after the plugin, load
        # the user cfg for this plugin and merge it with default values
        if(("predictors" in cfg) and (cfg["predictors"]) and (name in cfg["predictors"])):
            setting_info.update(
                {key: cfg["predictors"][name][key] for key in (setting_info.keys() & cfg["predictors"][name].keys())}
            )
    else:
        # If the user directly passed settings to this method, we ignore the config and use these settings.
        setting_info.update(
            {key: usr_passed_settings[key] for key in (setting_info.keys() & usr_passed_settings.keys())}
        )

    return setting_info


# Utility method used by GetPoseALL, gets a batch of frames, stores them in frame_store and returns the size of the batch
def GetVideoBatch(cap, batch_size, cfg, frame_store) -> int:
    """ Gets a batch size of frames, and returns them """
    current_frame = 0

    # While the cap is still going and the current frame is less then the batch size...
    while(cap.isOpened() and current_frame < batch_size):
        # Read a frame
        ret_val, frame = cap.read()

        # If we got an actual frame, store it in the frame store.
        if(ret_val):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if cfg['cropping']:
                frame_store[current_frame] = img_as_ubyte(frame[cfg['y1']:cfg['y2'], cfg['x1']:cfg['x2']])
            else:
                frame_store[current_frame] = img_as_ubyte(frame)
        else:
            # If we don't we have reached the end most likely.
            return current_frame

        current_frame += 1

    return current_frame


# Replaces old system of getting poses and uses a new plugin system for predicting poses...
def GetPoseAll(cfg, dlc_cfg, sess, inputs, outputs, cap, nframes, batchsize, predictor):
    """ Gets the poses for any batch size, including batch size of only 1 """
    # Create a numpy array to hold all pose prediction data...
    pose_prediction_data = np.zeros((nframes, 3 * len(dlc_cfg["all_joints_names"]) * dlc_cfg["num_outputs"]))

    pbar = tqdm(total=nframes)

    ny, nx = int(cap.get(4)), int(cap.get(3))

    if(cfg["cropping"]):
        ny, nx = checkcropping(cfg, cap)

    # Create the temporary batch frame store for storing video frames...
    frame_store = np.empty((batchsize, ny, nx, 3), dtype='ubyte')

    # Create a counter to keep track of the progress bar
    counter = 0
    prog_step = max(10, int(nframes / 100))
    current_step = 0

    frames_done = 0

    while(True):
        size = GetVideoBatch(cap, batchsize, cfg, frame_store)
        counter += size

        # If we pass the current step or phase, update the progress bar
        if(counter // prog_step > current_step):
            pbar.update(prog_step)
            current_step += 1

        if(size > 0):
            # If we received any frames, process them...
            scmap, locref = predict.extract_cnn_outputmulti(sess.run(outputs, feed_dict={inputs: frame_store}), dlc_cfg)
            down_scale = dlc_cfg.stride

            if len(scmap.shape) == 2:  # If there is a single body part, add a dimension at the end
                scmap = np.expand_dims(scmap, axis=2)

            pose = predictor.on_frames(processing.TrackingData(scmap[:size], locref, down_scale))

            if(pose is not None):
                # If the predictor returned a pose, add it to the final data.
                pose_prediction_data[frames_done:frames_done + pose.get_frame_count()] = pose.get_all()
                frames_done += pose.get_frame_count()

        if(size < batchsize):
            # If the output frames by the video capture were less then a full batch, we have reached the end of the
            # video...
            break

    pbar.update(counter - (prog_step * current_step))
    pbar.close()

    # Phase 2: Post processing...

    # Get all of the final poses that are still held by the predictor
    post_pbar = tqdm(total = nframes - frames_done)
    final_poses = predictor.on_end(post_pbar)
    post_pbar.close()

    # Add any post-processed frames
    if(final_poses is not None):
        pose_prediction_data[frames_done:frames_done + final_poses.get_frame_count()] = final_poses.get_all()
        frames_done += final_poses.get_frame_count()

    # Check and make sure the predictor returned all frames, otherwise throw an error.
    if(frames_done != nframes):
        raise ValueError(f"The predictor algorithm did not return the same amount of frames as are in the video.\n"
                         f"Expected Amount: {nframes}, Actual Amount Returned: {frames_done}")

    return pose_prediction_data, nframes


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
    predictor_cls=None,
    predictor_settings=None
):
    """ Helper function for analyzing a video. """
    print("Starting to analyze % ", video)

    if destfolder is None:
        destfolder = str(Path(video).resolve().parents[0])
    auxiliaryfunctions.attempttomakefolder(destfolder)
    vname = Path(video).stem
    try:
        _ = auxiliaryfunctions.load_analyzed_data(destfolder, vname, DLCscorer)
    except FileNotFoundError:
        dataname = os.path.join(destfolder, vname + DLCscorer + ".h5") # Path of the final .h5 file.

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
            if TFGPUinference:
                if(dlc_cfg["batch_size"] > 1):
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
                    PredictedData, nframes = GetPoseS_GTF(
                        cfg, dlc_cfg, sess, inputs, outputs, cap, nframes
                    )
            else:
                # Passed to the plugin to give it some info about the video...
                video_metadata = {
                    "fps": fps,
                    "duration": duration,
                    "size": size,
                    "h5-file-name": str(Path(dataname).resolve()),
                    "orig-video-path": str(Path(video).resolve()),
                    "cropping-offset": (int(cfg["y1"]), int(cfg["x1"])) if (cfg["cropping"]) else None
                }

                # Create a predictor plugin instance...
                predictor_settings = GetPredictorSettings(cfg, predictor_cls,
                                                          predictor_settings)  # Grab the plugin settings for this plugin...
                print(f"Plugin {predictor_cls.get_name()} Settings: {predictor_settings}")
                predictor_inst = predictor_cls(dlc_cfg['all_joints_names'], dlc_cfg["num_outputs"], nframes,
                                               predictor_settings, video_metadata)

                PredictedData, nframes = GetPoseAll(cfg, dlc_cfg, sess, inputs, outputs, cap, nframes,
                                                    int(dlc_cfg["batch_size"]), predictor_inst)

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

    vers = (tf.__version__).split(".")
    if int(vers[0]) == 1 and int(vers[1]) > 12:
        TF = tf.compat.v1
    else:
        TF = tf

    TF.reset_default_graph()
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


def convert_detections2tracklets(
    config,
    videos,
    videotype="avi",
    shuffle=1,
    trainingsetindex=0,
    overwrite=False,
    destfolder=None,
    BPTS=None,
    iBPTS=None,
    PAF=None,
    printintermediate=False,
    inferencecfg=None,
    modelprefix="",
    track_method="box",
    edgewisecondition=True,
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
        Method uses to track animals, either 'box' or 'skeleton'.
        By default, a constant velocity Kalman filter is used to track individual bounding boxes.

    BPTS: Default is None: all bodyparts are used.
        Pass list of indices if only certain bodyparts should be used (advanced).

    iBPTS: Default is None: all bodyparts are used.
        The inverse indices from BPTS.
        TODO: calculate from BPTS

    PAF: Default is None: all connections are used.
        Pass list of indices if only certain connections should be used (advanced).

    printintermediate: ## TODO
        Default is false.

    inferencecfg: Default is None.
        Configuaration file for inference (assembly of individuals). Ideally
        should be optained from cross validation (during evaluation). By default
        the parameters are loaded from inference_cfg.yaml, but these get_level_values
        can be overwritten.

    edgewisecondition: bool, default False.
        If true pairwise Euclidean distances of limbs (connections in PAF) will be
        estimated from the annotated data and used for excluding possible connections.

    Examples
    --------
    If you want to convert detections to tracklets:
    >>> deeplabcut.convert_detections2tracklets('/analysis/project/reaching-task/config.yaml',[]'/analysis/project/video1.mp4'], videotype='.mp4')
    --------

    """
    from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils, trackingutils
    from deeplabcut.utils import auxfun_multianimal
    from easydict import EasyDict as edict
    import pickle

    if track_method not in ("box", "skeleton"):
        raise ValueError(
            "Invalid tracking method. Only `box` and `skeleton` are currently supported."
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
        inferencecfg = edict(inferencecfg)
        auxfun_multianimal.check_inferencecfg_sanity(cfg, inferencecfg)

    if edgewisecondition:
        path_inferencebounds_config = (
            Path(modelfolder) / "test" / "inferencebounds.yaml"
        )
        try:
            inferenceboundscfg = auxiliaryfunctions.read_plainconfig(
                path_inferencebounds_config
            )
        except FileNotFoundError:
            print("Computing distances...")
            from deeplabcut.pose_estimation_tensorflow import calculatepafdistancebounds

            inferenceboundscfg = calculatepafdistancebounds(
                config, shuffle, trainingsetindex
            )
            auxiliaryfunctions.write_plainconfig(
                path_inferencebounds_config, inferenceboundscfg
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
            dataname = os.path.join(videofolder, vname + DLCscorer + ".h5")
            data, metadata = auxfun_multianimal.LoadFullMultiAnimalData(dataname)
            method = "sk" if track_method == "skeleton" else "bx"
            trackname = dataname.split(".h5")[0] + f"_{method}.pickle"
            trackname = trackname.replace(videofolder, destfolder)
            if (
                os.path.isfile(trackname) and not overwrite
            ):  # TODO: check if metadata are identical (same parameters!)
                print("Tracklets already computed", trackname)
                print("Set overwrite = True to overwrite.")
            else:
                print("Analyzing", dataname)
                DLCscorer = metadata["data"]["Scorer"]
                dlc_cfg = metadata["data"]["DLC-model-config file"]
                nms_radius = data["metadata"]["nms radius"]
                minconfidence = data["metadata"]["minimal confidence"]
                partaffinityfield_graph = data["metadata"]["PAFgraph"]
                all_joints = data["metadata"]["all_joints"]
                all_jointnames = data["metadata"]["all_joints_names"]

                if edgewisecondition:
                    upperbound = np.array(
                        [
                            float(
                                inferenceboundscfg[str(edge[0]) + "_" + str(edge[1])][
                                    "intra_max"
                                ]
                            )
                            for edge in partaffinityfield_graph
                        ]
                    )
                    lowerbound = np.array(
                        [
                            float(
                                inferenceboundscfg[str(edge[0]) + "_" + str(edge[1])][
                                    "intra_min"
                                ]
                            )
                            for edge in partaffinityfield_graph
                        ]
                    )
                    upperbound *= 1.25
                    lowerbound *= 0.5  # SLACK!
                else:
                    lowerbound = None
                    upperbound = None

                if PAF is None:
                    PAF = np.arange(
                        len(partaffinityfield_graph)
                    )  # THIS CAN BE A SUBSET!

                partaffinityfield_graph = [partaffinityfield_graph[l] for l in PAF]
                linkingpartaffinityfield_graph = partaffinityfield_graph

                numjoints = len(all_jointnames)
                if BPTS is None and iBPTS is None:
                    # NOTE: this can be used if only a subset is relevant. I.e. [0,1] for only first and second joint!
                    BPTS = range(numjoints)
                    iBPTS = range(numjoints)  # the corresponding inverse!

                # TODO: adjust this for multi + unique bodyparts!
                # this is only for multianimal parts and uniquebodyparts as one (not one uniquebodyparts guy tracked etc. )
                bodypartlabels = sum([3 * [all_jointnames[bpt]] for bpt in BPTS], [])
                numentries = len(bodypartlabels)

                scorers = numentries * [DLCscorer]
                xylvalue = int(len(bodypartlabels) / 3) * ["x", "y", "likelihood"]
                pdindex = pd.MultiIndex.from_arrays(
                    np.vstack([scorers, bodypartlabels, xylvalue]),
                    names=["scorer", "bodyparts", "coords"],
                )

                imnames = [fn for fn in data if fn != "metadata"]

                if track_method == "box":
                    mot_tracker = trackingutils.Sort(inferencecfg)
                else:
                    mot_tracker = trackingutils.SORT(
                        numjoints,
                        inferencecfg["max_age"],
                        inferencecfg["min_hits"],
                        inferencecfg.get("oks_threshold", 0.5),
                    )

                Tracks = {}
                for index, imname in tqdm(enumerate(imnames)):
                    animals = inferenceutils.assemble_individuals(
                        inferencecfg,
                        data[imname],
                        numjoints,
                        BPTS,
                        iBPTS,
                        PAF,
                        partaffinityfield_graph,
                        linkingpartaffinityfield_graph,
                        lowerbound,
                        upperbound,
                        printintermediate,
                    )
                    if track_method == "box":
                        # get corresponding bounding boxes!
                        bb = inferenceutils.individual2boundingbox(
                            inferencecfg, animals, 0
                        )  # TODO: get cropping parameters and utilize!
                        trackers = mot_tracker.update(bb)
                    else:
                        temp = [arr.reshape((-1, 3))[:, :2] for arr in animals]
                        trackers = mot_tracker.track(temp)
                    trackingutils.fill_tracklets(Tracks, trackers, animals, imname)

                    # Test whether the unique bodyparts have been assembled
                    # TODO Perhaps easier to check whether links were defined in the PAF graph?
                    inds_unique = [
                        all_jointnames.index(bp) for bp in cfg["uniquebodyparts"]
                    ]
                    if not any(
                        np.isfinite(a.reshape((-1, 3))[inds_unique]).all()
                        for a in animals
                    ):
                        single = np.full((numjoints, 3), np.nan)
                        single_dets = inferenceutils.convertdetectiondict2listoflist(
                            data[imname], inds_unique
                        )
                        for ind, dets in zip(inds_unique, single_dets):
                            if len(dets) == 1:
                                single[ind] = dets[0][:3]
                            elif len(dets) > 1:
                                best = sorted(dets, key=lambda x: x[2], reverse=True)[0]
                                single[ind] = best[:3]
                        # Find an unused tracklet ID for the 'unique' bodyparts
                        tracklet_id = 0
                        while True:
                            if tracklet_id not in Tracks:
                                break
                            tracklet_id += 1
                        Tracks[tracklet_id] = {}
                        Tracks[tracklet_id][imname] = single.flatten()

                Tracks["header"] = pdindex
                with open(trackname, "wb") as f:
                    # Pickle the 'labeled-data' dictionary using the highest protocol available.
                    pickle.dump(Tracks, f, pickle.HIGHEST_PROTOCOL)

        os.chdir(str(start_path))

        print("The tracklets were created. Now you can 'refine_tracklets'.")
        # print("If the tracking is not satisfactory for some videos, consider expanding the training set. You can use the function 'extract_outlier_frames' to extract any outlier frames!")
    else:
        print("No video(s) found. Please check your path!")


def list_predictor_plugins():
    """
    Retrieve and print all currently installed and available predictor plugins that can be used with deeplabcut to
    the console...

    :return: Nothing, if one wants to get the plugins for other code look at processing.get_predictor_plugins().
    """
    # Load the plugins...
    predictors = processing.get_predictor_plugins()

    for predictor in predictors:
        print(f"Plugin Name: '{predictor.get_name()}'")
        print("Description: ")
        print(predictor.get_description())
        print()


def get_predictor_settings(predictor_name = None):
    """
    Gets the available/modifiable settings for a specified predictor plugin...
    :param predictor_name: The string or list of strings being the names of the predictor plugins to view customizable
                           settings for. If None, will print settings for all currently available predictors.
                           Defaults to None.

    :return: Nothing, prints to console....
    """
    from typing import Iterable

    # Convert whatever the predictor_name argument is to a list of predictor plugins
    if(predictor_name is None):
        predictors = processing.get_predictor_plugins()
    elif(isinstance(predictor_name, str)):
        predictors = [processing.get_predictor(predictor_name)]
    elif(isinstance(predictor_name, Iterable)):
        predictors = [processing.get_predictor(name) for name in predictor_name]
    else:
        raise ValueError("Argument 'predictor_name' not of type Iterable[str], string, or None!!!")

    # Print name, and settings for each plugin.
    for predictor in predictors:
        print(f"Plugin Name: {predictor.get_name()}")
        print("Arguments: ")
        if(predictor.get_settings() is None):
            print("None")
        else:
            for name, desc, def_val in predictor.get_settings():
                print(f"Name: '{name}'")
                print(f"Description: \n{desc}")
                print(f"Default Value: {def_val} \n")
        print()


def test_predictor_plugin(predictor_name = None, interactive = False):
    """
    Run the tests for a predictor plugin.

    :param predictor_name: The name of the predictor or to run tests for, or a list of names of the predictors to run.
                           If the predictor_name is not specified or set to None, then run tests for all of the
                           predictor plugins...
                           Note: names are strings...
    :param interactive: A boolean. If True, the program will wait for user input after every test, to allow the user
                        to easily read tests one by one... If false, all tests will be run at once with no user
                        interaction. Defaults to false.
    :return: Nothing, prints test info to console...
    """
    from typing import Iterable
    import traceback

    # Convert whatever the predictor_name argument is to a list of predictor plugins
    if(predictor_name is None):
        predictors = processing.get_predictor_plugins()
    elif(isinstance(predictor_name, str)):
        predictors = [processing.get_predictor(predictor_name)]
    elif(isinstance(predictor_name, Iterable)):
        predictors = [processing.get_predictor(name) for name in predictor_name]
    else:
        raise ValueError("Argument 'predictor_name' not of type Iterable[str], string, or None!!!")

    # Test plugins by calling there tests...
    for predictor in predictors:
        print(f"Testing Plugin: {predictor.get_name()}")\
        # Get the tests...
        tests = predictor.get_tests()

        # If this test contains no test, let the user know and move to the next plugin.
        if(tests is None):
            print(f"Plugin {predictor.get_name()} has no tests...")
            print()
            continue

        # Iterate tests printing there results...
        for test_meth in tests:
            print(f"Running Test {test_meth.__name__}:")
            try:
                passed, expected, actual = test_meth()

                print(f"Results: {'Passed' if passed else 'Failed'}")
                if(not passed):
                    print(f"Expected Results: {expected}")
                    print(f"Actual Results: {actual}")

            except Exception as excep:
                print("Results: Failed With Exception")
                traceback.print_exception(excep, excep, excep.__traceback__)
            finally:
                print()
                if(interactive):
                    input("Press Enter To Continue: ")
                    print()
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("config")
    cli_args = parser.parse_args()
