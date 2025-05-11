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


####################################################
# Dependencies
####################################################

import argparse
import os
import os.path
import pickle
import re
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from skimage.util import img_as_ubyte
from tqdm import tqdm

from deeplabcut.core import trackingutils, inferenceutils
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.core import predict

from deeplabcut.refine_training_dataset.stitch import stitch_tracklets
from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal, auxfun_models
from deeplabcut.pose_estimation_tensorflow.core.openvino.session import (
    GetPoseF_OV,
    is_openvino_available,
)


####################################################
# Loading data, and defining model folder
####################################################


def create_tracking_dataset(
    config,
    videos,
    track_method,
    videotype="",
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
    n_triplets=1000,
):
    try:
        from deeplabcut.pose_tracking_pytorch import create_triplets_dataset
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Unsupervised identity learning requires PyTorch. Please run `pip install torch`."
        )

    from deeplabcut.pose_estimation_tensorflow.predict_multianimal import (
        extract_bpt_feature_from_video,
    )

    # allow_growth must be true here because tensorflow does not automatically free gpu memory and setting it as false occupies all gpu memory so that pytorch cannot kick in
    allow_growth = True

    if "TF_CUDNN_USE_AUTOTUNE" in os.environ:
        del os.environ["TF_CUDNN_USE_AUTOTUNE"]  # was potentially set during training

    if gputouse is not None:  # gpu selection
        auxfun_models.set_visible_devices(gputouse)

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
            auxiliaryfunctions.get_model_folder(
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

    Snapshots = auxiliaryfunctions.get_snapshots_from_folder(
        train_folder=Path(modelfolder) / "train",
    )

    if cfg["snapshotindex"] == "all":
        print(
            "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!"
        )
        snapshotindex = -1
    else:
        snapshotindex = cfg["snapshotindex"]

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

    if batchsize is None:
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
    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.get_scorer_name(
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
        sess, inputs, outputs = predict.setup_GPUpose_prediction(
            dlc_cfg, allow_growth=allow_growth
        )
    else:
        sess, inputs, outputs, extra_dict = predict.setup_pose_prediction(
            dlc_cfg, allow_growth=allow_growth, collect_extra=True
        )

    pdindex = pd.MultiIndex.from_product(
        [[DLCscorer], dlc_cfg["all_joints_names"], xyz_labs],
        names=["scorer", "bodyparts", "coords"],
    )

    ##################################################
    # Looping over videos
    ##################################################
    Videos = auxiliaryfunctions.get_list_of_videos(videos, videotype)
    if len(Videos) > 0:
        if "multi-animal" in dlc_cfg["dataset_type"]:
            for video in Videos:
                extract_bpt_feature_from_video(
                    video,
                    DLCscorer,
                    trainFraction,
                    cfg,
                    dlc_cfg,
                    sess,
                    inputs,
                    outputs,
                    extra_dict,
                    destfolder=destfolder,
                    robust_nframes=robust_nframes,
                )

            # should close tensorflow session here in order to free gpu
            sess.close()
            tf.keras.backend.clear_session()
            create_triplets_dataset(
                Videos,
                DLCscorer,
                track_method,
                n_triplets=n_triplets,
                destfolder=destfolder,
            )

        else:
            raise NotImplementedError("not implemented")

        os.chdir(str(start_path))
        if "multi-animal" in dlc_cfg["dataset_type"]:
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
        print("No video(s) were found. Please check your paths and/or 'videotype'.")
        return DLCscorer


def analyze_videos(
    config,
    videos,
    videotype="",
    shuffle=1,
    trainingsetindex=0,
    gputouse=None,
    save_as_csv=False,
    in_random_order=True,
    destfolder=None,
    batchsize=None,
    cropping=None,
    TFGPUinference=True,
    dynamic=(False, 0.5, 10),
    modelprefix="",
    robust_nframes=False,
    allow_growth=False,
    use_shelve=False,
    auto_track=True,
    n_tracks=None,
    animal_names=None,
    calibrate=False,
    identity_only=False,
    use_openvino="CPU" if is_openvino_available else None,
):
    """Makes prediction based on a trained network.

    The index of the trained network is specified by parameters in the config file
    (in particular the variable 'snapshotindex').

    The labels are stored as MultiIndex Pandas Array, which contains the name of
    the network, body part name, (x, y) label position in pixels, and the
    likelihood for each frame per body part. These arrays are stored in an
    efficient Hierarchical Data Format (HDF) in the same directory where the video
    is stored. However, if the flag save_as_csv is set to True, the data can also
    be exported in comma-separated values format (.csv), which in turn can be
    imported in many programs, such as MATLAB, R, Prism, etc.

    Parameters
    ----------
    config: str
        Full path of the config.yaml file.

    videos: list[str]
        A list of strings containing the full paths to videos for analysis or a path to
        the directory, where all the videos with same extension are stored.

    videotype: str, optional, default=""
        Checks for the extension of the video in case the input to the video is a
        directory. Only videos with this extension are analyzed. If left unspecified,
        videos with common extensions ('avi', 'mp4', 'mov', 'mpeg', 'mkv') are kept.

    shuffle: int, optional, default=1
        An integer specifying the shuffle index of the training dataset used for
        training the network.

    trainingsetindex: int, optional, default=0
        Integer specifying which TrainingsetFraction to use.
        By default the first (note that TrainingFraction is a list in config.yaml).

    gputouse: int or None, optional, default=None
        Indicates the GPU to use (see number in ``nvidia-smi``). If you do not have a
        GPU put ``None``.
        See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    save_as_csv: bool, optional, default=False
        Saves the predictions in a .csv file.

    in_random_order: bool, optional (default=True)
        Whether or not to analyze videos in a random order.
        This is only relevant when specifying a video directory in `videos`.

    destfolder: string or None, optional, default=None
        Specifies the destination folder for analysis data. If ``None``, the path of
        the video is used. Note that for subsequent analysis this folder also needs to
        be passed.

    batchsize: int or None, optional, default=None
        Change batch size for inference; if given overwrites value in ``pose_cfg.yaml``.

    cropping: list or None, optional, default=None
        List of cropping coordinates as [x1, x2, y1, y2].
        Note that the same cropping parameters will then be used for all videos.
        If different video crops are desired, run ``analyze_videos`` on individual
        videos with the corresponding cropping coordinates.

    TFGPUinference: bool, optional, default=True
        Perform inference on GPU with TensorFlow code. Introduced in "Pretraining
        boosts out-of-domain robustness for pose estimation" by Alexander Mathis,
        Mert Yüksekgönül, Byron Rogers, Matthias Bethge, Mackenzie W. Mathis.
        Source: https://arxiv.org/abs/1909.11229

    dynamic: tuple(bool, float, int) triple containing (state, detectiontreshold, margin)
        If the state is true, then dynamic cropping will be performed. That means that if an object is detected (i.e. any body part > detectiontreshold),
        then object boundaries are computed according to the smallest/largest x position and smallest/largest y position of all body parts. This  window is
        expanded by the margin and from then on only the posture within this crop is analyzed (until the object is lost, i.e. <detectiontreshold). The
        current position is utilized for updating the crop window for the next frame (this is why the margin is important and should be set large
        enough given the movement of the animal).

    modelprefix: str, optional, default=""
        Directory containing the deeplabcut models to use when evaluating the network.
        By default, the models are assumed to exist in the project folder.

    robust_nframes: bool, optional, default=False
        Evaluate a video's number of frames in a robust manner.
        This option is slower (as the whole video is read frame-by-frame),
        but does not rely on metadata, hence its robustness against file corruption.

    allow_growth: bool, optional, default=False.
        For some smaller GPUs the memory issues happen. If ``True``, the memory
        allocator does not pre-allocate the entire specified GPU memory region, instead
        starting small and growing as needed.
        See issue: https://forum.image.sc/t/how-to-stop-running-out-of-vram/30551/2

    use_shelve: bool, optional, default=False
        By default, data are dumped in a pickle file at the end of the video analysis.
        Otherwise, data are written to disk on the fly using a "shelf"; i.e., a
        pickle-based, persistent, database-like object by default, resulting in
        constant memory footprint.

    The following parameters are only relevant for multi-animal projects:

    auto_track: bool, optional, default=True
        By default, tracking and stitching are automatically performed, producing the
        final h5 data file. This is equivalent to the behavior for single-animal
        projects.

        If ``False``, one must run ``convert_detections2tracklets`` and
        ``stitch_tracklets`` afterwards, in order to obtain the h5 file.

    This function has 3 related sub-calls:

    identity_only: bool, optional, default=False
        If ``True`` and animal identity was learned by the model, assembly and tracking
        rely exclusively on identity prediction.

    calibrate: bool, optional, default=False
        If ``True``, use training data to calibrate the animal assembly procedure. This
        improves its robustness to wrong body part links, but requires very little
        missing data.

    n_tracks: int or None, optional, default=None
        Number of tracks to reconstruct. By default, taken as the number of individuals
        defined in the config.yaml. Another number can be passed if the number of
        animals in the video is different from the number of animals the model was
        trained on.

    animal_names: list[str], optional
        If you want the names given to individuals in the labeled data file, you can
        specify those names as a list here. If given and `n_tracks` is None, `n_tracks`
        will be set to `len(animal_names)`. If `n_tracks` is not None, then it must be
        equal to `len(animal_names)`. If it is not given, then `animal_names` will
        be loaded from the `individuals` in the project config.yaml file.

    use_openvino: str, optional
        Use "CPU" for inference if OpenVINO is available in the Python environment.

    Returns
    -------
    DLCScorer: str
        the scorer used to analyze the videos

    Examples
    --------

    Analyzing a single video on Windows

    >>> deeplabcut.analyze_videos(
            'C:\\myproject\\reaching-task\\config.yaml',
            ['C:\\yourusername\\rig-95\\Videos\\reachingvideo1.avi'],
        )

    Analyzing a single video on Linux/MacOS

    >>> deeplabcut.analyze_videos(
            '/analysis/project/reaching-task/config.yaml',
            ['/analysis/project/videos/reachingvideo1.avi'],
        )

    Analyze all videos of type ``avi`` in a folder

    >>> deeplabcut.analyze_videos(
            '/analysis/project/reaching-task/config.yaml',
            ['/analysis/project/videos'],
            videotype='.avi',
        )

    Analyze multiple videos

    >>> deeplabcut.analyze_videos(
            '/analysis/project/reaching-task/config.yaml',
            [
                '/analysis/project/videos/reachingvideo1.avi',
                '/analysis/project/videos/reachingvideo2.avi',
            ],
        )

    Analyze multiple videos with ``shuffle=2``

    >>> deeplabcut.analyze_videos(
            '/analysis/project/reaching-task/config.yaml',
            [
                '/analysis/project/videos/reachingvideo1.avi',
                '/analysis/project/videos/reachingvideo2.avi',
            ],
            shuffle=2,
        )

    Analyze multiple videos with ``shuffle=2``, save results as an additional csv file

    >>> deeplabcut.analyze_videos(
            '/analysis/project/reaching-task/config.yaml',
            [
                '/analysis/project/videos/reachingvideo1.avi',
                '/analysis/project/videos/reachingvideo2.avi',
            ],
            shuffle=2,
            save_as_csv=True,
        )
    """
    if "TF_CUDNN_USE_AUTOTUNE" in os.environ:
        del os.environ["TF_CUDNN_USE_AUTOTUNE"]  # was potentially set during training

    if gputouse is not None:  # gpu selection
        auxfun_models.set_visible_devices(gputouse)

    tf.compat.v1.reset_default_graph()
    start_path = os.getcwd()  # record cwd to return to this directory in the end

    cfg = auxiliaryfunctions.read_config(config)
    trainFraction = cfg["TrainingFraction"][trainingsetindex]
    iteration = cfg["iteration"]

    if cropping is not None:
        cfg["cropping"] = True
        cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"] = cropping
        print("Overwriting cropping parameters:", cropping)
        print("These are used for all videos, but won't be save to the cfg file.")

    modelfolder = os.path.join(
        cfg["project_path"],
        str(
            auxiliaryfunctions.get_model_folder(
                trainFraction, shuffle, cfg, modelprefix=modelprefix
            )
        ),
    )
    path_test_config = Path(modelfolder) / "test" / "pose_cfg.yaml"
    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError(
            "It seems the model for iteration %s and shuffle %s and trainFraction %s does not exist."
            % (iteration, shuffle, trainFraction)
        )

    Snapshots = auxiliaryfunctions.get_snapshots_from_folder(
        train_folder=Path(modelfolder) / "train",
    )

    if cfg["snapshotindex"] == "all":
        print(
            "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!"
        )
        snapshotindex = -1
    else:
        snapshotindex = cfg["snapshotindex"]

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

    if batchsize is None:
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
    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.get_scorer_name(
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

    if use_openvino:
        sess, inputs, outputs = predict.setup_openvino_pose_prediction(
            dlc_cfg, device=use_openvino
        )
    elif TFGPUinference:
        sess, inputs, outputs = predict.setup_GPUpose_prediction(
            dlc_cfg, allow_growth=allow_growth
        )
    else:
        sess, inputs, outputs = predict.setup_pose_prediction(
            dlc_cfg, allow_growth=allow_growth
        )

    pdindex = pd.MultiIndex.from_product(
        [[DLCscorer], dlc_cfg["all_joints_names"], xyz_labs],
        names=["scorer", "bodyparts", "coords"],
    )

    ##################################################
    # Looping over videos
    ##################################################
    Videos = auxiliaryfunctions.get_list_of_videos(videos, videotype, in_random_order)
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
                    use_shelve=use_shelve,
                )
                if auto_track:  # tracker type is taken from default in cfg
                    convert_detections2tracklets(
                        config,
                        [video],
                        videotype,
                        shuffle,
                        trainingsetindex,
                        destfolder=destfolder,
                        modelprefix=modelprefix,
                        calibrate=calibrate,
                        identity_only=identity_only,
                    )
                    stitch_tracklets(
                        config,
                        [video],
                        videotype,
                        shuffle,
                        trainingsetindex,
                        destfolder=destfolder,
                        n_tracks=n_tracks,
                        animal_names=animal_names,
                        modelprefix=modelprefix,
                        save_as_csv=save_as_csv,
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
                    use_openvino,
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
        and cfg["x2"] < int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 1)
        and cfg["y1"] >= 0
        and cfg["y2"] < int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 1)
    ):
        pass  # good cropping box
    else:
        raise Exception("Please check the boundary of cropping!")
    return int(ny), int(nx)


def GetPoseF(cfg, dlc_cfg, sess, inputs, outputs, cap, nframes, batchsize):
    """Batchwise prediction of pose"""
    PredictedData = np.zeros(
        (nframes, dlc_cfg["num_outputs"] * 3 * len(dlc_cfg["all_joints_names"]))
    )
    batch_ind = 0  # keeps track of which image within a batch should be written to
    batch_num = 0  # keeps track of which batch you are at
    ny, nx = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
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
        if counter != 0 and counter % step == 0:
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
    """Non batch wise pose estimation for video cap."""
    if cfg["cropping"]:
        ny, nx = checkcropping(cfg, cap)

    PredictedData = np.zeros(
        (nframes, dlc_cfg["num_outputs"] * 3 * len(dlc_cfg["all_joints_names"]))
    )
    pbar = tqdm(total=nframes)
    counter = 0
    step = max(10, int(nframes / 100))
    while cap.isOpened():
        if counter != 0 and counter % step == 0:
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
            PredictedData[counter, :] = (
                pose.flatten()
            )  # NOTE: thereby cfg['all_joints_names'] should be same order as bodyparts!
        elif counter >= nframes:
            break
        counter += 1

    pbar.close()
    return PredictedData, nframes


def GetPoseS_GTF(cfg, dlc_cfg, sess, inputs, outputs, cap, nframes):
    """Non batch wise pose estimation for video cap."""
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
        if counter != 0 and counter % step == 0:
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
            PredictedData[counter, :] = (
                pose.flatten()
            )  # NOTE: thereby cfg['all_joints_names'] should be same order as bodyparts!
        elif counter >= nframes:
            break
        counter += 1

    pbar.close()
    return PredictedData, nframes


def GetPoseF_GTF(cfg, dlc_cfg, sess, inputs, outputs, cap, nframes, batchsize):
    """Batchwise prediction of pose"""
    PredictedData = np.zeros((nframes, 3 * len(dlc_cfg["all_joints_names"])))
    batch_ind = 0  # keeps track of which image within a batch should be written to
    batch_num = 0  # keeps track of which batch you are at
    ny = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if cfg["cropping"]:
        ny, nx = checkcropping(cfg, cap)

    # Flip x, y, confidence and reshape
    pose_tensor = predict.extract_GPUprediction(outputs, dlc_cfg)
    pose_tensor = tf.gather(pose_tensor, [1, 0, 2], axis=1)
    pose_tensor = tf.reshape(pose_tensor, (batchsize, -1))

    frames = np.empty((batchsize, ny, nx, 3), dtype="ubyte")
    pbar = tqdm(total=nframes)
    counter = -1
    inds = []
    while cap.isOpened() and counter < nframes - 1:
        ret, frame = cap.read()
        counter += 1
        if not ret:
            warnings.warn(f"Could not decode frame #{counter}.")
            continue

        if cfg["cropping"]:
            frame = img_as_ubyte(frame[cfg["y1"] : cfg["y2"], cfg["x1"] : cfg["x2"]])
        else:
            frame = img_as_ubyte(frame)
        frames[batch_ind] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inds.append(counter)
        if batch_ind == batchsize - 1:
            pose = sess.run(pose_tensor, feed_dict={inputs: frames})
            PredictedData[inds] = pose
            batch_ind = 0
            batch_num += 1
            inds.clear()
            pbar.update(batchsize)
        else:
            batch_ind += 1

    if batch_ind > 0:
        pose = sess.run(pose_tensor, feed_dict={inputs: frames})
        PredictedData[inds[:batch_ind]] = pose[:batch_ind]
        pbar.update(batch_ind)

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
    """Non batch wise pose estimation for video cap by dynamically cropping around previously detected parts."""
    if cfg["cropping"]:
        ny, nx = checkcropping(cfg, cap)
    else:
        ny, nx = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
    x1, x2, y1, y2 = 0, nx, 0, ny
    detected = False
    # TODO: perform detection on resized image (For speed)

    PredictedData = np.zeros((nframes, 3 * len(dlc_cfg["all_joints_names"])))
    pbar = tqdm(total=nframes)
    counter = 0
    step = max(10, int(nframes / 100))
    while cap.isOpened():
        if counter != 0 and counter % step == 0:
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
    use_openvino="CPU" if is_openvino_available else None,
):
    """Helper function for analyzing a video."""
    print("Starting to analyze % ", video)

    if destfolder is None:
        destfolder = str(Path(video).parents[0])
    auxiliaryfunctions.attempt_to_make_folder(destfolder)
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
        # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
        fps = cap.get(cv2.CAP_PROP_FPS)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = nframes * 1.0 / fps
        size = (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )
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
                args = (
                    cfg,
                    dlc_cfg,
                    sess,
                    inputs,
                    outputs,
                    cap,
                    nframes,
                    int(dlc_cfg["batch_size"]),
                )
                if use_openvino:
                    PredictedData, nframes = GetPoseF_OV(*args)
                elif TFGPUinference:
                    PredictedData, nframes = GetPoseF_GTF(*args)
                else:
                    PredictedData, nframes = GetPoseF(*args)
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
            "cropping_parameters": coords,
            # "gpu_info": device_lib.list_local_devices()
        }
        metadata = {"data": dictionary}

        print(f"Saving results in {destfolder}...")
        dataname = os.path.join(destfolder, vname + DLCscorer + ".h5")
        auxiliaryfunctions.save_data(
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
    cfg, dlc_cfg, sess, inputs, outputs, directory, framelist, nframes, batchsize
):
    """Batchwise prediction of pose for frame list in directory"""
    from deeplabcut.utils.auxfun_videos import imread

    print("Starting to extract posture")
    im = imread(os.path.join(directory, framelist[0]), mode="skimage")

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
            im = imread(os.path.join(directory, framename), mode="skimage")

            if counter != 0 and counter % step == 0:
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
            im = imread(os.path.join(directory, framename), mode="skimage")

            if counter != 0 and counter % step == 0:
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

    Examples
    --------
    If you want to analyze all frames in /analysis/project/timelapseexperiment1
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml','/analysis/project/timelapseexperiment1')
    --------

    Note: for test purposes one can extract all frames from a video with ffmeg, e.g. ffmpeg -i testvideo.avi thumb%04d.png
    """
    if "TF_CUDNN_USE_AUTOTUNE" in os.environ:
        del os.environ["TF_CUDNN_USE_AUTOTUNE"]  # was potentially set during training

    if gputouse is not None:  # gpu selection
        auxfun_models.set_visible_devices(gputouse)

    tf.compat.v1.reset_default_graph()
    start_path = os.getcwd()  # record cwd to return to this directory in the end

    cfg = auxiliaryfunctions.read_config(config)
    trainFraction = cfg["TrainingFraction"][trainingsetindex]
    modelfolder = os.path.join(
        cfg["project_path"],
        str(
            auxiliaryfunctions.get_model_folder(
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

    Snapshots = auxiliaryfunctions.get_snapshots_from_folder(
        train_folder=Path(modelfolder) / "train",
    )

    if cfg["snapshotindex"] == "all":
        print(
            "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!"
        )
        snapshotindex = -1
    else:
        snapshotindex = cfg["snapshotindex"]

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
    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.get_scorer_name(
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
        auxfun_models.set_visible_devices(gputouse)

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
        notanalyzed, dataname, DLCscorer = auxiliaryfunctions.check_if_not_analyzed(
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

                auxiliaryfunctions.save_data(
                    PredictedData[:nframes, :],
                    metadata,
                    dataname,
                    pdindex,
                    framelist,
                    save_as_csv,
                )
                print("The folder was analyzed. Now your research can truly start!")
                print(
                    "If the tracking is not satisfactory for some frame, consider expanding the training set."
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
    greedy=False,
    calibrate=False,
):
    track_method = cfg.get("default_track_method", "ellipse")
    if track_method not in trackingutils.TRACK_METHODS:
        raise ValueError(
            f"Invalid tracking method. Only {', '.join(trackingutils.TRACK_METHODS)} are currently supported."
        )

    if track_method == "ctd":
        raise ValueError(
            "CTD tracking is only available for BUCTD models with the PyTorch engine."
        )

    joints = data["metadata"]["all_joints_names"]
    partaffinityfield_graph = data["metadata"]["PAFgraph"]
    paf_inds = data["metadata"]["PAFinds"]
    paf_graph = [partaffinityfield_graph[l] for l in paf_inds]
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

    assembly_builder = inferenceutils.Assembler(
        data,
        max_n_individuals=inference_cfg["topktoretain"],
        n_multibodyparts=len(cfg["multianimalbodyparts"]),
        graph=paf_graph,
        paf_inds=list(paf_inds),
        greedy=greedy,
        pcutoff=inference_cfg.get("pcutoff", 0.1),
        min_affinity=inference_cfg.get("pafthreshold", 0.05),
        min_n_links=inference_cfg["minimalnumberofconnections"]
    )
    if calibrate:
        trainingsetfolder = auxiliaryfunctions.get_training_set_folder(cfg)
        train_data_file = os.path.join(
            cfg["project_path"],
            str(trainingsetfolder),
            "CollectedData_" + cfg["scorer"] + ".h5",
        )
        assembly_builder.calibrate(train_data_file)
    assembly_builder.assemble()

    output_path, _ = os.path.splitext(output_path)
    output_path += ".pickle"
    assembly_builder.to_pickle(output_path.replace(".pickle", "_assemblies.pickle"))

    if cfg["uniquebodyparts"]:
        tracklets["single"] = {}
        tracklets["single"].update(assembly_builder.unique)

    for i, imname in tqdm(enumerate(assembly_builder.metadata["imnames"])):
        assemblies = assembly_builder.assemblies.get(i)
        if assemblies is None:
            continue
        animals = np.stack(
            [assembly_builder.data[:, :3] for assembly_builder in assemblies]
        )
        if track_method == "box":
            xy = trackingutils.calc_bboxes_from_keypoints(
                animals, inference_cfg.get("boundingboxslack", 0)
            )  # TODO: get cropping parameters and utilize!
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
    videotype="",
    shuffle=1,
    trainingsetindex=0,
    overwrite=False,
    destfolder=None,
    ignore_bodyparts=None,
    inferencecfg=None,
    modelprefix="",
    greedy=False,
    calibrate=False,
    window_size=0,
    identity_only=False,
    track_method="",
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
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed.
        If left unspecified, videos with common extensions ('avi', 'mp4', 'mov', 'mpeg', 'mkv') are kept.

    shuffle: int, optional
        An integer specifying the shuffle index of the training dataset used for training the network. The default is 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    overwrite: bool, optional.
        Overwrite tracks file i.e. recompute tracks from full detections and overwrite.

    destfolder: string, optional
        Specifies the destination folder for analysis data (default is the path of the video). Note that for subsequent analysis this
        folder also needs to be passed.

    ignore_bodyparts: optional
        List of body part names that should be ignored during tracking (advanced).
        By default, all the body parts are used.

    inferencecfg: Default is None.
        Configuration file for inference (assembly of individuals). Ideally
        should be obtained from cross validation (during evaluation). By default
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

    track_method: string, optional
         Specifies the tracker used to generate the pose estimation data.
         For multiple animals, must be either 'box', 'skeleton', or 'ellipse'
         and will be taken from the config.yaml file if none is given.


    Examples
    --------
    If you want to convert detections to tracklets:
    >>> deeplabcut.convert_detections2tracklets('/analysis/project/reaching-task/config.yaml',[]'/analysis/project/video1.mp4'], videotype='.mp4')

    If you want to convert detections to tracklets based on box_tracker:
    >>> deeplabcut.convert_detections2tracklets('/analysis/project/reaching-task/config.yaml',[]'/analysis/project/video1.mp4'], videotype='.mp4',track_method='box')

    --------

    """
    cfg = auxiliaryfunctions.read_config(config)
    track_method = auxfun_multianimal.get_track_method(cfg, track_method=track_method)

    if len(cfg["multianimalbodyparts"]) == 1 and track_method != "box":
        warnings.warn("Switching to `box` tracker for single point tracking...")
        track_method = "box"
        cfg["default_track_method"] = track_method
        auxiliaryfunctions.write_config(config, cfg)

    trainFraction = cfg["TrainingFraction"][trainingsetindex]
    start_path = os.getcwd()  # record cwd to return to this directory in the end

    # TODO: add cropping as in video analysis!
    # if cropping is not None:
    #    cfg['cropping']=True
    #    cfg['x1'],cfg['x2'],cfg['y1'],cfg['y2']=cropping
    #    print("Overwriting cropping parameters:", cropping)
    #    print("These are used for all videos, but won't be save to the cfg file.")

    modelfolder = os.path.join(
        cfg["project_path"],
        str(
            auxiliaryfunctions.get_model_folder(
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

    if len(cfg["multianimalbodyparts"]) == 1 and track_method != "box":
        warnings.warn("Switching to `box` tracker for single point tracking...")
        track_method = "box"
        # Also ensure `boundingboxslack` is greater than zero, otherwise overlap
        # between trackers cannot be evaluated, resulting in empty tracklets.
        inferencecfg["boundingboxslack"] = max(inferencecfg["boundingboxslack"], 40)

    Snapshots = auxiliaryfunctions.get_snapshots_from_folder(
        train_folder=Path(modelfolder) / "train",
    )

    if cfg["snapshotindex"] == "all":
        print(
            "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!"
        )
        snapshotindex = -1
    else:
        snapshotindex = cfg["snapshotindex"]

    print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)
    dlc_cfg["init_weights"] = os.path.join(
        modelfolder, "train", Snapshots[snapshotindex]
    )
    trainingsiterations = (dlc_cfg["init_weights"].split(os.sep)[-1]).split("-")[-1]

    # Name for scorer:
    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.get_scorer_name(
        cfg,
        shuffle,
        trainFraction,
        trainingsiterations=trainingsiterations,
        modelprefix=modelprefix,
    )

    ##################################################
    # Looping over videos
    ##################################################
    Videos = auxiliaryfunctions.get_list_of_videos(videos, videotype)
    if len(Videos) > 0:
        for video in Videos:
            print("Processing... ", video)
            videofolder = str(Path(video).parents[0])
            if destfolder is None:
                destfolder = videofolder
            auxiliaryfunctions.attempt_to_make_folder(destfolder)
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
                bodypartlabels = [
                    bpt for i, bpt in enumerate(all_jointnames) for _ in range(3)
                ]
                scorers = len(bodypartlabels) * [DLCscorer]
                xylvalue = int(len(bodypartlabels) / 3) * ["x", "y", "likelihood"]
                pdindex = pd.MultiIndex.from_arrays(
                    np.vstack([scorers, bodypartlabels, xylvalue]),
                    names=["scorer", "bodyparts", "coords"],
                )

                imnames = [fn for fn in data if fn != "metadata"]

                if track_method == "box":
                    mot_tracker = trackingutils.SORTBox(
                        inferencecfg["max_age"],
                        inferencecfg["min_hits"],
                        inferencecfg.get("iou_threshold", 0.3),
                    )
                elif track_method == "skeleton":
                    mot_tracker = trackingutils.SORTSkeleton(
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
                assembly_builder = inferenceutils.Assembler(
                    data,
                    max_n_individuals=inferencecfg["topktoretain"],
                    n_multibodyparts=len(multi_bpts),
                    greedy=greedy,
                    pcutoff=inferencecfg.get("pcutoff", 0.1),
                    min_affinity=inferencecfg.get("pafthreshold", 0.05),
                    window_size=window_size,
                    identity_only=identity_only,
                    min_n_links=inferencecfg["minimalnumberofconnections"]
                )
                assemblies_filename = dataname.split(".h5")[0] + "_assemblies.pickle"
                if not os.path.exists(assemblies_filename) or overwrite:
                    if calibrate:
                        trainingsetfolder = auxiliaryfunctions.get_training_set_folder(
                            cfg
                        )
                        train_data_file = os.path.join(
                            cfg["project_path"],
                            str(trainingsetfolder),
                            "CollectedData_" + cfg["scorer"] + ".h5",
                        )
                        assembly_builder.calibrate(train_data_file)
                    assembly_builder.assemble()
                    assembly_builder.to_pickle(assemblies_filename)
                else:
                    assembly_builder.from_pickle(assemblies_filename)
                    print(f"Loading assemblies from {assemblies_filename}")
                try:
                    data.close()
                except AttributeError:
                    pass

                if cfg[
                    "uniquebodyparts"
                ]:  # Initialize storage of the 'single' individual track
                    tracklets["single"] = {}
                    _single = {}
                    for index, imname in enumerate(imnames):
                        single_detection = assembly_builder.unique.get(index)
                        if single_detection is None:
                            continue
                        imindex = int(re.findall(r"\d+", imname)[0])
                        _single[imindex] = single_detection
                    tracklets["single"].update(_single)

                if inferencecfg["topktoretain"] == 1:
                    tracklets[0] = {}
                    for index, imname in tqdm(enumerate(imnames)):
                        assemblies = assembly_builder.assemblies.get(index)
                        if assemblies is None:
                            continue
                        tracklets[0][imname] = assemblies[0].data
                else:
                    keep = set(multi_bpts).difference(ignore_bodyparts or [])
                    keep_inds = sorted(multi_bpts.index(bpt) for bpt in keep)
                    for index, imname in tqdm(enumerate(imnames)):
                        assemblies = assembly_builder.assemblies.get(index)
                        if assemblies is None:
                            continue
                        animals = np.stack(
                            [assembly_builder.data for assembly_builder in assemblies]
                        )
                        if not identity_only:
                            if track_method == "box":
                                xy = trackingutils.calc_bboxes_from_keypoints(
                                    animals[:, keep_inds],
                                    inferencecfg["boundingboxslack"],
                                )  # TODO: get cropping parameters and utilize!
                            else:
                                xy = animals[:, keep_inds, :2]
                            trackers = mot_tracker.track(xy)
                        else:
                            # Optimal identity assignment based on soft voting
                            mat = np.zeros(
                                (len(assemblies), inferencecfg["topktoretain"])
                            )
                            for nrow, assembly in enumerate(assemblies):
                                for k, v in assembly.soft_identity.items():
                                    mat[nrow, k] = v
                            inds = linear_sum_assignment(mat, maximize=True)
                            trackers = np.c_[inds][:, ::-1]
                        trackingutils.fill_tracklets(
                            tracklets, trackers, animals, imname
                        )

                tracklets["header"] = pdindex
                with open(trackname, "wb") as f:
                    pickle.dump(tracklets, f, pickle.HIGHEST_PROTOCOL)

        os.chdir(str(start_path))

        print(
            "The tracklets were created (i.e., under the hood deeplabcut.convert_detections2tracklets was run). Now you can 'refine_tracklets' in the GUI, or run 'deeplabcut.stitch_tracklets'."
        )
    else:
        print("No video(s) found. Please check your path!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("config")
    cli_args = parser.parse_args()
