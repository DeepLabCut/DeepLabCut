import os
import os.path
import pickle
import time
from pathlib import Path

import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
from skimage.util import img_as_ubyte
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from deeplabcut.modelzoo.utils import parse_available_supermodels
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.core import \
    predict as single_predict
from deeplabcut.pose_estimation_tensorflow.core import \
    predict_multianimal as predict
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.auxfun_videos import VideoWriter
from dlclibrary.dlcmodelzoo.modelzoo_download import (
    download_huggingface_model,
    MODELOPTIONS,
)
from scipy import signal

import glob



# instead of having these in a lengthy function, I made this a separate function
def get_nuances(
    videos,
    test_cfg,
    videotype="avi",
    destfolder=None,
    batchsize=None,
    allow_growth=False,
    init_weights="",
):
    test_cfg["num_outputs"] = 1
    test_cfg["batch_size"] = batchsize

    sess, inputs, outputs = single_predict.setup_pose_prediction(
        test_cfg, allow_growth=allow_growth
    )
    DLCscorer = "DLC_" + Path(init_weights).stem

    Videos = auxiliaryfunctions.get_list_of_videos(videos, videotype)

    ret = {}
    ret["videos"] = Videos
    ret["DLCscorer"] = DLCscorer
    ret["test_cfg"] = test_cfg
    ret["sess"] = sess
    ret["inputs"] = inputs
    ret["outputs"] = outputs
    ret["destfolder"] = destfolder
    ret["init_weights"] = init_weights
    return ret


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
    preds, scale_list, num_kpts, cos_dist_threshold=0.997, confidence_threshold=0.1
):

    ret_pred = {}

    ret_pred["coordinates"] = [[[]] * num_kpts]
    ret_pred["confidence"] = [[]] * num_kpts

    for scale_id, pred in enumerate(preds):
        # better handle the case where the pred is empty
        if not len(pred):
            coordinate = [[]] * num_kpts
            confidence = [[]] * num_kpts
        else:
            coordinate = pred["coordinates"][0]
            confidence = pred["confidence"]

        for kpt_id, coord_list in enumerate(coordinate):
            if len(ret_pred["coordinates"][0][kpt_id]) == 0:
                ret_pred["coordinates"][0][kpt_id] = [[]] * len(scale_list)
                ret_pred["confidence"][kpt_id] = [[]] * len(scale_list)

            temp_coord = np.expand_dims(coord_list, axis=0)
            ret_pred["coordinates"][0][kpt_id][scale_id] = temp_coord
            temp_confidence = np.expand_dims(confidence[kpt_id], axis=0)
            ret_pred["confidence"][kpt_id][scale_id] = temp_confidence

    for kpt_id in range(num_kpts):

        remove_indices = []
        for idx, ele in enumerate(ret_pred["coordinates"][0][kpt_id]):
            if len(ele) == 0 or len(ele[0]) == 0:
                remove_indices.append(idx)

        for idx, ele in enumerate(ret_pred["coordinates"][0][kpt_id]):
            if idx in remove_indices:
                # using [0,0] instead of [nan,nan] for cosine similarity to correctly pick up distances
                ret_pred["coordinates"][0][kpt_id][idx] = np.array([[0, 0]])
                ret_pred["confidence"][kpt_id][idx] = np.array([[0]])

        mean_vec = np.nanmedian(np.array(ret_pred["coordinates"][0][kpt_id]), axis=0)
        candidates = np.array(ret_pred["coordinates"][0][kpt_id])
        dist = []

        for i in range(len(candidates)):
            # In case where the the predictions do not exist
            dist.append(cosine_similarity(candidates[i], mean_vec))

        filter_indices = []

        for idx, ele in enumerate(ret_pred["coordinates"][0][kpt_id]):
            if (
                dist[idx] < cos_dist_threshold
                or ret_pred["confidence"][kpt_id][idx] < confidence_threshold
            ):

                filter_indices.append(idx)

        for idx, ele in enumerate(ret_pred["coordinates"][0][kpt_id]):
            if idx in filter_indices:
                ret_pred["coordinates"][0][kpt_id][idx] = np.array([[np.nan, np.nan]])
                ret_pred["confidence"][kpt_id][idx] = np.array([[np.nan]])

        if len(ret_pred["coordinates"][0][kpt_id]) != 0:

            ret_pred["coordinates"][0][kpt_id] = np.concatenate(
                ret_pred["coordinates"][0][kpt_id], axis=0
            )
            ret_pred["confidence"][kpt_id] = np.concatenate(
                ret_pred["confidence"][kpt_id], axis=0
            )

            # need np.array for wrapping the list for evaluation code to work correctly
            ret_pred["coordinates"][0][kpt_id] = np.array(
                [np.nanmedian(np.array(ret_pred["coordinates"][0][kpt_id]), axis=0)]
            )
            ret_pred["confidence"][kpt_id] = np.array(
                [np.nanmedian(np.array(ret_pred["confidence"][kpt_id]), axis=0)]
            )
        else:
            ret_pred["coordinates"][0][kpt_id] = np.array([[np.nan, np.nan]])
            ret_pred["confidence"][kpt_id] = np.array([[np.nan]])
    return ret_pred


def _video_inference(
    test_cfg,
    sess,
    inputs,
    outputs,
    cap,
    nframes,
    batchsize,
    invert_color=False,
    scale_list=[],
    apply_filter = False
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
            if invert_color:
                frame = 255 - frame

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


def video_inference_superanimal(
    videos,
    superanimal_name,
    scale_list=[],
    invert_color=False,
    videotype="avi",
    destfolder=None,
    batchsize=1,
    robust_nframes=False,
    allow_growth=False,
    init_weights = "",
    customized_test_config="",
    apply_filter = False,
):
    """
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
        The name of the superanimal model. We currently only support supertopview and superquadruped
    scale_list: list
        A list of int containing the target height of the multi scale test time augmentation. By default it uses the original size. Users are advised to try a wide range of scale list when the super model does not give reasonable results

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed. The default is ``.avi``

    destfolder: string, optional
        Specifies the destination folder for analysis data (default is the path of the video). Note that for subsequent analysis this
        folder also needs to be passed.

    batchsize: int, default from pose_cfg.yaml
        Change batch size for inference; if given overwrites value in pose_cfg.yaml

    TFGPUinference: bool, default: True
        Perform inference on GPU with TensorFlow code. Introduced in "Pretraining boosts out-of-domain robustness for pose estimation" by
        Alexander Mathis, Mert Yüksekgönül, Byron Rogers, Matthias Bethge, Mackenzie W. Mathis Source: https://arxiv.org/abs/1909.11229

    robust_nframes: bool, optional (default=False)
        Evaluate a video's number of frames in a robust manner.
        This option is slower (as the whole video is read frame-by-frame),
        but does not rely on metadata, hence its robustness against file corruption.

    allow_growth: bool, default false.
        For some smaller GPUs the memory issues happen. If true, the memory allocator does not pre-allocate the entire specified
        GPU memory region, instead starting small and growing as needed. See issue: https://forum.image.sc/t/how-to-stop-running-out-of-vram/30551/2
    init_weights: str, default "".
        Customized path for inference

    Given a list of scales for spatial pyramid, i.e. [600, 700]

    scale_list = range(600,800,100)

    superanimal_name = 'superanimal_topviewmouse'
    videotype = 'mp4'
    >>>  init_weights = deeplabcut.video_inference_superanimal(
                                      [video_path],
                                      superanimal_name,
                                      videotype=videotype,
                                      scale_list = scale_list,
                                     )
    Note we do not need to pass a config in this case
    >>> deeplabcut.create_labeled_video("",
                                [video_path],
                                videotype = videotype,
                                superanimal_name = superanimal_name,
                                init_weights = init_weights,
                                pcutoff = 0.6
                                )


    """

    dlc_root_path = auxiliaryfunctions.get_deeplabcut_path()

    if customized_test_config == "":
        supermodels = parse_available_supermodels()
        test_cfg = load_config(
            os.path.join(
                dlc_root_path,
                "pose_estimation_tensorflow",
                "superanimal_configs",
                supermodels[superanimal_name],
            )
        )
    else:
        test_cfg = load_config(customized_test_config)


    # add a temp folder for checkpoint

    weight_folder = superanimal_name + '_weights'
    if superanimal_name in MODELOPTIONS:
        if not os.path.exists(weight_folder):
            download_huggingface_model(superanimal_name, weight_folder)
        else:
            print (f"{weight_folder} exists, using the downloaded weights")
    else:
        print ('do not have that weight yet')
    
    snapshots = glob.glob(
        os.path.join(weight_folder, 'snapshot-*.index')
    )
    if init_weights == "":
        init_weights = os.path.abspath(snapshots[0]).replace('.index', '')
            
    test_cfg["partaffinityfield_graph"] = []
    test_cfg["partaffinityfield_predict"] = False

    if init_weights != "":
        test_cfg["init_weights"] = init_weights

    setting = get_nuances(
        videos,
        test_cfg,
        videotype=videotype,
        destfolder=destfolder,
        batchsize=batchsize,
        allow_growth=allow_growth,
        init_weights=init_weights,
    )

    videos = setting["videos"]
    destfolder = setting["destfolder"]
    DLCscorer = setting["DLCscorer"]
    sess = setting["sess"]
    inputs = setting["inputs"]
    outputs = setting["outputs"]

    for video in videos:

        vname = Path(video).stem

        videofolder = str(Path(video).parents[0])
        if destfolder is None:
            destfolder = videofolder
            auxiliaryfunctions.attempttomakefolder(destfolder)

        dataname = os.path.join(destfolder, vname + DLCscorer + ".h5")

        if os.path.isfile(dataname):
            print("Video already analyzed!", dataname)
        else:
            print("Loading ", video)
            vid = VideoWriter(video)
            if len(scale_list) == 0:
                # if the scale_list is empty, by default we use the original one
                scale_list = [vid.height]
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
            print("before inference")
            PredicteData, nframes = _video_inference(
                test_cfg,
                sess,
                inputs,
                outputs,
                vid,
                nframes,
                int(test_cfg["batch_size"]),
                invert_color=invert_color,
                scale_list=scale_list,
                apply_filter = apply_filter
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
                        temp[n, :2] = xy[0]
                        temp[n, 2] = c[0]
                    data[i] = temp.flatten()
            df = pd.DataFrame(data, columns=columnindex, index=imagenames)

            if apply_filter:
                data = df.copy()
                
                mask = df.columns.get_level_values("coords") != "likelihood"
                data.loc[:, mask] = df.loc[:, mask].apply(
                        signal.medfilt, args=(41,), axis=0
                ).to_numpy()
                df = data
                
                
            df.to_hdf(dataname, key="df_with_missing")
            
    return init_weights
