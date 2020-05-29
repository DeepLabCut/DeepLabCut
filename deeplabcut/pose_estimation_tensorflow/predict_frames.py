"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

"""
from typing import Iterable, Union, Optional, List, Type, Dict, Any
from os import PathLike

from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.pose_estimation_tensorflow.nnet import processing
from deeplabcut.pose_estimation_tensorflow.predict_videos import GetPredictorSettings
from pathlib import Path
import pandas as pd
import numpy as np
import deeplabcut.pose_estimation_tensorflow.util.frame_store_fmt as frame_store_fmt
import tqdm
import time

# Represents strings and any filesystem path-like type....
Pathy = Union[PathLike, str]

def analyze_frame_store(config_path: Pathy, frame_stores: Union[Iterable[Pathy], Pathy],
                        predictor: Optional[str] = None, save_as_csv: bool = False,
                        multi_output_format: str = "default", video_folders: Union[None, Pathy, Iterable[Pathy]]=None,
                        num_outputs: Optional[int] = None, shuffle = 1, trainingsetindex = 0,
                        predictor_settings: Optional[Dict[str, Any]] = None):
    """
    Takes a DeepLabCut Frame Store file (.dlcf) and makes predictions for the stored frames, using whatever predictor
    plugin is selected. This allows for the video to be run through the Deep Neural Network once, and then run through
    several prediction algorithms as many times as desired, saving time. It also allows for frames to be processed
    on one computer to be transferred to another computer for post-processing and predictions.

    :param config_path: The path to the DLC config to use to interpret this data. The .DLCF will inherit the neural
                        network of this project, allowing for frame labeling using this project.
    :param frame_stores: The paths to the frame stores (.dlcf files), string or list of strings.
    :param predictor: A String, the name of the predictor plugin to be used to make predictions.
    :param save_as_csv: A Boolean, True to save the results to the human readable .csv format, otherwise false.
    :param multi_output_format: A string. Determines the multi output format used. "default" uses the default format,
                                while "separate-bodyparts" separates the multi output predictions such that each is its
                                own body part.
    :param video_folders: None, a string, or a list of strings, folders to search through to find videos which
                          correlate to the .dlcf files. If set to None, this method will search for the corresponding
                          videos in the directory each .dlcf file is contained in.
    :param num_outputs: int, default: from config.yaml, or 1 if not set in config.yaml.
                        Allows the user to set the number of predictions for bodypart,
                        overriding the option in the config file.
    :param shuffle: int, optional. An integer specifying the shuffle index of the training dataset used for training
                    the network. The default is 1.
    :param trainingsetindex: int, optional. Integer specifying which TrainingsetFraction to use. By default the first
                             (note that TrainingFraction is a list in config.yaml).
    :param predictor_settings: Optional dictionary of strings to any. This will specify what settings a predictor should use,
                        completely ignoring any settings specified in the config.yaml. Default value is None, which
                        tells this method to use the settings specified in the config.yaml.

    :return: The labels are stored as MultiIndex Pandas Array, which contains the name of the network, body part name,
            (x, y) label position in pixels, and the likelihood for each frame per body part. These arrays are stored
            in an efficient Hierarchical Data Format (HDF) in the same directory, where the video is stored. However,
            if the flag save_as_csv is set to True, the data can also be exported in comma-separated values format
            (.csv), which in turn can be imported in many programs, such as MATLAB, R, Prism, etc.
    """
    # Grab the name of the current DLC Scorer, hack as DLCs Plot functions require a scorer, which is dumb. If it fails,
    # we just call the model 'Unknown' :). Simply means user won't be able to use create_labeled_video, data is still
    # 100% valid.
    cfg = auxiliaryfunctions.read_config(config_path)
    train_frac = cfg["TrainingFraction"][trainingsetindex]

    try:
        dlc_scorer, dlc_scorer_legacy = auxiliaryfunctions.GetScorerName(cfg, shuffle, train_frac)
    except Exception:
        dlc_scorer, dlc_scorer_legacy = "Unknown", "Unknown"

    # Convert all of the path lists to Path objects, useful later...
    frame_stores = _sanitize_path_arg(frame_stores)
    if(frame_stores is None):
        raise ValueError("Path must be PathLike, a string, a list of PathLike, or a list of strings.")

    video_folders = _sanitize_path_arg(video_folders)
    # Video files are not required, but some plugins do rely on them, so this code tries to resolve videos...
    video_files = _resolve_videos(frame_stores, video_folders)

    # Get the number of outputs...
    num_outputs = max(1, cfg.get('num_outputs', 1)) if(num_outputs is None) else max(1, int(num_outputs))

    # Loading the predictor plugin
    if(predictor is None):
        predictor = "argmax"
    predictor_cls = processing.get_predictor(predictor)

    # Check and make sure that this predictor supports multi output if we are currently in that mode...
    if((num_outputs > 1) and (not predictor_cls.supports_multi_output())):
        raise NotImplementedError("The selected predictor plugin doesn't support multiple outputs!!!")

    for frame_store_path, video_path in zip(frame_stores, video_files):
        _analyze_frame_store(cfg, frame_store_path, video_path, dlc_scorer, dlc_scorer_legacy, predictor_cls,
                             multi_output_format, num_outputs, train_frac, save_as_csv, predictor_settings)

    print("Analysis and Predictions are Done! Now your research can truly start!")
    return dlc_scorer


def _resolve_videos(frame_store_paths: List[Path], video_folders: Optional[List[Path]]) -> List[Optional[Path]]:
    """
    Private: Resolves the video paths of the frame stores. The entry for a frame store will be None if the original
    video path can't be found.
    """
    video_paths = [None] * len(frame_store_paths)
    expected_video_names = {"~".join(path.stem.split("~")[:-1]): idx for idx, path in enumerate(frame_store_paths)}
    expected_video_names = {name: idx for name, idx in expected_video_names.items() if(name.strip() != "")}
    print(expected_video_names)

    # If the user passed video folders to check, check them, searching for all matching videos with the same name.
    if(video_folders is not None):
        for path in video_folders:
            if(path.is_dir()):
                for subpath in path.iterdir():
                    if((subpath.name in expected_video_names) and (video_paths[expected_video_names[subpath.name]] is None)):
                        video_paths[expected_video_names[subpath.name]] = subpath

    # Check if the video exists in the same folder as the .dlcf, if so add it. Overrides video folder search above...
    for idx, path in enumerate(frame_store_paths):
        name = "~".join(path.stem.split("~")[:-1])
        if(name.strip() != ""):
            suspect_video = (path.parent) / (name)
            if(suspect_video.exists()):
                video_paths[idx] = suspect_video

    return video_paths


def _sanitize_path_arg(paths: Union[None, Iterable[Pathy], Pathy]) -> Optional[List[Path]]:
    """
    Sanitizes a pathlike or list of pathlike argument and returns a list of Path, or None if rogue data was passed...
    """
    if(isinstance(paths, (PathLike, str))):
        return [Path(str(paths))]
    elif(isinstance(paths, Iterable) ):
        paths = list(paths)
        if(len(paths) > 0):
            return [Path(str(path)) for path in paths]
        else:
            return None
    else:
        return None


def _get_pandas_header(body_parts: List[str], num_outputs: int, out_format: str, dlc_scorer: str) -> pd.MultiIndex:
    """
    Creates the pandas data header for the passed body parts and number of outputs.

    :param body_parts: The list of body part names. List of strings.
    :param num_outputs: The number of outputs per body part, and integer.
    :param out_format: The output format, either 'separate-bodyparts' or 'default'.
    :param dlc_scorer: A string, being the name of the DLC Scorer for this DLC instance.
    :return: A pandas MultiIndex, being the header entries for the DLC output data.
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
        out_format = "default"
        suffixes = [str(i + 1) for i in range(num_outputs)]
        suffixes[0] = ""
        sub_headers = [state + s for s in suffixes for state in ['x', 'y', 'likelihood']]
        return pd.MultiIndex.from_product([[dlc_scorer], body_parts, sub_headers],
                                             names=['scorer', 'bodyparts', 'coords'])


def _analyze_frame_store(cfg: dict, frame_store_path: Path, video_name: Optional[str], dlc_scorer: str,
                         dlc_scorer_legacy: str, predictor_cls: Type[Predictor], multi_output_format: str,
                         num_outputs: int, train_frac: str, save_as_csv: bool,
                         predictor_settings: Optional[Dict[str, Any]]) -> str:
    # Check if the data was analyzed yet...
    v_name_sanitized = Path(video_name).resolve().stem if(video_name is not None) else "unknownVideo"
    print(v_name_sanitized)
    not_analyzed, data_name, dlc_scorer = auxiliaryfunctions.CheckifNotAnalyzed(str(frame_store_path.parent),
                                                                                v_name_sanitized, dlc_scorer,
                                                                                dlc_scorer_legacy)

    if(not_analyzed):
        # Read the frame store into memory:
        with frame_store_path.open("rb") as fb:
            print(f"Processing '{frame_store_path.name}'")
            start = time.time()

            # Read in the header, setup the settings.
            frame_reader = frame_store_fmt.DLCFSReader(fb)
            num_f, f_h, f_w, f_rate, stride, vid_h, vid_w, off_y, off_x, bp_lst = frame_reader.get_header().to_list()
            pd_index = _get_pandas_header(bp_lst, num_outputs, multi_output_format, dlc_scorer)

            predictor_settings = GetPredictorSettings(cfg, predictor_cls, predictor_settings)

            video_metadata = {
                "fps": f_rate,
                "duration": float(num_f) / f_rate,
                "size": (vid_h, vid_w),
                "h5-file-name": data_name,
                "orig-video-path": video_name, # This may be None if we were unable to find the video...
                "cropping-offset": None if(off_x is None or off_y is None) else (off_y, off_x)
            }

            # Create the plugin instance...
            print(f"Plugin {predictor_cls.get_name()} Settings: {predictor_settings}")
            predictor_inst = predictor_cls(bp_lst, num_outputs, num_f, predictor_settings, video_metadata)

            # The pose prediction final output array...
            pose_prediction_data = np.zeros((num_f, 3 * len(bp_lst) * num_outputs))

            # Begin running through frames...
            p_bar = tqdm.tqdm(total=num_f)
            frames_done = 0

            while(frame_reader.has_next()):
                frame = frame_reader.read_frames()
                pose = predictor_inst.on_frames(frame)
                if(pose is not None):
                    # If the predictor returned a pose, add it to the final data.
                    pose_prediction_data[frames_done:frames_done + pose.get_frame_count()] = pose.get_all()
                    frames_done += pose.get_frame_count()

                p_bar.update()

            p_bar.close()

            # Post-Processing Phase:
            # Phase 2: Post processing...

            # Get all of the final poses that are still held by the predictor
            post_pbar = tqdm.tqdm(total=num_f - frames_done)
            final_poses = predictor_inst.on_end(post_pbar)
            post_pbar.close()

            # Add any post-processed frames
            if (final_poses is not None):
                pose_prediction_data[frames_done:frames_done + final_poses.get_frame_count()] = final_poses.get_all()
                frames_done += final_poses.get_frame_count()

            # Check and make sure the predictor returned all frames, otherwise throw an error.
            if (frames_done != num_f):
                raise ValueError(
                    f"The predictor algorithm did not return the same amount of frames as are in the video.\n"
                    f"Expected Amount: {num_f}, Actual Amount Returned: {frames_done}")

            stop = time.time()

            if cfg['cropping']:
                coords = [cfg['x1'], cfg['x2'], cfg['y1'], cfg['y2']]
            else:
                coords = [0, vid_w, 0, vid_h]

            sub_meta = {
                "start": start,
                "stop": stop,
                "run_duration": stop - start,
                "Scorer": dlc_scorer,
                "DLC-model-config file": None, # We don't have access to this, so don't even try....
                "fps": f_rate,
                "num_outputs": num_outputs,
                "batch_size": 1,
                "multi_output_format": multi_output_format,
                "frame_dimensions": (f_h * stride, f_w * stride),
                "nframes": num_f,
                "iteration (active-learning)": cfg["iteration"],
                "training set fraction": train_frac,
                "cropping": cfg['cropping'],
                "cropping_parameters": coords
            }
            metadata = {'data': sub_meta}

            # We are Done!!! Save data and return...
            auxiliaryfunctions.SaveData(pose_prediction_data, metadata, data_name, pd_index, range(num_f), save_as_csv)

    return dlc_scorer


