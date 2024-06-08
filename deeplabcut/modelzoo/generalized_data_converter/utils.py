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
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.modelzoo.generalized_data_converter.datasets.materialize import (
    SingleDLC_config,
)


def threshold_kpts(config_path, h5path, threshold_mean=0.9, threshold_min=0.1):

    df = pd.read_hdf(h5path)

    scorer = df.columns.get_level_values("scorer").unique()[0]
    try:
        data = df[scorer]["individual0"]
    except:
        data = df[scorer]

    cfg = auxiliaryfunctions.read_config(config_path)

    bodyparts = cfg["multianimalbodyparts"]

    thresholded_bpts = []

    for bpt in bodyparts:
        _mean = data[bpt]["likelihood"].mean()
        _min = data[bpt]["likelihood"].min()
        _var = data[bpt]["likelihood"].var()
        if _mean > threshold_mean and _min > threshold_min:
            thresholded_bpts.append(bpt)
        print(bpt, "mean", _mean)
        print(bpt, "min", _min)
        print(bpt, "var", _var)

    print("thresholded kpts", thresholded_bpts)
    return thresholded_bpts
    ret = []
    print(ret)
    return ret


def create_dummy_config_file_from_h5(
    proj_root, reference_h5, taskname="dummytask", scorer="dummyscorer", date="March30"
):
    """
    Assuming at least labeled-data folder is there
    """

    cfg_template = SingleDLC_config()

    df = pd.read_hdf(reference_h5)

    print(df)

    pattern = glob.glob(os.path.join(proj_root, "labeled-data", "*"))

    labeled_folders = [f.split("/")[-1] for f in pattern]

    video_sets = {
        f"{folder}.mp4": {"crop": "0, 400, 0, 400"} for folder in labeled_folders
    }

    # bodyparts = df[scorer]['bodyparts']

    bodyparts = list(df.columns.get_level_values("bodyparts").unique())
    scorer = df.columns.get_level_values("scorer").unique()[0]

    modify_dict = dict(
        Task=taskname,
        project_path=proj_root,
        scorer=scorer,
        date=date,
        video_sets=video_sets,
        bodyparts=bodyparts,
        TrainingFraction=[0.95],
    )

    cfg_template.create_cfg(proj_root, modify_dict)


def create_dummy_config_file_from_pickle(
    proj_root,
    reference_pickle,
    video_path,
    taskname="dummytask",
    scorer="dummyscorer",
    date="March30",
):
    """
    Assuming at least labeled-data folder is there
    """

    cfg_template = SingleDLC_config()

    with open(reference_pickle, "rb") as f:

        pickle_obj = pickle.load(f)

    # bodyparts  = pickle_obj['keypoint_names']
    bodyparts = [
        "tail",
        "spine4",
        "spine3",
        "spine2",
        "spine1",
        "head",
        "nose",
        "right ear",
        "left ear",
    ]

    video_name = video_path.split("/")[-1]

    video_sets = {f"{video_path}": {"crop": "0, 400, 0, 400"}}

    modify_dict = dict(
        Task=taskname,
        project_path=proj_root,
        scorer=scorer,
        date=date,
        video_sets=video_sets,
        bodyparts=bodyparts,
        TrainingFraction=[0.95],
    )

    cfg_template.create_cfg(".", modify_dict)


def create_video_h5_from_pickle(proj_root, cfg, reference_pickle, videopath):

    with open(reference_pickle, "rb") as f:

        pickle_obj = pickle.load(f)

    # bodyparts  = pickle_obj['keypoint_names']

    bodyparts = [
        "tail",
        "spine4",
        "spine3",
        "spine2",
        "spine1",
        "head",
        "nose",
        "right ear",
        "left ear",
    ]

    video_name = videopath.split("/")[-1]

    video_key = f"{video_name}"  # .replace('.top.ir.mp4', '')

    print("video_key", video_key)

    print(list(pickle_obj.keys()))

    detections = pickle_obj[video_key]

    nframes = len(detections)

    xyz_labs = ["x", "y", "likelihood"]

    scorer = cfg["scorer"]

    keypoint_names = cfg["bodyparts"]

    product = [[scorer], keypoint_names, xyz_labs]

    names = ["scorer", "bodyparts", "coords"]
    columnindex = pd.MultiIndex.from_product(product, names=names)
    imagenames = [f"frame{i}" for i in range(nframes)]
    data = np.zeros((len(imagenames), len(columnindex))) * np.nan
    df = pd.DataFrame(data, columns=columnindex, index=imagenames)

    for imagename, kpts in zip(imagenames, detections):

        for kpt_id, kpt_name in enumerate(keypoint_names):

            df.loc[imagename][scorer, kpt_name, "x"] = kpts[kpt_id, 0]
            df.loc[imagename][scorer, kpt_name, "y"] = kpts[kpt_id, 1]
            df.loc[imagename][scorer, kpt_name, "likelihood"] = kpts[kpt_id, 2]

    vname = Path(videopath).stem
    DLCscorer = ""

    coords = [0, 400, 0, 400]
    trainFraction = cfg["TrainingFraction"][0]
    modelfolder = os.path.join(
        cfg["project_path"],
        str(auxiliaryfunctions.get_model_folder(trainFraction, 0, cfg)),
    )

    path_test_config = Path(modelfolder) / "test" / "pose_cfg.yaml"
    test_cfg = auxiliaryfunctions.read_plainconfig(path_test_config)

    start = 0
    stop = 10
    fps = 10
    dictionary = {
        "start": start,
        "stop": stop,
        "run_duration": stop - start,
        "Scorer": DLCscorer,
        "DLC-model-config file": test_cfg,
        "fps": fps,
        "batch_size": test_cfg["batch_size"],
        "frame_dimensions": (400, 400),
        "nframes": nframes,
        "iteration (active-learning)": cfg["iteration"],
        "cropping": cfg["cropping"],
        "training set fraction": trainFraction,
        "cropping_parameters": coords,
    }
    metadata = {"data": dictionary}

    dataname = os.path.join(proj_root, vname + DLCscorer + ".h5")

    metadata_path = dataname.split(".h5")[0] + "_meta.pickle"

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)

    df.to_hdf(dataname, "df_with_missing", format="table", mode="w")


def add_skeleton(config_path, pretrain_model_name):

    modelzoo_names = ["superquadruped", "supertopview"]

    assert pretrain_model_name in modelzoo_names

    super_quadruped = [
        ("left_eye", "right_eye"),
        ("left_eye", "left_earbase"),
        ("right_eye", "right_earbase"),
        ("left_eye", "nose"),
        ("right_eye", "nose"),
        ("nose", "throat_base"),
        ("throat_base", "back_base"),
        ("tail_base", "back_base"),
        ("throat_base", "front_left_thai"),
        ("front_left_thai", "front_left_knee"),
        ("front_left_knee", "front_left_paw"),
        ("throat_base", "front_right_thai"),
        ("front_right_thai", "front_right_knee"),
        ("front_right_knee", "front_right_paw"),
        ("tail_base", "back_left_thai"),
        ("back_left_thai", "back_left_knee"),
        ("back_left_knee", "back_left_paw"),
        ("tail_base", "back_right_thai"),
        ("back_right_thai", "back_right_knee"),
        ("back_right_knee", "back_right_paw"),
    ]

    skeleton_dict = {"superquadruped": super_quadruped, "supertopview": None}

    skeleton = skeleton_dict[pretrain_model_name]

    cfg = auxiliaryfunctions.read_config(config_path)
    cfg["skeleton"] = skeleton
    print(f"overwriting skeleton for {config_path}")
    auxiliaryfunctions.write_config(config_path, cfg)


def customized_colormap(config_path):
    # look for all symmetric keypoints
    # make symmetric keypoints the same color

    cfg = auxiliaryfunctions.read_config(config_path)
    bodyparts = cfg["multianimalbodyparts"]
    n_bodyparts = len(cfg["multianimalbodyparts"])

    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap("rainbow", n_bodyparts)

    colors = [cmap(i) for i in range(n_bodyparts)]

    visited = set()
    for kpt_id in range(len(bodyparts)):

        bodypart = bodyparts[kpt_id]
        if "left" in bodypart:
            ref_color = colors[kpt_id]
            temp = bodypart.replace("left", "right")
            if temp in bodyparts:
                temp_id = bodyparts.index(temp)
                colors[temp_id] = ref_color

    def ret_function(i):
        return colors[i]

    return ret_function


def create_modelprefix(modelprefix):
    import shutil

    shutil.copytree(
        "template-dlc-models",
        os.path.join(modelprefix, "dlc-models"),
        dirs_exist_ok=True,
    )


if __name__ == "__main__":

    customized_colormap("hei")
