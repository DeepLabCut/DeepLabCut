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
import json
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import scipy.io as sio
import yaml

import deeplabcut.compat as compat
from deeplabcut.generate_training_dataset.multiple_individuals_trainingsetmanipulation import (
    create_multianimaltraining_dataset,
    format_multianimal_training_data,
)
from deeplabcut.generate_training_dataset.trainingsetmanipulation import (
    create_training_dataset,
)
from deeplabcut.generate_training_dataset.trainingsetmanipulation import (
    format_training_data as format_single_training_data,
)
from deeplabcut.utils import auxiliaryfunctions


def get_filename(filename):
    if type(filename) == tuple:
        filename = os.path.join(*filename)
    return filename


def modify_train_test_cfg(config_path, shuffle=1, modelprefix=""):
    # get train_cfg from main cfg
    # use dlcr net
    # use gradient masking
    # set batch size as 8
    trainposeconfigfile, testposeconfigfile, snapshotfolder = (
        compat.return_train_network_path(
            config_path, shuffle=shuffle, modelprefix=modelprefix, trainingsetindex=0
        )
    )

    train_cfg = auxiliaryfunctions.read_plainconfig(trainposeconfigfile)
    train_cfg["multi_stage"] = True
    train_cfg["batch_size"] = 8
    train_cfg["gradient_masking"] = True

    auxiliaryfunctions.write_plainconfig(trainposeconfigfile, train_cfg)

    test_cfg = auxiliaryfunctions.read_plainconfig(testposeconfigfile)
    test_cfg["multi_stage"] = True
    test_cfg["batch_size"] = 8
    test_cfg["gradient_masking"] = True

    auxiliaryfunctions.write_plainconfig(testposeconfigfile, test_cfg)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class SingleDLC_config:
    def __init__(self):
        Task = ""  # could be dataset name
        project_path = ""
        scorer = ""  # random stuff
        date = ""  # random stuff
        video_sets = ""  # has to be used for labeled data
        skeleton = ""  # could be arbitrary
        bodyparts = ""  # either single or multi
        start = 0  # not sure
        stop = 1  # not sure
        numframes2pick = 42  # does not matter
        skeleton_color = "black"
        pcutoff = 0.6
        dotsize = 8
        alphavalue = 0.7
        colormap = "rainbow"
        TrainingFraction = ""  # need to be filled correctly
        iteration = 0
        default_net_type = "resnet_50"
        default_augmenter = "imgaug"
        snapshotindex = -1
        batch_size = 8
        cropping = False
        croppedtraining = False
        multianimalproject = False
        uniquebodyparts = []
        x1 = 0
        x2 = 640
        y1 = 277
        y2 = 624
        corer2move2 = [50, 50]
        move2corner = True
        identity = False
        self.cfg = {
            k: v for k, v in vars().items() if "__" not in k and "self" not in k
        }

    def create_cfg(self, proj_root, kwargs):
        self.cfg.update(kwargs)
        with open(os.path.join(proj_root, "config.yaml"), "w") as f:
            yaml.dump(self.cfg, f)


class MaDLC_config:
    def __init__(self):
        """
        Plain text only for generating templates
        Some variables can be configured by the user later
        """

        Task = ""  # could be dataset name
        project_path = ""
        scorer = ""  # random stuff
        date = ""  # random stuff
        video_sets = ""  # has to be used for labeled data
        individuals = ""  # number of individuals
        multianimalbodyparts = ""  # keypoints
        skeleton = ""  # could be arbitrary
        bodyparts = ""  # either single or multi
        start = 0  # not sure
        stop = 1  # not sure
        numframes2pick = 42  # does not matter
        skeleton_color = "black"
        pcutoff = 0.6
        dotsize = 8
        alphavalue = 0.7
        colormap = "rainbow"
        TrainingFraction = ""  # need to be filled correctly
        iteration = 0
        default_net_type = "resnet_50"
        default_augmenter = "multi-animal-imgaug"
        snapshotindex = -1
        batch_size = 8
        cropping = False
        croppedtraining = True
        multianimalproject = True
        uniquebodyparts = []
        x1 = 0
        x2 = 640
        y1 = 277
        y2 = 624
        corer2move2 = [50, 50]
        move2corner = True
        identity = False
        self.cfg = {
            k: v for k, v in vars().items() if "__" not in k and "self" not in k
        }

    def create_cfg(self, proj_root, kwargs):
        self.cfg.update(kwargs)
        with open(os.path.join(proj_root, "config.yaml"), "w") as f:
            yaml.dump(self.cfg, f)


def _generic2madlc(
    proj_root,
    train_images,
    test_images,
    train_annotations,
    test_annotations,
    meta,
    deepcopy=False,
    full_image_path=True,
    append_image_id=True,
):
    """
    Within DeepLabCut, if we don't explicitly call deeplabcut.create_traindataset(), the train and test split might just be arbitrarily messed up. So here we need to calculate train and test indices to

    Args:
    proj_root where to materialize the data

    """

    assert full_image_path, "DLC wants full image path"

    os.makedirs(os.path.join(proj_root, "labeled-data"), exist_ok=True)

    cfg_template = MaDLC_config()

    individuals = [f"individual{i}" for i in range(meta["max_individuals"])]

    bodyparts = meta["categories"]["keypoints"]

    scorer = "maDLC_scorer"
    # this line is taken from dlc's multi animal dataset creation function
    train_fraction = round(
        len(train_images) * 1.0 / (len(train_images) + len(test_images)), 2
    )

    # need to fake a video path
    # let's use individual dataset names as fake video name
    # merged_dataset_name = '_'.join(meta['mat_datasets'])
    video_sets = {
        f"{dataset_name}.mp4": {"crop": "0, 400, 0, 400"}
        for dataset_name in meta["mat_datasets"]
    }

    modify_dict = dict(
        Task=meta["dataset_name"],
        project_path=proj_root,
        individuals=individuals,
        scorer=scorer,
        date="March30",
        video_sets=video_sets,
        bodyparts="MULTI!",
        TrainingFraction=[train_fraction],
        multianimalbodyparts=list(bodyparts),
    )

    cfg_template.create_cfg(proj_root, modify_dict)
    # what's special in dlc or madlc creation is that we will need to
    # use dlc's code for creating the project structure
    # because you don't want to write your own. It's a lot of lines of code
    # But at least we can focus on labeled-data

    imageid2datasetname = meta["imageid2datasetname"]

    for dataset_name in meta["mat_datasets"]:
        os.makedirs(
            os.path.join(proj_root, "labeled-data", dataset_name), exist_ok=True
        )

    # also, to make sure the split is right, we will have to pass the right indices

    columnindex = pd.MultiIndex.from_product(
        [[scorer], individuals, bodyparts, ["x", "y"]],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )

    # it's important to put train first so the train_fraction parameter can work correctly
    total_images = train_images + test_images
    total_annotations = train_annotations + test_annotations

    # DLC uses relative dest as index into dataframe
    imageid2relativedest = {}
    count = 0
    for image in total_images:
        image_id = image["id"]
        file_name = image["file_name"]
        image_name = file_name.split(os.sep)[-1]
        pre, suffix = image_name.split(".")
        if append_image_id == True:
            dest_image_name = f"{pre}_{image_id}.{suffix}"
        else:
            dest_image_name = image_name
        # the generic data has original pointers to images in the original folders
        # Here, we have to change the image name and location of these to fit corresponding framework's convention

        dataset_name = imageid2datasetname[image_id]

        dest = os.path.join(proj_root, "labeled-data", dataset_name, dest_image_name)
        if deepcopy:
            shutil.copy(file_name, dest)
        else:
            try:
                os.symlink(file_name, dest)
            except Exception as e:
                pass

        relative_dest = os.path.join("labeled-data", dataset_name, dest_image_name)

        imageid2relativedest[image_id] = relative_dest

    temp_count = 0
    for dataset_name, dataset in meta["mat_datasets"].items():

        dataset_total_images = (
            dataset.generic_train_images + dataset.generic_test_images
        )
        dataset_total_annotations = (
            dataset.generic_train_annotations + dataset.generic_test_annotations
        )

        dataset_index = []

        for image in dataset_total_images:
            image_id = image["id"]
            relative_dest = imageid2relativedest[image_id]
            dataset_index.append(relative_dest)

        raw_data = np.zeros((len(dataset_total_images), len(columnindex))) * np.nan
        df = pd.DataFrame(raw_data, columns=columnindex, index=dataset_index)
        # so we know where to put the next annotation if there are multiple individuals in that image
        imageid2filledindividualcount = {}

        image_ids = []
        for anno in dataset_total_annotations:
            keypoints = anno["keypoints"]
            image_id = anno["image_id"]
            image_ids.append(image_id)
            if image_id not in imageid2filledindividualcount:
                imageid2filledindividualcount[image_id] = 0
            else:
                imageid2filledindividualcount[image_id] += 1
            individual_id = imageid2filledindividualcount[image_id]

            file_name = imageid2relativedest[image_id]
            for kpt_id, kpt_name in enumerate(meta["categories"]["keypoints"]):
                coord = keypoints[3 * kpt_id : 3 * kpt_id + 3]
                # note dlc does not yet have visibility flag
                # need to be careful here to assign right keypoints to right people
                if coord[0] > 0 and coord[1] > 0:
                    # leave them to NaN if values are 0
                    df.loc[file_name][
                        scorer, f"individual{individual_id}", kpt_name, "x"
                    ] = coord[0]
                    df.loc[file_name][
                        scorer, f"individual{individual_id}", kpt_name, "y"
                    ] = coord[1]
                elif coord[2] == -1:
                    df.loc[file_name][
                        scorer, f"individual{individual_id}", kpt_name, "x"
                    ] = -1
                    df.loc[file_name][
                        scorer, f"individual{individual_id}", kpt_name, "y"
                    ] = -1
        df.to_hdf(
            os.path.join(
                proj_root, "labeled-data", dataset_name, f"CollectedData_{scorer}.h5"
            ),
            key="df_with_missing",
            mode="w",
        )
    # paf_graph default as None. But I am not sure how to do better
    create_multianimaltraining_dataset(
        os.path.join(proj_root, "config.yaml"), paf_graph=None
    )

    # dlc's merge_annotation messes up my indices, so I will need to overwrite the documentation file
    # I could have done it in a more elegant way if I could modify part of DLC source code, but for backward compatibility reasons, overriding documentation is smarter

    config_path = os.path.join(proj_root, "config.yaml")

    cfg = auxiliaryfunctions.read_config(config_path)

    train_folder = os.path.join(proj_root, auxiliaryfunctions.GetTrainingSetFolder(cfg))

    datafilename, metafilename = auxiliaryfunctions.GetDataandMetaDataFilenames(
        train_folder, train_fraction, 1, cfg
    )

    modify_train_test_cfg(config_path)

    dlc_df = pd.read_hdf(os.path.join(train_folder, f"CollectedData_{scorer}.h5"))

    # I strip off video info from the naming. For horse10, I need to get it back
    parent_trace = {}

    def _filter(image):
        file_name = image["file_name"]

        image_name = file_name.split(os.sep)[-1]
        video_folder = file_name.split(os.sep)[-2]
        pre, suffix = image_name.split(".")
        image_id = image["id"]
        if append_image_id:
            ret = f"{pre}_{image_id}.{suffix}"
        else:
            ret = image_name
        parent_trace[ret] = video_folder
        return ret

    _filter_train_images = list(map(_filter, train_images))
    _filter_test_images = list(map(_filter, test_images))

    with open(os.path.join(train_folder, "parent_trace.pickle"), "wb") as f:
        pickle.dump(parent_trace, f)

    trainIndices = [
        idx
        for idx, image in enumerate(dlc_df.index)
        if get_filename(image).split(os.sep)[-1] in _filter_train_images
    ]
    testIndices = [
        idx
        for idx, image in enumerate(dlc_df.index)
        if get_filename(image).split(os.sep)[-1] in _filter_test_images
    ]

    with open(metafilename, "rb") as f:
        metafile = pickle.load(f)

    metafile[1] = trainIndices
    metafile[2] = testIndices

    with open(metafilename, "wb") as f:
        pickle.dump(metafile, f)

    # need to overwrite the data pickle file too

    nbodyparts = len(bodyparts)

    if "individuals" not in dlc_df.columns.names:
        old_idx = dlc_df.columns.to_frame()
        old_idx.insert(0, "individuals", "")
        dlc_df.columns = pd.MultiIndex.from_frame(old_idx)

    data = format_multianimal_training_data(dlc_df, trainIndices, cfg["project_path"])

    datafilename = datafilename.split(".mat")[0] + ".pickle"

    print(f"overwriting data file {datafilename}")

    with open(os.path.join(proj_root, datafilename), "wb") as f:

        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def _generic2sdlc(
    proj_root,
    train_images,
    test_images,
    train_annotations,
    test_annotations,
    meta,
    deepcopy=False,
    full_image_path=True,
    append_image_id=True,
):

    assert full_image_path, "DLC wants full image path"

    os.makedirs(os.path.join(proj_root, "labeled-data"), exist_ok=True)

    cfg_template = SingleDLC_config()

    bodyparts = meta["categories"]["keypoints"]
    scorer = "singleDLC_scorer"

    train_fraction = round(
        len(train_images) * 1.0 / (len(train_images) + len(test_images)), 2
    )

    # need to fake a video path
    # let's use individual dataset names as fake video name

    video_sets = {
        f"{dataset_name}.mp4": {"crop": "0, 400, 0, 400"}
        for dataset_name in meta["mat_datasets"].keys()
    }

    modify_dict = dict(
        Task=meta["dataset_name"],
        project_path=proj_root,
        scorer=scorer,
        date="March30",
        bodyparts=list(bodyparts),
        video_sets=video_sets,
        TrainingFraction=[train_fraction],
    )

    cfg_template.create_cfg(proj_root, modify_dict)

    imageid2datasetname = meta["imageid2datasetname"]

    for dataset_name in meta["mat_datasets"]:
        os.makedirs(
            os.path.join(proj_root, "labeled-data", dataset_name), exist_ok=True
        )

    columnindex = pd.MultiIndex.from_product(
        [[scorer], bodyparts, ["x", "y"]], names=["scorer", "bodyparts", "coords"]
    )

    total_images = train_images + test_images
    total_annotations = train_annotations + test_annotations

    # DLC uses relative dest as index
    imageid2relativedest = {}

    for image in total_images:
        imageid = image["id"]
        filename = image["file_name"]
        datasetname = imageid2datasetname[imageid]
    count = 0
    for image in total_images:
        image_id = image["id"]
        file_name = image["file_name"]

        image_name = file_name.split(os.sep)[-1]
        pre, suffix = image_name.split(".")

        if append_image_id == True:
            dest_image_name = f"{pre}_{image_id}.{suffix}"
        else:
            dest_image_name = image_name
        # the generic data has original pointers to images in the original folders
        # Here, we have to change the image name and location of these to fit corresponding framework's convention

        dataset_name = imageid2datasetname[image_id]

        dest = os.path.join(proj_root, "labeled-data", dataset_name, dest_image_name)
        if deepcopy:
            shutil.copy(file_name, dest)
        else:
            try:
                os.symlink(file_name, dest)
            except:
                pass

        if dataset_name == "AwA-Pose":
            count += 1

        relative_dest = os.path.join("labeled-data", dataset_name, dest_image_name)
        imageid2relativedest[image_id] = relative_dest

    # so we know where to put the next annotation if there are multiple individuals in that image

    for dataset_name, dataset in meta["mat_datasets"].items():

        dataset_total_images = (
            dataset.generic_train_images + dataset.generic_test_images
        )
        dataset_total_annotations = (
            dataset.generic_train_annotations + dataset.generic_test_annotations
        )

        dataset_index = []
        freq = {}
        for image in dataset_total_images:
            filename = image["file_name"]

            image_id = image["id"]
            relative_dest = imageid2relativedest[image_id]

            dataset_index.append(relative_dest)

        raw_data = np.zeros((len(dataset_total_images), len(columnindex))) * np.nan

        dataset_index = dataset_index

        df = pd.DataFrame(raw_data, columns=columnindex, index=dataset_index)

        for idx, anno in enumerate(dataset_total_annotations):
            keypoints = np.array(anno["keypoints"])
            image_id = anno["image_id"]

            file_name = imageid2relativedest[image_id]

            for kpt_id, kpt_name in enumerate(meta["categories"]["keypoints"]):
                coord = keypoints[3 * kpt_id : 3 * kpt_id + 3]
                # note dlc does not yet have visibility flag
                # need to be careful here to assign right keypoints to right people

                if coord[0] > 0 and coord[1] > 0:

                    df.loc[file_name][scorer, kpt_name, "x"] = coord[0]
                    df.loc[file_name][scorer, kpt_name, "y"] = coord[1]
                elif coord[2] == -1:
                    # if -1, this visibility flag means a given keypoint was not annotated in the original dataset
                    df.loc[file_name][scorer, kpt_name, "x"] = -1
                    df.loc[file_name][scorer, kpt_name, "y"] = -1

        df = df.dropna(how="all")
        df.to_hdf(
            os.path.join(
                proj_root, "labeled-data", dataset_name, f"CollectedData_{scorer}.h5"
            ),
            key="df_with_missing",
            mode="w",
        )

    create_training_dataset(os.path.join(proj_root, "config.yaml"))

    # dlc's merge_annotation messes up my indices, so I will need to overwrite the documentation file
    # I could have done it in a more elegant way if I could modify part of DLC source code, but for backward compatibility reasons, overriding documentation is smarter

    config_path = os.path.join(proj_root, "config.yaml")

    cfg = auxiliaryfunctions.read_config(config_path)

    train_folder = os.path.join(proj_root, auxiliaryfunctions.GetTrainingSetFolder(cfg))

    datafilename, metafilename = auxiliaryfunctions.GetDataandMetaDataFilenames(
        train_folder, train_fraction, 1, cfg
    )

    modify_train_test_cfg(config_path)

    dlc_df = pd.read_hdf(os.path.join(train_folder, f"CollectedData_{scorer}.h5"))

    parent_trace = {}

    def _filter(image):
        file_name = image["file_name"]
        image_name = file_name.split(os.sep)[-1]
        video_folder = file_name.split(os.sep)[-2]
        pre, suffix = image_name.split(".")
        image_id = image["id"]
        if append_image_id:
            ret = f"{pre}_{image_id}.{suffix}"
        else:
            ret = image_name

        parent_trace[ret] = video_folder

        return ret

    _filter_train_images = list(map(_filter, train_images))
    _filter_test_images = list(map(_filter, test_images))

    with open(os.path.join(train_folder, "parent_trace.pickle"), "wb") as f:
        pickle.dump(parent_trace, f)

    trainIndices = [
        idx
        for idx, image in enumerate(dlc_df.index)
        if get_filename(image).split(os.sep)[-1] in _filter_train_images
    ]
    testIndices = [
        idx
        for idx, image in enumerate(dlc_df.index)
        if get_filename(image).split(os.sep)[-1] in _filter_test_images
    ]

    with open(metafilename, "rb") as f:
        metafile = pickle.load(f)

    metafile[1] = trainIndices
    metafile[2] = testIndices

    with open(metafilename, "wb") as f:
        pickle.dump(metafile, f)

    # need to overwrite the true data file too
    nbodyparts = len(bodyparts)

    data, MatlabData = format_single_training_data(
        dlc_df, trainIndices, nbodyparts, cfg["project_path"]
    )

    print(f"overwriting data file {datafilename}")

    sio.savemat(os.path.join(datafilename), {"dataset": MatlabData})


def _generic2coco(
    proj_root,
    train_images,
    test_images,
    train_annotations,
    test_annotations,
    meta,
    deepcopy=False,
    full_image_path=True,
    append_image_id=True,
):
    """
    Take generic data and create coco structure
    My generic definition of coco structure:
    images
      ...
    annotations
    - train.json
    - test.json
    """

    os.makedirs(os.path.join(proj_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(proj_root, "annotations"), exist_ok=True)

    # from new path to old_path
    lookuptable = {}

    for annotation in train_annotations + test_annotations:
        if "iscrowd" not in annotation:
            annotation["iscrowd"] = 0

        keypoints = annotation["keypoints"]
        for kpt_id, kpt_name in enumerate(meta["categories"]["keypoints"]):
            coord = keypoints[3 * kpt_id : 3 * kpt_id + 3]
            if coord[0] < 0 or coord[1] < 0:
                coord[2] = -1

    broken_links = []
    # copying images via symbolic link
    for image in train_images + test_images:
        src = image["file_name"]
        image_id = image["id"]

        if not os.path.exists(src):
            print("problem comes from", image["source_dataset"])
            print(src)
            broken_links.append(image_id)
            continue
        else:
            pass
            # print ('success comes from', image['source_dataset'])
            # print (src)

        # in dlc, some images have same name but under different folder
        # we used to use a parent folder to distinguish them, but it's only applicable to DLC
        # so here it's easier to just append a id into the filename

        image_name = src.split(os.sep)[-1]

        if image_name.count(".") > 1:
            sep = image_name.rfind(".")
            pre, suffix = image_name[:sep], image_name[sep + 1 :]
        else:
            # this does not work for image file that looks like image9.5.jpg..
            pre, suffix = image_name.split(".")

        # not to repeatedly add image id in memory replay training
        if append_image_id:
            dest_image_name = f"{pre}_{image_id}.{suffix}"
        else:
            dest_image_name = image_name
        dest = os.path.join(proj_root, "images", dest_image_name)

        # now, we will also need to update the path in the config files

        if full_image_path:
            image["file_name"] = dest
        else:
            image["file_name"] = os.path.join("images", dest_image_name)

        if deepcopy:
            shutil.copy(src, dest)
        else:
            try:
                os.symlink(src, dest)
            except:
                pass

        lookuptable[dest] = src

    train_annotations = [
        train_anno
        for train_anno in train_annotations
        if train_anno["image_id"] not in broken_links
    ]
    test_annotations = [
        test_anno
        for test_anno in test_annotations
        if test_anno["image_id"] not in broken_links
    ]

    with open(os.path.join(proj_root, "annotations", "train.json"), "w") as f:

        train_json_obj = dict(
            images=train_images,
            annotations=train_annotations,
            categories=[meta["categories"]],
        )

        json.dump(train_json_obj, f, indent=4, cls=NpEncoder)

    with open(os.path.join(proj_root, "annotations", "test.json"), "w") as f:
        test_json_obj = dict(
            images=test_images,
            annotations=test_annotations,
            categories=[meta["categories"]],
        )

        json.dump(test_json_obj, f, indent=4, cls=NpEncoder)

    return lookuptable


def mat_func_factory(framework):
    assert framework in [
        "coco",
        "sdlc",
        "madlc",
    ], f"Does not support framework {framework}"
    if framework == "madlc":
        mat_func = _generic2madlc
    elif framework == "coco":
        mat_func = _generic2coco
    elif framework == "sdlc":
        mat_func = _generic2sdlc

    return mat_func
