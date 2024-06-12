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
from __future__ import annotations

import os
import os.path
import re
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from tqdm import tqdm

import deeplabcut.compat as compat
import deeplabcut.generate_training_dataset.metadata as metadata
from deeplabcut.core.engine import Engine
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.generate_training_dataset import (
    merge_annotateddatasets,
    read_image_shape_fast,
    SplitTrials,
    MakeTrain_pose_yaml,
    MakeTest_pose_yaml,
    MakeInference_yaml,
    pad_train_test_indices,
    validate_shuffles,
)
from deeplabcut.utils import (
    auxiliaryfunctions,
    auxfun_models,
    auxfun_multianimal,
)


def format_multianimal_training_data(
    df,
    train_inds,
    project_path,
    n_decimals=2,
):
    train_data = []
    nrows = df.shape[0]
    filenames = df.index.to_list()
    n_bodyparts = df.columns.get_level_values("bodyparts").unique().size
    individuals = df.columns.get_level_values("individuals")
    n_individuals = individuals.unique().size
    mask_single = individuals.str.contains("single")
    n_animals = n_individuals - 1 if np.any(mask_single) else n_individuals
    array = np.full(
        (nrows, n_individuals, n_bodyparts, 3), fill_value=np.nan, dtype=np.float32
    )
    array[..., 0] = np.arange(n_bodyparts)
    temp = df.to_numpy()
    temp_multi = temp[:, ~mask_single].reshape((nrows, n_animals, -1, 2))
    n_multibodyparts = temp_multi.shape[2]
    array[:, :n_animals, :n_multibodyparts, 1:] = temp_multi
    if n_animals != n_individuals:  # There is a unique individual
        n_uniquebodyparts = n_bodyparts - n_multibodyparts
        temp_single = np.reshape(temp[:, mask_single], (nrows, 1, n_uniquebodyparts, 2))
        array[:, -1:, -n_uniquebodyparts:, 1:] = temp_single
    array = np.round(array, decimals=n_decimals)
    for i in tqdm(train_inds):
        filename = filenames[i]
        img_shape = read_image_shape_fast(os.path.join(project_path, *filename))
        joints = dict()
        has_data = False
        for n, xy in enumerate(array[i]):
            # Drop missing body parts
            xy = xy[~np.isnan(xy).any(axis=1)]
            # Drop points lying outside the image
            inside = np.logical_and.reduce(
                (
                    xy[:, 1] < img_shape[2],
                    xy[:, 1] > 0,
                    xy[:, 2] < img_shape[1],
                    xy[:, 2] > 0,
                )
            )
            xy = xy[inside]
            if xy.size:
                has_data = True
                joints[n] = xy

        if has_data:
            data = {
                "image": filename,
                "size": np.asarray(img_shape),
                "joints": joints,
            }
            train_data.append(data)

    return train_data


def create_multianimaltraining_dataset(
    config,
    num_shuffles=1,
    Shuffles=None,
    windows2linux=False,
    net_type=None,
    numdigits=2,
    crop_size=(400, 400),
    crop_sampling="hybrid",
    paf_graph=None,
    trainIndices=None,
    testIndices=None,
    n_edges_threshold=105,
    paf_graph_degree=6,
    userfeedback: bool = True,
    weight_init: WeightInitialization | None = None,
    engine: Engine | None = None,
):
    """
    Creates a training dataset for multi-animal datasets. Labels from all the extracted
    frames are merged into a single .h5 file.\n
    Only the videos included in the config file are used to create this dataset.\n
    [OPTIONAL] Use the function 'add_new_videos' at any stage of the project to add more
    videos to the project.

    Important differences to standard:
     - stores coordinates with numdigits as many digits

    Parameter
    ----------
    config : string
        Full path of the config.yaml file as a string.

    num_shuffles : int, optional
        Number of shuffles of training dataset to create, i.e. [1,2,3] for num_shuffles=3. Default is set to 1.

    Shuffles: list of shuffles.
        Alternatively the user can also give a list of shuffles (integers!).

    net_type: string
        Type of networks. The options available depend on which engine is used. See
        Lauer et al. 2021 https://www.biorxiv.org/content/10.1101/2021.04.30.442096v1
        Currently supported options are:
            TensorFlow
                * ``resnet_50``
                * ``resnet_101``
                * ``resnet_152``
                * ``efficientnet-b0``
                * ``efficientnet-b1``
                * ``efficientnet-b2``
                * ``efficientnet-b3``
                * ``efficientnet-b4``
                * ``efficientnet-b5``
                * ``efficientnet-b6``
            PyTorch (call ``deeplabcut.pose_estimation.available_models()`` for a
            complete list)
                * ``resnet_50``
                * ``resnet_101``
                * ``dekr_w18``
                * ``dekr_w32``
                * ``dekr_w48``
                * ``top_down_resnet_50``
                * ``top_down_resnet_101``
                * ``top_down_hrnet_w18``
                * ``top_down_hrnet_w32``
                * ``top_down_hrnet_w48``
                * ``animaltokenpose_base``

    numdigits: int, optional

    crop_size: tuple of int, optional
        Only for the TensorFlow engine.
        Dimensions (width, height) of the crops for data augmentation.
        Default is 400x400.

    crop_sampling: str, optional
        Only for the TensorFlow engine.
        Crop centers sampling method. Must be either:
        "uniform" (randomly over the image),
        "keypoints" (randomly over the annotated keypoints),
        "density" (weighing preferentially dense regions of keypoints),
        or "hybrid" (alternating randomly between "uniform" and "density").
        Default is "hybrid".

    paf_graph: list of lists, or "config" optional (default=None)
        Only for the TensorFlow engine.
        If not None, overwrite the default complete graph. This is useful for advanced users who
        already know a good graph, or simply want to use a specific one. Note that, in that case,
        the data-driven selection procedure upon model evaluation will be skipped.

        "config" will use the skeleton defined in the config file.

    trainIndices: list of lists, optional (default=None)
        List of one or multiple lists containing train indexes.
        A list containing two lists of training indexes will produce two splits.

    testIndices: list of lists, optional (default=None)
        List of one or multiple lists containing test indexes.

    n_edges_threshold: int, optional (default=105)
        Only for the TensorFlow engine.
        Number of edges above which the graph is automatically pruned.

    paf_graph_degree: int, optional (default=6)
        Only for the TensorFlow engine.
        Degree of paf_graph when automatically pruning it (before training).

    userfeedback: bool, optional, default=True
        If ``False``, all requested train/test splits are created (no matter if they
        already exist). If you want to assure that previous splits etc. are not
        overwritten, set this to ``True`` and you will be asked for each split.

    weight_init: WeightInitialisation, optional, default=None
        PyTorch engine only. Specify how model weights should be initialized. The
        default mode uses transfer learning from ImageNet weights.

    engine: Engine, optional
        Whether to create a pose config for a Tensorflow or PyTorch model. Defaults to
        the value specified in the project configuration file. If no engine is specified
        for the project, defaults to ``deeplabcut.compat.DEFAULT_ENGINE``.

    Example
    --------
    >>> deeplabcut.create_multianimaltraining_dataset('/analysis/project/reaching-task/config.yaml',num_shuffles=1)

    >>> deeplabcut.create_multianimaltraining_dataset('/analysis/project/reaching-task/config.yaml', Shuffles=[0,1,2], trainIndices=[trainInd1, trainInd2, trainInd3], testIndices=[testInd1, testInd2, testInd3])

    Windows:
    >>> deeplabcut.create_multianimaltraining_dataset(r'C:\\Users\\Ulf\\looming-task\\config.yaml',Shuffles=[3,17,5])
    --------
    """
    if windows2linux:
        warnings.warn(
            "`windows2linux` has no effect since 2.2.0.4 and will be removed in 2.2.1.",
            FutureWarning,
        )

    if len(crop_size) != 2 or not all(isinstance(v, int) for v in crop_size):
        raise ValueError("Crop size must be a tuple of two integers (width, height).")

    if crop_sampling not in ("uniform", "keypoints", "density", "hybrid"):
        raise ValueError(
            f"Invalid sampling {crop_sampling}. Must be "
            f"either 'uniform', 'keypoints', 'density', or 'hybrid."
        )

    # Loading metadata from config file:
    cfg = auxiliaryfunctions.read_config(config)
    scorer = cfg["scorer"]
    project_path = cfg["project_path"]
    # Create path for training sets & store data there
    trainingsetfolder = auxiliaryfunctions.get_training_set_folder(cfg)
    full_training_path = Path(project_path, trainingsetfolder)
    auxiliaryfunctions.attempt_to_make_folder(full_training_path, recursive=True)

    # Create the trainset metadata file, if it doesn't yet exist
    if not metadata.TrainingDatasetMetadata.path(cfg).exists():
        trainset_metadata = metadata.TrainingDatasetMetadata.create(cfg)
        trainset_metadata.save()

    Data = merge_annotateddatasets(cfg, full_training_path)
    if Data is None:
        return
    Data = Data[scorer]

    if net_type is None:  # loading & linking pretrained models
        net_type = cfg.get("default_net_type", "dlcrnet_ms5")

    # load the engine to use to create the shuffle
    if engine is None:
        engine = compat.get_project_engine(cfg)

    if not (
        any(net in net_type for net in ("resnet", "eff", "dlc", "mob"))
        or engine == Engine.PYTORCH
    ):
        raise ValueError(f"Unsupported network {net_type} for engine {engine}.")

    multi_stage = False
    ### dlcnet_ms5: backbone resnet50 + multi-fusion & multi-stage module
    ### dlcr101_ms5/dlcr152_ms5: backbone resnet101/152 + multi-fusion & multi-stage module
    if (
        all(net in net_type for net in ("dlcr", "_ms5"))
        and engine != Engine.PYTORCH
    ):
        num_layers = re.findall("dlcr([0-9]*)", net_type)[0]
        if num_layers == "":
            num_layers = 50
        net_type = "resnet_{}".format(num_layers)
        multi_stage = True

    dataset_type = "multi-animal-imgaug"
    (
        individuals,
        uniquebodyparts,
        multianimalbodyparts,
    ) = auxfun_multianimal.extractindividualsandbodyparts(cfg)

    if paf_graph is None:  # Automatically form a complete PAF graph
        n_bpts = len(multianimalbodyparts)
        partaffinityfield_graph = [
            list(edge) for edge in combinations(range(n_bpts), 2)
        ]
        n_edges_orig = len(partaffinityfield_graph)
        # If the graph is unnecessarily large (with 15+ keypoints by default),
        # we randomly prune it to a size guaranteeing an average node degree of 6;
        # see Suppl. Fig S9c in Lauer et al., 2022.
        if n_edges_orig >= n_edges_threshold:
            partaffinityfield_graph = auxfun_multianimal.prune_paf_graph(
                partaffinityfield_graph,
                average_degree=paf_graph_degree,
            )
    else:
        if paf_graph == "config":
            # Use the skeleton defined in the config file
            skeleton = cfg["skeleton"]
            paf_graph = [
                sorted(
                    (multianimalbodyparts.index(bpt1), multianimalbodyparts.index(bpt2))
                )
                for bpt1, bpt2 in skeleton
            ]
            print(
                "Using `skeleton` from the config file as a paf_graph. Data-driven skeleton will not be computed."
            )

        # Ignore possible connections between 'multi' and 'unique' body parts;
        # one can never be too careful...
        to_ignore = auxfun_multianimal.filter_unwanted_paf_connections(cfg, paf_graph)
        partaffinityfield_graph = [
            edge for i, edge in enumerate(paf_graph) if i not in to_ignore
        ]
        auxfun_multianimal.validate_paf_graph(cfg, partaffinityfield_graph)

    print("Utilizing the following graph:", partaffinityfield_graph)
    # Disable the prediction of PAFs if the graph is empty
    partaffinityfield_predict = bool(partaffinityfield_graph)

    # Loading the encoder (if necessary downloading from TF)
    dlcparent_path = auxiliaryfunctions.get_deeplabcut_path()
    defaultconfigfile = os.path.join(dlcparent_path, "pose_cfg.yaml")

    if engine == Engine.PYTORCH:
        model_path = dlcparent_path
    else:
        model_path = auxfun_models.check_for_weights(
            net_type, Path(dlcparent_path)
        )

    Shuffles = validate_shuffles(cfg, Shuffles, num_shuffles, userfeedback)

    # print(trainIndices,testIndices, Shuffles, augmenter_type,net_type)
    if trainIndices is None and testIndices is None:
        splits = []
        for shuffle in Shuffles:  # Creating shuffles starting from 1
            for train_frac in cfg["TrainingFraction"]:
                train_inds, test_inds = SplitTrials(range(len(Data)), train_frac)
                splits.append((train_frac, shuffle, (train_inds, test_inds)))
    else:
        if len(trainIndices) != len(testIndices) != len(Shuffles):
            raise ValueError(
                "Number of Shuffles and train and test indexes should be equal."
            )
        splits = []
        for shuffle, (train_inds, test_inds) in enumerate(
            zip(trainIndices, testIndices)
        ):
            trainFraction = round(
                len(train_inds) * 1.0 / (len(train_inds) + len(test_inds)), 2
            )
            print(
                f"You passed a split with the following fraction: {int(100 * trainFraction)}%"
            )
            # Now that the training fraction is guaranteed to be correct,
            # the values added to pad the indices are removed.
            train_inds = np.asarray(train_inds)
            train_inds = train_inds[train_inds != -1]
            test_inds = np.asarray(test_inds)
            test_inds = test_inds[test_inds != -1]
            splits.append((trainFraction, Shuffles[shuffle], (train_inds, test_inds)))

    top_down = False
    if engine == Engine.PYTORCH and net_type.startswith("top_down_"):
        top_down = True
        net_type = net_type[len("top_down_"):]

    for trainFraction, shuffle, (trainIndices, testIndices) in splits:
        ####################################################
        # Generating data structure with labeled information & frame metadata (for deep cut)
        ####################################################
        print(
            "Creating training data for: Shuffle:",
            shuffle,
            "TrainFraction: ",
            trainFraction,
        )

        # Make training file!
        data = format_multianimal_training_data(
            Data,
            trainIndices,
            cfg["project_path"],
            numdigits,
        )

        if len(trainIndices) > 0:
            (
                datafilename,
                metadatafilename,
            ) = auxiliaryfunctions.get_data_and_metadata_filenames(
                trainingsetfolder, trainFraction, shuffle, cfg
            )
            ################################################################################
            # Saving metadata and data file (Pickle file)
            ################################################################################
            auxiliaryfunctions.save_metadata(
                os.path.join(project_path, metadatafilename),
                data,
                trainIndices,
                testIndices,
                trainFraction,
            )
            metadata.update_metadata(
                cfg=cfg,
                train_fraction=trainFraction,
                shuffle=shuffle,
                engine=engine,
                train_indices=trainIndices,
                test_indices=testIndices,
                overwrite=not userfeedback,
            )

            datafilename = datafilename.split(".mat")[0] + ".pickle"
            import pickle

            with open(os.path.join(project_path, datafilename), "wb") as f:
                # Pickle the 'labeled-data' dictionary using the highest protocol available.
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

            ################################################################################
            # Creating file structure for training &
            # Test files as well as pose_yaml files (containing training and testing information)
            #################################################################################

            modelfoldername = auxiliaryfunctions.get_model_folder(
                trainFraction, shuffle, cfg, engine=engine,
            )
            auxiliaryfunctions.attempt_to_make_folder(
                Path(config).parents[0] / modelfoldername, recursive=True
            )
            auxiliaryfunctions.attempt_to_make_folder(
                str(Path(config).parents[0] / modelfoldername / "train")
            )
            auxiliaryfunctions.attempt_to_make_folder(
                str(Path(config).parents[0] / modelfoldername / "test")
            )

            path_train_config = str(
                os.path.join(
                    cfg["project_path"],
                    Path(modelfoldername),
                    "train",
                    "pose_cfg.yaml",
                )
            )
            path_test_config = str(
                os.path.join(
                    cfg["project_path"],
                    Path(modelfoldername),
                    "test",
                    "pose_cfg.yaml",
                )
            )
            path_inference_config = str(
                os.path.join(
                    cfg["project_path"],
                    Path(modelfoldername),
                    "test",
                    "inference_cfg.yaml",
                )
            )

            jointnames = [str(bpt) for bpt in multianimalbodyparts]
            jointnames.extend([str(bpt) for bpt in uniquebodyparts])
            items2change = {
                "dataset": datafilename,
                "engine": engine.aliases[0],
                "metadataset": metadatafilename,
                "num_joints": len(multianimalbodyparts)
                + len(uniquebodyparts),  # cfg["uniquebodyparts"]),
                "all_joints": [
                    [i] for i in range(len(multianimalbodyparts) + len(uniquebodyparts))
                ],  # cfg["uniquebodyparts"]))],
                "all_joints_names": jointnames,
                "init_weights": str(model_path),
                "project_path": str(cfg["project_path"]),
                "net_type": net_type,
                "multi_stage": multi_stage,
                "pairwise_loss_weight": 0.1,
                "pafwidth": 20,
                "partaffinityfield_graph": partaffinityfield_graph,
                "partaffinityfield_predict": partaffinityfield_predict,
                "weigh_only_present_joints": False,
                "num_limbs": len(partaffinityfield_graph),
                "dataset_type": dataset_type,
                "optimizer": "adam",
                "batch_size": 8,
                "multi_step": [[1e-4, 7500], [5 * 1e-5, 12000], [1e-5, 200000]],
                "save_iters": 10000,
                "display_iters": 500,
                "num_idchannel": len(cfg["individuals"])
                if cfg.get("identity", False)
                else 0,
                "crop_size": list(crop_size),
                "crop_sampling": crop_sampling,
            }

            trainingdata = MakeTrain_pose_yaml(
                items2change, path_train_config, defaultconfigfile
            )
            keys2save = [
                "dataset",
                "num_joints",
                "all_joints",
                "all_joints_names",
                "net_type",
                "multi_stage",
                "init_weights",
                "global_scale",
                "location_refinement",
                "locref_stdev",
                "dataset_type",
                "partaffinityfield_predict",
                "pairwise_predict",
                "partaffinityfield_graph",
                "num_limbs",
                "dataset_type",
                "num_idchannel",
            ]

            MakeTest_pose_yaml(
                trainingdata,
                keys2save,
                path_test_config,
                nmsradius=5.0,
                minconfidence=0.01,
                sigma=1,
                locref_smooth=False,
            )  # setting important def. values for inference

            # Setting inference cfg file:
            defaultinference_configfile = os.path.join(
                dlcparent_path, "inference_cfg.yaml"
            )
            items2change = {
                "minimalnumberofconnections": int(len(cfg["multianimalbodyparts"]) / 2),
                "topktoretain": len(cfg["individuals"]),
                "withid": cfg.get("identity", False),
            }
            MakeInference_yaml(
                items2change, path_inference_config, defaultinference_configfile
            )

            # Populate the pytorch config yaml file
            if engine == Engine.PYTORCH:
                from deeplabcut.pose_estimation_pytorch.config.make_pose_config import make_pytorch_pose_config
                from deeplabcut.pose_estimation_pytorch.modelzoo.config import make_super_animal_finetune_config

                pose_cfg_path = path_train_config.replace("pose_cfg.yaml", "pytorch_config.yaml")
                if weight_init is not None and weight_init.with_decoder:
                    pytorch_cfg = make_super_animal_finetune_config(
                        project_config=cfg,
                        pose_config_path=path_train_config,
                        net_type=net_type,
                        weight_init=weight_init,
                    )
                else:
                    pytorch_cfg = make_pytorch_pose_config(
                        project_config=cfg,
                        pose_config_path=path_train_config,
                        net_type=net_type,
                        top_down=top_down,
                        weight_init=weight_init,
                    )

                auxiliaryfunctions.write_plainconfig(pose_cfg_path, pytorch_cfg)

            print(
                "The training dataset is successfully created. Use the function 'train_network' to start training. Happy training!"
            )
        else:
            pass


def convert_cropped_to_standard_dataset(
    config_path,
    recreate_datasets=True,
    delete_crops=True,
    back_up=True,
):
    import pandas as pd
    import pickle
    import shutil
    from deeplabcut.generate_training_dataset import trainingsetmanipulation
    from deeplabcut.utils import read_plainconfig, write_config

    cfg = auxiliaryfunctions.read_config(config_path)
    videos_orig = cfg.pop("video_sets_original")
    is_cropped = cfg.pop("croppedtraining")
    if videos_orig is None or not is_cropped:
        print(
            "Labeled data do not appear to be cropped. "
            "Project will remain unchanged..."
        )
        return

    project_path = cfg["project_path"]

    if back_up:
        print("Backing up project...")
        shutil.copytree(project_path, project_path + "_bak", symlinks=True)

    if delete_crops:
        print("Deleting crops...")
        data_path = os.path.join(project_path, "labeled-data")
        for video in cfg["video_sets"]:
            _, filename, _ = trainingsetmanipulation._robust_path_split(video)
            if "_cropped" in video:  # One can never be too safe...
                shutil.rmtree(os.path.join(data_path, filename), ignore_errors=True)

    cfg["video_sets"] = videos_orig
    write_config(config_path, cfg)

    if not recreate_datasets:
        return

    datasets_folder = os.path.join(
        project_path,
        auxiliaryfunctions.get_training_set_folder(cfg),
    )
    df_old = pd.read_hdf(
        os.path.join(datasets_folder, "CollectedData_" + cfg["scorer"] + ".h5"),
    )

    def strip_cropped_image_name(path):
        head, filename = os.path.split(path)
        head = head.replace("_cropped", "")
        file, ext = filename.split(".")
        file = file.split("c")[0]
        return os.path.join(head, file + "." + ext)

    img_names_old = np.asarray(
        [strip_cropped_image_name(img) for img in df_old.index.to_list()]
    )
    df = merge_annotateddatasets(cfg, datasets_folder)
    img_names = df.index.to_numpy()
    train_idx = []
    test_idx = []
    pickle_files = []
    for filename in os.listdir(datasets_folder):
        if filename.endswith("pickle"):
            pickle_file = os.path.join(datasets_folder, filename)
            pickle_files.append(pickle_file)
            if filename.startswith("Docu"):
                with open(pickle_file, "rb") as f:
                    _, train_inds, test_inds, train_frac = pickle.load(f)
                    train_inds_temp = np.flatnonzero(
                        np.isin(img_names, img_names_old[train_inds])
                    )
                    test_inds_temp = np.flatnonzero(
                        np.isin(img_names, img_names_old[test_inds])
                    )
                    train_inds, test_inds = pad_train_test_indices(
                        train_inds_temp, test_inds_temp, train_frac
                    )
                    train_idx.append(train_inds)
                    test_idx.append(test_inds)

    # Search a pose_config.yaml file to parse missing information
    pose_config_path = ""
    for dirpath, _, filenames in os.walk(os.path.join(project_path, "dlc-models")):
        for file in filenames:
            if file.endswith("pose_cfg.yaml"):
                pose_config_path = os.path.join(dirpath, file)
                break
    pose_cfg = read_plainconfig(pose_config_path)
    net_type = pose_cfg["net_type"]
    if net_type == "resnet_50" and pose_cfg.get("multi_stage", False):
        net_type = "dlcrnet_ms5"

    # Clean the training-datasets folder prior to recreating the data pickles
    shuffle_inds = set()
    for file in pickle_files:
        os.remove(file)
        shuffle_inds.add(int(re.findall(r"shuffle(\d+)", file)[0]))
    create_multianimaltraining_dataset(
        config_path,
        trainIndices=train_idx,
        testIndices=test_idx,
        Shuffles=sorted(shuffle_inds),
        net_type=net_type,
        paf_graph=pose_cfg["partaffinityfield_graph"],
        crop_size=pose_cfg.get("crop_size", [400, 400]),
        crop_sampling=pose_cfg.get("crop_sampling", "hybrid"),
    )
