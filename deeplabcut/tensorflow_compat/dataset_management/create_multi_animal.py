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
"""TensorFlow implementation of multi-animal training dataset creation."""

from __future__ import annotations

import os
import pickle
import re
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np

import deeplabcut.generate_training_dataset.metadata as metadata
from deeplabcut.core.engine import Engine
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.generate_training_dataset.multiple_individuals_trainingsetmanipulation import (
    format_multianimal_training_data,
)
from deeplabcut.generate_training_dataset.trainingsetmanipulation import (
    SplitTrials,
    merge_annotateddatasets,
    validate_shuffles,
)
from deeplabcut.tensorflow_compat.dataset_management.pose_yaml import (
    MakeInference_yaml,
    MakeTest_pose_yaml,
    MakeTrain_pose_yaml,
)
from deeplabcut.utils import (
    auxfun_models,
    auxfun_multianimal,
    auxiliaryfunctions,
)

_ENGINE = Engine.TF


def create_multianimaltraining_dataset(
    config,
    num_shuffles=1,
    Shuffles=None,
    windows2linux=False,
    net_type=None,
    detector_type=None,
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
    ctd_conditions: int | str | Path | tuple[int, str] | tuple[int, int] | None = None,
):
    if windows2linux:
        warnings.warn(
            "`windows2linux` has no effect since 2.2.0.4 and will be removed in 2.2.1.",
            FutureWarning,
            stacklevel=2,
        )

    if len(crop_size) != 2 or not all(isinstance(v, int) for v in crop_size):
        raise ValueError("Crop size must be a tuple of two integers (width, height).")

    if crop_sampling not in ("uniform", "keypoints", "density", "hybrid"):
        raise ValueError(
            f"Invalid sampling {crop_sampling}. Must be either 'uniform', 'keypoints', 'density', or 'hybrid."
        )

    cfg = auxiliaryfunctions.read_config(config)
    scorer = cfg["scorer"]
    project_path = cfg["project_path"]
    trainingsetfolder = auxiliaryfunctions.get_training_set_folder(cfg)
    full_training_path = Path(project_path, trainingsetfolder)
    auxiliaryfunctions.attempt_to_make_folder(full_training_path, recursive=True)

    if not metadata.TrainingDatasetMetadata.path(cfg).exists():
        trainset_metadata = metadata.TrainingDatasetMetadata.create(cfg)
        trainset_metadata.save()

    Data = merge_annotateddatasets(cfg, full_training_path)
    if Data is None:
        return
    Data = Data[scorer]

    if net_type is None:
        net_type = cfg.get("default_net_type", "dlcrnet_ms5")

    if not any(net in net_type for net in ("resnet", "eff", "dlc", "mob")):
        raise ValueError(f"Unsupported network {net_type} for TensorFlow.")

    multi_stage = False
    if all(net in net_type for net in ("dlcr", "_ms5")):
        num_layers = re.findall("dlcr([0-9]*)", net_type)[0]
        if num_layers == "":
            num_layers = 50
        net_type = f"resnet_{num_layers}"
        multi_stage = True

    dataset_type = "multi-animal-imgaug"
    (
        individuals,
        uniquebodyparts,
        multianimalbodyparts,
    ) = auxfun_multianimal.extractindividualsandbodyparts(cfg)

    if paf_graph is None:
        n_bpts = len(multianimalbodyparts)
        partaffinityfield_graph = [list(edge) for edge in combinations(range(n_bpts), 2)]
        n_edges_orig = len(partaffinityfield_graph)
        if n_edges_orig >= n_edges_threshold:
            partaffinityfield_graph = auxfun_multianimal.prune_paf_graph(
                partaffinityfield_graph,
                average_degree=paf_graph_degree,
            )
    else:
        if paf_graph == "config":
            skeleton = cfg["skeleton"]
            paf_graph = [
                sorted((multianimalbodyparts.index(bpt1), multianimalbodyparts.index(bpt2))) for bpt1, bpt2 in skeleton
            ]
            print("Using `skeleton` from the config file as a paf_graph. Data-driven skeleton will not be computed.")

        to_ignore = auxfun_multianimal.filter_unwanted_paf_connections(cfg, paf_graph)
        partaffinityfield_graph = [edge for i, edge in enumerate(paf_graph) if i not in to_ignore]
        auxfun_multianimal.validate_paf_graph(cfg, partaffinityfield_graph)

    print("Utilizing the following graph:", partaffinityfield_graph)
    partaffinityfield_predict = bool(partaffinityfield_graph)

    dlcparent_path = auxiliaryfunctions.get_deeplabcut_path()
    defaultconfigfile = os.path.join(dlcparent_path, "pose_cfg.yaml")
    model_path = auxfun_models.check_for_weights(net_type, Path(dlcparent_path))

    Shuffles = validate_shuffles(cfg, Shuffles, num_shuffles, userfeedback)

    if trainIndices is None and testIndices is None:
        splits = []
        for shuffle in Shuffles:
            for train_frac in cfg["TrainingFraction"]:
                train_inds, test_inds = SplitTrials(range(len(Data)), train_frac)
                splits.append((train_frac, shuffle, (train_inds, test_inds)))
    else:
        if len(trainIndices) != len(testIndices) != len(Shuffles):
            raise ValueError("Number of Shuffles and train and test indexes should be equal.")
        splits = []
        for shuffle, (train_inds, test_inds) in enumerate(zip(trainIndices, testIndices, strict=False)):
            trainFraction = round(len(train_inds) * 1.0 / (len(train_inds) + len(test_inds)), 2)
            print(f"You passed a split with the following fraction: {int(100 * trainFraction)}%")
            train_inds = np.asarray(train_inds)
            train_inds = train_inds[train_inds != -1]
            test_inds = np.asarray(test_inds)
            test_inds = test_inds[test_inds != -1]
            splits.append((trainFraction, Shuffles[shuffle], (train_inds, test_inds)))

    for trainFraction, shuffle, (trainIndices, testIndices) in splits:
        print(
            "Creating training data for: Shuffle:",
            shuffle,
            "TrainFraction: ",
            trainFraction,
        )

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
            ) = auxiliaryfunctions.get_data_and_metadata_filenames(trainingsetfolder, trainFraction, shuffle, cfg)
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
                engine=_ENGINE,
                train_indices=trainIndices,
                test_indices=testIndices,
                overwrite=not userfeedback,
            )

            datafilename = datafilename.split(".mat")[0] + ".pickle"
            with open(os.path.join(project_path, datafilename), "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

            modelfoldername = auxiliaryfunctions.get_model_folder(
                trainFraction,
                shuffle,
                cfg,
                engine=_ENGINE,
            )
            auxiliaryfunctions.attempt_to_make_folder(Path(config).parents[0] / modelfoldername, recursive=True)
            auxiliaryfunctions.attempt_to_make_folder(str(Path(config).parents[0] / modelfoldername / "train"))
            auxiliaryfunctions.attempt_to_make_folder(str(Path(config).parents[0] / modelfoldername / "test"))

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
                "engine": _ENGINE.aliases[0],
                "metadataset": metadatafilename,
                "num_joints": len(multianimalbodyparts) + len(uniquebodyparts),
                "all_joints": [[i] for i in range(len(multianimalbodyparts) + len(uniquebodyparts))],
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
                "num_idchannel": (len(cfg["individuals"]) if cfg.get("identity", False) else 0),
                "crop_size": list(crop_size),
                "crop_sampling": crop_sampling,
            }

            trainingdata = MakeTrain_pose_yaml(
                items2change,
                path_train_config,
                defaultconfigfile,
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
            )

            default_inf_path = Path(dlcparent_path) / "inference_cfg.yaml"
            inf_updates = dict(
                minimalnumberofconnections=int(len(cfg["multianimalbodyparts"]) / 2),
                topktoretain=len(cfg["individuals"]),
                withid=cfg.get("identity", False),
            )
            MakeInference_yaml(inf_updates, path_inference_config, default_inf_path)

            print(
                "The training dataset is successfully created. Use the function "
                "'train_network' to start training. Happy training!"
            )
