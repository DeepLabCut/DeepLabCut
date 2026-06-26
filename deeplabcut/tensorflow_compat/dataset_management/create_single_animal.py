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
"""TensorFlow implementation of single-animal training dataset creation."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio

import deeplabcut.generate_training_dataset.metadata as metadata
from deeplabcut.core.engine import Engine, get_available_aug_methods
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.generate_training_dataset.trainingsetmanipulation import (
    SplitTrials,
    format_training_data,
    merge_annotateddatasets,
    validate_shuffles,
)
from deeplabcut.tensorflow_compat.dataset_management.create_multi_animal import (
    create_multianimaltraining_dataset,
)
from deeplabcut.tensorflow_compat.dataset_management.pose_yaml import (
    MakeTest_pose_yaml,
    MakeTrain_pose_yaml,
)
from deeplabcut.tensorflow_compat.tensorflow_api import return_train_network_path
from deeplabcut.utils import auxfun_models, auxiliaryfunctions


def create_training_dataset(
    config,
    num_shuffles=1,
    Shuffles=None,
    windows2linux=False,
    userfeedback=True,
    trainIndices=None,
    testIndices=None,
    net_type=None,
    detector_type=None,
    augmenter_type=None,
    posecfg_template=None,
    superanimal_name="",
    weight_init: WeightInitialization | None = None,
    engine: Engine | None = None,
    ctd_conditions: int | str | Path | tuple[int, str] | tuple[int, int] | None = None,
):
    if windows2linux:
        warnings.warn(
            "`windows2linux` has no effect since 2.2.0.4 and will be removed in 2.2.1.",
            FutureWarning,
            stacklevel=2,
        )

    cfg = auxiliaryfunctions.read_config(config)
    auxiliaryfunctions.get_deeplabcut_path()

    if superanimal_name != "":
        raise ValueError(
            "Invalid argument superanimal_name. This functionality has been "
            "removed. Please use modelzoo.build_weight_init() instead."
        )

    if posecfg_template:
        if (
            not posecfg_template.endswith("pose_cfg.yaml")
            and not posecfg_template.endswith("superquadruped.yaml")
            and not posecfg_template.endswith("supertopview.yaml")
        ):
            raise ValueError("posecfg_template argument must contain path to a pose_cfg.yaml file")
        else:
            print("Reloading pose_cfg parameters from " + posecfg_template + "\n")
            from deeplabcut.utils.auxiliaryfunctions import read_plainconfig

        prior_cfg = read_plainconfig(posecfg_template)

    if cfg.get("multianimalproject", False):
        return create_multianimaltraining_dataset(
            config,
            num_shuffles,
            Shuffles,
            net_type=net_type,
            detector_type=detector_type,
            trainIndices=trainIndices,
            testIndices=testIndices,
            userfeedback=userfeedback,
            engine=Engine.TF,
            weight_init=weight_init,
            ctd_conditions=ctd_conditions,
        )

    engine = Engine.TF
    scorer = cfg["scorer"]
    project_path = cfg["project_path"]

    trainingsetfolder = auxiliaryfunctions.get_training_set_folder(cfg)
    auxiliaryfunctions.attempt_to_make_folder(Path(os.path.join(project_path, str(trainingsetfolder))), recursive=True)

    if not metadata.TrainingDatasetMetadata.path(cfg).exists():
        trainset_metadata = metadata.TrainingDatasetMetadata.create(cfg)
        trainset_metadata.save()

    Data = merge_annotateddatasets(
        cfg,
        Path(os.path.join(project_path, trainingsetfolder)),
    )
    if Data is None:
        return
    Data = Data[scorer]

    if net_type is None:
        net_type = cfg.get("default_net_type", "resnet_50")
    elif not ("resnet" in net_type or "mobilenet" in net_type or "efficientnet" in net_type or "dlcrnet" in net_type):
        raise ValueError("Invalid network type:", net_type)

    augmenters = get_available_aug_methods(engine)
    default_augmenter = augmenters[0]
    if augmenter_type is None:
        augmenter_type = cfg.get("default_augmenter", default_augmenter)

        if augmenter_type is None:
            augmenter_type = default_augmenter
            auxiliaryfunctions.edit_config(config, {"default_augmenter": augmenter_type})
        elif augmenter_type not in augmenters:
            augmenter_type = default_augmenter
            logging.info(
                f"Default augmenter {augmenter_type} not available for engine "
                f"{engine}: using {default_augmenter} instead"
            )

    if augmenter_type not in augmenters:
        raise ValueError(f"Invalid augmenter type: {augmenter_type} (available: for engine={engine}: {augmenters})")

    if posecfg_template:
        if net_type != prior_cfg["net_type"]:
            print(
                "WARNING: Specified net_type does not match net_type from "
                "posecfg_template path entered. Proceed with caution."
            )
        if augmenter_type != prior_cfg["dataset_type"]:
            print(
                "WARNING: Specified augmenter_type does not match dataset_type "
                "from posecfg_template path entered. Proceed with caution."
            )

    dlcparent_path = auxiliaryfunctions.get_deeplabcut_path()
    if not posecfg_template:
        defaultconfigfile = os.path.join(dlcparent_path, "pose_cfg.yaml")
    else:
        defaultconfigfile = posecfg_template

    model_path = auxfun_models.check_for_weights(net_type, Path(dlcparent_path))

    Shuffles = validate_shuffles(cfg, Shuffles, num_shuffles, userfeedback)

    if trainIndices is None and testIndices is None:
        splits = [
            (
                trainFraction,
                shuffle,
                SplitTrials(range(len(Data.index)), trainFraction),
            )
            for trainFraction in cfg["TrainingFraction"]
            for shuffle in Shuffles
        ]
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

    bodyparts = auxiliaryfunctions.get_bodyparts(cfg)
    nbodyparts = len(bodyparts)
    for trainFraction, shuffle, (trainIndices, testIndices) in splits:
        if len(trainIndices) > 0:
            if userfeedback:
                trainposeconfigfile, _, _ = return_train_network_path(
                    config,
                    shuffle=shuffle,
                    trainingsetindex=cfg["TrainingFraction"].index(trainFraction),
                    engine=engine,
                )
                if trainposeconfigfile.is_file():
                    askuser = input(
                        "The model folder is already present. "
                        "If you continue, it will overwrite the existing model (split). "
                        "Do you want to continue?(yes/no): "
                    )
                    if askuser == "no" or askuser == "No" or askuser == "N" or askuser == "No":
                        raise Exception(
                            "Use the Shuffles argument as a list to specify a different shuffle index. "
                            "Check out the help for more details."
                        )

            (
                datafilename,
                metadatafilename,
            ) = auxiliaryfunctions.get_data_and_metadata_filenames(trainingsetfolder, trainFraction, shuffle, cfg)

            data, MatlabData = format_training_data(Data, trainIndices, nbodyparts, project_path)
            sio.savemat(os.path.join(project_path, datafilename), {"dataset": MatlabData})

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

            modelfoldername = auxiliaryfunctions.get_model_folder(
                trainFraction,
                shuffle,
                cfg,
                engine=engine,
            )
            auxiliaryfunctions.attempt_to_make_folder(Path(config).parents[0] / modelfoldername, recursive=True)
            auxiliaryfunctions.attempt_to_make_folder(str(Path(config).parents[0] / modelfoldername) + "/train")
            auxiliaryfunctions.attempt_to_make_folder(str(Path(config).parents[0] / modelfoldername) + "/test")

            path_train_config = str(
                os.path.join(
                    cfg["project_path"],
                    Path(modelfoldername),
                    "train",
                    engine.pose_cfg_name,
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

            if weight_init is not None:
                raise ValueError(
                    "Weight initialization is not supported for TensorFlow engine. "
                    "Pretrained weights are automatically downloaded."
                )
            items2change = {
                "dataset": datafilename,
                "engine": engine.aliases[0],
                "metadataset": metadatafilename,
                "num_joints": len(bodyparts),
                "all_joints": [[i] for i in range(len(bodyparts))],
                "all_joints_names": [str(bpt) for bpt in bodyparts],
                "init_weights": model_path,
                "project_path": str(cfg["project_path"]),
                "net_type": net_type,
                "dataset_type": augmenter_type,
            }

            items2drop = {}
            if augmenter_type == "scalecrop":
                items2drop = {"rotation": 0, "rotratio": 0.0}
            for key in [
                "pre_resize",
                "crop_size",
                "max_shift",
                "crop_sampling",
            ]:
                items2drop[key] = None

            trainingdata = MakeTrain_pose_yaml(
                items2change,
                path_train_config,
                defaultconfigfile,
                items2drop,
            )

            keys2save = [
                "dataset",
                "num_joints",
                "all_joints",
                "all_joints_names",
                "net_type",
                "init_weights",
                "global_scale",
                "location_refinement",
                "locref_stdev",
            ]
            MakeTest_pose_yaml(trainingdata, keys2save, path_test_config)
            print(
                "The training dataset is successfully created. Use the function"
                "'train_network' to start training. Happy training!"
            )

    return splits
