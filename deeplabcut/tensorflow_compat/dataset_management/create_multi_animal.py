#
# DeepLabCut Toolbox (deeplabcut.org)
# (c) A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Frozen TensorFlow-specific create_multianimaltraining_dataset.

NOT actively maintained. Exists only as a reference for legacy TF support.
The functions in this module are called from the ``if engine == Engine.TF:``
branches in ``generate_training_dataset.multiple_individuals_trainingsetmanipulation``.
"""

from __future__ import annotations


def _tf_create_multianimal_pose_config_files(
    datafilename: str,
    metadatafilename: str,
    multianimalbodyparts: list,
    uniquebodyparts: list,
    model_path: str,
    project_path: str,
    net_type: str,
    multi_stage: bool,
    partaffinityfield_graph: list,
    partaffinityfield_predict: bool,
    dataset_type: str,
    crop_size: tuple,
    crop_sampling: str,
    cfg: dict,
    path_train_config: str,
    defaultconfigfile: str,
    path_test_config: str,
) -> None:
    """Create TF-specific multi-animal pose_cfg.yaml config files."""
    from deeplabcut.generate_training_dataset.trainingsetmanipulation import (
        MakeTest_pose_yaml,
        MakeTrain_pose_yaml,
    )

    jointnames = [str(bpt) for bpt in multianimalbodyparts]
    jointnames.extend([str(bpt) for bpt in uniquebodyparts])
    items2change = {
        "dataset": datafilename,
        "engine": "tensorflow",
        "metadataset": metadatafilename,
        "num_joints": len(multianimalbodyparts) + len(uniquebodyparts),
        "all_joints": [[i] for i in range(len(multianimalbodyparts) + len(uniquebodyparts))],
        "all_joints_names": jointnames,
        "init_weights": str(model_path),
        "project_path": project_path,
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
        save=True,
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
