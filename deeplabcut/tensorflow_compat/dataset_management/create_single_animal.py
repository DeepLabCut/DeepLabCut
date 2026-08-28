#
# DeepLabCut Toolbox (deeplabcut.org)
# (c) A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Frozen TensorFlow-specific create_training_dataset (single animal).

NOT actively maintained. Exists only as a reference for legacy TF support.
The functions in this module are called from the ``if engine == Engine.TF:``
branches in ``generate_training_dataset.trainingsetmanipulation``.
"""

from __future__ import annotations

from pathlib import Path


def _tf_get_model_path(net_type: str, dlcparent_path: str | Path) -> str:
    """Resolve the pretrained encoder weights path for TensorFlow.

    Returns:
        The path to the downloaded/verified weights file.
    """
    from deeplabcut.utils import auxfun_models

    return auxfun_models.check_for_weights(net_type, Path(dlcparent_path))


def _tf_create_pose_config_files(
    datafilename: str,
    metadatafilename: str,
    bodyparts: list,
    model_path: str,
    project_path: str,
    net_type: str,
    augmenter_type: str,
    path_train_config: str,
    defaultconfigfile: str | Path,
    path_test_config: str,
    weight_init,
) -> None:
    """Create TF-specific pose_cfg.yaml train and test configuration files.

    Raises:
        ValueError: If ``weight_init`` is not None (not supported for TF).
    """
    from deeplabcut.generate_training_dataset.trainingsetmanipulation import (
        MakeTest_pose_yaml,
        MakeTrain_pose_yaml,
    )

    if weight_init is not None:
        raise ValueError(
            "Weight initialization is not supported for TensorFlow engine. "
            "Pretrained weights are automatically downloaded."
        )

    items2change = {
        "dataset": datafilename,
        "engine": "tensorflow",
        "metadataset": metadatafilename,
        "num_joints": len(bodyparts),
        "all_joints": [[i] for i in range(len(bodyparts))],
        "all_joints_names": [str(bpt) for bpt in bodyparts],
        "init_weights": model_path,
        "project_path": project_path,
        "net_type": net_type,
        "dataset_type": augmenter_type,
    }

    items2drop: dict = {}
    if augmenter_type == "scalecrop":
        # these values are dropped as scalecrop
        # doesn't have rotation implemented
        items2drop = {"rotation": 0, "rotratio": 0.0}
    # Also drop maDLC smart cropping augmentation parameters
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
        save=True,
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
