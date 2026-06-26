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
"""TensorFlow implementation of create_training_dataset_from_existing_split."""

from __future__ import annotations

from pathlib import Path

import deeplabcut.generate_training_dataset.metadata as metadata
from deeplabcut.core.engine import Engine
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.generate_training_dataset.trainingsetmanipulation import _compute_padding
from deeplabcut.tensorflow_compat.dataset_management.create_single_animal import (
    create_training_dataset,
)
from deeplabcut.utils import auxiliaryfunctions


def create_training_dataset_from_existing_split(
    config: str,
    from_shuffle: int,
    from_trainsetindex: int = 0,
    num_shuffles: int = 1,
    shuffles: list[int] | None = None,
    userfeedback: bool = True,
    net_type: str | None = None,
    detector_type: str | None = None,
    augmenter_type: str | None = None,
    ctd_conditions: int | str | Path | tuple[int, str] | tuple[int, int] | None = None,
    posecfg_template: dict | None = None,
    superanimal_name: str = "",
    weight_init: WeightInitialization | None = None,
    engine: Engine | None = None,
) -> None | list[int]:
    cfg = auxiliaryfunctions.read_config(config)
    trainset_meta_path = metadata.TrainingDatasetMetadata.path(cfg)
    if not trainset_meta_path.exists():
        meta = metadata.TrainingDatasetMetadata.create(cfg)
        meta.save()
    else:
        meta = metadata.TrainingDatasetMetadata.load(cfg, load_splits=False)

    shuffle = meta.get(trainset_index=from_trainsetindex, index=from_shuffle)
    shuffle = shuffle.load_split(cfg, trainset_path=trainset_meta_path.parent)

    num_copies = num_shuffles
    if shuffles is not None:
        num_copies = len(shuffles)

    train_idx = list(shuffle.split.train_indices)
    test_idx = list(shuffle.split.test_indices)
    n_train, n_test = len(train_idx), len(test_idx)

    train_fraction = round(cfg["TrainingFraction"][from_trainsetindex], 2)
    if round(n_train / (n_train + n_test), 2) != train_fraction:
        train_padding, test_padding = _compute_padding(train_fraction, n_train, n_test)
        train_idx = train_idx + (train_padding * [-1])
        test_idx = test_idx + (test_padding * [-1])

    return create_training_dataset(
        config=config,
        num_shuffles=num_shuffles,
        Shuffles=shuffles,
        userfeedback=userfeedback,
        trainIndices=[train_idx for _ in range(num_copies)],
        testIndices=[test_idx for _ in range(num_copies)],
        net_type=net_type,
        detector_type=detector_type,
        augmenter_type=augmenter_type,
        posecfg_template=posecfg_template,
        superanimal_name=superanimal_name,
        weight_init=weight_init,
        engine=Engine.TF,
        ctd_conditions=ctd_conditions,
    )
