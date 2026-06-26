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
"""TensorFlow implementation of create_training_model_comparison."""

from __future__ import annotations

import logging
import os
import warnings

from deeplabcut.generate_training_dataset.trainingsetmanipulation import (
    get_existing_shuffle_indices,
    mergeandsplit,
)
from deeplabcut.tensorflow_compat.dataset_management.create_single_animal import (
    create_training_dataset,
)
from deeplabcut.utils import auxiliaryfunctions


def create_training_model_comparison(
    config,
    trainindex=0,
    num_shuffles=1,
    net_types=None,
    augmenter_types=None,
    userfeedback=False,
    windows2linux=False,
):
    if augmenter_types is None:
        augmenter_types = ["imgaug"]
    if net_types is None:
        net_types = ["resnet_50"]
    cfg = auxiliaryfunctions.read_config(config)

    if windows2linux:
        warnings.warn(
            "`windows2linux` has no effect since 2.2.0.4 and will be removed in 2.2.1.",
            FutureWarning,
            stacklevel=2,
        )

    log_file_name = os.path.join(cfg["project_path"], "training_model_comparison.log")
    logger = logging.getLogger("training_model_comparison")
    if not logger.handlers:
        logger = logging.getLogger("training_model_comparison")
        hdlr = logging.FileHandler(log_file_name)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)

    existing_shuffles = get_existing_shuffle_indices(cfg)
    if len(existing_shuffles) == 0:
        largestshuffleindex = 0
    else:
        largestshuffleindex = existing_shuffles[-1] + 1

    shuffle_list = []
    for shuffle in range(num_shuffles):
        trainIndices, testIndices = mergeandsplit(config, trainindex=trainindex, uniform=True)
        for idx_net, net in enumerate(net_types):
            for idx_aug, aug in enumerate(augmenter_types):
                get_max_shuffle_idx = (
                    largestshuffleindex
                    + idx_aug
                    + idx_net * len(augmenter_types)
                    + shuffle * len(augmenter_types) * len(net_types)
                )

                shuffle_list.append(get_max_shuffle_idx)
                log_info = str(
                    "Shuffle index:"
                    + str(get_max_shuffle_idx)
                    + ", net_type:"
                    + net
                    + ", augmenter_type:"
                    + aug
                    + ", trainsetindex:"
                    + str(trainindex)
                    + ", frozen shuffle ID:"
                    + str(shuffle)
                )
                create_training_dataset(
                    config,
                    Shuffles=[get_max_shuffle_idx],
                    net_type=net,
                    trainIndices=[trainIndices],
                    testIndices=[testIndices],
                    augmenter_type=aug,
                    userfeedback=userfeedback,
                )
                logger.info(log_info)

    return shuffle_list
