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
"""DeepLabCut training dataset management API"""

from __future__ import annotations

from pathlib import Path

from deeplabcut.api._tf_routing import with_tensorflow_fallback
from deeplabcut.core.engine import Engine
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.utils.deprecation import renamed_parameter


@renamed_parameter(old="Shuffles", new="shuffles", since="3.1")
@with_tensorflow_fallback
def create_training_dataset(
    config,
    num_shuffles=1,
    shuffles=None,
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
    from deeplabcut.generate_training_dataset.trainingsetmanipulation import (
        create_training_dataset as _create_training_dataset,
    )

    return _create_training_dataset(
        config,
        num_shuffles=num_shuffles,
        shuffles=shuffles,
        windows2linux=windows2linux,
        userfeedback=userfeedback,
        trainIndices=trainIndices,
        testIndices=testIndices,
        net_type=net_type,
        detector_type=detector_type,
        augmenter_type=augmenter_type,
        posecfg_template=posecfg_template,
        superanimal_name=superanimal_name,
        weight_init=weight_init,
        ctd_conditions=ctd_conditions,
    )


@renamed_parameter(old="Shuffles", new="shuffles", since="3.1")
@with_tensorflow_fallback
def create_multianimaltraining_dataset(
    config,
    num_shuffles=1,
    shuffles=None,
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
    engine: Engine | None = None,
    ctd_conditions: int | str | Path | tuple[int, str] | tuple[int, int] | None = None,
):
    from deeplabcut.generate_training_dataset.multiple_individuals_trainingsetmanipulation import (
        create_multianimaltraining_dataset as _create_multianimaltraining_dataset,
    )

    return _create_multianimaltraining_dataset(
        config,
        num_shuffles=num_shuffles,
        shuffles=shuffles,
        windows2linux=windows2linux,
        net_type=net_type,
        detector_type=detector_type,
        numdigits=numdigits,
        crop_size=crop_size,
        crop_sampling=crop_sampling,
        paf_graph=paf_graph,
        trainIndices=trainIndices,
        testIndices=testIndices,
        n_edges_threshold=n_edges_threshold,
        paf_graph_degree=paf_graph_degree,
        userfeedback=userfeedback,
        weight_init=weight_init,
        ctd_conditions=ctd_conditions,
    )


@with_tensorflow_fallback
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
    from deeplabcut.generate_training_dataset.trainingsetmanipulation import (
        create_training_dataset_from_existing_split as _create_training_dataset_from_existing_split,
    )

    return _create_training_dataset_from_existing_split(
        config=config,
        from_shuffle=from_shuffle,
        from_trainsetindex=from_trainsetindex,
        num_shuffles=num_shuffles,
        shuffles=shuffles,
        userfeedback=userfeedback,
        net_type=net_type,
        detector_type=detector_type,
        augmenter_type=augmenter_type,
        ctd_conditions=ctd_conditions,
        posecfg_template=posecfg_template,
        superanimal_name=superanimal_name,
        weight_init=weight_init,
    )


@with_tensorflow_fallback
def create_training_model_comparison(
    config,
    trainindex=0,
    num_shuffles=1,
    net_types=None,
    augmenter_types=None,
    userfeedback=False,
    windows2linux=False,
    engine: Engine | None = None,
):
    from deeplabcut.generate_training_dataset.trainingsetmanipulation import (
        create_training_model_comparison as _create_training_model_comparison,
    )

    return _create_training_model_comparison(
        config,
        trainindex=trainindex,
        num_shuffles=num_shuffles,
        net_types=net_types,
        augmenter_types=augmenter_types,
        userfeedback=userfeedback,
        windows2linux=windows2linux,
    )
