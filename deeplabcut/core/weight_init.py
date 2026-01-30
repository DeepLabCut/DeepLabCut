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
"""Classes to configure how to initialize model weights"""

from __future__ import annotations

import warnings
from pydantic.dataclasses import dataclass
from pathlib import Path

import numpy as np

from deeplabcut.core.types import PydanticNDArray


@dataclass
class WeightInitialization:
    """Configures weights initialization when transfer learning or fine-tuning models

    Args:
        snapshot_path: The path to the snapshot used to initialize pose model weights
            when training a model.
        detector_snapshot_path: The path to the snapshot used to initialize detector
            weights when training a model.
        dataset: Optionally, the dataset on which the snapshots were trained. Required
            when fine-tuning SuperAnimal models.
        with_decoder: Whether to load the decoder weights as well.
        memory_replay: Only when ``with_decoder=True``. Whether to train the model with
            memory replay, so that it predicts all SuperAnimal (or previous project)
            bodyparts.
        conversion_array: The mapping from SuperAnimal (or other project, on which the
            weights were trained) to project bodyparts. Required when
            `with_decoder=True`.
            An array [7, 0, 1] means the project has 3 bodyparts, where the 1st bodypart
            corresponds to the 8th bodypart in the pretrained model, the 2nd to the 1st
            and the 3rd to the 2nd (as arrays are 0-indexed).
        bodyparts: Optionally, the name of each bodypart entry in the conversion array.
    """

    snapshot_path: Path | None = None
    detector_snapshot_path: Path | None = None
    dataset: str | None = None
    with_decoder: bool = False
    memory_replay: bool = False
    conversion_array: PydanticNDArray | None = None
    bodyparts: list[str] | None = None

    def __post_init__(self):
        if self.memory_replay and not self.with_decoder:
            raise ValueError(
                "You cannot train a model with memory replay if you do not keep the "
                "decoder layers (``with_decoder=True``), but you passed "
                "`memory_replay=True` and `with_decoder=False`. Please change your "
                "WeightInitialization parameters."
            )

        if self.with_decoder and self.conversion_array is None:
            raise ValueError(
                f"You must specify a conversion_array to initialize decoder weights "
                f"(``with_decoder=True``)."
            )

        if self.bodyparts is not None and self.conversion_array is None:
            raise ValueError(
                f"Specifying bodyparts should only be done when `with_decoder=True` and"
                f" the conversion array is specified."
            )

        if self.conversion_array is not None and self.bodyparts is not None:
            if not len(self.conversion_array) == len(self.bodyparts):
                raise ValueError(
                    f"There must be the same number of elements in the bodyparts list "
                    "and conv. array; found {self.bodyparts}, {self.conversion_array}"
                )

    def to_dict(self) -> dict:
        """Returns: the weight initialization as a dict"""
        data = dict()
        if self.dataset is not None:
            data["dataset"] = self.dataset

        data["snapshot_path"] = str(self.snapshot_path)
        if self.detector_snapshot_path is not None:
            data["detector_snapshot_path"] = str(self.detector_snapshot_path)

        data["with_decoder"] = self.with_decoder
        data["memory_replay"] = self.memory_replay

        if self.conversion_array is not None:
            data["conversion_array"] = self.conversion_array.tolist()

        if self.bodyparts is not None:
            data["bodyparts"] = self.bodyparts

        return data

    @staticmethod
    def from_dict(data: dict) -> "WeightInitialization":
        if "snapshot_path" not in data:
            return WeightInitialization.from_dict_legacy(data)

        detector_snapshot_path = data.get("detector_snapshot_path")
        if detector_snapshot_path is not None:
            detector_snapshot_path = Path(detector_snapshot_path)

        conversion_array = data.get("conversion_array")
        if conversion_array is not None:
            conversion_array = np.array(conversion_array, dtype=int)

        return WeightInitialization(
            snapshot_path=Path(data["snapshot_path"]),
            detector_snapshot_path=detector_snapshot_path,
            dataset=data.get("dataset"),
            with_decoder=data["with_decoder"],
            memory_replay=data["memory_replay"],
            conversion_array=conversion_array,
            bodyparts=data.get("bodyparts"),
        )

    @staticmethod
    def from_dict_legacy(data: dict) -> "WeightInitialization":
        """Deals with weight initialization that were created before 3.0.0rc5"""
        import deeplabcut.pose_estimation_pytorch.modelzoo.utils as utils

        conversion_array = data.get("conversion_array")
        if conversion_array is not None:
            conversion_array = np.array(conversion_array, dtype=int)

        return WeightInitialization(
            snapshot_path=utils.get_super_animal_snapshot_path(
                dataset=data["dataset"],
                model_name="hrnet_w32",
            ),
            detector_snapshot_path=utils.get_super_animal_snapshot_path(
                dataset=data["dataset"],
                model_name="fasterrcnn_resnet50_fpn_v2",
            ),
            with_decoder=data["with_decoder"],
            memory_replay=data["memory_replay"],
            conversion_array=conversion_array,
            bodyparts=data.get("bodyparts"),
        )

    @staticmethod
    def build(
        cfg: dict,
        super_animal: str,
        model_name: str = "hrnet_w32",
        detector_name: str = "fasterrcnn_resnet50_fpn_v2",
        with_decoder: bool = False,
        memory_replay: bool = False,
        customized_pose_checkpoint: str | None = None,
        customized_detector_checkpoint: str | None = None,
    ) -> "WeightInitialization":
        """Builds a WeightInitialization for a project

        `WeightInitialization.build` is deprecated and will be removed in a future
        version of DeepLabCut. Please use `build_weight_init` from `deeplabcut.modelzoo`
        instead.

        Args:
            cfg: The project's configuration.
            super_animal: The SuperAnimal model with which to initialize weights.
            model_name: The name of the model architecture for which to load the weights
                (defaults to "hrnet_w32" for backwards compatibility).
            detector_name: The name of the detector architecture for which to load the
                weights (defaults to "fasterrcnn_resnet50_fpn_v2" for backwards
                compatibility).
            with_decoder: Whether to load the decoder weights as well. If this is true,
                a conversion table must be specified for the given SuperAnimal in the
                project configuration file. See
                ``deeplabcut.modelzoo.utils.create_conversion_table`` to create a
                conversion table.
            memory_replay: Only when ``with_decoder=True``. Whether to train the model
                with memory replay, so that it predicts all SuperAnimal bodyparts.
            customized_pose_checkpoint: A customized SuperAnimal pose checkpoint, as an
                alternative to the Hugging Face one
            customized_detector_checkpoint: A customized SuperAnimal detector
                checkpoint, as an alternative to the Hugging Face one

        Returns:
            The built WeightInitialization.
        """
        from deeplabcut.modelzoo import build_weight_init
        deprecation_warning = (
            "The `WeightInitialization.build` is deprecated and will be removed in a "
            "future version of DeepLabCut. Please use `build_weight_init` from "
            "`deeplabcut.modelzoo` instead."
        )
        warnings.warn(deprecation_warning, DeprecationWarning)

        return build_weight_init(
            cfg,
            super_animal,
            model_name,
            detector_name,
            with_decoder,
            memory_replay,
            customized_pose_checkpoint,
            customized_detector_checkpoint,
        )
