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

from dataclasses import dataclass

import numpy as np

import deeplabcut.modelzoo.utils as modelzoo_utils


@dataclass
class WeightInitialization:
    """The dataset from which to initialize weights

    To build a WeightInitialization instance for a project using the conversion table
    specified in the project configuration file, use:

        ```
        from pathlib import Path
        from deeplabcut.utils.auxiliaryfunctions import read_config

        project_cfg = read_config("/path/to/my/project/config.yaml")
        super_animal = "superanimal_quadruped"
        weight_init = WeightInitialization.build(
            cfg=project_cfg,
            super_animal="superanimal_quadruped",
            with_decoder=True,
        )
        ```

    Args:
        dataset: The dataset on which the model weights were trained. Must be one of the
            SuperAnimal weights.
        with_decoder: Whether to load the decoder weights as well.
        conversion_array: The mapping from SuperAnimal to project bodyparts. Required
            when `with_decoder=True`.

            An array [7, 0, 1] means the project has 3 bodyparts, where the 1st bodypart
            corresponds to the 8th bodypart in the pretrained model, the 2nd to the 1st
            and the 3rd to the 2nd (as arrays are 0-indexed).
        bodyparts: Optionally, the name of each bodypart entry in the conversion array.
    """
    dataset: str
    with_decoder: bool = False
    conversion_array: np.ndarray | None = None
    bodyparts: list[str] | None = None

    def __post_init__(self):
        # check that the dataset exists; raises a ValueError if it doesn't
        _ = modelzoo_utils.get_super_animal_project_cfg(self.dataset)
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
        data = {
            "dataset": self.dataset,
            "with_decoder": self.with_decoder,
        }
        if self.conversion_array is not None:
            data["conversion_array"] = self.conversion_array.tolist()

        return data

    @staticmethod
    def from_dict(data: dict) -> "WeightInitialization":
        conversion_array = data.get("conversion_array")
        if conversion_array is not None:
            conversion_array = np.array(conversion_array, dtype=int)

        return WeightInitialization(
            dataset=data["dataset"],
            with_decoder=data["with_decoder"],
            conversion_array=conversion_array,
        )

    @staticmethod
    def build(
        cfg: dict,
        super_animal: str,
        with_decoder: bool = False,
    ) -> "WeightInitialization":
        """Builds a WeightInitialization for a project

        Args:
            cfg: The project's configuration.
            super_animal: The SuperAnimal model with which to initialize weights.
            with_decoder: Whether to load the decoder weights as well. If this is true,
                a conversion table must be specified for the given SuperAnimal in the
                project configuration file. See
                ``deeplabcut.modelzoo.utils.create_conversion_table`` to create a
                conversion table.

        Returns:
            The built WeightInitialization.
        """
        conversion_array = None
        bodyparts = None
        if with_decoder:
            conversion_table = modelzoo_utils.get_conversion_table(cfg, super_animal)
            conversion_array = conversion_table.to_array()
            bodyparts = conversion_table.converted_bodyparts()

        return WeightInitialization(
            dataset=super_animal,
            with_decoder=with_decoder,
            conversion_array=conversion_array,
            bodyparts=bodyparts,
        )
