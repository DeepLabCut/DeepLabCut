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
"""Defines conversion tables mapping DeepLabCut project bodyparts to SA bodyparts"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConversionTable:
    """Maps DLC project bodyparts to the corresponding SuperAnimal bodyparts

    The conversion table must satisfy the following conditions (checked by validate):
        - All SuperAnimal bodyparts must be valid (defined for the SuperAnimal model)
        - All project bodyparts must be valid (defined for the DLC project)
    """

    super_animal: str
    project_bodyparts: list[str]
    super_animal_bodyparts: list[str]
    table: dict[str, str]

    def __post_init__(self):
        """Validates the table"""
        self.validate()

    def to_array(self) -> np.ndarray:
        """
        Returns:
            An array mapping the indices of SuperAnimal bodyparts

        Raises:
            ValueError: If the conversion table is misconfigured.
        """
        self.validate()
        sa_indices = {sa_bpt: i for i, sa_bpt in enumerate(self.super_animal_bodyparts)}
        sa_bpt_ordering = [self.table[bpt] for bpt in self.converted_bodyparts()]
        return np.array([sa_indices[sa_bpt] for sa_bpt in sa_bpt_ordering])

    def converted_bodyparts(self) -> list[str]:
        """Returns: The project bodyparts included in this ordered"""
        return [bpt for bpt in self.project_bodyparts if bpt in self.table]

    def validate(self) -> None:
        """
        Raises:
            ValueError: If the conversion table is misconfigured.
        """
        project_bpts = set(self.project_bodyparts)
        sa_bpts = set(self.super_animal_bodyparts)

        mapped_sa = set(self.table.values())
        mapped_project = set(self.table.keys())

        # check all mapped SuperAnimal bodyparts are in the config
        if len(mapped_sa.difference(sa_bpts)) != 0:
            extra_bodyparts = set(mapped_sa).difference(sa_bpts)
            raise ValueError(
                f"Some bodyparts in your mapping are not in the {self.super_animal} "
                f"model: {extra_bodyparts}. Available bodyparts are {' '.join(sa_bpts)}"
            )

        # check all given bodyparts are in the project configuration
        if len(mapped_project.difference(project_bpts)) != 0:
            extra_bodyparts = mapped_project.difference(project_bpts)
            raise ValueError(
                "Some bodyparts in your mapping are not in your project configuration: "
                f"{extra_bodyparts}. Defined bodyparts are {' '.join(project_bpts)}"
            )
