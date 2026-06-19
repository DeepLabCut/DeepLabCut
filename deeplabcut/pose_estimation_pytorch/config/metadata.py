from pathlib import Path

from pydantic import Field
from typing_extensions import Self

from deeplabcut.core.config import DLCBaseConfig, ProjectConfig
from deeplabcut.core.config.validation import UniqueStrList


# TODO @deruyter92 2026-06-08: This is duplicated from ProjectConfig. Note that field names are misaligned!
# Once field names are aligned (in v1), we should merge these ProjectConfig subsets.
class PoseMetadata(DLCBaseConfig):
    project_path: Path | None = None
    pose_config_path: Path | None = None
    bodyparts: UniqueStrList = Field(default_factory=list)
    unique_bodyparts: UniqueStrList = Field(default_factory=list, json_schema_extra={"aliases": ["uniquebodyparts"]})
    individuals: UniqueStrList = Field(default_factory=lambda: ["individual_1"])

    # TODO @deruyter92 2026-06-09: Nullable field to support old configs with empty identity field -> fix in v1
    with_identity: bool | None = Field(default=None, json_schema_extra={"aliases": ["identity"]})

    @property
    def num_individuals(self) -> int:
        return len(self.individuals)

    @property
    def num_bodyparts(self) -> int:
        return len(self.bodyparts)

    @property
    def num_unique_bodyparts(self) -> int:
        return len(self.unique_bodyparts) if self.unique_bodyparts is not None else 0

    @classmethod
    def build(
        cls,
        project_config: ProjectConfig | dict | Path | str,
        *,
        project_path: Path | str | None = None,
        pose_config_path: Path | str | None = None,
        bodyparts: list[str] | None = None,
        unique_bodyparts: list[str] | None = None,
        individuals: list[str] | None = None,
        with_identity: bool | None = None,
    ) -> Self:
        """Get metadata from a project configuration with optional overrides"""
        cfg = ProjectConfig.from_any(project_config)

        # Conversions for diverging fields in ProjectConfig
        cfg_bodyparts = cfg.bodyparts_list
        cfg_unique_bodyparts = cfg.uniquebodyparts
        cfg_with_identity = cfg.identity

        return cls(
            project_path=project_path if project_path is not None else cfg.project_path,
            pose_config_path=pose_config_path if pose_config_path is not None else cfg.pose_config_path,
            bodyparts=bodyparts if bodyparts is not None else cfg_bodyparts,
            unique_bodyparts=unique_bodyparts if unique_bodyparts is not None else cfg_unique_bodyparts,
            individuals=individuals if individuals is not None else cfg.individuals,
            with_identity=with_identity if with_identity is not None else cfg_with_identity,
        )

    @classmethod
    def build_for_superanimal(cls, super_animal: str, model_name: str, max_individuals: int) -> Self:
        from deeplabcut.pose_estimation_pytorch.modelzoo.config import build_superanimal_metadata

        metadata = build_superanimal_metadata(
            super_animal=super_animal,
            model_name=model_name,
            max_individuals=max_individuals,
        )
        return cls.from_dict(metadata)

    # NOTE @deruyter92 2026-06-12
    # This serves as a replacement for pose_estimation_pytorch.config.make_basic_project_config.
    # It can safely be removed when we stop supporting that API. Prefer PoseMetadata(..) instead.
    def to_dict_legacy(self) -> dict:
        return dict(
            project_path=self.project_path,
            multianimalproject=self.num_individuals > 1,
            bodyparts=self.bodyparts if self.num_individuals <= 1 else "MULTI!",
            multianimalbodyparts=self.bodyparts if self.num_individuals > 1 else None,
            uniquebodyparts=[],
            individuals=self.individuals,
        )
