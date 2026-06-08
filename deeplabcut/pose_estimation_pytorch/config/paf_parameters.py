from pathlib import Path

from pydantic import Field
from typing_extensions import Self

from deeplabcut.core.config import DLCBaseConfig, ProjectConfig
from deeplabcut.core.config.validation import NonNegativeInt


class PAFParameters(DLCBaseConfig):
    paf_graph: list[list[NonNegativeInt]] = Field(default_factory=list)
    num_limbs: NonNegativeInt = 0
    paf_edges_to_keep: list[NonNegativeInt] = Field(default_factory=list)

    @classmethod
    def build(
        cls,
        project_config: ProjectConfig | dict | Path | str,
        *,
        bodyparts: list[str] | None = None,
        num_limbs_threshold: int = 105,
        paf_graph_degree: int = 6,
    ) -> Self:

        # Normalize ProjectConfig + set defaults if not provided
        project_config = ProjectConfig.from_any(project_config)
        if bodyparts is None:
            bodyparts = project_config.bodyparts_list

        # Build the PAF parameters
        return cls.from_dict(get_paf_parameters(project_config, bodyparts, num_limbs_threshold, paf_graph_degree))


def get_paf_parameters(
    project_config: dict,
    bodyparts: list[str],
    num_limbs_threshold: int = 105,
    paf_graph_degree: int = 6,
) -> dict:
    """Gets values for PAF parameters from the project configuration."""
    from deeplabcut.utils import auxfun_multianimal

    paf_graph = [[i, j] for i in range(len(bodyparts)) for j in range(i + 1, len(bodyparts))]
    num_limbs = len(paf_graph)
    # If the graph is unnecessarily large (with 15+ keypoints by default),
    # we randomly prune it to a size guaranteeing an average node degree of 6;
    # see Suppl. Fig S9c in Lauer et al., 2022.
    if num_limbs >= num_limbs_threshold:
        paf_graph = auxfun_multianimal.prune_paf_graph(
            paf_graph,
            average_degree=paf_graph_degree,
        )
        num_limbs = len(paf_graph)
    return {
        "paf_graph": paf_graph,
        "num_limbs": num_limbs,
        "paf_edges_to_keep": project_config.get("paf_best", list(range(num_limbs))),
    }
