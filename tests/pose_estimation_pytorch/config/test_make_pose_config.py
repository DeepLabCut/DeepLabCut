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
"""Tests the pre-processors"""
import pytest

from deeplabcut.pose_estimation_pytorch.config.make_pose_config import make_pytorch_pose_config
from deeplabcut.pose_estimation_pytorch.config.utils import pretty_print, update_config


@pytest.mark.parametrize("bodyparts", [["nose"], ["nose", "ear", "eye"]])
@pytest.mark.parametrize(
    "net_type", ["resnet_50", "resnet_101", "hrnet_w18", "hrnet_w32", "hrnet_w48"]
)
def test_make_single_animal_config(bodyparts: list[str], net_type: str):
    # Single animal projects can't have unique bodyparts
    project_config = _make_project_config(
        project_path="my/little/project",
        multianimal=False,
        identity=False,
        individuals=[],
        bodyparts=bodyparts,
        unique_bodyparts=[],
    )
    pytorch_pose_config = make_pytorch_pose_config(
        project_config,
        "pytorch_config.yaml",
        net_type=net_type,
    )
    pretty_print(pytorch_pose_config)

    # check heads are there
    assert "bodypart" in pytorch_pose_config["model"]["heads"].keys()
    # check that the bodypart head has locref and heatmaps and the correct output shapes
    bodypart_head = pytorch_pose_config["model"]["heads"]["bodypart"]

    outputs = [("heatmap_config", len(bodyparts))]
    if bodypart_head["predictor"]["location_refinement"]:
        outputs += [("locref_config", 2 * len(bodyparts))]

    for name, output_channels in outputs:
        head = bodypart_head[name]
        if "final_conv" in head:
            actual_output_channels = head["final_conv"]["out_channels"]
        else:
            actual_output_channels = head["channels"][-1]
        assert name in bodypart_head
        assert actual_output_channels == output_channels


@pytest.mark.parametrize("multianimal", [True])
@pytest.mark.parametrize("individuals", [["single"], ["bugs", "daffy"]])
@pytest.mark.parametrize("bodyparts", [["nose"], ["nose", "ear", "eye"]])
@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("unique_bodyparts", [[], ["tail"]])
@pytest.mark.parametrize(
    "net_type", ["resnet_50", "resnet_101", "hrnet_w18", "hrnet_w32", "hrnet_w48"]
)
def test_backbone_plus_paf_config(
    multianimal: bool,
    individuals: list[str],
    bodyparts: list[str],
    identity: bool,
    unique_bodyparts: list[str],
    net_type: str,
):
    # Single animal projects can't have unique bodyparts
    project_config = _make_project_config(
        project_path="my/little/project",
        multianimal=multianimal,
        identity=identity,
        individuals=individuals,
        bodyparts=bodyparts,
        unique_bodyparts=unique_bodyparts,
    )
    pytorch_pose_config = make_pytorch_pose_config(
        project_config,
        "pytorch_config.yaml",
        net_type=net_type,
    )
    pretty_print(pytorch_pose_config)

    graph = [
        [i, j]
        for i in range(len(bodyparts))
        for j in range(i + 1, len(bodyparts))
    ]
    num_limbs = len(graph) * 2

    # check heads are there
    assert "bodypart" in pytorch_pose_config["model"]["heads"].keys()
    bodypart_head = pytorch_pose_config["model"]["heads"]["bodypart"]

    # check PAF head
    assert bodypart_head["type"] == "DLCRNetHead"
    assert bodypart_head["predictor"]["type"] == "PartAffinityFieldPredictor"

    for name, output_channels in [
        ("heatmap_config", len(bodyparts)),
        ("locref_config", len(bodyparts) * 2),
        ("paf_config", num_limbs)
    ]:
        print(name, bodypart_head[name]["channels"])
        assert name in bodypart_head
        assert bodypart_head[name]["channels"][-1] == output_channels

    if len(unique_bodyparts) > 0:
        assert "unique_bodypart" in pytorch_pose_config["model"]["heads"].keys()
        unique_bodypart_head = pytorch_pose_config["model"]["heads"]["unique_bodypart"]
        for name, output_channels in [
            ("heatmap_config", len(unique_bodyparts)),
            ("locref_config", 2 * len(unique_bodyparts)),
        ]:
            assert name in unique_bodypart_head
            assert unique_bodypart_head[name]["channels"][-1] == output_channels
        assert unique_bodypart_head["target_generator"]["heatmap_mode"] == "KEYPOINT"

    if identity:
        assert "identity" in pytorch_pose_config["model"]["heads"].keys()
        id_head = pytorch_pose_config["model"]["heads"]["identity"]
        assert "heatmap_config" in id_head
        assert id_head["heatmap_config"]["channels"][-1] == len(individuals)
        assert "locref_config" not in id_head
        assert id_head["target_generator"]["heatmap_mode"] == "INDIVIDUAL"


@pytest.mark.parametrize("individuals", [["single"], ["bugs", "daffy"]])
@pytest.mark.parametrize("bodyparts", [["nose"], ["nose", "ear", "eye"]])
@pytest.mark.parametrize(
    "net_type", ["resnet_50", "resnet_101", "hrnet_w18", "hrnet_w32", "hrnet_w48"]
)
def test_top_down_config(
    individuals: list[str],
    bodyparts: list[str],
    net_type: str,
):
    # Single animal projects can't have unique bodyparts
    project_config = _make_project_config(
        project_path="my/little/project",
        multianimal=True,
        identity=False,
        individuals=individuals,
        bodyparts=bodyparts,
        unique_bodyparts=[],
    )
    pytorch_pose_config = make_pytorch_pose_config(
        project_config,
        "pytorch_config.yaml",
        net_type=net_type,
        top_down=True,
    )
    pretty_print(pytorch_pose_config)

    # check no collate function
    collate = pytorch_pose_config["data"]["train"].get("collate")
    print(f"Collate: {collate}")
    assert not collate

    # check heads are there
    assert "bodypart" in pytorch_pose_config["model"]["heads"].keys()
    bodypart_head = pytorch_pose_config["model"]["heads"]["bodypart"]

    for name, output_channels in [
        ("heatmap_config", len(bodyparts)),
    ]:
        print(name, bodypart_head[name]["channels"])
        assert name in bodypart_head
        assert bodypart_head[name]["final_conv"]["out_channels"] == output_channels


@pytest.mark.parametrize("multianimal", [True])
@pytest.mark.parametrize("individuals", [["single"], ["bugs", "daffy"]])
@pytest.mark.parametrize("bodyparts", [["nose"], ["nose", "ear", "eye"]])
@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("unique_bodyparts", [[], ["tail"]])
@pytest.mark.parametrize("net_type", ["dekr_w18", "dekr_w32", "dekr_w48"])
def test_make_dekr_config(
    multianimal: bool,
    individuals: list[str],
    bodyparts: list[str],
    identity: bool,
    unique_bodyparts: list[str],
    net_type: str,
):
    project_config = _make_project_config(
        project_path="my/little/project",
        multianimal=multianimal,
        identity=identity,
        individuals=individuals,
        bodyparts=bodyparts,
        unique_bodyparts=unique_bodyparts
    )
    pytorch_pose_config = make_pytorch_pose_config(
        project_config,
        "pytorch_config.yaml",
        net_type=net_type,
    )
    pretty_print(pytorch_pose_config)

    # check heads are there
    assert "bodypart" in pytorch_pose_config["model"]["heads"].keys()
    bodypart_head = pytorch_pose_config["model"]["heads"]["bodypart"]
    for name, output_channels in [
        ("heatmap_config", len(bodyparts) + 1),
        ("offset_config", len(bodyparts)),
    ]:
        print(name, bodypart_head[name]["channels"])
        assert name in bodypart_head
        assert bodypart_head[name]["channels"][-1] == output_channels

    if len(unique_bodyparts) > 0:
        assert "unique_bodypart" in pytorch_pose_config["model"]["heads"].keys()
        unique_bodypart_head = pytorch_pose_config["model"]["heads"]["unique_bodypart"]
        for name, output_channels in [
            ("heatmap_config", len(unique_bodyparts)),
            ("locref_config", 2 * len(unique_bodyparts)),
        ]:
            assert name in unique_bodypart_head
            assert unique_bodypart_head[name]["channels"][-1] == output_channels
        assert unique_bodypart_head["target_generator"]["heatmap_mode"] == "KEYPOINT"

    if identity:
        assert "identity" in pytorch_pose_config["model"]["heads"].keys()
        id_head = pytorch_pose_config["model"]["heads"]["identity"]
        assert "heatmap_config" in id_head
        assert id_head["heatmap_config"]["channels"][-1] == len(individuals)
        assert "locref_config" not in id_head
        assert id_head["target_generator"]["heatmap_mode"] == "INDIVIDUAL"


@pytest.mark.parametrize("multianimal", [True])
@pytest.mark.parametrize("individuals", [["single"], ["bugs", "daffy"]])
@pytest.mark.parametrize("bodyparts", [["nose", "ears"], ["nose", "ear", "eye"]])
@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("unique_bodyparts", [[], ["tail"]])
@pytest.mark.parametrize("net_type", ["dlcrnet_stride16_ms5", "dlcrnet_stride32_ms5"])
def test_make_dlcrnet_config(
    multianimal: bool,
    individuals: list[str],
    bodyparts: list[str],
    identity: bool,
    unique_bodyparts: list[str],
    net_type: str,
):
    project_config = _make_project_config(
        project_path="my/little/project",
        multianimal=multianimal,
        identity=identity,
        individuals=individuals,
        bodyparts=bodyparts,
        unique_bodyparts=unique_bodyparts
    )
    pytorch_pose_config = make_pytorch_pose_config(
        project_config,
        "pytorch_config.yaml",
        net_type=net_type,
    )
    pretty_print(pytorch_pose_config)
    paf_graph = [
        [i, j]
        for i in range(len(bodyparts))
        for j in range(i + 1, len(bodyparts))
    ]
    num_limbs = len(paf_graph)

    # check heads are there
    assert "bodypart" in pytorch_pose_config["model"]["heads"].keys()
    bodypart_head = pytorch_pose_config["model"]["heads"]["bodypart"]
    for name, output_channels in [
        ("heatmap_config", len(bodyparts)),
        ("locref_config", 2 * len(bodyparts)),
        ("paf_config", 2 * num_limbs),
    ]:
        print(name, bodypart_head[name]["channels"])
        assert name in bodypart_head
        assert bodypart_head[name]["channels"][-1] == output_channels

    if len(unique_bodyparts) > 0:
        assert "unique_bodypart" in pytorch_pose_config["model"]["heads"].keys()
        unique_bodypart_head = pytorch_pose_config["model"]["heads"]["unique_bodypart"]
        for name, output_channels in [
            ("heatmap_config", len(unique_bodyparts)),
            ("locref_config", 2 * len(unique_bodyparts)),
        ]:
            assert name in unique_bodypart_head
            assert unique_bodypart_head[name]["channels"][-1] == output_channels
        assert unique_bodypart_head["target_generator"]["heatmap_mode"] == "KEYPOINT"

    if identity:
        assert "identity" in pytorch_pose_config["model"]["heads"].keys()
        id_head = pytorch_pose_config["model"]["heads"]["identity"]
        assert "heatmap_config" in id_head
        assert id_head["heatmap_config"]["channels"][-1] == len(individuals)
        assert "locref_config" not in id_head
        assert id_head["target_generator"]["heatmap_mode"] == "INDIVIDUAL"


@pytest.mark.parametrize("individuals", [["single"], ["bugs", "daffy"]])
@pytest.mark.parametrize("bodyparts", [["nose", "eyes"], ["nose", "ear", "eye"]])
@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("unique_bodyparts", [[], ["tail"]])
@pytest.mark.parametrize("net_type", ["animaltokenpose_base"])
def test_make_tokenpose_config(
    individuals: list[str],
    bodyparts: list[str],
    identity: bool,
    unique_bodyparts: list[str],
    net_type: str,
):
    project_config = _make_project_config(
        project_path="my/little/project",
        multianimal=True,
        identity=identity,
        individuals=individuals,
        bodyparts=bodyparts,
        unique_bodyparts=unique_bodyparts
    )

    if identity or len(unique_bodyparts) > 0:
        with pytest.raises(ValueError) as err_info:
            # Not yet implemented!
            _ = make_pytorch_pose_config(
                project_config,
                "pytorch_config.yaml",
                net_type=net_type,
            )
    else:
        pytorch_pose_config = make_pytorch_pose_config(
            project_config,
            "pytorch_config.yaml",
            net_type=net_type,
        )
        pretty_print(pytorch_pose_config)

        # check no collate function
        collate = pytorch_pose_config["data"]["train"].get("collate")
        print(f"Collate: {collate}")
        assert not collate

        # check detector is there
        assert "detector" in pytorch_pose_config
        assert "data" in pytorch_pose_config["detector"]


@pytest.mark.parametrize("data", [
    {
        "config": {"a": 0, "b": 0},
        "updates": {"b": 1},
        "expected_result": {"a": 0, "b": 1},
    },
    {
        "config": {"a": 0, "b": {"i0": 1, "i1": 2}},
        "updates": {"b": 1},
        "expected_result": {"a": 0, "b": 1},
    },
    {
        "config": {"a": 0, "b": {"i0": 1, "i1": 2}},
        "updates": {"b": {"i0": [1, 2, 3]}},
        "expected_result": {"a": 0, "b": {"i0": [1, 2, 3], "i1": 2}},
    },
    {
        "config": {"detector": {"batch_size": 1, "epochs": 10, "save_epochs": 5}},
        "updates": {"batch_size": 1, "detector": {"batch_size": 8, "save_epochs": 1}},
        "expected_result": {"batch_size": 1, "detector": {"batch_size": 8, "epochs": 10, "save_epochs": 1}},
    },
])
def test_update_config(data: dict):
    result = update_config(config=data["config"], updates=data["updates"])
    print("\nResult")
    pretty_print(result)
    assert result == data["expected_result"]


def _make_project_config(
    project_path: str,
    multianimal: bool,
    identity: bool,
    individuals: list[str],
    bodyparts: list[str],
    unique_bodyparts: list[str],
) -> dict:
    project_config = {
        "project_path": project_path,
        "multianimalproject": multianimal,
        "identity": identity,
        "uniquebodyparts": unique_bodyparts,
    }

    if multianimal:
        project_config["multianimalbodyparts"] = bodyparts
        project_config["bodyparts"] = "MULTI!"
        project_config["individuals"] = individuals
    else:
        project_config["bodyparts"] = bodyparts

    return project_config
