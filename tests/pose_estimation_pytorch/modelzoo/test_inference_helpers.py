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

import copy
from types import SimpleNamespace

import pytest

import deeplabcut.pose_estimation_pytorch.modelzoo.inference_helpers as helpers


def _dummy_cfg(method: str = "TD") -> dict:
    return {
        "method": method,
        "metadata": {"bodyparts": ["nose"], "unique_bodyparts": []},
    }


def test_create_superanimal_inference_runners_uses_custom_config_path(monkeypatch):
    cfg = _dummy_cfg("TD")
    read_calls = []

    def fake_from_any(config):
        read_calls.append(config)
        return cfg

    monkeypatch.setattr(helpers.PoseConfig, "from_any", fake_from_any)
    monkeypatch.setattr(
        helpers,
        "get_inference_runners",
        lambda **kwargs: ("pose_runner", "det_runner"),
    )

    import deeplabcut.modelzoo.weight_initialization as wi

    monkeypatch.setattr(
        wi,
        "build_weight_init",
        lambda **kwargs: SimpleNamespace(
            snapshot_path="pose.pt",
            detector_snapshot_path="det.pt",
        ),
    )

    pose_runner, detector_runner, model_cfg = helpers.create_superanimal_inference_runners(
        superanimal_name="superanimal_quadruped",
        model_name="hrnet_w32",
        detector_name="fasterrcnn_resnet50_fpn_v2",
        customized_model_config="/tmp/custom_model_cfg.yaml",
    )

    assert read_calls == ["/tmp/custom_model_cfg.yaml"]
    assert pose_runner == "pose_runner"
    assert detector_runner == "det_runner"
    assert model_cfg is cfg


def test_create_superanimal_inference_runners_does_not_mutate_custom_dict(monkeypatch):
    custom_cfg = _dummy_cfg("TD")
    original_bodyparts = list(custom_cfg["metadata"]["bodyparts"])

    monkeypatch.setattr(
        helpers.PoseConfig,
        "from_any",
        lambda config: copy.deepcopy(config),
    )
    monkeypatch.setattr(
        helpers,
        "get_inference_runners",
        lambda **kwargs: ("pose_runner", None),
    )

    import deeplabcut.modelzoo.weight_initialization as wi

    monkeypatch.setattr(
        wi,
        "build_weight_init",
        lambda **kwargs: SimpleNamespace(
            snapshot_path="pose.pt",
            detector_snapshot_path=None,
        ),
    )

    _, _, model_cfg = helpers.create_superanimal_inference_runners(
        superanimal_name="superanimal_quadruped",
        model_name="hrnet_w32",
        detector_name=None,
        customized_model_config=custom_cfg,
    )

    assert custom_cfg["metadata"]["bodyparts"] == original_bodyparts
    assert model_cfg is not custom_cfg


@pytest.mark.parametrize("input_device", ["auto", None])
def test_create_superanimal_inference_runners_auto_device_selection(monkeypatch, input_device):
    captured = {}

    def fake_build_for_superanimal_inference(
        cls,
        super_animal,
        *,
        model_name,
        detector_name=None,
        max_individuals=30,
        device=None,
    ):
        captured["device"] = device
        return _dummy_cfg("TD")

    monkeypatch.setattr(
        helpers.PoseConfig,
        "build_for_superanimal_inference",
        classmethod(fake_build_for_superanimal_inference),
    )
    monkeypatch.setattr(
        helpers,
        "get_inference_runners",
        lambda **kwargs: ("pose_runner", "det_runner"),
    )

    import deeplabcut.modelzoo.weight_initialization as wi

    monkeypatch.setattr(
        wi,
        "build_weight_init",
        lambda **kwargs: SimpleNamespace(
            snapshot_path="pose.pt",
            detector_snapshot_path="det.pt",
        ),
    )

    helpers.create_superanimal_inference_runners(
        superanimal_name="superanimal_quadruped",
        model_name="hrnet_w32",
        detector_name="fasterrcnn_resnet50_fpn_v2",
        customized_model_config=None,
        device=input_device,
    )
    assert captured["device"] == "auto"


def test_create_superanimal_inference_runners_raises_for_fmpose3d():
    with pytest.raises(NotImplementedError, match="FMPose3D"):
        helpers.create_superanimal_inference_runners(
            superanimal_name="superanimal_quadruped",
            model_name="FMPose3D_resnet",
            detector_name="fasterrcnn_resnet50_fpn_v2",
            customized_model_config=_dummy_cfg("TD"),
        )


def test_create_superanimal_inference_runners_propagates_unsupported_dataset_error(
    monkeypatch,
):
    def fake_build_for_superanimal_inference(cls, *args, **kwargs):
        raise ValueError("Unsupported dataset for model zoo config")

    monkeypatch.setattr(
        helpers.PoseConfig,
        "build_for_superanimal_inference",
        classmethod(fake_build_for_superanimal_inference),
    )

    with pytest.raises(ValueError, match="Unsupported dataset"):
        helpers.create_superanimal_inference_runners(
            superanimal_name="superanimal_unknown",
            model_name="hrnet_w32",
            detector_name="fasterrcnn_resnet50_fpn_v2",
            customized_model_config=None,
        )
