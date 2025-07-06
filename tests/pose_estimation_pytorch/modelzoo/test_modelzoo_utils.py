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

import pytest

import deeplabcut.pose_estimation_pytorch.modelzoo as modelzoo

# TODO: make a proper test incl. human model, bird model and that skips the require... at least once per week.

@pytest.mark.parametrize(
    "super_animal", ["superanimal_quadruped", "superanimal_topviewmouse"]
)
@pytest.mark.parametrize("model_name", ["hrnet_w32"])
@pytest.mark.parametrize("detector_name", [None, "fasterrcnn_resnet50_fpn_v2"])
def test_get_config_model_paths(super_animal, model_name, detector_name):
    model_config = modelzoo.load_super_animal_config(
        super_animal=super_animal,
        model_name=model_name,
        detector_name=detector_name,
    )

    assert isinstance(model_config, dict)
    if detector_name is None:
        assert model_config["method"].lower() == "bu"
        assert "detector" not in model_config
    else:
        assert model_config["method"].lower() == "td"
        assert "detector" in model_config
