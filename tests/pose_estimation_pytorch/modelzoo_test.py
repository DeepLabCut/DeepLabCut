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
import os

import dlclibrary
import pytest
from dlclibrary.dlcmodelzoo.modelzoo_download import MODELOPTIONS

from deeplabcut.utils import auxiliaryfunctions


def test_download_huggingface_model(tmp_path_factory, model="full_cat"):
    folder = tmp_path_factory.mktemp("temp")
    dlclibrary.download_huggingface_model(model, str(folder))

    assert os.path.exists(folder / "pose_cfg.yaml")
    assert any(f.startswith("snapshot-") for f in os.listdir(folder))
    # Verify that the Hugging Face folder was removed
    assert not any(f.startswith("models--") for f in os.listdir(folder))


def test_download_huggingface_wrong_model():
    with pytest.raises(ValueError):
        dlclibrary.download_huggingface_model("wrong_model_name")


@pytest.mark.skip
@pytest.mark.parametrize("model", MODELOPTIONS)
def test_download_all_models(tmp_path_factory, model):
    test_download_huggingface_model(tmp_path_factory, model)


examples_folder = os.path.join(
    auxiliaryfunctions.get_deeplabcut_path(),
    "examples",
    "openfield-Pranav-2018-10-30",
    "labeled-data",
    "m4s1",
)


@pytest.mark.parametrize(
    "image_path",
    [
        f"{examples_folder}/img0001.png",
        f"{examples_folder}/img0004.png",
        f"{examples_folder}/img0009.png",
    ],
)
@pytest.mark.parametrize("max_individuals", [1, 3])
@pytest.mark.parametrize(
    "project_name", ["superanimal_quadruped", "superanimal_topview"]
)
def test_webapp_init(project_name, max_individuals):
    inference_pipeline = SuperanimalPyTorchInference(
        project_name, pose_model_type, max_individuals=max_individuals
    )
