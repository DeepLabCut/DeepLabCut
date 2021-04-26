import numpy as np
import os
import pickle
import pytest
from deeplabcut.create_project import create_new_project
from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils, crossvalutils
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config


TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
videos = [
    os.path.join(TEST_DATA_DIR, "vid1.mov"),
    os.path.join(TEST_DATA_DIR, "vid2.mov"),
]
SINGLE_CONFIG_PATH = os.path.join(TEST_DATA_DIR, "single-dlc-2020-04-21/config.yaml")
MULTI_CONFIG_PATH = os.path.join(TEST_DATA_DIR, "multi-dlc-2020-04-21/config.yaml")
SCORER = "dlc"
NUM_FRAMES = 5
TRAIN_SIZE = 0.8


@pytest.fixture(scope="session")
def real_assemblies():
    with open(os.path.join(TEST_DATA_DIR, "trimouse_assemblies.pickle"), "rb") as file:
        temp = pickle.load(file)
    data = np.stack(list(temp.values()))
    return inferenceutils._parse_ground_truth_data(data[..., :3])


@pytest.fixture(scope="session")
def real_tracklets():
    with open(os.path.join(TEST_DATA_DIR, "trimouse_tracklets.pickle"), "rb") as file:
        return pickle.load(file)


@pytest.fixture(scope="session")
def uncropped_data_and_metadata():
    full_data_file = os.path.join(TEST_DATA_DIR, "trimouse_eval.pickle")
    metadata_file = full_data_file.replace("eval", "meta")
    with open(full_data_file, "rb") as file:
        data = pickle.load(file)
    with open(metadata_file, "rb") as file:
        metadata = pickle.load(file)
    params = crossvalutils._set_up_evaluation(data)
    data_unc, _ = crossvalutils._rebuild_uncropped_data(data, params)
    meta_unc = crossvalutils._rebuild_uncropped_metadata(metadata, params["imnames"])
    return data_unc, meta_unc


@pytest.fixture()
def cfg_single():
    return read_config(SINGLE_CONFIG_PATH)


@pytest.fixture()
def cfg_multi():
    return read_config(MULTI_CONFIG_PATH)


@pytest.fixture(scope="session")
def project_single(tmp_path_factory):
    config_path = create_new_project(
        "single", SCORER, [videos[1]], str(tmp_path_factory.getbasetemp())
    )
    _ = edit_config(
        config_path, {"numframes2pick": NUM_FRAMES, "TrainingFraction": [TRAIN_SIZE]}
    )
    return config_path, tmp_path_factory


@pytest.fixture(scope="session")
def project_multi(tmp_path_factory):
    config_path = create_new_project(
        "multi",
        SCORER,
        [videos[1]],
        str(tmp_path_factory.getbasetemp()),
        multianimal=True,
    )
    _ = edit_config(
        config_path, {"numframes2pick": NUM_FRAMES, "TrainingFraction": [TRAIN_SIZE]}
    )
    return config_path, tmp_path_factory


@pytest.fixture()
def fake_project(request):
    return request.getfixturevalue(request.param)
