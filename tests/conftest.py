import numpy as np
import os
import pickle
import pytest
from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils, crossvalutils
from PIL import Image


TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


@pytest.fixture(scope="session")
def ground_truth_detections():
    with open(os.path.join(TEST_DATA_DIR, "dets.pickle"), "rb") as file:
        return pickle.load(file)


@pytest.fixture(scope="session")
def model_outputs():
    with open(os.path.join(TEST_DATA_DIR, "outputs.pickle"), "rb") as file:
        scmaps, locrefs, pafs = pickle.load(file)
    locrefs = np.reshape(locrefs, (*locrefs.shape[:3], -1, 2))
    locrefs *= 7.2801
    pafs = np.reshape(pafs, (*pafs.shape[:3], -1, 2))
    return scmaps, locrefs, pafs


@pytest.fixture(scope="session")
def sample_image():
    return np.asarray(Image.open(os.path.join(TEST_DATA_DIR, "image.png")))


@pytest.fixture(scope="session")
def sample_keypoints():
    with open(os.path.join(TEST_DATA_DIR, "trimouse_assemblies.pickle"), "rb") as file:
        temp = pickle.load(file)
    return np.concatenate(temp[0])[:, :2]


@pytest.fixture(scope="session")
def real_assemblies():
    with open(os.path.join(TEST_DATA_DIR, "trimouse_assemblies.pickle"), "rb") as file:
        temp = pickle.load(file)
    data = np.stack(list(temp.values()))
    return inferenceutils._parse_ground_truth_data(data)


@pytest.fixture(scope="session")
def real_tracklets():
    with open(os.path.join(TEST_DATA_DIR, "trimouse_tracklets.pickle"), "rb") as file:
        return pickle.load(file)


@pytest.fixture(scope="session")
def evaluation_data_and_metadata():
    full_data_file = os.path.join(TEST_DATA_DIR, "trimouse_eval.pickle")
    metadata_file = full_data_file.replace("eval", "meta")
    with open(full_data_file, "rb") as file:
        data = pickle.load(file)
    with open(metadata_file, "rb") as file:
        metadata = pickle.load(file)
    return data, metadata
