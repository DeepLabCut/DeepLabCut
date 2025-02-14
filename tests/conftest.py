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
import numpy as np
import os
import pickle
import pytest
import shutil
import urllib.request
import zipfile
from deeplabcut.core import inferenceutils
from io import BytesIO
from PIL import Image
from tqdm import tqdm


TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


def unzip_from_url(url, dest_folder):
    """Directly extract files without writing the archive to disk."""
    os.makedirs(dest_folder, exist_ok=True)
    resp = urllib.request.urlopen(url)
    with zipfile.ZipFile(BytesIO(resp.read())) as zf:
        for member in tqdm(zf.infolist(), desc="Extracting"):
            try:
                zf.extract(member, path=dest_folder)
            except zipfile.error:
                pass


def pytest_sessionstart(session):
    unzip_from_url(
        "https://github.com/DeepLabCut/UnitTestData/raw/main/data.zip",
        os.path.split(TEST_DATA_DIR)[0],
    )
    session.__DATA_FOLDER = TEST_DATA_DIR


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(session.__DATA_FOLDER)


@pytest.fixture(scope="function")
def ground_truth_detections():
    with open(os.path.join(TEST_DATA_DIR, "dets.pickle"), "rb") as file:
        return pickle.load(file)


@pytest.fixture(scope="function")
def model_outputs():
    with open(os.path.join(TEST_DATA_DIR, "outputs.pickle"), "rb") as file:
        scmaps, locrefs, pafs = pickle.load(file)
    locrefs = np.reshape(locrefs, (*locrefs.shape[:3], -1, 2))
    locrefs *= 7.2801
    pafs = np.reshape(pafs, (*pafs.shape[:3], -1, 2))
    return scmaps, locrefs, pafs


@pytest.fixture(scope="function")
def sample_image():
    return np.asarray(Image.open(os.path.join(TEST_DATA_DIR, "image.png")))


@pytest.fixture(scope="function")
def sample_keypoints():
    with open(os.path.join(TEST_DATA_DIR, "trimouse_assemblies.pickle"), "rb") as file:
        temp = pickle.load(file)
    return np.concatenate(temp[0])[:, :2]


@pytest.fixture(scope="function")
def real_assemblies():
    with open(os.path.join(TEST_DATA_DIR, "trimouse_assemblies.pickle"), "rb") as file:
        temp = pickle.load(file)
    data = np.stack(list(temp.values()))
    return inferenceutils._parse_ground_truth_data(data)


@pytest.fixture(scope="function")
def real_assemblies_montblanc():
    with open(os.path.join(TEST_DATA_DIR, "montblanc_assemblies.pickle"), "rb") as file:
        temp = pickle.load(file)
    single = temp.pop("single")
    data = np.full((max(temp) + 1, 3, 4, 4), np.nan)
    for k, assemblies in temp.items():
        for i, assembly in enumerate(assemblies):
            data[k, i] = assembly
    return inferenceutils._parse_ground_truth_data(data), single


@pytest.fixture(scope="function")
def real_tracklets():
    with open(os.path.join(TEST_DATA_DIR, "trimouse_tracklets.pickle"), "rb") as file:
        return pickle.load(file)


@pytest.fixture(scope="function")
def real_tracklets_montblanc():
    with open(os.path.join(TEST_DATA_DIR, "montblanc_tracklets.pickle"), "rb") as file:
        return pickle.load(file)


@pytest.fixture(scope="function")
def evaluation_data_and_metadata():
    full_data_file = os.path.join(TEST_DATA_DIR, "trimouse_eval.pickle")
    metadata_file = full_data_file.replace("eval", "meta")
    with open(full_data_file, "rb") as file:
        data = pickle.load(file)
    with open(metadata_file, "rb") as file:
        metadata = pickle.load(file)
    return data, metadata


@pytest.fixture(scope="function")
def evaluation_data_and_metadata_montblanc():
    full_data_file = os.path.join(TEST_DATA_DIR, "montblanc_eval.pickle")
    metadata_file = full_data_file.replace("eval", "meta")
    with open(full_data_file, "rb") as file:
        data = pickle.load(file)
    with open(metadata_file, "rb") as file:
        metadata = pickle.load(file)
    return data, metadata
