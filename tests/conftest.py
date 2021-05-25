import numpy as np
import os
import pickle
import pytest
from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils, crossvalutils


TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


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
