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
import pickle
import urllib.request
import zipfile
from enum import Enum, unique
from io import BytesIO
from typing import NamedTuple

import numpy as np
import pytest
from packaging.version import Version
from PIL import Image
from tqdm import tqdm

# Enable pandas future mode warnings if DLC_PANDAS_FUTURE env var is set
from deeplabcut.utils.pandas_future_mode import configure_pandas_future_if_enabled

configure_pandas_future_if_enabled()

from deeplabcut.core import inferenceutils  # noqa: E402
from deeplabcut.version import __version__ as _dlc_version  # noqa: E402

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_DATA_DIR = os.path.join(TESTS_DIR, "data")

REQUIRED_TEST_FILES = [
    os.path.join(TEST_DATA_DIR, "dets.pickle"),
    os.path.join(TEST_DATA_DIR, "outputs.pickle"),
    os.path.join(TEST_DATA_DIR, "image.png"),
    os.path.join(TEST_DATA_DIR, "trimouse_assemblies.pickle"),
    os.path.join(TEST_DATA_DIR, "montblanc_tracks.h5"),
    os.path.join(TEST_DATA_DIR, "trimouse_calib.h5"),
]


def unzip_from_url(url: str, dest_folder: str) -> None:
    """Directly extract files without writing the archive to disk."""
    os.makedirs(dest_folder, exist_ok=True)
    resp = urllib.request.urlopen(url)
    with zipfile.ZipFile(BytesIO(resp.read())) as zf:
        for member in tqdm(zf.infolist(), desc="Extracting"):
            try:
                zf.extract(member, path=dest_folder)
            except zipfile.error:
                pass


def _test_data_ready() -> bool:
    return all(os.path.exists(path) for path in REQUIRED_TEST_FILES)


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """Ensure shared test data exists once per pytest session.

    This is autouse so tests that directly open files under tests/data/
    keep working without being rewritten.
    """
    if not _test_data_ready():
        unzip_from_url(
            "https://github.com/DeepLabCut/UnitTestData/raw/main/data.zip",
            TESTS_DIR,
        )
    yield


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to shared test data under tests/data/."""
    return TEST_DATA_DIR


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


# -----------------------------------------------------------------------------
# Known defects
#
# Mark a test that captures a known defect with
# ``@pytest.mark.known_defect("NAME")``.
#
# Through the defect's ``affects_through`` version, the test is treated as a
# strict xfail. Only an AssertionError is considered an expected failure;
# setup, import, and other errors still fail the test.
#
# After ``affects_through``, the xfail is disabled and the test result is
# determined by its assertions. Each marked test also emits a PytestWarning
# indicating that the marker should be removed or assigned a later version.
#
# If the defect is fixed before or during the affected version range, the test
# produces XPASS(strict). Remove the registry entry and all markers that
# reference it once the defect is fixed.
# -----------------------------------------------------------------------------


class KnownDefectInfo(NamedTuple):
    affects_through: Version  # last DeepLabCut version known to carry the defect
    reason: str


@unique
class KnownDefect(Enum):
    """Known defects pinned by strict-xfail tests, one entry per topic."""


_DLC_VERSION = Version(_dlc_version)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "known_defect(name): strict xfail for the KnownDefect entry `name`, see tests/conftest.py",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        for marker in item.iter_markers("known_defect"):
            (name,) = marker.args
            try:
                defect = KnownDefect[name].value
            except KeyError:
                raise pytest.UsageError(f"{item.nodeid}: unknown KnownDefect {name!r}") from None
            if _DLC_VERSION > defect.affects_through:
                item.warn(
                    pytest.PytestWarning(
                        f"KnownDefect.{name} is past {defect.affects_through}, so known_defect no longer applies. "
                        "If this test passes, delete the marker and the entry; "
                        "if it fails, fix the defect or move affects_through forward."
                    )
                )
            item.add_marker(
                pytest.mark.xfail(
                    _DLC_VERSION <= defect.affects_through,
                    reason=f"{name} (known through {defect.affects_through}): {defect.reason}",
                    raises=AssertionError,
                    strict=True,
                )
            )
