import os

os.environ["DLClight"] = "True"
import pytest
from deeplabcut.pose_estimation_tensorflow import analyze_videos
from deeplabcut.utils.auxiliaryfunctions import grab_files_in_folder
from tests import conftest


@pytest.mark.parametrize(
    "config_path, dynamic",
    [
        (conftest.SINGLE_CONFIG_PATH, (True, 0.1, 5)),
        (conftest.MULTI_CONFIG_PATH, (False, 0.1, 5)),
    ],
)
def test_analyze_videos(tmp_path, config_path, dynamic):
    analyze_videos(
        config_path, [conftest.videos[1]], "mov", dynamic=dynamic, destfolder=tmp_path
    )
    assert len(list(grab_files_in_folder(tmp_path))) == 2
