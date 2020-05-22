import os

os.environ["DLClight"] = "True"
import pytest
from deeplabcut import add_new_videos, create_new_project
from deeplabcut.utils import auxiliaryfunctions
from tests import conftest


@pytest.fixture()
def empty_project_single(tmp_path):
    project = "single"
    experimenter = "dlc"
    config_path = create_new_project(project, experimenter, [], str(tmp_path))
    return config_path, tmp_path


@pytest.fixture()
def empty_project_multi(tmp_path):
    project = "multi"
    experimenter = "dlc"
    config_path = create_new_project(
        project, experimenter, [], str(tmp_path), multianimal=True
    )
    return config_path, tmp_path


# Hacky solution to pass fixtures to 'parametrize'
# See https://github.com/pytest-dev/pytest/issues/349
@pytest.mark.parametrize(
    "fake_project",
    ["empty_project_single", "empty_project_multi"],
    indirect=["fake_project"],
)
def test_new_project_no_video(fake_project):
    config_path, _ = fake_project
    assert config_path == "nothingcreated"
    project_path = os.path.dirname(config_path)
    assert not os.path.isdir(project_path)


@pytest.mark.parametrize(
    "videos, copy_videos, multi",
    [([conftest.TEST_DATA_DIR], False, False), (conftest.videos, True, True)],
)
def test_new_project(tmpdir, videos, copy_videos, multi):
    project = "single"
    experimenter = "dlc"
    config_path = create_new_project(
        project,
        experimenter,
        videos,
        working_directory=tmpdir,
        copy_videos=copy_videos,
        videotype=".mov",
        multianimal=multi,
    )
    assert os.path.isfile(config_path)
    project_path = os.path.dirname(config_path)
    assert os.path.isdir(os.path.join(project_path, "videos"))
    assert os.path.isdir(os.path.join(project_path, "labeled-data"))
    assert os.path.isdir(os.path.join(project_path, "training-datasets"))
    assert os.path.isdir(os.path.join(project_path, "dlc-models"))
    cfg = auxiliaryfunctions.read_config(config_path)
    assert len(cfg["video_sets"]) == 2
    config_folder = os.path.split(config_path)[0]
    video_path = os.path.join(config_folder, "videos", "vid1.mov")
    if copy_videos:
        assert os.path.isfile(video_path)
    else:
        assert os.path.islink(video_path)


@pytest.mark.parametrize(
    "fake_project, videos, copy_videos",
    [
        ("project_single", [conftest.videos[0]], False),
        ("project_multi", [conftest.videos[0]], True),
    ],
    indirect=["fake_project"],
)
def test_add_videos(fake_project, videos, copy_videos):
    config_path, _ = fake_project
    add_new_videos(config_path, videos, copy_videos)
    cfg = auxiliaryfunctions.read_config(config_path)
    assert len(cfg["video_sets"]) == 2
    config_folder = os.path.dirname(config_path)
    video_path = os.path.join(config_folder, "videos", os.path.split(videos[0])[1])
    if copy_videos:
        assert os.path.isfile(video_path)
    else:
        assert os.path.islink(video_path)


@pytest.mark.parametrize(
    "fake_project", ["project_single", "project_multi"], indirect=["fake_project"]
)
def test_add_videos_corrupted(capsys, fake_project):
    config_path, folder = fake_project
    empty_dir = folder.mktemp("empty")
    fake_vid = empty_dir / "fake.avi"
    add_new_videos(config_path, [fake_vid])
    out, _ = capsys.readouterr()
    assert "Cannot open" in out
