import os
os.environ['DLClight'] = 'True'
import pytest
from deeplabcut import create_new_project
from deeplabcut.utils import auxiliaryfunctions


TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
videos = [os.path.join(TEST_DATA_DIR, 'vid1.mov'),
          os.path.join(TEST_DATA_DIR, 'vid2.mov')]


@pytest.fixture()
def empty_project_single(tmpdir):
    project = 'single'
    experimenter = 'dlc'
    config_path = create_new_project(project, experimenter, [], tmpdir)
    return config_path, tmpdir


@pytest.fixture()
def empty_project_multi(tmpdir):
    project = 'multi'
    experimenter = 'dlc'
    config_path = create_new_project(project, experimenter, [], tmpdir, multianimal=True)
    return config_path, tmpdir


@pytest.fixture()
def empty_projects(request):
    return request.getfixturevalue(request.param)


# Hacky solution to pass fixtures to 'parametrize'
# See https://github.com/pytest-dev/pytest/issues/349
@pytest.mark.parametrize('empty_projects', ['empty_project_single', 'empty_project_multi'],
                         indirect=['empty_projects'])
def test_new_project_no_video(empty_projects):
    config_path, _ = empty_projects
    assert config_path == 'nothingcreated'
    project_path = os.path.dirname(config_path)
    assert not os.path.isdir(project_path)


@pytest.mark.parametrize('videos, copy_videos, multi',
                         [([TEST_DATA_DIR], False, False),
                          (videos, True, True)])
def test_new_project(tmpdir, videos, copy_videos, multi):
    project = 'single'
    experimenter = 'dlc'
    config_path = create_new_project(project, experimenter, videos, working_directory=tmpdir,
                                     copy_videos=copy_videos, videotype='.mov', multianimal=multi)
    assert os.path.isfile(config_path)
    project_path = os.path.dirname(config_path)
    assert os.path.isdir(os.path.join(project_path, 'videos'))
    assert os.path.isdir(os.path.join(project_path, 'labeled-data'))
    assert os.path.isdir(os.path.join(project_path, 'training-datasets'))
    assert os.path.isdir(os.path.join(project_path, 'dlc-models'))
    cfg = auxiliaryfunctions.read_config(config_path)
    assert len(cfg['video_sets']) == 2
    config_folder = os.path.split(config_path)[0]
    video_path = os.path.join(config_folder, 'videos', 'vid1.mov')
    if copy_videos:
        assert os.path.isfile(video_path)
    else:
        assert os.path.islink(video_path)
