import os
os.environ['DLClight'] = 'True'
import pytest
from deeplabcut.create_project import create_new_project
from deeplabcut.utils import auxiliaryfunctions


TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
videos = [os.path.join(TEST_DATA_DIR, 'vid1.mov'),
          os.path.join(TEST_DATA_DIR, 'vid2.mov')]
SINGLE_CONFIG_PATH = os.path.join(TEST_DATA_DIR, 'single-dlc-2020-04-17/config.yaml')
MULTI_CONFIG_PATH = os.path.join(TEST_DATA_DIR, 'multi-dlc-2020-04-17/config.yaml')


@pytest.fixture()
def cfg_single():
    return auxiliaryfunctions.read_config(SINGLE_CONFIG_PATH)


@pytest.fixture()
def cfg_multi():
    return auxiliaryfunctions.read_config(MULTI_CONFIG_PATH)


@pytest.fixture()
def project_single(tmpdir):
    project = 'single'
    experimenter = 'dlc'
    config_path = create_new_project(project, experimenter, [videos[0]], tmpdir)
    return config_path, tmpdir


@pytest.fixture()
def project_multi(tmpdir):
    project = 'multi'
    experimenter = 'dlc'
    config_path = create_new_project(project, experimenter, [videos[0]], tmpdir, multianimal=True)
    return config_path, tmpdir


@pytest.fixture()
def fake_project(request):
    return request.getfixturevalue(request.param)
