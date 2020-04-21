import os
os.environ['DLClight'] = 'True'
import pytest
from deeplabcut.create_project import create_new_project
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config


TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
videos = [os.path.join(TEST_DATA_DIR, 'vid1.mov'),
          os.path.join(TEST_DATA_DIR, 'vid2.mov')]
SINGLE_CONFIG_PATH = os.path.join(TEST_DATA_DIR, 'single-dlc-2020-04-17/config.yaml')
MULTI_CONFIG_PATH = os.path.join(TEST_DATA_DIR, 'multi-dlc-2020-04-17/config.yaml')
SCORER = 'dlc'
NUM_FRAMES = 5
TRAIN_SIZE = 0.8


@pytest.fixture()
def cfg_single():
    return read_config(SINGLE_CONFIG_PATH)


@pytest.fixture()
def cfg_multi():
    return read_config(MULTI_CONFIG_PATH)


@pytest.fixture(scope='session')
def project_single(tmp_path_factory):
    config_path = create_new_project('single', SCORER, [videos[0]],
                                     str(tmp_path_factory.getbasetemp()))
    _ = edit_config(config_path, {'numframes2pick': NUM_FRAMES,
                                  'TrainingFraction': [TRAIN_SIZE]})
    return config_path, tmp_path_factory


@pytest.fixture(scope='session')
def project_multi(tmp_path_factory):
    config_path = create_new_project('multi', SCORER, [videos[0]],
                                     str(tmp_path_factory.getbasetemp()), multianimal=True)
    _ = edit_config(config_path, {'numframes2pick': NUM_FRAMES,
                                  'TrainingFraction': [TRAIN_SIZE]})
    return config_path, tmp_path_factory


@pytest.fixture()
def fake_project(request):
    return request.getfixturevalue(request.param)
