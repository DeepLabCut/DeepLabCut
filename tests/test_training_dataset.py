import numpy as np
import os
os.environ['DLClight'] = 'True'
import pandas as pd
import pytest
from deeplabcut import create_multianimaltraining_dataset
from deeplabcut.generate_training_dataset import trainingsetmanipulation, frame_extraction, check_labels
from deeplabcut.utils.auxiliaryfunctions import read_config, grab_files_in_folder, get_labeled_data_folder
from deeplabcut.utils import auxfun_models
from deeplabcut.utils.auxfun_multianimal import extractindividualsandbodyparts
from tests import conftest
from skimage import io


@pytest.mark.parametrize('fake_project, algo, crop',
                         [('project_single', 'kmeans', False),
                          ('project_multi', 'uniform', True)],
                         indirect=['fake_project'])
def test_extract_frames(fake_project, algo, crop):
    config_path, tmpdir = fake_project
    frame_extraction.extract_frames(config_path, algo=algo, crop=crop, userfeedback=False)
    cfg = read_config(config_path)
    video = list(cfg['video_sets'])[0]
    image_folder = get_labeled_data_folder(cfg, video)
    assert len(list(grab_files_in_folder(image_folder, 'png'))) == cfg['numframes2pick']


@pytest.mark.parametrize('network, aug',
                         [('resnet_50', 'default'),
                          ('resnet_101', 'imgaug'),
                          ('mobilenet_v2_1.0', 'tensorpack'),
                          ('mobilenet_v2_0.35', 'deterministic')])
def test_create_training_dataset(mocker, project_single, network, aug):
    config_path, tmpdir = project_single

    # Add artificial data
    cfg = read_config(config_path)
    video = list(cfg['video_sets'])[0]
    image_folder = get_labeled_data_folder(cfg, video)
    rel_folder = os.path.relpath(image_folder, cfg['project_path'])
    bodyparts = cfg['bodyparts']
    columns = pd.MultiIndex.from_product(([conftest.SCORER], bodyparts, ['x', 'y']),
                                         names=['scorer', 'bodyparts', 'coords'])
    index = [os.path.join(rel_folder, image) for image in grab_files_in_folder(image_folder, 'png')]
    fake_data = np.tile(np.repeat(50 * np.arange(len(bodyparts)) + 100, 2), (len(index), 1))
    df = pd.DataFrame(fake_data, index=index, columns=columns)
    output_path = os.path.join(image_folder, f'CollectedData_{conftest.SCORER}.csv')
    df.to_csv(output_path)
    df.to_hdf(output_path.replace('csv', 'h5'), 'df_with_missing', format='table', mode='w')

    fracs = [0.4, 0.6, 0.8]
    train_inds, test_inds = zip(*[trainingsetmanipulation.SplitTrials(range(conftest.NUM_FRAMES), frac)
                                  for frac in fracs])
    shuffles = [1] * len(fracs)
    mocker.patch.object(auxfun_models, 'Downloadweights', return_value=None)  # Avoid downloading weights
    splits = trainingsetmanipulation.create_training_dataset(config_path, Shuffles=shuffles,
                                                             trainIndexes=train_inds, testIndexes=test_inds,
                                                             net_type=network, augmenter_type=aug)
    actual_fracs = [split[0] for split in splits]
    np.testing.assert_array_almost_equal(fracs, actual_fracs, decimal=1)


@pytest.mark.parametrize('network', ['resnet_50', 'resnet_101', 'resnet_152'])
def test_create_training_dataset_multi(mocker, project_multi, network):
    config_path, tmpdir = project_multi

    # Add artificial data
    cfg = read_config(config_path)
    video = list(cfg['video_sets'])[0]
    image_folder = get_labeled_data_folder(cfg, video)
    rel_folder = os.path.relpath(image_folder, cfg['project_path'])
    n_animals = len(cfg['individuals'])
    animals, bodyparts_single, bodyparts_multi = extractindividualsandbodyparts(cfg)
    animals_id = [i for i in range(n_animals) for _ in bodyparts_multi] + [n_animals] * len(bodyparts_single)
    map_ = dict(zip(range(len(animals)), animals))
    individuals = [map_[ind] for ind in animals_id for _ in range(2)]
    scorer = [conftest.SCORER] * len(individuals)
    coords = ['x', 'y'] * len(animals_id)
    bodyparts = [bp for _ in range(n_animals) for bp in bodyparts_multi for _ in range(2)]
    bodyparts += [bp for bp in bodyparts_single for _ in range(2)]
    columns = pd.MultiIndex.from_arrays([scorer, individuals, bodyparts, coords],
                                        names=['scorer', 'individuals', 'bodyparts', 'coords'])
    index = [os.path.join(rel_folder, image) for image in grab_files_in_folder(image_folder, 'png')]
    fake_data = np.tile(np.repeat(50 * np.arange(len(animals_id)) + 100, 2), (len(index), 1))
    df = pd.DataFrame(fake_data, index=index, columns=columns)
    output_path = os.path.join(image_folder, f'CollectedData_{conftest.SCORER}.csv')
    df.to_csv(output_path)
    df.to_hdf(output_path.replace('csv', 'h5'), 'df_with_missing', format='table', mode='w')

    mocker.patch.object(auxfun_models, 'Downloadweights', return_value=None)  # Avoid downloading weights
    create_multianimaltraining_dataset(config_path, net_type=network)


@pytest.mark.parametrize('fake_project, scale, draw_skeleton, visualizeindividuals',
                         [('project_single', 1, True, False),
                          ('project_multi', 0.5, False, True)],
                         indirect=['fake_project'])
def test_check_labels(fake_project, scale, draw_skeleton, visualizeindividuals):
    config_path, tmpdir = fake_project
    labels = ['+', '.', 'x']
    check_labels(config_path, labels, scale, draw_skeleton, visualizeindividuals)
    cfg = read_config(config_path)
    video = list(cfg['video_sets'])[0]
    _, width, _, height = list(map(int, cfg['video_sets'][video]['crop'].split(',')))
    image_folder = get_labeled_data_folder(cfg, video) + '_labeled'
    images = list(grab_files_in_folder(image_folder, 'png', False))
    assert len(images) == conftest.NUM_FRAMES
    image = io.imread(images[0])
    assert image.shape[0] == int(height * scale)
    assert image.shape[1] == int(width * scale)
