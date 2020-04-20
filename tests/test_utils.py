import os
os.environ['DLClight'] = 'True'
import pytest
from deeplabcut.utils import auxiliaryfunctions
from tests import conftest


def test_read_config_invalid_path(tmpdir):
    fake_config = tmpdir.join('fake_config.yaml')
    with pytest.raises(FileNotFoundError):
        auxiliaryfunctions.read_config(fake_config)


def test_write_config(tmpdir, monkeypatch, cfg_single):
    output_path = tmpdir.join('config.yaml')
    cfg_single.pop('project_path')
    cfg_single.pop('skeleton')
    auxiliaryfunctions.write_config(output_path, cfg_single)
    cfg = auxiliaryfunctions.read_config(output_path)
    assert 'skeleton' in cfg
    assert 'project_path' in cfg
    assert cfg['project_path'] is None


def test_edit_config(tmpdir):
    config_path = conftest.SINGLE_CONFIG_PATH
    output_path = tmpdir.join('config.yaml')
    edits = {'project_path': 'nowhere',
             'new_key': 'new_value',
             'numframes2pick': 4,
             'TrainingFraction': [0.8],
             'multianimalproject': True}
    auxiliaryfunctions.edit_config(config_path, edits, output_path)
    cfg = auxiliaryfunctions.read_config(output_path)
    assert all(cfg[k] == v for k, v in edits.items())


def test_get_list_of_videos():
    vids = auxiliaryfunctions.Getlistofvideos([conftest.TEST_DATA_DIR], 'mov')
    assert isinstance(vids, list)
    assert len(vids) == 2

    assert len(auxiliaryfunctions.Getlistofvideos(conftest.videos, 'mov')) == 2

    vids = auxiliaryfunctions.Getlistofvideos(conftest.videos[0], 'mov')
    assert isinstance(vids, list)
    assert len(vids) == 1

    assert len(auxiliaryfunctions.Getlistofvideos(conftest.videos[0], 'avi')) == 1
    assert len(vids) == 1

    assert not len(auxiliaryfunctions.Getlistofvideos([conftest.TEST_DATA_DIR], 'avi'))


def test_grab_files_in_folder(mocker):
    mocker.patch.object(os, 'listdir', return_value=['file.h5', 'datah5.pickle'])
    files = list(auxiliaryfunctions.grab_files_in_folder('folder', 'h5', relative=False))
    assert len(files) == 1
    assert files[0].startswith('folder')


def test_intersection_bodyparts(cfg_single, cfg_multi):
    assert len(auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(cfg_single, 'all')) == 4
    assert len(auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(cfg_single, ['bodypart1'])) == 1
    assert not len(auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(cfg_single, ['none']))
    assert len(auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(cfg_multi, 'all')) == 4


@pytest.mark.parametrize('suffix', ['_meta', 'includingmetadata'])
def test_find_video_metadata(tmpdir, mocker, suffix):
    videoname = 'vid'
    scorer = 'DLCscorer'
    file = f'{videoname}{scorer}{suffix}.pickle'
    mocker.patch.object(auxiliaryfunctions, 'grab_files_in_folder', return_value=[file])
    meta = auxiliaryfunctions.find_video_metadata(tmpdir, videoname, scorer)
    assert meta == tmpdir.join(file)


def test_find_video_metadata_not_found(tmpdir):
    with pytest.raises(FileNotFoundError):
        _ = auxiliaryfunctions.load_video_metadata(tmpdir, '', '')


@pytest.mark.parametrize('filtered, track_method',
                         [(False, ''), (True, ''),
                          (False, 'box'), (True, 'box'),
                          (False, 'skeleton'), (True, 'skeleton')])
def test_find_analyzed_data(tmpdir, capsys, mocker, filtered, track_method):
    videoname = 'vid'
    scorer = 'DLCscorer'
    files = [f'{videoname}{scorer}.h5',
             f'{videoname}{scorer}_filtered.h5',
             f'{videoname}{scorer}_bx.h5',
             f'{videoname}{scorer}_bx_filtered.h5',
             f'{videoname}{scorer}_sk.h5',
             f'{videoname}{scorer}_sk_filtered.h5']
    mocker.patch.object(auxiliaryfunctions, 'grab_files_in_folder', return_value=files)
    filepath, scorer_found, suffix_found = auxiliaryfunctions.find_analyzed_data(tmpdir, videoname, scorer,
                                                                                 filtered, track_method)
    tracker = ''
    if track_method == 'skeleton':
        tracker = '_sk'
    elif track_method == 'box':
        tracker = '_bx'
    suffix = '_filtered' if filtered else ''
    assert filepath == tmpdir.join(f'{videoname}{scorer}{tracker}{suffix}.h5')
    assert scorer_found == scorer
    assert suffix_found == suffix


@pytest.mark.parametrize('filtered, track_method',
                         [(True, ''), (True, 'box'), (False, 'skeleton')])
def test_find_analyzed_data_not_found(tmpdir, mocker, filtered, track_method):
    videoname = 'vid'
    scorer = 'DLCscorer'
    # Let us try to fool the data finder
    files = [f'{videoname}{scorer}.h5',
             f'{videoname}{scorer}_bx.h5',
             f'{videoname}{scorer}_sk_filtered.h5']
    mocker.patch.object(auxiliaryfunctions, 'grab_files_in_folder', return_value=files)
    with pytest.raises(FileNotFoundError):
        _ = auxiliaryfunctions.find_analyzed_data(tmpdir, videoname, scorer,
                                                  filtered, track_method)
