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
from pathlib import Path
from unittest.mock import patch

import pytest

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.auxfun_videos import SUPPORTED_VIDEOS


def test_find_analyzed_data(tmpdir_factory):
    fake_folder = tmpdir_factory.mktemp("videos")
    SUPPORTED_VIDEOS = ["avi"]
    len(SUPPORTED_VIDEOS)

    SCORER = "DLC_dlcrnetms5_multi_mouseApr11shuffle1_5"
    WRONG_SCORER = "DLC_dlcrnetms5_multi_mouseApr11shuffle3_5"

    def _create_fake_file(filename):
        path = str(fake_folder.join(filename))
        with open(path, "w") as f:
            f.write("")
        return path

    for ind, ext in enumerate(SUPPORTED_VIDEOS):
        vname = "video" + str(ind)
        _ = _create_fake_file(vname + "." + ext)
        _ = _create_fake_file(vname + SCORER + ".pickle")
        _ = _create_fake_file(vname + SCORER + ".h5")

    for ind, ext in enumerate(SUPPORTED_VIDEOS):
        # test if existing models are found:
        assert auxiliaryfunctions.find_analyzed_data(fake_folder, "video" + str(ind), SCORER)

        # Test if nonexisting models are not found
        with pytest.raises(FileNotFoundError):
            auxiliaryfunctions.find_analyzed_data(fake_folder, "video" + str(ind), WRONG_SCORER)

        with pytest.raises(FileNotFoundError):
            auxiliaryfunctions.find_analyzed_data(fake_folder, "video" + str(ind), SCORER, filtered=True)


@pytest.mark.deprecated
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_get_list_of_videos(tmpdir_factory):
    fake_folder = tmpdir_factory.mktemp("videos")
    n_ext = len(SUPPORTED_VIDEOS)

    def _create_fake_file(filename):
        path = str(fake_folder.join(filename))
        with open(path, "w") as f:
            f.write("")
        return path

    fake_videos = []
    for ext in SUPPORTED_VIDEOS:
        path = _create_fake_file(f"fake.{ext}")
        fake_videos.append(path)

    # Add some other office files:
    path = _create_fake_file("fake.xls")
    path = _create_fake_file("fake.pptx")

    # Add a .pickle and .h5 files
    _ = _create_fake_file("fake.pickle")
    _ = _create_fake_file("fake.h5")

    # By default, all videos with common extensions are taken from a directory
    videos = auxiliaryfunctions.get_list_of_videos(
        str(fake_folder),
        videotype="",
    )
    assert len(videos) == n_ext

    # A list of extensions can also be passed in
    videos = auxiliaryfunctions.get_list_of_videos(
        str(fake_folder),
        videotype=SUPPORTED_VIDEOS,
    )
    assert len(videos) == n_ext

    for ext in SUPPORTED_VIDEOS:
        videos = auxiliaryfunctions.get_list_of_videos(
            str(fake_folder),
            videotype=ext,
        )
        assert len(videos) == 1

    videos = auxiliaryfunctions.get_list_of_videos(
        str(fake_folder),
        videotype="unknown",
    )
    assert not len(videos)

    videos = auxiliaryfunctions.get_list_of_videos(
        fake_videos,
        videotype="",
    )
    assert len(videos) == n_ext

    for video in fake_videos:
        videos = auxiliaryfunctions.get_list_of_videos([video], videotype="")
        assert len(videos) == 1

    for ext in SUPPORTED_VIDEOS:
        videos = auxiliaryfunctions.get_list_of_videos(
            fake_videos,
            videotype=ext,
        )
        assert len(videos) == 1


def test_write_config_has_skeleton(tmpdir_factory):
    """Required for backward compatibility."""
    fake_folder = tmpdir_factory.mktemp("fakeConfigs")
    fake_config_file = fake_folder / Path("fakeConfig")
    auxiliaryfunctions.write_config(fake_config_file, {})
    config_data = auxiliaryfunctions.read_config(fake_config_file)
    assert "skeleton" in config_data


@pytest.mark.parametrize(
    "multianimal, bodyparts, ma_bpts, unique_bpts, comparison_bpts, expected_bpts",
    [
        (
            False,
            ["head", "shoulders", "knees", "toes"],
            None,
            None,
            {"knees", "others", "toes"},
            ["knees", "toes"],
        ),
        (
            True,
            None,
            ["head", "shoulders", "knees"],
            ["toes"],
            {"knees", "others", "toes"},
            ["knees", "toes"],
        ),
    ],
)
def test_intersection_of_body_parts_and_ones_given_by_user(
    multianimal, bodyparts, ma_bpts, unique_bpts, comparison_bpts, expected_bpts
):
    cfg = {
        "multianimalproject": multianimal,
        "bodyparts": bodyparts,
        "multianimalbodyparts": ma_bpts,
        "uniquebodyparts": unique_bpts,
    }

    if multianimal:
        all_bodyparts = list(set(ma_bpts + unique_bpts))
    else:
        all_bodyparts = bodyparts

    filtered_bpts = auxiliaryfunctions.intersection_of_body_parts_and_ones_given_by_user(cfg, comparisonbodyparts="all")
    print(all_bodyparts)
    print(filtered_bpts)
    assert len(all_bodyparts) == len(filtered_bpts)
    assert all([bpt in all_bodyparts for bpt in filtered_bpts])

    filtered_bpts = auxiliaryfunctions.intersection_of_body_parts_and_ones_given_by_user(
        cfg,
        comparisonbodyparts=comparison_bpts,
    )
    print(filtered_bpts)
    assert len(expected_bpts) == len(filtered_bpts)
    assert all([bpt in expected_bpts for bpt in filtered_bpts])


class MockPath:
    def __init__(self, path: Path, st_mtime: int):
        self.path = path
        self.parent = self.path.parent
        self.st_mtime = st_mtime

    def lstat(self):
        return self


# labeled_folders: (has_H5, H5_st_mtime, folder_name)
@pytest.mark.parametrize(
    "labeled_folders, next_folder_name",
    [
        ([(True, 1, "a"), (False, None, "b"), (False, None, "c")], "b"),
        ([(False, None, "a"), (True, 123, "d"), (False, None, "f")], "f"),
    ],
)
def test_find_next_unlabeled_folder(
    tmpdir_factory,
    monkeypatch,
    labeled_folders,
    next_folder_name,
):
    project_folder = tmpdir_factory.mktemp("project")
    fake_cfg = Path(project_folder / "cfg.yaml")
    auxiliaryfunctions.write_config(fake_cfg, {"project_path": str(project_folder)})

    data_folder = project_folder / "labeled-data"
    data_folder.mkdir()
    rglob_results = []
    for has_h5, h5_last_mod_time, folder_name in labeled_folders:
        labeled_folder_path = Path(data_folder / folder_name)
        labeled_folder_path.mkdir()
        if has_h5:
            h5_path = Path(labeled_folder_path / "data.h5")
            rglob_results.append(MockPath(h5_path, h5_last_mod_time))

    def get_rglob_results(*args, **kwargs):
        return rglob_results

    monkeypatch.setattr(Path, "rglob", get_rglob_results)
    next_folder = auxiliaryfunctions.find_next_unlabeled_folder(fake_cfg)
    assert str(next_folder) == str(Path(data_folder / next_folder_name))


@pytest.fixture
def mock_snapshot_folder(tmp_path):
    """Mock folder with snapshots."""
    folder = tmp_path / "train"
    folder.mkdir()

    # mock files
    snapshot_files = [
        "snapshot-4.index",
        "snapshot-5.index",
        "snapshot-6.index",
        "snapshot-3.data-00000-of-00001",
        "snapshot-3.index",
        "snapshot-3.meta",
    ]
    for file_name in snapshot_files:
        (folder / file_name).touch()

    return folder


@pytest.fixture
def mock_no_snapshots_folder(tmp_path):
    """Mock folder with no snapshots."""
    folder = tmp_path / "train"
    folder.mkdir()

    # mock files
    snapshot_files = ["log.txt", "pose_cfg.yaml"]
    for file_name in snapshot_files:
        (folder / file_name).touch()

    return folder


def test_get_snapshots_from_folder(mock_snapshot_folder):
    """Test returns expected snapshots in order."""
    snapshot_names = auxiliaryfunctions.get_snapshots_from_folder(mock_snapshot_folder)
    assert snapshot_names == ["snapshot-3", "snapshot-4", "snapshot-5", "snapshot-6"]


def test_get_snapshots_from_folder_none(mock_no_snapshots_folder):
    """Test raises ValueError if no snapshots are found."""
    with pytest.raises(FileNotFoundError):
        auxiliaryfunctions.get_snapshots_from_folder(mock_no_snapshots_folder)


# ---------------------------------------------------------------------------
# Tests for safe_resolve() and read_config() network-drive path safety
# https://github.com/DeepLabCut/DeepLabCut/issues/3348
# ---------------------------------------------------------------------------


class TestSafeResolve:
    """safe_resolve() must return a Path whose str() representation can be
    opened by plain string-based I/O — i.e. it must not return Windows 11 SMB
    Volume GUID paths like \\\\?\\Volume{...}\\...
    """

    def test_normal_path_is_returned_unchanged(self, tmp_path):
        """On a normal local filesystem, safe_resolve returns the resolved path."""
        f = tmp_path / "config.yaml"
        f.touch()
        result = auxiliaryfunctions.safe_resolve(f)
        assert result == f.resolve()
        assert result.exists()

    def test_fallback_when_resolve_produces_unusable_path(self, tmp_path):
        """When resolve() returns a path that cannot be opened as a string,
        safe_resolve must fall back to abspath."""
        f = tmp_path / "config.yaml"
        f.write_text("project_path: .")

        fake_volume_guid = Path(r"\\?\Volume{DEADBEEF-0000-0000-0000-000000000000}\fake")

        with patch.object(Path, "resolve", return_value=fake_volume_guid):
            result = auxiliaryfunctions.safe_resolve(f)

        # Must NOT return the unusable Volume GUID path
        assert "Volume{" not in str(result)
        # Fallback must be the absolute (non-resolved) path, which exists and is openable
        assert result == f.absolute()
        open(result).close()


class TestReadConfigProjectPath:
    """read_config() must never persist a \\\\?\\Volume{GUID}\\... path into
    project_path, even on Windows 11 SMB network drives."""

    def test_project_path_not_written_as_volume_guid(self, tmp_path):
        """If resolve() would produce a Volume GUID path, read_config must fall
        back to abspath and write a usable path to config.yaml."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()
        config_file = project_dir / "config.yaml"

        auxiliaryfunctions.write_config(config_file, {"project_path": str(project_dir)})

        fake_volume_guid = Path(r"\\?\Volume{DEADBEEF-0000-0000-0000-000000000000}\my_project")

        with patch.object(Path, "resolve", return_value=fake_volume_guid):
            cfg = auxiliaryfunctions.read_config(config_file)

        assert "Volume{" not in cfg["project_path"]
        # The stored value must be openable as a plain string
        assert os.path.isdir(cfg["project_path"])

    def test_project_path_updated_when_moved(self, tmp_path):
        """read_config() must still update project_path when a project is moved
        to a new directory (the original feature that resolve() was meant for)."""
        project_dir = tmp_path / "original_location"
        project_dir.mkdir()
        config_file = project_dir / "config.yaml"

        auxiliaryfunctions.write_config(config_file, {"project_path": "/some/old/path/that/no/longer/exists"})

        cfg = auxiliaryfunctions.read_config(config_file)

        expected = str(project_dir.absolute())
        assert cfg["project_path"] == expected
