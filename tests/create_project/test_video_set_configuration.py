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
"""Unit tests for deeplabcut.create_project.new module"""

import warnings
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
from collections.abc import Mapping

import deeplabcut.create_project.new as new_module
from deeplabcut.utils.auxfun_videos import VideoReader

@pytest.fixture
def project_directory(tmpdir_factory) -> Path:
    proj_dir = Path(tmpdir_factory.mktemp("test-project"))
    return proj_dir

@pytest.fixture
def mock_video_file(tmpdir_factory) -> Path:
    """Create a mock video file for testing"""
    fake_folder = tmpdir_factory.mktemp("some_video")
    video_path = Path(fake_folder) / "test_video.avi"
    video_path.write_bytes(b"fake video content")
    return video_path


@pytest.fixture
def mock_video_reader() -> VideoReader:
    """Create a mock VideoReader"""
    mock_reader = Mock(spec=VideoReader)
    mock_reader.get_bbox.return_value = (0, 640, 277, 624)
    return mock_reader


@pytest.fixture
def video_directory(tmpdir_factory) -> Path:
    """Create a directory with multiple video files"""
    video_dir = Path(tmpdir_factory.mktemp("some_videos"))
    video_dir.mkdir(exist_ok=True)
    
    # Create multiple video files with different extensions
    (video_dir / "video1.avi").write_bytes(b"fake video 1")
    (video_dir / "video2.mp4").write_bytes(b"fake video 2")
    (video_dir / "video3.mov").write_bytes(b"fake video 3")
    (video_dir / "not_a_video.txt").write_text("text file")
    
    return video_dir


def test_project_directory_creation_basic(
    tmpdir: Path, 
    mock_video_file: Path, 
    mock_video_reader: VideoReader,
):
    """Test that project directories are created correctly"""
    with patch("deeplabcut.create_project.new.VideoReader", return_value=mock_video_reader):
        config_path = new_module.create_new_project(
            project="test-project",
            experimenter="test-user",
            videos=[str(mock_video_file)],
            working_directory=str(tmpdir),
            copy_videos=False,
        )
    
    project_path = Path(config_path).parent
    assert project_path.exists()
    assert (project_path / "videos").exists()
    assert (project_path / "labeled-data").exists()
    assert (project_path / "training-datasets").exists()
    assert (project_path / "dlc-models").exists()


@pytest.mark.parametrize('copy_videos', [True, False])
def test_single_video_file(
    tmpdir: Path,
    mock_video_file: Path,
    mock_video_reader: VideoReader,
    copy_videos: bool,
):
    """Test adding a single video file"""
    with patch("deeplabcut.create_project.new.VideoReader", return_value=mock_video_reader):
        config_path = new_module.create_new_project(
            project="test",
            experimenter="user",
            videos=[str(mock_video_file)],
            working_directory=str(tmpdir),
            copy_videos=copy_videos,
        )
    
    project_path = Path(config_path).parent
    video_path = project_path / "videos" / 'test_video.avi'
    assert video_path.exists() or video_path.is_symlink()

    # Content should match
    if copy_videos:
        assert mock_video_file.read_bytes() == video_path.read_bytes()


@pytest.mark.parametrize('copy_videos', [True, False])    
def test_video_directory(
    tmpdir: Path,
    video_directory: Path,
    mock_video_reader: VideoReader,
    copy_videos: bool,
):
    """Test adding videos from a directory"""
    with patch("deeplabcut.create_project.new.VideoReader", return_value=mock_video_reader):
        config_path = new_module.create_new_project(
            project="test",
            experimenter="user",
            videos=[str(video_directory)],
            working_directory=str(tmpdir),
            videotype=".avi",
            copy_videos=copy_videos,
        )
    
    project_path = Path(config_path).parent
    assert (project_path / "videos" / "video1.avi").exists() or (project_path / "videos" / "video1.avi").is_symlink()
    
    # Content should match
    if copy_videos:
        assert (project_path / "videos" / "video1.avi").read_bytes() == (video_directory / "video1.avi").read_bytes()


@pytest.mark.parametrize('copy_videos', [True, False])  
def test_mixed_video_files_and_directories(
    tmpdir,
    mock_video_file: Path,
    video_directory: Path,
    mock_video_reader: VideoReader,
    copy_videos: bool,
):
    """Test adding both video files and directories"""
    with patch("deeplabcut.create_project.new.VideoReader", return_value=mock_video_reader):
        config_path = new_module.create_new_project(
            project="test",
            experimenter="user",
            videos=[str(mock_video_file), str(video_directory)],
            working_directory=str(tmpdir),
            videotype=".avi",
            copy_videos=copy_videos,
        )
    
    project_path = Path(config_path).parent
    videos_dir = project_path / "videos"
    # Should have both the single file and files from directory
    assert (videos_dir / mock_video_file.name).exists() or (videos_dir / mock_video_file.name).is_symlink()
    assert (videos_dir / "video1.avi").exists() or (videos_dir / "video1.avi").is_symlink()


def test_empty_video_directory(
    tmpdir: Path,
    mock_video_reader: VideoReader,
):
    """Test handling of empty video directory"""
    empty_dir = tmpdir / "empty_videos"
    empty_dir.mkdir()
    
    with patch("deeplabcut.create_project.new.VideoReader", return_value=mock_video_reader):
        with warnings.catch_warnings(record=True) as w:
            result = new_module.create_new_project(
                project="test",
                experimenter="user",
                videos=[str(empty_dir)],
                working_directory=str(tmpdir),
                videotype=".avi",
                copy_videos=False,
            )
            # Should return "nothingcreated" when no valid videos found
            assert result == "nothingcreated" or len(w) > 0


def test_valid_video_included_in_config(
    tmpdir: Path,
    mock_video_file: Path,
    mock_video_reader: VideoReader,
):
    """Test that valid videos are included in the config file"""
    with patch("deeplabcut.create_project.new.VideoReader", return_value=mock_video_reader):
        config_path = new_module.create_new_project(
            project="test",
            experimenter="user",
            videos=[str(mock_video_file)],
            working_directory=str(tmpdir),
            copy_videos=False,
        )
    
    from deeplabcut.utils import auxiliaryfunctions
    cfg = auxiliaryfunctions.read_config(config_path)
    
    assert "video_sets" in cfg
    assert len(cfg["video_sets"]) > 0
    # Check that video path is in video_sets
    video_path_str = str(Path(mock_video_file).resolve())
    assert any(video_path_str in key for key in cfg["video_sets"].keys())


def test_invalid_video_removed_from_project(
    tmpdir: Path,
    mock_video_file: Path,
):
    """Test that invalid videos are removed from the project"""
    # Mock VideoReader to raise IOError
    mock_reader = Mock(side_effect=IOError("Cannot open video"))
    
    with patch("deeplabcut.create_project.new.VideoReader", mock_reader):
        with warnings.catch_warnings(record=True):
            result = new_module.create_new_project(
                project="test",
                experimenter="user",
                videos=[str(mock_video_file)],
                working_directory=str(tmpdir),
                copy_videos=False,
            )
    
    # Should return "nothingcreated" when no valid videos
    assert result == "nothingcreated"

    
def test_config_file_video_sets_format(
    tmpdir: Path,
    mock_video_file: Path,
    mock_video_reader: VideoReader,
):
    """Test that video_sets in config has correct format"""
    with patch("deeplabcut.create_project.new.VideoReader", return_value=mock_video_reader):
        config_path = new_module.create_new_project(
            project="test",
            experimenter="user",
            videos=[str(mock_video_file)],
            working_directory=str(tmpdir),
            copy_videos=False,
        )
    
    from deeplabcut.utils import auxiliaryfunctions
    cfg = auxiliaryfunctions.read_config(config_path)
    
    assert "video_sets" in cfg
    assert isinstance(cfg["video_sets"], Mapping)
    
    # Check format of video_sets entries
    for video_path, video_info in cfg["video_sets"].items():
        assert isinstance(video_info, Mapping)
        assert "crop" in video_info
        assert isinstance(video_info["crop"], str)
