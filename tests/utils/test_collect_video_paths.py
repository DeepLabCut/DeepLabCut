#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests for ``collect_video_paths``.

These tests pin down the rule:

* When ``video_type`` is not set, directory enumeration filters by
  ``SUPPORTED_VIDEOS`` but explicitly-supplied files are trusted (returned
  as-is, even if they have no suffix).
* When ``video_type`` is set, it is honoured everywhere — both for files
  pulled from directories and for files supplied by the caller.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deeplabcut.utils.auxfun_videos import SUPPORTED_VIDEOS, collect_video_paths
from deeplabcut.utils.deprecation import DLCDeprecationWarning


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_keeps_suffixless_files_when_explicitly_listed(tmp_path):
    """Regression test: a caller-supplied file without an extension (e.g.
    a content-addressed cache entry) must not be silently dropped."""
    suffixed = _touch(tmp_path / "video.mp4")
    hashed = _touch(tmp_path / "abcd1234")

    result = collect_video_paths([suffixed, hashed], extensions=None)

    assert {p.name for p in result} == {"video.mp4", "abcd1234"}


def test_accepts_path_objects_and_strings(tmp_path):
    suffixed = _touch(tmp_path / "video.mp4")
    hashed = _touch(tmp_path / "abcd1234")

    result = collect_video_paths([str(suffixed), hashed], extensions=None)

    assert {p.name for p in result} == {"video.mp4", "abcd1234"}


def test_accepts_single_path_argument(tmp_path):
    """A single path (not wrapped in a list) is also valid input."""
    hashed = _touch(tmp_path / "abcd1234")

    result = collect_video_paths(hashed, extensions=None)

    assert [p.name for p in result] == ["abcd1234"]


def test_explicit_video_type_filters_listed_files(tmp_path):
    """When ``extensions`` is set, it filters explicitly-supplied files too."""
    mp4 = _touch(tmp_path / "video.mp4")
    avi = _touch(tmp_path / "video.avi")

    result = collect_video_paths([mp4, avi], extensions="mp4")

    assert {p.name for p in result} == {"video.mp4"}


def test_explicit_video_type_accepts_leading_dot(tmp_path):
    mp4 = _touch(tmp_path / "video.mp4")
    avi = _touch(tmp_path / "video.avi")

    result = collect_video_paths([mp4, avi], extensions=".mp4")

    assert {p.name for p in result} == {"video.mp4"}


def test_explicit_video_type_case_insensitive(tmp_path):
    """Extension matching must be case-insensitive."""
    mp4 = _touch(tmp_path / "video.mp4")
    avi = _touch(tmp_path / "video.avi")

    result = collect_video_paths([mp4, avi], extensions="MP4")

    assert {p.name for p in result} == {"video.mp4"}


def test_multiple_extensions_filter_directory(tmp_path):
    """A sequence of extensions filters directory contents to only matching files."""
    mp4 = _touch(tmp_path / "video.mp4")
    avi = _touch(tmp_path / "video.avi")
    _touch(tmp_path / "video.mkv")

    result = collect_video_paths(tmp_path, extensions=["mp4", "avi"])

    assert {p.name for p in result} == {mp4.name, avi.name}


def test_directory_enumeration_filters_by_supported_videos(tmp_path):
    """Directory scans must continue to discriminate videos from non-videos."""
    mp4 = _touch(tmp_path / "video.mp4")
    _touch(tmp_path / "notes.txt")
    _touch(tmp_path / "results.h5")
    _touch(tmp_path / "abcd1234")  # suffix-less file in a directory: not a video

    result = collect_video_paths(tmp_path, extensions=None)

    assert [p.name for p in result] == [mp4.name]


def test_directory_enumeration_skips_dlc_artifacts(tmp_path):
    """``*_labeled.*`` and ``*_full.*`` are DLC outputs, not inputs."""
    mp4 = _touch(tmp_path / "video.mp4")
    _touch(tmp_path / "video_labeled.mp4")
    _touch(tmp_path / "video_full.mp4")

    result = collect_video_paths(tmp_path, extensions=None)

    assert {p.name for p in result} == {mp4.name}


def test_disable_exclude_patterns_includes_dlc_artifacts(tmp_path):
    """Setting ``exclude_patterns=[]`` disables all pattern exclusion."""
    mp4 = _touch(tmp_path / "video.mp4")
    labeled = _touch(tmp_path / "video_labeled.mp4")
    full = _touch(tmp_path / "video_full.mp4")

    result = collect_video_paths(tmp_path, extensions=None, exclude_patterns=[])

    assert {p.name for p in result} == {mp4.name, labeled.name, full.name}


def test_mixed_files_and_directories(tmp_path):
    """The function handles a mix of explicit files and directories."""
    folder = tmp_path / "folder"
    in_folder = _touch(folder / "from_dir.mp4")
    _touch(folder / "ignored.txt")

    explicit_mp4 = _touch(tmp_path / "explicit.mp4")
    explicit_hashed = _touch(tmp_path / "abcd1234")

    result = collect_video_paths(
        [folder, explicit_mp4, explicit_hashed],
        extensions=None,
    )

    assert {p.name for p in result} == {
        in_folder.name,
        explicit_mp4.name,
        explicit_hashed.name,
    }


def test_duplicates_are_removed(tmp_path):
    mp4 = _touch(tmp_path / "video.mp4")

    result = collect_video_paths([mp4, mp4, str(mp4)], extensions=None)

    assert len(result) == 1
    assert result[0].name == "video.mp4"


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect_video_paths([tmp_path / "does_not_exist.mp4"], extensions=None)


@pytest.mark.parametrize("ext", SUPPORTED_VIDEOS)
def test_each_supported_extension_picked_up_in_directory(tmp_path, ext):
    expected = _touch(tmp_path / f"clip.{ext}")

    result = collect_video_paths(tmp_path, extensions=None)

    assert [p.name for p in result] == [expected.name]


def test_sorted_by_default_when_not_shuffled(tmp_path):
    a = _touch(tmp_path / "a.mp4")
    b = _touch(tmp_path / "b.mp4")
    c = _touch(tmp_path / "c.mp4")

    result = collect_video_paths([c, a, b], extensions=None, shuffle=False)

    assert [p.name for p in result] == ["a.mp4", "b.mp4", "c.mp4"]


@pytest.mark.parametrize("deprecated_value", ["", [""], ("",), {""}])
def test_deprecated_empty_extensions_warns(tmp_path, deprecated_value):
    """Empty / blank extension values are deprecated and should emit a warning."""
    _touch(tmp_path / "video.mp4")

    with pytest.warns(DLCDeprecationWarning):
        collect_video_paths(tmp_path, extensions=deprecated_value)


def test_empty_sequence_raises(tmp_path):
    """An empty sequence is not a valid filter; callers must pass None instead."""
    with pytest.raises(ValueError):
        collect_video_paths(tmp_path, extensions=[])
