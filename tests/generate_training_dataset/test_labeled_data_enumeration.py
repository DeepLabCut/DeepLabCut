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
"""Enumeration of labeled-data folders versus check_labels ``<stem>_labeled`` plot folders."""

from pathlib import Path

import pandas as pd
import pytest

from deeplabcut.core.config.utils import read_config_as_dict, write_config
from deeplabcut.generate_training_dataset import trainingsetmanipulation
from deeplabcut.utils import conversioncode


def _configured_videos(project_root: Path) -> dict[str, dict]:
    """Return ``video_sets`` as DeepLabCut would read it back from disk."""

    return read_config_as_dict(project_root / "config.yaml")["video_sets"]


def _declare_videos(project_root: Path, stems: tuple[str, ...]) -> None:
    """Point ``video_sets`` at ``videos/<stem>.mp4`` without creating the files.

    The files are deliberately absent.
    """

    config_path = project_root / "config.yaml"
    config = read_config_as_dict(config_path)

    config["video_sets"] = {str(project_root / "videos" / f"{stem}.mp4"): {"crop": "0, 100, 0, 100"} for stem in stems}

    write_config(config_path, config)


def _add_dataset_folder(project_root: Path, name: str) -> Path:
    folder = project_root / "labeled-data" / name
    folder.mkdir(parents=True, exist_ok=True)

    return folder


def test_normally_named_folder_keeps_its_video_entry(valid_project: Path) -> None:
    """Control: the reconciliation runs and leaves a matched pair alone.

    Without this, a test asserting that some entry survives cannot distinguish
    "filter is correct" from "function did nothing".
    """

    _declare_videos(valid_project, ("session-01",))
    _add_dataset_folder(valid_project, "session-01")

    trainingsetmanipulation.adddatasetstovideolistandviceversa(str(valid_project / "config.yaml"))

    assert [Path(video).stem for video in _configured_videos(valid_project)] == ["session-01"]


@pytest.mark.known_defect("LABELED_SUFFIX_PLOTS_FILTER")
def test_a_video_named_labeled_keeps_its_config_entry(valid_project: Path) -> None:
    """A user's video may be called ``trial_labeled_v2``.

    Its dataset folder is filtered out as if it were ``check_labels`` output,
    the stem is then reported missing, and the entry is deleted from
    ``config.yaml`` on disk.
    """

    _declare_videos(valid_project, ("trial_labeled_v2",))
    _add_dataset_folder(valid_project, "trial_labeled_v2")

    trainingsetmanipulation.adddatasetstovideolistandviceversa(str(valid_project / "config.yaml"))

    assert [Path(video).stem for video in _configured_videos(valid_project)] == ["trial_labeled_v2"]


@pytest.mark.known_defect("STEM_SUBSTRING_REMOVAL")
def test_dropping_one_stem_does_not_drop_names_containing_it(valid_project: Path) -> None:
    """``Sample1`` has no folder; ``Sample10`` has one and must be kept."""

    _declare_videos(valid_project, ("Sample1", "Sample10"))
    _add_dataset_folder(valid_project, "Sample10")

    trainingsetmanipulation.adddatasetstovideolistandviceversa(str(valid_project / "config.yaml"))

    assert "Sample10" in [Path(video).stem for video in _configured_videos(valid_project)]


@pytest.mark.known_defect("LABELED_SUFFIX_PLOTS_FILTER")
def test_comparison_does_not_report_a_labeled_named_folder_as_missing(
    valid_project: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The read-only report should agree that the folder is present."""

    _declare_videos(valid_project, ("trial_labeled_v2",))
    _add_dataset_folder(valid_project, "trial_labeled_v2")

    trainingsetmanipulation.comparevideolistsanddatafolders(str(valid_project / "config.yaml"))

    assert "is missing as a folder" not in capsys.readouterr().out


@pytest.mark.known_defect("UNFILTERED_LABELED_DATA_MERGE")
def test_a_directory_the_config_does_not_name_is_not_merged(valid_project: Path) -> None:
    """A copy of a dataset folder is merged as though it were a second dataset.

    ``merge_windowsannotationdataONlinuxsystem`` lists ``labeled-data/`` with
    ``d.is_dir()`` and no filter, so a directory belongs to the training set by
    holding a ``CollectedData`` file rather than by being named in the config.

    Reached from ``create_training_dataset`` through ``merge_annotateddatasets``'
    empty-result branch.

    ``check_labels`` output is not an instance of this: ``<stem>_labeled/`` holds
    rendered images only, so the loop skips it.
    """

    _declare_videos(valid_project, ("session-01",))

    config = read_config_as_dict(valid_project / "config.yaml")
    scorer = config["scorer"]

    def write_collected_data(folder: Path, frame: str) -> None:
        columns = pd.MultiIndex.from_tuples(
            [(scorer, "nose", "x"), (scorer, "nose", "y")],
            names=["scorer", "bodyparts", "coords"],
        )
        frame_index = [("labeled-data", folder.name, frame)]
        pd.DataFrame([[1.0, 2.0]], columns=columns, index=pd.MultiIndex.from_tuples(frame_index)).to_hdf(
            folder / f"CollectedData_{scorer}.h5",
            key="df_with_missing",
            mode="w",
        )

    dataset = _add_dataset_folder(valid_project, "session-01")
    copy = _add_dataset_folder(valid_project, "session-01_backup")

    write_collected_data(dataset, "img001.png")
    write_collected_data(copy, "img001.png")

    merged = conversioncode.merge_windowsannotationdataONlinuxsystem(config)

    merged_folders = {entry[1] for entry in pd.concat(merged).index}

    assert merged_folders == {"session-01"}
