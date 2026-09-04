import warnings
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from scipy.spatial import KDTree

from deeplabcut.utils import skeleton as skeleton_mod
from deeplabcut.utils.skeleton import SkeletonBuilder, write_config

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_config(project_path, scorer="TestScorer", skeleton=None):
    return {
        "project_path": str(project_path),
        "scorer": scorer,
        "skeleton": skeleton or [],
        "skeleton_color": "red",
        "dotsize": 4,
    }


def make_test_builder():
    """
    Construct a SkeletonBuilder instance without calling __init__,
    so individual methods can be unit-tested in isolation.
    """
    builder = SkeletonBuilder.__new__(SkeletonBuilder)
    return builder


def attach_fake_canvas(builder):
    builder.fig = Figure()
    builder._ax = builder.fig.add_subplot(111)
    builder._ax.set_xlim(-5, 25)
    builder._ax.set_ylim(-5, 5)
    builder.fig.canvas.draw_idle = lambda: None


def patch_builder_ui(monkeypatch, imread_calls=None):
    """Stub out the interactive parts of SkeletonBuilder.__init__."""

    def fake_imread(path):
        if imread_calls is not None:
            imread_calls.append(path)
        return np.zeros((5, 5, 3), dtype=np.uint8)

    monkeypatch.setattr(skeleton_mod.io, "imread", fake_imread)
    monkeypatch.setattr(SkeletonBuilder, "build_ui", lambda self: None)
    monkeypatch.setattr(SkeletonBuilder, "display", lambda self: None)
    monkeypatch.setattr(np.random, "shuffle", lambda x: None)


# ---------------------------------------------------------------------
# Annotation data fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def collected_data():
    """Factory for a one-row CollectedData frame.

    NaN values stand for unlabeled bodyparts. Pass ``individuals`` to get the
    multi-animal column layout.
    """

    def _make(video, values, bodyparts=("nose", "tail"), individuals=None):
        index = pd.MultiIndex.from_tuples(
            [("labeled-data", video, "img001.png")],
            names=["data_folder", "video", "image"],
        )
        levels = [["TestScorer"]]
        names = ["scorer"]
        if individuals is not None:
            levels.append(list(individuals))
            names.append("individuals")
        levels += [list(bodyparts), ["x", "y"]]
        names += ["bodyparts", "coords"]
        columns = pd.MultiIndex.from_product(levels, names=names)
        return pd.DataFrame([values], index=index, columns=columns)

    return _make


@pytest.fixture
def write_collected_data():
    """Factory writing a CollectedData frame into a labeled-data folder."""

    def _write(folder, df, scorer="TestScorer"):
        folder.mkdir(parents=True, exist_ok=True)
        df.to_hdf(folder / f"CollectedData_{scorer}.h5", key="df", mode="w")
        return folder

    return _write


@pytest.fixture
def project(tmp_path):
    """Factory for a project directory with an empty labeled-data folder.

    Returns ``(project_path, config_path)``.
    """

    def _make(skeleton=None, scorer="TestScorer"):
        project_path = tmp_path / "project"
        (project_path / "labeled-data").mkdir(parents=True)
        cfg_path = project_path / "config.yaml"
        write_config(cfg_path, make_config(project_path, scorer=scorer, skeleton=skeleton))
        return project_path, cfg_path

    return _make


# ---------------------------------------------------------------------
# pick_labeled_frame
# ---------------------------------------------------------------------


def test_pick_labeled_frame_multi_animal_drops_single(monkeypatch, collected_data):
    builder = make_test_builder()
    # "single" is fully labeled too, but should be dropped before choosing.
    builder.df = collected_data(
        "session1",
        [1.0, 2.0, 3.0, 4.0] + [10.0, 20.0, 30.0, 40.0],
        individuals=["single", "mouseA"],
    )

    monkeypatch.setattr(np.random, "shuffle", lambda x: None)

    picked_row, picked_col = builder.pick_labeled_frame()

    assert picked_row == ("labeled-data", "session1", "img001.png")
    assert picked_col == "mouseA"


def test_pick_labeled_frame_returns_none_when_nothing_is_labeled(collected_data):
    builder = make_test_builder()
    builder.df = collected_data("session1", [np.nan] * 4)

    assert builder.pick_labeled_frame() is None


def test_pick_labeled_frame_returns_none_for_an_empty_frame(collected_data):
    builder = make_test_builder()
    builder.df = collected_data("session1", [np.nan] * 4).iloc[:0]

    assert builder.pick_labeled_frame() is None


def test_pick_labeled_frame_returns_none_when_only_single_is_labeled(collected_data):
    """'single' is dropped before counting, leaving nothing to pick."""
    builder = make_test_builder()
    builder.df = collected_data(
        "session1",
        [1.0, 2.0, 3.0, 4.0] + [np.nan] * 4,
        individuals=["single", "mouseA"],
    )

    assert builder.pick_labeled_frame() is None


def test_pick_labeled_frame_without_individuals(monkeypatch, collected_data):
    builder = make_test_builder()
    builder.df = collected_data("session1", [1.0, 2.0, 3.0, 4.0])

    monkeypatch.setattr(np.random, "shuffle", lambda x: None)

    picked_row, picked_col = builder.pick_labeled_frame()

    assert picked_row == ("labeled-data", "session1", "img001.png")
    # fallback path uses count(...).to_frame(), so the single column is usually 0
    assert picked_col == 0


# ---------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------


def test_clear_resets_indices_segments_and_linecollection():
    builder = make_test_builder()
    builder.inds = {(0, 1), (1, 2)}
    builder.segs = {
        ((0.0, 0.0), (10.0, 0.0)),
        ((10.0, 0.0), (20.0, 0.0)),
    }
    builder.lines = LineCollection([np.array([[0.0, 0.0], [10.0, 0.0]]), np.array([[10.0, 0.0], [20.0, 0.0]])])
    attach_fake_canvas(builder)

    builder.clear()

    assert builder.inds == set()
    assert builder.segs == set()
    assert list(builder.lines.get_segments()) == []


# ---------------------------------------------------------------------
# export
# ---------------------------------------------------------------------


def test_export_sorts_pairs_and_warns_for_unconnected(monkeypatch, caplog):
    builder = make_test_builder()
    builder.config_path = "dummy_config.yaml"
    builder.xy = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],  # intentionally left unconnected
        ]
    )
    builder.bpts = pd.Index(["nose", "tail", "paw", "ear"], name="bodyparts")
    builder.inds = {(1, 2), (0, 1)}  # intentionally unordered
    builder.cfg = {"skeleton": []}

    captured = {}

    def fake_write_config(path, cfg):
        captured["path"] = path
        captured["cfg"] = cfg.copy()

    monkeypatch.setattr(skeleton_mod, "write_config", fake_write_config)

    with caplog.at_level("INFO"):
        builder.export()
    assert "Not all bodyparts are connected" in caplog.text

    assert captured["path"] == "dummy_config.yaml"
    assert captured["cfg"]["skeleton"] == [
        ("nose", "tail"),
        ("tail", "paw"),
    ]


def test_export_without_warning_when_all_bodyparts_connected(monkeypatch):
    builder = make_test_builder()
    builder.config_path = "dummy_config.yaml"
    builder.xy = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
        ]
    )
    builder.bpts = pd.Index(["nose", "tail", "paw"], name="bodyparts")
    builder.inds = {(0, 1), (1, 2)}
    builder.cfg = {"skeleton": []}

    monkeypatch.setattr(skeleton_mod, "write_config", lambda path, cfg: None)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        builder.export()

    assert not any("didn't connect all the bodyparts" in str(w.message) for w in record)
    assert builder.cfg["skeleton"] == [
        ("nose", "tail"),
        ("tail", "paw"),
    ]


# ---------------------------------------------------------------------
# on_select
# ---------------------------------------------------------------------


def test_on_select_adds_pairs_segments_and_updates_canvas():
    builder = make_test_builder()
    builder.xy = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
        ]
    )
    builder.tree = KDTree(builder.xy)
    builder.inds = set()
    builder.segs = set()
    builder.lines = LineCollection([])
    attach_fake_canvas(builder)

    verts = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    builder.on_select(verts)

    assert builder.inds == {(0, 1), (1, 2)}
    assert ((0.0, 0.0), (10.0, 0.0)) in builder.segs
    assert ((10.0, 0.0), (20.0, 0.0)) in builder.segs
    assert len(builder.lines.get_segments()) == 2


def test_on_select_ignores_duplicate_hits():
    builder = make_test_builder()
    builder.xy = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
        ]
    )
    builder.tree = KDTree(builder.xy)
    builder.inds = set()
    builder.segs = set()
    builder.lines = LineCollection([])
    attach_fake_canvas(builder)

    # Repeated nearby vertices should not create duplicate pairs
    verts = [(0.0, 0.0), (0.1, 0.0), (10.0, 0.0), (10.1, 0.0), (20.0, 0.0)]
    builder.on_select(verts)

    assert builder.inds == {(0, 1), (1, 2)}
    assert len(builder.segs) == 2


# ---------------------------------------------------------------------
# on_pick
# ---------------------------------------------------------------------


def test_on_pick_right_click_removes_segment_and_pair():
    builder = make_test_builder()
    builder.xy = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
        ]
    )
    builder.tree = KDTree(builder.xy)
    builder.inds = {(0, 1)}
    builder.segs = {((0.0, 0.0), (10.0, 0.0))}
    builder.lines = LineCollection([np.array([[0.0, 0.0], [10.0, 0.0]])])
    attach_fake_canvas(builder)

    event = SimpleNamespace(
        mouseevent=SimpleNamespace(button=3),
        artist=builder.lines,
        ind=[0],
    )

    builder.on_pick(event)

    assert builder.inds == set()
    assert builder.segs == set()
    assert list(builder.lines.get_segments()) == []


def test_on_pick_non_right_click_does_nothing():
    builder = make_test_builder()
    builder.xy = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
        ]
    )
    builder.tree = KDTree(builder.xy)
    builder.inds = {(0, 1)}
    builder.segs = {((0.0, 0.0), (10.0, 0.0))}
    builder.lines = LineCollection([np.array([[0.0, 0.0], [10.0, 0.0]])])
    attach_fake_canvas(builder)

    event = SimpleNamespace(
        mouseevent=SimpleNamespace(button=1),
        artist=builder.lines,
        ind=[0],
    )

    builder.on_pick(event)

    assert builder.inds == {(0, 1)}
    assert builder.segs == {((0.0, 0.0), (10.0, 0.0))}
    assert len(builder.lines.get_segments()) == 1


# ---------------------------------------------------------------------
# __init__ lightweight integration
# ---------------------------------------------------------------------


def test_init_loads_dataframe_image_and_existing_skeleton(monkeypatch, project, collected_data, write_collected_data):
    project_path, cfg_path = project(
        skeleton=[
            ["nose", "tail"],
            ["missing", "nose"],
        ],  # second pair should be ignored
    )
    write_collected_data(
        project_path / "labeled-data" / "session1",
        collected_data("session1", [0.0, 0.0, 10.0, 0.0]),
    )
    patch_builder_ui(monkeypatch)

    builder = SkeletonBuilder(str(cfg_path))

    assert builder.config_path == str(cfg_path)
    assert list(builder.bpts) == ["nose", "tail"]
    assert builder.xy.shape == (2, 2)
    assert builder.image.shape == (5, 5, 3)
    assert builder.inds == {(0, 1)}
    assert ((0.0, 0.0), (10.0, 0.0)) in builder.segs


def test_init_raises_if_no_labeled_data_found(monkeypatch, project):
    _project_path, cfg_path = project()

    monkeypatch.setattr(SkeletonBuilder, "build_ui", lambda self: None)
    monkeypatch.setattr(SkeletonBuilder, "display", lambda self: None)

    with pytest.raises(IOError, match="No labeled data were found"):
        SkeletonBuilder(str(cfg_path))


# ---------------------------------------------------------------------
# __init__ labeled-data folder search
#
# The search loop skips folders it cannot use instead of letting the
# first unusable one abort the whole search. The tests below pin down
# both the unchanged selection behaviour and the skipping.
# ---------------------------------------------------------------------


def test_init_prefers_a_fully_labeled_folder_over_a_partial_one(
    monkeypatch, project, collected_data, write_collected_data
):
    """Selection is by completeness, not by folder order."""
    project_path, cfg_path = project()
    labeled_data = project_path / "labeled-data"
    write_collected_data(
        labeled_data / "aaa_partial",
        collected_data("aaa_partial", [1.0, 2.0, np.nan, np.nan]),
    )
    write_collected_data(
        labeled_data / "zzz_complete",
        collected_data("zzz_complete", [0.0, 0.0, 10.0, 0.0]),
    )
    patch_builder_ui(monkeypatch)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        builder = SkeletonBuilder(str(cfg_path))

    assert not np.isnan(builder.xy).any()
    assert not any("fully labeled animal could not be found" in str(w.message) for w in record)


def test_init_loads_the_image_of_the_picked_frame(monkeypatch, project, collected_data, write_collected_data):
    """The image path is rebuilt from the picked row, relative to project_path."""
    project_path, cfg_path = project()
    write_collected_data(
        project_path / "labeled-data" / "session1",
        collected_data("session1", [0.0, 0.0, 10.0, 0.0]),
    )
    imread_calls = []
    patch_builder_ui(monkeypatch, imread_calls=imread_calls)

    builder = SkeletonBuilder(str(cfg_path))

    assert builder.image.shape == (5, 5, 3)
    assert imread_calls == [project_path / "labeled-data" / "session1" / "img001.png"]


def test_init_drops_the_individuals_level_for_multi_animal_data(
    monkeypatch, project, collected_data, write_collected_data
):
    """self.df is narrowed to the picked individual, so bpts excludes it."""
    project_path, cfg_path = project()
    write_collected_data(
        project_path / "labeled-data" / "session1",
        collected_data(
            "session1",
            [np.nan] * 4 + [0.0, 0.0, 10.0, 0.0],
            individuals=["single", "mouseA"],
        ),
    )
    patch_builder_ui(monkeypatch)

    builder = SkeletonBuilder(str(cfg_path))

    assert "individuals" not in builder.df.columns.names
    assert list(builder.bpts) == ["nose", "tail"]
    assert builder.xy.shape == (2, 2)


def test_init_skips_a_folder_without_collected_data(monkeypatch, caplog, project, collected_data, write_collected_data):
    """Frames extracted but never labeled must not abort the search."""
    project_path, cfg_path = project()
    labeled_data = project_path / "labeled-data"
    (labeled_data / "aaa_not_labeled_yet").mkdir()
    write_collected_data(
        labeled_data / "zzz_complete",
        collected_data("zzz_complete", [0.0, 0.0, 10.0, 0.0]),
    )
    patch_builder_ui(monkeypatch)

    with caplog.at_level("WARNING"):
        builder = SkeletonBuilder(str(cfg_path))

    assert list(builder.bpts) == ["nose", "tail"]
    assert "aaa_not_labeled_yet" in caplog.text


def test_init_skips_a_folder_with_no_labeled_rows(monkeypatch, caplog, project, collected_data, write_collected_data):
    project_path, cfg_path = project()
    labeled_data = project_path / "labeled-data"
    write_collected_data(
        labeled_data / "aaa_all_nan",
        collected_data("aaa_all_nan", [np.nan] * 4),
    )
    write_collected_data(
        labeled_data / "zzz_complete",
        collected_data("zzz_complete", [0.0, 0.0, 10.0, 0.0]),
    )
    patch_builder_ui(monkeypatch)

    with caplog.at_level("WARNING"):
        builder = SkeletonBuilder(str(cfg_path))

    assert not np.isnan(builder.xy).any()
    assert "aaa_all_nan" in caplog.text


def test_init_skips_a_folder_whose_image_is_missing(monkeypatch, project, collected_data, write_collected_data):
    """A row pointing at a deleted frame skips that folder, not the search."""
    project_path, cfg_path = project()
    labeled_data = project_path / "labeled-data"
    write_collected_data(
        labeled_data / "aaa_image_gone",
        collected_data("aaa_image_gone", [1.0, 2.0, 3.0, 4.0]),
    )
    write_collected_data(
        labeled_data / "zzz_complete",
        collected_data("zzz_complete", [0.0, 0.0, 10.0, 0.0]),
    )

    def fake_imread(path):
        if "aaa_image_gone" in str(path):
            raise FileNotFoundError(path)
        return np.zeros((5, 5, 3), dtype=np.uint8)

    monkeypatch.setattr(skeleton_mod.io, "imread", fake_imread)
    monkeypatch.setattr(SkeletonBuilder, "build_ui", lambda self: None)
    monkeypatch.setattr(SkeletonBuilder, "display", lambda self: None)
    monkeypatch.setattr(np.random, "shuffle", lambda x: None)

    builder = SkeletonBuilder(str(cfg_path))

    assert builder.xy.tolist() == [[0.0, 0.0], [10.0, 0.0]]


def test_init_error_reports_inspected_and_skipped_folders(monkeypatch, project, collected_data, write_collected_data):
    project_path, cfg_path = project()
    labeled_data = project_path / "labeled-data"
    (labeled_data / "no_h5_a").mkdir()
    (labeled_data / "no_h5_b").mkdir()
    write_collected_data(
        labeled_data / "all_nan",
        collected_data("all_nan", [np.nan] * 4),
    )
    patch_builder_ui(monkeypatch)

    with pytest.raises(IOError, match="No labeled data were found") as excinfo:
        SkeletonBuilder(str(cfg_path))

    message = str(excinfo.value)
    assert "3 folder(s) were inspected" in message
    assert "3 of which had to be skipped" in message


def test_init_still_ignores_cropped_and_labeled_folders(monkeypatch, project, collected_data, write_collected_data):
    """Derived output folders are not annotation sources."""
    project_path, cfg_path = project()
    labeled_data = project_path / "labeled-data"
    write_collected_data(
        labeled_data / "session1_labeled",
        collected_data("session1_labeled", [0.0, 0.0, 10.0, 0.0]),
    )
    write_collected_data(
        labeled_data / "session1cropped",
        collected_data("session1cropped", [0.0, 0.0, 10.0, 0.0]),
    )
    patch_builder_ui(monkeypatch)

    with pytest.raises(IOError, match="No labeled data were found") as excinfo:
        SkeletonBuilder(str(cfg_path))

    assert "0 folder(s) were inspected" in str(excinfo.value)
