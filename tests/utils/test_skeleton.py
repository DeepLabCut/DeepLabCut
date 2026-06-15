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
    builder.fig.canvas.draw_idle = lambda: None


# ---------------------------------------------------------------------
# pick_labeled_frame
# ---------------------------------------------------------------------


def test_pick_labeled_frame_multi_animal_drops_single(monkeypatch):
    builder = make_test_builder()

    index = pd.MultiIndex.from_tuples(
        [("labeled-data/session1", "img001.png")],
        names=["folder", "image"],
    )
    columns = pd.MultiIndex.from_product(
        [["TestScorer"], ["single", "mouseA"], ["nose", "tail"], ["x", "y"]],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )

    # "single" is fully labeled too, but should be dropped before choosing.
    row = [
        1.0,
        2.0,
        3.0,
        4.0,  # single
        10.0,
        20.0,
        30.0,
        40.0,  # mouseA
    ]
    builder.df = pd.DataFrame([row], index=index, columns=columns)

    monkeypatch.setattr(np.random, "shuffle", lambda x: None)

    picked_row, picked_col = builder.pick_labeled_frame()

    assert picked_row == ("labeled-data/session1", "img001.png")
    assert picked_col == "mouseA"


def test_pick_labeled_frame_without_individuals(monkeypatch):
    builder = make_test_builder()

    index = pd.MultiIndex.from_tuples(
        [("labeled-data/session1", "img001.png")],
        names=["folder", "image"],
    )
    columns = pd.MultiIndex.from_product(
        [["TestScorer"], ["nose", "tail"], ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )

    builder.df = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0]],
        index=index,
        columns=columns,
    )

    monkeypatch.setattr(np.random, "shuffle", lambda x: None)

    picked_row, picked_col = builder.pick_labeled_frame()

    assert picked_row == ("labeled-data/session1", "img001.png")
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


def test_export_sorts_pairs_and_warns_for_unconnected(monkeypatch):
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

    with pytest.warns(UserWarning, match="didn't connect all the bodyparts"):
        builder.export()

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


def test_init_loads_dataframe_image_and_existing_skeleton(tmp_path, monkeypatch):
    project_path = tmp_path / "project"
    labeled_data = project_path / "labeled-data" / "session1"
    labeled_data.mkdir(parents=True)

    cfg_path = project_path / "config.yaml"
    cfg = make_config(
        project_path=project_path,
        scorer="TestScorer",
        skeleton=[
            ["nose", "tail"],
            ["missing", "nose"],
        ],  # second pair should be ignored
    )
    write_config(cfg_path, cfg)

    index = pd.MultiIndex.from_tuples(
        [("labeled-data/session1", "img001.png")],
        names=["folder", "image"],
    )
    columns = pd.MultiIndex.from_product(
        [["TestScorer"], ["nose", "tail"], ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    df = pd.DataFrame(
        [[0.0, 0.0, 10.0, 0.0]],
        index=index,
        columns=columns,
    )
    h5_path = labeled_data / "CollectedData_TestScorer.h5"
    df.to_hdf(h5_path, key="df", mode="w")

    monkeypatch.setattr(skeleton_mod.io, "imread", lambda path: np.zeros((5, 5, 3), dtype=np.uint8))
    monkeypatch.setattr(SkeletonBuilder, "build_ui", lambda self: None)
    monkeypatch.setattr(SkeletonBuilder, "display", lambda self: None)
    monkeypatch.setattr(np.random, "shuffle", lambda x: None)

    builder = SkeletonBuilder(str(cfg_path))

    assert builder.config_path == str(cfg_path)
    assert list(builder.bpts) == ["nose", "tail"]
    assert builder.xy.shape == (2, 2)
    assert builder.image.shape == (5, 5, 3)
    assert builder.inds == {(0, 1)}
    assert ((0.0, 0.0), (10.0, 0.0)) in builder.segs


def test_init_raises_if_no_labeled_data_found(tmp_path, monkeypatch):
    project_path = tmp_path / "project"
    (project_path / "labeled-data").mkdir(parents=True)

    cfg_path = project_path / "config.yaml"
    cfg = make_config(project_path=project_path, scorer="TestScorer")
    write_config(cfg_path, cfg)

    monkeypatch.setattr(SkeletonBuilder, "build_ui", lambda self: None)
    monkeypatch.setattr(SkeletonBuilder, "display", lambda self: None)

    with pytest.raises(IOError, match="No labeled data were found"):
        SkeletonBuilder(str(cfg_path))
