import os

os.environ["DLClight"] = "True"
import numpy as np
import pytest
from deeplabcut.refine_training_dataset.stitch import Tracklet, TrackletStitcher


TRACKLET_LEN = 1000
TRACKLET_START = 50
TRACKLET_ID = 0
N_DETS = 5
N_TRACKLETS = 20


@pytest.fixture()
def fake_tracklet():
    inds = np.arange(TRACKLET_START, TRACKLET_START + TRACKLET_LEN)
    data = np.empty((TRACKLET_LEN, N_DETS, 4))
    data[..., :2] = np.arange(N_DETS).reshape(-1, 1) * [1, 1]
    data[..., 2] = 1
    data[..., 3] = TRACKLET_ID
    return Tracklet(data, inds)


@pytest.fixture()
def fake_stitcher():
    inds = np.arange(TRACKLET_LEN)
    data = np.random.rand(inds.size, N_DETS, 3)
    track = Tracklet(data, inds)
    idx = np.linspace(0, inds.size, N_TRACKLETS + 1, dtype=int)
    tracklets = TrackletStitcher.split_tracklet(track, idx[1:-1])
    return TrackletStitcher(
        tracklets,
        n_tracks=2,
    )


def test_tracklet_wrong_inputs(fake_tracklet):
    with pytest.raises(ValueError):
        _ = Tracklet(fake_tracklet.data[..., :2], fake_tracklet.inds)
        _ = Tracklet(fake_tracklet.data[:TRACKLET_LEN - 2], fake_tracklet.inds)


def test_tracklet_monotonic_indices(fake_tracklet):
    tracklet_inv = Tracklet(
        fake_tracklet.data[::-1], fake_tracklet.inds[::-1]
    )
    np.testing.assert_equal(fake_tracklet.inds, tracklet_inv.inds)
    np.testing.assert_equal(fake_tracklet.xy, tracklet_inv.xy)


def test_tracklet(fake_tracklet):
    assert len(fake_tracklet) == TRACKLET_LEN
    assert fake_tracklet.likelihood == 1
    assert fake_tracklet.identity == TRACKLET_ID
    assert fake_tracklet.start == TRACKLET_START
    assert fake_tracklet.end == TRACKLET_START + TRACKLET_LEN - 1
    np.testing.assert_equal(
        fake_tracklet.centroid,
        np.full((TRACKLET_LEN, 2), np.arange(N_DETS).mean())
    )
    fake_tracklet2 = Tracklet(
        fake_tracklet.data,
        fake_tracklet.inds + TRACKLET_LEN
    )
    assert fake_tracklet not in fake_tracklet2
    tracklet = fake_tracklet + fake_tracklet2
    tracklet -= fake_tracklet
    np.testing.assert_equal(tracklet.data, fake_tracklet2.data)
    np.testing.assert_equal(tracklet.inds, fake_tracklet2.inds)
    tracklet2 = fake_tracklet + fake_tracklet
    assert tracklet2.contains_duplicates()


def test_tracklet_default_identity(fake_tracklet):
    fake_tracklet.data = fake_tracklet.data[..., :3]
    assert fake_tracklet.identity == -1


def test_tracklet_data_access(fake_tracklet):
    np.testing.assert_equal(
        fake_tracklet.get_data_at(TRACKLET_START), fake_tracklet.data[0]
    )
    fake_tracklet.set_data_at(TRACKLET_START + 1, fake_tracklet.data[0] * 2)
    np.testing.assert_equal(
        fake_tracklet.data[1], fake_tracklet.data[0] * 2
    )
    fake_tracklet.del_data_at(TRACKLET_START + 1)
    assert not fake_tracklet.is_continuous
    assert TRACKLET_START + 1 not in fake_tracklet.inds


@pytest.mark.parametrize(
    "where, norm",
    [("head", False), ("tail", True)]
)
def test_tracklet_calc_velocity(fake_tracklet, where, norm):
    _ = fake_tracklet.calc_velocity(where, norm)


def test_tracklet_affinities(fake_tracklet):
    other_tracklet = Tracklet(
        fake_tracklet.data,
        fake_tracklet.inds + TRACKLET_LEN
    )
    _ = fake_tracklet.dynamic_similarity_with(other_tracklet)
    _ = fake_tracklet.dynamic_dissimilarity_with(other_tracklet)
    _ = fake_tracklet.shape_dissimilarity_with(other_tracklet)
    _ = fake_tracklet.box_overlap_with(other_tracklet)
    _ = fake_tracklet.motion_affinity_with(other_tracklet)
    _ = fake_tracklet.distance_to(other_tracklet)


def test_stitcher_wrong_inputs(fake_tracklet):
    with pytest.raises(IOError):
        _ = TrackletStitcher([], n_tracks=2)
        _ = TrackletStitcher([fake_tracklet], n_tracks=1)
        _ = TrackletStitcher([fake_tracklet], n_tracks=2, min_length=2)


def test_purify_tracklets(fake_tracklet):
    fake_tracklet.data = np.full_like(fake_tracklet.data, np.nan)
    assert TrackletStitcher.purify_tracklet(fake_tracklet) is None
    fake_tracklet.data[0] = 1
    tracklet = TrackletStitcher.purify_tracklet(fake_tracklet)
    assert len(tracklet) == 1
    assert tracklet.inds == fake_tracklet.inds[0]


def test_stitcher(tmpdir_factory, fake_stitcher):
    assert len(fake_stitcher) == N_TRACKLETS
    assert fake_stitcher.n_frames == TRACKLET_LEN
    assert fake_stitcher.compute_max_gap() == 1
    fake_stitcher.build_graph(max_gap=1)
    fake_stitcher.stitch(add_back_residuals=True)
    output_name = tmpdir_factory.mktemp('data').join('fake.h5')
    fake_stitcher.write_tracks(output_name)
