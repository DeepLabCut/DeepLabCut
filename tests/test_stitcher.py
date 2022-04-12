import numpy as np
import pandas as pd
import pytest
from deeplabcut.refine_training_dataset.stitch import Tracklet, TrackletStitcher


TRACKLET_LEN = 1000
TRACKLET_START = 50
TRACKLET_ID = 0
N_DETS = 5
N_TRACKLETS = 20


def fake_tracklet():
    inds = np.arange(TRACKLET_START, TRACKLET_START + TRACKLET_LEN)
    data = np.empty((TRACKLET_LEN, N_DETS, 4))
    data[..., :2] = np.arange(N_DETS).reshape(-1, 1) * [1, 1]
    data[..., 2] = 1
    data[..., 3] = TRACKLET_ID
    return Tracklet(data, inds)


def make_fake_tracklets():
    tracklet = fake_tracklet()
    tracklet_single = Tracklet(tracklet.data[:, :1], tracklet.inds)
    return tracklet, tracklet_single


@pytest.fixture()
def fake_stitcher():
    inds = np.arange(TRACKLET_LEN)
    data = np.random.rand(inds.size, N_DETS, 3)
    track = Tracklet(data, inds)
    idx = np.linspace(0, inds.size, N_TRACKLETS + 1, dtype=int)
    tracklets = TrackletStitcher.split_tracklet(track, idx[1:-1])
    return TrackletStitcher(tracklets, n_tracks=2)


@pytest.mark.parametrize("tracklet", make_fake_tracklets())
def test_tracklet_wrong_inputs(tracklet):
    with pytest.raises(ValueError):
        _ = Tracklet(tracklet.data[..., :2], tracklet.inds)
    with pytest.raises(ValueError):
        _ = Tracklet(tracklet.data[: TRACKLET_LEN - 2], tracklet.inds)


@pytest.mark.parametrize("tracklet", make_fake_tracklets())
def test_tracklet_monotonic_indices(tracklet):
    tracklet_inv = Tracklet(tracklet.data[::-1], tracklet.inds[::-1])
    np.testing.assert_equal(tracklet.inds, tracklet_inv.inds)
    np.testing.assert_equal(tracklet.xy, tracklet_inv.xy)


@pytest.mark.parametrize("tracklet", make_fake_tracklets())
def test_tracklet(tracklet):
    assert len(tracklet) == TRACKLET_LEN
    assert tracklet.likelihood == 1
    assert tracklet.identity == TRACKLET_ID
    assert tracklet.start == TRACKLET_START
    assert tracklet.end == TRACKLET_START + TRACKLET_LEN - 1
    np.testing.assert_equal(
        tracklet.centroid,
        np.full((TRACKLET_LEN, 2), np.arange(tracklet.data.shape[1]).mean()),
    )
    tracklet2 = Tracklet(tracklet.data, tracklet.inds + TRACKLET_LEN)
    assert tracklet not in tracklet2
    tracklet_new = tracklet + tracklet2
    tracklet_new -= tracklet
    np.testing.assert_equal(tracklet_new.data, tracklet2.data)
    np.testing.assert_equal(tracklet_new.inds, tracklet2.inds)
    tracklet2 = tracklet + tracklet
    assert tracklet2.contains_duplicates()


@pytest.mark.parametrize("tracklet", make_fake_tracklets())
def test_tracklet_default_identity(tracklet):
    tracklet.data = tracklet.data[..., :3]
    assert tracklet.identity == -1


@pytest.mark.parametrize("tracklet", make_fake_tracklets())
def test_tracklet_data_access(tracklet):
    np.testing.assert_equal(tracklet.get_data_at(TRACKLET_START), tracklet.data[0])
    tracklet.set_data_at(TRACKLET_START + 1, tracklet.data[0] * 2)
    np.testing.assert_equal(tracklet.data[1], tracklet.data[0] * 2)
    tracklet.del_data_at(TRACKLET_START + 1)
    assert not tracklet.is_continuous
    assert TRACKLET_START + 1 not in tracklet.inds


@pytest.mark.parametrize(
    "tracklet, where, norm",
    list(zip(make_fake_tracklets(), ("head", "tail"), (False, True))),
)
def test_tracklet_calc_velocity(tracklet, where, norm):
    _ = tracklet.calc_velocity(where, norm)


@pytest.mark.parametrize("tracklet", make_fake_tracklets())
def test_tracklet_calc_rate_of_turn(tracklet):
    for where in ("head", "tail"):
        _ = tracklet.calc_rate_of_turn(where)


@pytest.mark.parametrize("tracklet", make_fake_tracklets())
def test_tracklet_affinities(tracklet):
    other_tracklet = Tracklet(tracklet.data, tracklet.inds + TRACKLET_LEN)
    _ = tracklet.dynamic_similarity_with(other_tracklet)
    _ = tracklet.dynamic_dissimilarity_with(other_tracklet)
    _ = tracklet.shape_dissimilarity_with(other_tracklet)
    _ = tracklet.box_overlap_with(other_tracklet)
    _ = tracklet.motion_affinity_with(other_tracklet)
    _ = tracklet.distance_to(other_tracklet)


@pytest.mark.parametrize("tracklet", make_fake_tracklets())
def test_stitcher_wrong_inputs(tracklet):
    with pytest.raises(IOError):
        _ = TrackletStitcher([], n_tracks=2)
    with pytest.raises(ValueError):
        _ = TrackletStitcher([tracklet], n_tracks=2, min_length=2)


@pytest.mark.parametrize("tracklet", make_fake_tracklets())
def test_purify_tracklets(tracklet):
    tracklet.data = np.full_like(tracklet.data, np.nan)
    assert TrackletStitcher.purify_tracklet(tracklet) is None
    tracklet.data[0] = 1
    tracklet_pure = TrackletStitcher.purify_tracklet(tracklet)
    assert len(tracklet_pure) == 1
    assert tracklet_pure.inds == tracklet.inds[0]


def test_stitcher(tmpdir_factory, fake_stitcher):
    assert len(fake_stitcher) == N_TRACKLETS
    assert fake_stitcher.n_frames == TRACKLET_LEN
    assert fake_stitcher.compute_max_gap(fake_stitcher.tracklets) == 1
    fake_stitcher.build_graph(max_gap=1)
    fake_stitcher.stitch(add_back_residuals=True)
    output_name = tmpdir_factory.mktemp("data").join("fake.h5")
    fake_stitcher.write_tracks(output_name)

    # Break the graph to test stitching failure
    fake_stitcher.G.remove_edge("source", "0in")
    with pytest.warns(UserWarning):
        fake_stitcher.stitch(add_back_residuals=True)


def test_stitcher_plot(fake_stitcher):
    fake_stitcher.build_graph(max_gap=1)
    fake_stitcher.draw_graph(with_weights=True)
    fake_stitcher.stitch(add_back_residuals=True)
    fake_stitcher.plot_tracklets()
    fake_stitcher.plot_paths()
    fake_stitcher.plot_tracks()


def test_tracklet_interpolate(real_tracklets):
    data = np.stack(list(real_tracklets[0].values()))[:10]
    inds = np.arange(len(data))
    gap = 2
    inds[len(inds) // 2 :] += gap
    tracklet = Tracklet(data, inds)
    assert len(tracklet) == len(data)
    new_tracklet = tracklet.interpolate(max_gap=1)
    assert len(new_tracklet) == len(data)
    new_tracklet = tracklet.interpolate(max_gap=gap)
    assert len(new_tracklet) == len(data) + gap
    missing_inds = list(set(range(inds.max())).difference(inds))
    assert np.all(new_tracklet.data[missing_inds, :, 2] == 0.5)


def test_stitcher_real(tmpdir_factory, real_tracklets):
    stitcher = TrackletStitcher.from_dict_of_dict(real_tracklets, n_tracks=3)
    assert len(stitcher) == 3
    assert all(tracklet.is_continuous for tracklet in stitcher.tracklets)
    assert all(tracklet.identity == -1 for tracklet in stitcher.tracklets)
    assert not stitcher.residuals
    assert stitcher.compute_max_gap(stitcher.tracklets) == 0

    stitcher.build_graph()
    assert stitcher.G.number_of_edges() == 9
    assert all(weight is None for *_, weight in stitcher.G.edges.data("weight"))

    stitcher.stitch()
    assert len(stitcher.tracks) == 3
    assert all(len(track) == 50 for track in stitcher.tracks)
    assert all(0.998 <= track.likelihood <= 1 for track in stitcher.tracks)

    output_name = tmpdir_factory.mktemp("data").join("fake.h5")
    stitcher.write_tracks(output_name, ["mickey", "minnie", "bianca"])


def test_stitcher_montblanc(real_tracklets_montblanc):
    stitcher = TrackletStitcher.from_dict_of_dict(real_tracklets_montblanc, n_tracks=3,)
    assert len(stitcher) == 5
    assert all(tracklet.is_continuous for tracklet in stitcher.tracklets)
    assert all(tracklet.identity == -1 for tracklet in stitcher.tracklets)
    assert len(stitcher.residuals) == 1
    assert len(stitcher.residuals[0]) == 2
    assert stitcher.compute_max_gap(stitcher.tracklets) == 5

    stitcher.build_graph()
    assert stitcher.G.number_of_edges() == 18
    weights = [w for *_, w in stitcher.G.edges.data("weight") if w]
    assert weights == [2453, 24498, 5428]

    stitcher.stitch()
    assert len(stitcher.tracks) == 3
    assert all(len(track) >= 176 for track in stitcher.tracks)
    assert all(0.996 <= track.likelihood <= 1 for track in stitcher.tracks)

    df_gt = pd.read_hdf("tests/data/montblanc_tracks.h5")
    df = stitcher.format_df()
    np.testing.assert_equal(df.to_numpy(), df_gt.to_numpy())


def test_stitcher_with_identity(real_tracklets):
    # Add fake IDs
    for i in range(3):
        tracklet = real_tracklets[i]
        for v in tracklet.values():
            v[:, -1] = i
    stitcher = TrackletStitcher.from_dict_of_dict(real_tracklets, n_tracks=3)
    tracklets = sorted(stitcher, key=lambda t: t.identity)
    assert all(tracklet.identity == i for i, tracklet in enumerate(tracklets))

    # Split all tracklets in half
    tracklets = [t for track in stitcher for t in stitcher.split_tracklet(track, [25])]
    stitcher = TrackletStitcher(tracklets, n_tracks=3)
    assert len(stitcher) == 6

    stitcher.build_graph()
    weight = stitcher.G.edges[("0out", "3in")]["weight"]

    def weight_func(t1, t2):
        w = 0.01 if t1.identity == t2.identity else 1
        return w * t1.distance_to(t2)

    stitcher.build_graph(weight_func=weight_func)
    assert stitcher.G.number_of_edges() == 27
    new_weight = stitcher.G.edges[("0out", "3in")]["weight"]
    assert new_weight == weight // 100

    stitcher.stitch()
    assert len(stitcher.tracks) == 3
    assert all(len(track) == 50 for track in stitcher.tracks)
    assert all(0.998 <= track.likelihood <= 1 for track in stitcher.tracks)
    tracks = sorted(stitcher.tracks, key=lambda t: t.identity)
    assert all(track.identity == i for i, track in enumerate(tracks))
