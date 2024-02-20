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
import numpy as np
import pytest
from deeplabcut.core import trackingutils


@pytest.fixture()
def ellipse():
    params = {"x": 0, "y": 0, "width": 2, "height": 4, "theta": np.pi / 2}
    return trackingutils.Ellipse(**params)


def test_ellipse(ellipse):
    assert ellipse.aspect_ratio == 2
    np.testing.assert_equal(
        ellipse.contains_points(np.asarray([[0, 0], [10, 10]])), [True, False]
    )


def test_ellipse_similarity(ellipse):
    assert ellipse.calc_similarity_with(ellipse) == 1


def test_ellipse_fitter():
    fitter = trackingutils.EllipseFitter()
    assert fitter.fit(np.random.rand(2, 2)) is None
    xy = np.asarray([[-2, 0], [2, 0], [0, 1], [0, -1]], dtype=float)
    assert fitter.fit(xy) is not None
    fitter.sd = 0
    el = fitter.fit(xy)
    assert np.isclose(el.parameters, [0, 0, 4, 2, 0]).all()


def test_ellipse_tracker(ellipse):
    tracker1 = trackingutils.EllipseTracker(ellipse.parameters)
    tracker2 = trackingutils.EllipseTracker(ellipse.parameters)
    assert tracker1.id != tracker2.id
    tracker1.update(ellipse.parameters)
    assert tracker1.hit_streak == 1
    state = tracker1.predict()
    np.testing.assert_equal(ellipse.parameters, state)
    _ = tracker1.predict()
    assert tracker1.hit_streak == 0


def test_sort_ellipse():
    tracklets = dict()
    mot_tracker = trackingutils.SORTEllipse(1, 1, 0.6)
    poses = np.random.rand(2, 10, 3)
    trackers = mot_tracker.track(poses[..., :2])
    assert trackers.shape == (2, 7)
    trackingutils.fill_tracklets(tracklets, trackers, poses, imname=0)
    assert all(id_ in tracklets for id_ in trackers[:, -2])
    assert all(np.array_equal(tracklets[n][0], pose) for n, pose in enumerate(poses))


def test_tracking_ellipse(real_assemblies, real_tracklets):
    tracklets_ref = real_tracklets.copy()
    _ = tracklets_ref.pop("header", None)
    tracklets = dict()
    mot_tracker = trackingutils.SORTEllipse(1, 1, 0.6)
    for ind, assemblies in real_assemblies.items():
        animals = np.stack([ass.data for ass in assemblies])
        trackers = mot_tracker.track(animals[..., :2])
        trackingutils.fill_tracklets(tracklets, trackers, animals, ind)
    assert len(tracklets) == len(tracklets_ref)
    assert [len(tracklet) for tracklet in tracklets.values()] == [
        len(tracklet) for tracklet in tracklets_ref.values()
    ]
    assert all(
        t.shape[1] == 4 for tracklet in tracklets.values() for t in tracklet.values()
    )


def test_box_tracker():
    bbox = 0, 0, 100, 100
    tracker1 = trackingutils.BoxTracker(bbox)
    tracker2 = trackingutils.BoxTracker(bbox)
    assert tracker1.id != tracker2.id
    tracker1.update(bbox)
    assert tracker1.hit_streak == 1
    state = tracker1.predict()
    np.testing.assert_equal(bbox, state)
    _ = tracker1.predict()
    assert tracker1.hit_streak == 0


def test_tracking_box(real_assemblies, real_tracklets):
    tracklets_ref = real_tracklets.copy()
    _ = tracklets_ref.pop("header", None)
    tracklets = dict()
    mot_tracker = trackingutils.SORTBox(1, 1, 0.1)
    for ind, assemblies in real_assemblies.items():
        animals = np.stack([ass.data for ass in assemblies])
        bboxes = trackingutils.calc_bboxes_from_keypoints(animals)
        trackers = mot_tracker.track(bboxes)
        trackingutils.fill_tracklets(tracklets, trackers, animals, ind)
    assert len(tracklets) == len(tracklets_ref)
    assert [len(tracklet) for tracklet in tracklets.values()] == [
        len(tracklet) for tracklet in tracklets_ref.values()
    ]
    assert all(
        t.shape[1] == 4 for tracklet in tracklets.values() for t in tracklet.values()
    )


def test_tracking_montblanc(
    real_assemblies_montblanc,
    real_tracklets_montblanc,
):
    tracklets_ref = real_tracklets_montblanc.copy()
    _ = tracklets_ref.pop("header", None)
    tracklets = dict()
    tracklets["single"] = real_assemblies_montblanc[1]
    mot_tracker = trackingutils.SORTEllipse(1, 1, 0.6)
    for ind, assemblies in real_assemblies_montblanc[0].items():
        animals = np.stack([ass.data for ass in assemblies])
        trackers = mot_tracker.track(animals[..., :2])
        trackingutils.fill_tracklets(tracklets, trackers, animals, ind)
    assert len(tracklets) == len(tracklets_ref)
    assert [len(tracklet) for tracklet in tracklets.values()] == [
        len(tracklet) for tracklet in tracklets_ref.values()
    ]
    for k, assemblies in tracklets.items():
        ref = tracklets_ref[k]
        for ind, data in assemblies.items():
            frame = f"frame{str(ind).zfill(3)}" if k != "single" else ind
            np.testing.assert_equal(data, ref[frame])


def test_calc_bboxes_from_keypoints():
    # Test bounding box from a single keypoint
    xy = np.asarray([[[0, 0, 1]]])
    np.testing.assert_equal(
        trackingutils.calc_bboxes_from_keypoints(xy, 10), [[-10, -10, 10, 10, 1]]
    )
    np.testing.assert_equal(
        trackingutils.calc_bboxes_from_keypoints(xy, 20, 10), [[-10, -20, 30, 20, 1]]
    )

    width = 200
    height = width * 2
    xyp = np.zeros((1, 2, 3))
    xyp[:, 1, :2] = width, height
    xyp[:, 1, 2] = 1
    with pytest.raises(ValueError):
        _ = trackingutils.calc_bboxes_from_keypoints(xyp[..., :2])

    bboxes = trackingutils.calc_bboxes_from_keypoints(xyp)
    np.testing.assert_equal(bboxes, [[0, 0, width, height, 0.5]])

    slack = 20
    bboxes = trackingutils.calc_bboxes_from_keypoints(xyp, slack=slack)
    np.testing.assert_equal(
        bboxes, [[-slack, -slack, width + slack, height + slack, 0.5]]
    )

    offset = 50
    bboxes = trackingutils.calc_bboxes_from_keypoints(xyp, offset=offset)
    np.testing.assert_equal(bboxes, [[offset, 0, width + offset, height, 0.5]])
