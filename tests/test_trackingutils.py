import os

os.environ["DLClight"] = "True"
import numpy as np
import pytest
from deeplabcut.pose_estimation_tensorflow.lib import trackingutils


@pytest.fixture()
def ellipse():
    params = {
        'x': 0,
        'y': 0,
        'width': 2,
        'height': 4,
        'theta': np.pi / 2,
    }
    return trackingutils.Ellipse(**params)


def test_ellipse(ellipse):
    assert ellipse.aspect_ratio == 2
    assert ellipse.geometry is not None
    np.testing.assert_equal(
        ellipse.contains_points(np.asarray([[0, 0], [10, 10]])),
        [True, False],
    )


def test_ellipse_similarity(ellipse):
    assert ellipse.calc_iou_with(ellipse) == 1
    assert ellipse.calc_similarity_with(ellipse) == 1


def test_ellipse_fitter():
    fitter = trackingutils.EllipseFitter()
    assert fitter.fit(np.random.rand(2, 2)) is None
    xy = np.asarray([[-2, 0], [2, 0], [0, 1], [0, -1]], dtype=np.float)
    assert fitter.fit(xy) is not None
    fitter.sd = 0
    el = fitter.fit(xy)
    assert np.isclose(el.parameters, [0, 0, 4, 2, 0]).all()


def test_ellipse_tracker(ellipse):
    tracker1 = trackingutils.EllipseTracker(ellipse.parameters)
    assert tracker1.id == 0
    tracker2 = trackingutils.EllipseTracker(ellipse.parameters)
    assert tracker2.id == 1
    tracker1.update(ellipse.parameters)
    assert tracker1.hit_streak == 1
    state = tracker1.predict()
    np.testing.assert_equal(ellipse.parameters, state)
    _ = tracker1.predict()
    assert tracker1.hit_streak == 0


def test_sort_ellipse():
    tracklets = dict()
    mot = trackingutils.SORTEllipse(1, 1, 0.6)
    poses = np.random.rand(2, 10, 3)
    trackers = mot.track(poses[..., :2])
    assert trackers.shape == (2, 7)
    trackingutils.fill_tracklets(
        tracklets,
        trackers,
        poses,
        imname=0
    )
    assert all(id_ in tracklets for id_ in trackers[:, -2])


def test_calc_bboxes_from_keypoints():
    xy = np.asarray([[[0, 0, 1]]])
    np.testing.assert_equal(
        trackingutils.calc_bboxes_from_keypoints(xy, 10),
        [[-10, -10, 10, 10, 1]],
    )
    np.testing.assert_equal(
        trackingutils.calc_bboxes_from_keypoints(xy, 20, 10),
        [[-10, -20, 30, 20, 1]],
    )
