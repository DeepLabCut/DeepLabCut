"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

class SORT code adapted from:

SORT: A Simple, Online and Realtime Tracker
Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

class SkeletonTracker developed for DLC 2.2.
"""
import math
import numpy as np
import warnings
from collections import defaultdict
from filterpy.common import kinematic_kf
from filterpy.kalman import KalmanFilter
from matplotlib import patches
from numba import jit
from numba.core.errors import NumbaPerformanceWarning
from scipy.optimize import linear_sum_assignment
from scipy.stats import mode
from tqdm import tqdm


warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


@jit
def iou(bb_test, bb_gt):
    """
    Computes intersection of union (IOU) metric for pair of bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
        - wh
    )
    return o


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        ).reshape((1, 4))
    else:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[
            4:, 4:
        ] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


class Ellipse:
    def __init__(self, x, y, width, height, theta):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.theta = theta  # in radians
        self._geometry = None

    @property
    def parameters(self):
        return self.x, self.y, self.width, self.height, self.theta

    @property
    def aspect_ratio(self):
        return max(self.width, self.height) / min(self.width, self.height)

    def calc_similarity_with(self, other_ellipse):
        max_dist = max(
            self.height, self.width, other_ellipse.height, other_ellipse.width
        )
        dist = math.sqrt(
            (self.x - other_ellipse.x) ** 2 + (self.y - other_ellipse.y) ** 2
        )
        cost1 = 1 - min(dist / max_dist, 1)
        cost2 = abs(math.cos(self.theta - other_ellipse.theta))
        return 0.8 * cost1 + 0.2 * cost2 * cost1

    def contains_points(self, xy, tol=0.1):
        ca = math.cos(self.theta)
        sa = math.sin(self.theta)
        x_demean = xy[:, 0] - self.x
        y_demean = xy[:, 1] - self.y
        return (
            ((ca * x_demean + sa * y_demean) ** 2 / (0.5 * self.width) ** 2)
            + ((sa * x_demean - ca * y_demean) ** 2 / (0.5 * self.height) ** 2)
        ) <= 1 + tol

    def draw(self, show_axes=True, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.transforms import Affine2D

        if ax is None:
            ax = plt.subplot(111, aspect="equal")
        el = patches.Ellipse(
            xy=(self.x, self.y),
            width=self.width,
            height=self.height,
            angle=np.rad2deg(self.theta),
            **kwargs,
        )
        ax.add_patch(el)
        if show_axes:
            major = Line2D([-self.width / 2, self.width / 2], [0, 0], lw=3, zorder=3)
            minor = Line2D([0, 0], [-self.height / 2, self.height / 2], lw=3, zorder=3)
            trans = (
                Affine2D().rotate(self.theta).translate(self.x, self.y) + ax.transData
            )
            major.set_transform(trans)
            minor.set_transform(trans)
            ax.add_artist(major)
            ax.add_artist(minor)


class EllipseFitter:
    def __init__(self, sd=2):
        self.sd = sd
        self.x = None
        self.y = None
        self.params = None
        self._coeffs = None

    def fit(self, xy):
        self.x, self.y = xy[np.isfinite(xy).all(axis=1)].T
        if len(self.x) < 3:
            return None
        if self.sd:
            self.params = self._fit_error(self.x, self.y, self.sd)
        else:
            self._coeffs = self._fit(self.x, self.y)
            self.params = self.calc_parameters(self._coeffs)
        if not np.isnan(self.params).any():
            el = Ellipse(*self.params)
            # Regularize by forcing AR <= 5
            # max_ar = 5
            # if el.aspect_ratio >= max_ar:
            #     if el.height > el.width:
            #         el.width = el.height / max_ar
            #     else:
            #         el.height = el.width / max_ar
            # Orient the ellipse such that it encompasses most points
            # n_inside = el.contains_points(np.c_[self.x, self.y]).sum()
            # el.theta += 0.5 * np.pi
            # if el.contains_points(np.c_[self.x, self.y]).sum() < n_inside:
            #     el.theta -= 0.5 * np.pi
            return el
        return None

    @staticmethod
    @jit(nopython=True)
    def _fit(x, y):
        """
        Least Squares ellipse fitting algorithm
        Fit an ellipse to a set of X- and Y-coordinates.
        See Halir and Flusser, 1998 for implementation details

        :param x: ndarray, 1D trajectory
        :param y: ndarray, 1D trajectory
        :return: 1D ndarray of 6 coefficients of the general quadratic curve:
            ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0
        """
        D1 = np.vstack((x * x, x * y, y * y))
        D2 = np.vstack((x, y, np.ones_like(x)))
        S1 = D1 @ D1.T
        S2 = D1 @ D2.T
        S3 = D2 @ D2.T
        T = -np.linalg.inv(S3) @ S2.T
        temp = S1 + S2 @ T
        M = np.zeros_like(temp)
        M[0] = temp[2] * 0.5
        M[1] = -temp[1]
        M[2] = temp[0] * 0.5
        E, V = np.linalg.eig(M)
        cond = 4 * V[0] * V[2] - V[1] ** 2
        a1 = V[:, cond > 0][:, 0]
        a2 = T @ a1
        return np.hstack((a1, a2))

    @staticmethod
    @jit(nopython=True)
    def _fit_error(x, y, sd):
        """
        Fit a sd-sigma covariance error ellipse to the data.

        :param x: ndarray, 1D input of X coordinates
        :param y: ndarray, 1D input of Y coordinates
        :param sd: int, size of the error ellipse in 'standard deviation'
        :return: ellipse center, semi-axes length, angle to the X-axis
        """
        cov = np.cov(x, y)
        E, V = np.linalg.eigh(cov)  # Returns the eigenvalues in ascending order
        # r2 = chi2.ppf(2 * norm.cdf(sd) - 1, 2)
        # height, width = np.sqrt(E * r2)
        height, width = 2 * sd * np.sqrt(E)
        a, b = V[:, 1]
        rotation = math.atan2(b, a) % np.pi
        return [np.mean(x), np.mean(y), width, height, rotation]

    @staticmethod
    @jit(nopython=True)
    def calc_parameters(coeffs):
        """
        Calculate ellipse center coordinates, semi-axes lengths, and
        the counterclockwise angle of rotation from the x-axis to the ellipse major axis.
        Visit http://mathworld.wolfram.com/Ellipse.html
        for how to estimate ellipse parameters.

        :param coeffs: list of fitting coefficients
        :return: center: 1D ndarray, semi-axes: 1D ndarray, angle: float
        """
        # The general quadratic curve has the form:
        # ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0
        a, b, c, d, f, g = coeffs
        b *= 0.5
        d *= 0.5
        f *= 0.5

        # Ellipse center coordinates
        x0 = (c * d - b * f) / (b * b - a * c)
        y0 = (a * f - b * d) / (b * b - a * c)

        # Semi-axes lengths
        num = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
        den1 = (b * b - a * c) * (np.sqrt((a - c) ** 2 + 4 * b * b) - (a + c))
        den2 = (b * b - a * c) * (-np.sqrt((a - c) ** 2 + 4 * b * b) - (a + c))
        major = np.sqrt(num / den1)
        minor = np.sqrt(num / den2)

        # Angle to the horizontal
        if b == 0:
            if a < c:
                phi = 0
            else:
                phi = np.pi / 2
        else:
            if a < c:
                phi = np.arctan(2 * b / (a - c)) / 2
            else:
                phi = np.pi / 2 + np.arctan(2 * b / (a - c)) / 2

        return [x0, y0, 2 * major, 2 * minor, phi]


class EllipseTracker:
    n_trackers = 0

    def __init__(self, params):
        self.kf = kinematic_kf(5, order=1, dim_z=5, order_by_dim=False)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[
            5:, 5:
        ] *= 1000.0  # High uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0
        self.kf.Q[5:, 5:] *= 0.01
        self.state = params
        self.time_since_update = 0
        self.id = EllipseTracker.n_trackers
        EllipseTracker.n_trackers += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, ellipse):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(ellipse)

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.state

    @property
    def state(self):
        return self.kf.x.squeeze()[:5]

    @state.setter
    def state(self, params):
        self.kf.x[:5] = np.asarray(params).reshape((-1, 1))


class SORTEllipse:
    def __init__(self, max_age, min_hits, iou_threshold, sd=2):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.fitter = EllipseFitter(sd)
        self.n_frames = 0
        self.trackers = []
        # Reset tracker IDs
        EllipseTracker.n_trackers = 0

    def track(self, poses, identities=None):
        self.n_frames += 1

        trackers = np.zeros((len(self.trackers), 6))
        for i in range(len(trackers)):
            trackers[i, :5] = self.trackers[i].predict()
        empty = np.isnan(trackers).any(axis=1)
        trackers = trackers[~empty]
        for ind in np.flatnonzero(empty)[::-1]:
            self.trackers.pop(ind)

        ellipses = []
        pred_ids = []
        for i, pose in enumerate(poses):
            el = self.fitter.fit(pose)
            if el is not None:
                ellipses.append(el)
                if identities is not None:
                    pred_ids.append(mode(identities[i])[0][0])
        if not len(trackers):
            matches = np.empty((0, 2), dtype=int)
            unmatched_detections = np.arange(len(ellipses))
            unmatched_trackers = np.empty((0, 6), dtype=int)
        else:
            ellipses_trackers = [Ellipse(*t[:5]) for t in trackers]
            cost_matrix = np.zeros((len(ellipses), len(ellipses_trackers)))
            for i, el in enumerate(ellipses):
                for j, el_track in enumerate(ellipses_trackers):
                    cost = el.calc_similarity_with(el_track)
                    if identities is not None:
                        match = 2 if pred_ids[i] == self.trackers[j].id_ else 1
                        cost *= match
                    cost_matrix[i, j] = cost
            row_indices, col_indices = linear_sum_assignment(cost_matrix, maximize=True)
            unmatched_detections = [
                i for i, _ in enumerate(ellipses) if i not in row_indices
            ]
            unmatched_trackers = [
                j for j, _ in enumerate(trackers) if j not in col_indices
            ]
            matches = []
            for row, col in zip(row_indices, col_indices):
                val = cost_matrix[row, col]
                # diff = val - cost_matrix
                # diff[row, col] += val
                # if (
                #         val < self.iou_threshold
                #         or np.any(diff[row] <= 0.2)
                #         or np.any(diff[:, col] <= 0.2)
                # ):
                if val < self.iou_threshold:
                    unmatched_detections.append(row)
                    unmatched_trackers.append(col)
                else:
                    matches.append([row, col])
            if not len(matches):
                matches = np.empty((0, 2), dtype=int)
            else:
                matches = np.stack(matches)
            unmatched_trackers = np.asarray(unmatched_trackers)
            unmatched_detections = np.asarray(unmatched_detections)

        animalindex = []
        for t, tracker in enumerate(self.trackers):
            if t not in unmatched_trackers:
                ind = matches[matches[:, 1] == t, 0][0]
                animalindex.append(ind)
                tracker.update(ellipses[ind].parameters)
            else:
                animalindex.append(-1)

        for i in unmatched_detections:
            trk = EllipseTracker(ellipses[i].parameters)
            if identities is not None:
                trk.id_ = mode(identities[i])[0][0]
            self.trackers.append(trk)
            animalindex.append(i)

        i = len(self.trackers)
        ret = []
        for trk in reversed(self.trackers):
            d = trk.state
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.n_frames <= self.min_hits
            ):
                ret.append(
                    np.concatenate((d, [trk.id, int(animalindex[i - 1])])).reshape(
                        1, -1
                    )
                )  # for DLC we also return the original animalid
                # +1 as MOT benchmark requires positive >> this is removed for DLC!
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))


class SkeletonTracker:
    n_trackers = 0

    def __init__(self, n_bodyparts):
        # TODO Try particle filter (since we already have the keypoints)
        self.kf = kinematic_kf(
            n_bodyparts * 2, order=1, dim_z=n_bodyparts, order_by_dim=False
        )
        self.kf.Q[self.kf.dim_z :, self.kf.dim_z :] *= 10
        self.kf.R[self.kf.dim_z :, self.kf.dim_z :] *= 0.01
        self.kf.P[self.kf.dim_z :, self.kf.dim_z :] *= 1000
        self.id = SkeletonTracker.n_trackers
        SkeletonTracker.n_trackers += 1
        self.time_since_update = 0
        self.age = 0
        self.hits = 0
        self.hit_streak = 0

    def update(self, pose):
        flat = pose.reshape((-1, 1))
        empty = np.isnan(flat).squeeze()
        if empty.any():
            H = self.kf.H.copy()
            H[empty] = 0
            flat[empty] = 0
            self.kf.update(flat, H=H)
        else:
            self.kf.update(flat)
            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.state

    @property
    def empty_state(self):
        return np.full(self.kf.dim_z, np.nan)

    @property
    def state(self):
        return self.kf.x.squeeze()[: self.kf.dim_z]

    @state.setter
    def state(self, pose):
        curr_pose = pose.copy()
        empty = np.isnan(curr_pose).all(axis=1)
        if empty.any():
            fill = np.nanmean(pose, axis=0)
            curr_pose[empty] = fill
        self.kf.x[: self.kf.dim_z] = curr_pose.reshape((-1, 1))


class SORT:
    def __init__(self, n_bodyparts, max_age=20, min_hits=3, oks_threshold=0.5):
        self.n_bodyparts = n_bodyparts
        self.max_age = max_age
        self.min_hits = min_hits
        self.oks_threshold = oks_threshold
        self.trackers = []
        self.frame_count = 0

    @staticmethod
    def weighted_hausdorff(x, y):
        # Modified from scipy source code:
        # - to restrict its use to 2D
        # - to get rid of shuffling (since arrays are only (nbodyparts * 3) element long)
        # TODO - factor in keypoint confidence (and weight by # of observations??)
        cmax = 0
        for i in range(x.shape[0]):
            no_break_occurred = True
            cmin = np.inf
            for j in range(y.shape[0]):
                d = (x[i, 0] - y[j, 0]) ** 2 + (x[i, 1] - y[j, 1]) ** 2
                if d < cmax:
                    no_break_occurred = False
                    break
                if d < cmin:
                    cmin = d
            if cmin != np.inf and cmin > cmax and no_break_occurred:
                cmax = cmin
        return np.sqrt(cmax)

    @staticmethod
    def object_keypoint_similarity(x, y):
        mask = ~np.isnan(x * y).all(axis=1)  # Intersection visible keypoints
        xx = x[mask]
        yy = y[mask]
        dist = np.linalg.norm(xx - yy, axis=1)
        scale = np.sqrt(
            np.product(np.ptp(yy, axis=0))
        )  # square root of bounding box area
        oks = np.exp(-0.5 * (dist / (0.05 * scale)) ** 2)
        return np.mean(oks)

    def calc_pairwise_hausdorff_dist(self, poses, poses_ref):
        mat = np.zeros((len(poses), len(poses_ref)))
        for i, pose in enumerate(poses):
            for j, pose_ref in enumerate(poses_ref):
                mat[i, j] = self.weighted_hausdorff(pose, pose_ref)
        return mat

    def calc_pairwise_oks(self, poses, poses_ref):
        mat = np.zeros((len(poses), len(poses_ref)))
        for i, pose in enumerate(poses):
            for j, pose_ref in enumerate(poses_ref):
                mat[i, j] = self.object_keypoint_similarity(pose, pose_ref)
        return mat

    def track(self, poses):
        self.frame_count += 1

        if not len(self.trackers):
            for pose in poses:
                tracker = SkeletonTracker(self.n_bodyparts)
                tracker.state = pose
                self.trackers.append(tracker)

        poses_ref = []
        for i, tracker in enumerate(self.trackers):
            pose_ref = tracker.predict()
            poses_ref.append(pose_ref.reshape((-1, 2)))

        # mat = self.calc_pairwise_oks(poses, poses_ref)
        mat = self.calc_pairwise_hausdorff_dist(poses, poses_ref)
        row_indices, col_indices = linear_sum_assignment(mat, maximize=False)

        unmatched_poses = [p for p, _ in enumerate(poses) if p not in row_indices]
        unmatched_trackers = [
            t for t, _ in enumerate(poses_ref) if t not in col_indices
        ]
        # Remove matched detections with low OKS
        # matches = []
        # for row, col in zip(row_indices, col_indices):
        #     if mat[row, col] < self.oks_threshold:
        #         unmatched_poses.append(row)
        #         unmatched_trackers.append(col)
        #     else:
        #         matches.append([row, col])
        # if not len(matches):
        #     matches = np.empty((0, 2), dtype=int)
        # else:
        #     matches = np.stack(matches)
        matches = np.c_[row_indices, col_indices]

        animalindex = []
        for t, tracker in enumerate(self.trackers):
            if t not in unmatched_trackers:
                ind = matches[matches[:, 1] == t, 0][0]
                animalindex.append(ind)
                tracker.update(poses[ind])
            else:
                animalindex.append(-1)

        for i in unmatched_poses:
            tracker = SkeletonTracker(self.n_bodyparts)
            tracker.state = poses[i]
            self.trackers.append(tracker)
            animalindex.append(i)

        states = []
        i = len(self.trackers)
        for tracker in reversed(self.trackers):
            i -= 1
            if tracker.time_since_update > self.max_age:
                self.trackers.pop()
                continue
            state = tracker.predict()
            states.append(np.r_[state, [tracker.id, int(animalindex[i])]])
        if len(states) > 0:
            return np.stack(states)
        return np.empty((0, self.n_bodyparts * 2 + 2))


def associate_detections_to_trackers(detections, trackers, iou_threshold):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if not len(trackers):
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    row_indices, col_indices = linear_sum_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in row_indices:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in col_indices:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for row, col in zip(row_indices, col_indices):
        if iou_matrix[row, col] < iou_threshold:
            unmatched_detections.append(row)
            unmatched_trackers.append(col)
        else:
            matches.append([row, col])
    if not len(matches):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.stack(matches)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort:
    def __init__(self, cfg):
        """
        Sets key parameters for SORT
        """
        self.max_age = cfg.get("max_age", 1)
        self.min_hits = cfg.get("min_hits", 3)
        self.trackers = []
        self.frame_count = 0
        self.iou_threshold = cfg.get("iou_threshold", 0.3)
        KalmanBoxTracker.count = 0

    def update(self, dets):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))

        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        # update matched trackers with assigned detections
        animalindex = []
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                animalindex.append(d[0])
                trk.update(dets[d, :][0])  # update coordinates
            else:
                animalindex.append("nix")  # lost trk!

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
            animalindex.append(i)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(
                    np.concatenate((d, [trk.id, int(animalindex[i - 1])])).reshape(
                        1, -1
                    )
                )  # for DLC we also return the original animalid
                # +1 as MOT benchmark requires positive >> this is removed for DLC!
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


def fill_tracklets(tracklets, trackers, animals, imname):
    for content in trackers:
        tracklet_id, pred_id = content[-2:].astype(np.int)
        if tracklet_id not in tracklets:
            tracklets[tracklet_id] = {}
        if pred_id != -1:
            tracklets[tracklet_id][imname] = animals[pred_id]
        else:  # Resort to the tracker prediction
            xy = np.asarray(content[:-2])
            pred = np.insert(xy, range(2, len(xy) + 1, 2), 1)
            tracklets[tracklet_id][imname] = pred


def calc_bboxes_from_keypoints(data, slack=0, offset=0):
    data = np.asarray(data)
    if data.shape[-1] < 3:
        raise ValueError("Data should be of shape (n_animals, n_bodyparts, 3)")

    if data.ndim != 3:
        data = np.expand_dims(data, axis=0)
    bboxes = np.full((data.shape[0], 5), np.nan)
    bboxes[:, :2] = np.nanmin(data[..., :2], axis=1) - slack  # X1, Y1
    bboxes[:, 2:4] = np.nanmax(data[..., :2], axis=1) + slack  # X2, Y2
    bboxes[:, -1] = np.nanmean(data[..., 2])  # Average confidence
    bboxes[:, [0, 2]] += offset
    return bboxes


def reconstruct_all_ellipses(data, sd):
    xy = data.droplevel("scorer", axis=1).drop("likelihood", axis=1, level=-1)
    if "single" in xy:
        xy.drop("single", axis=1, level="individuals", inplace=True)
    animals = xy.columns.get_level_values("individuals").unique()
    nrows = xy.shape[0]
    ellipses = np.full((len(animals), nrows, 5), np.nan)
    fitter = EllipseFitter(sd)
    for n, animal in enumerate(animals):
        data = xy.xs(animal, axis=1, level="individuals").values.reshape((nrows, -1, 2))
        for i, coords in enumerate(tqdm(data)):
            el = fitter.fit(coords.astype(np.float64))
            if el is not None:
                ellipses[n, i] = el.parameters
    return ellipses


def _track_individuals(
    individuals, min_hits=1, max_age=5, similarity_threshold=0.6, track_method="ellipse"
):
    if track_method not in ("box", "skeleton", "ellipse"):
        raise ValueError(f"Unknown {track_method} tracker.")

    if track_method == "ellipse":
        tracker = SORTEllipse(max_age, min_hits, similarity_threshold)
    elif track_method == "box":
        tracker = Sort(
            {
                "max_age": max_age,
                "min_hits": min_hits,
                "iou_threshold": similarity_threshold,
            }
        )
    else:
        n_bodyparts = individuals[0][0].shape[0]
        tracker = SORT(n_bodyparts, max_age, min_hits, similarity_threshold)

    tracklets = defaultdict(dict)
    all_hyps = dict()
    for i, (multi, single) in enumerate(tqdm(individuals)):
        if single is not None:
            tracklets["single"][i] = single
        if multi is None:
            continue
        if track_method == "box":
            # TODO: get cropping parameters and utilize!
            bboxes = calc_bboxes_from_keypoints(multi)
            hyps = tracker.update(bboxes)
        else:
            xy = multi[..., :2]
            hyps = tracker.track(xy)
        all_hyps[i] = hyps
        for hyp in hyps:
            tracklet_id, pred_id = hyp[-2:].astype(int)
            if pred_id != -1:
                tracklets[tracklet_id][i] = multi[pred_id]
    return tracklets, all_hyps
