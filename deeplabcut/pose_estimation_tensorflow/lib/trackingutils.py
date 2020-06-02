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

import numpy as np
from filterpy.common import kinematic_kf
from filterpy.kalman import KalmanFilter
from numba import jit
from scipy.optimize import linear_sum_assignment


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


class KalmanBoxTracker(object):
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
