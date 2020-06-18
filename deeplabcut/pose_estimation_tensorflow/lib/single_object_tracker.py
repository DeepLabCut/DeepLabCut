"""
DeepLabCut 2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

This module was contributed by Tabet Ehsainieh - https://github.com/ehsainit
"""

from deeplabcut.pose_estimation_tensorflow.lib.trackingutils import *


class TrackByDetectionTracker:
    def __init__(self, max_frames_to_skip, jointsnum):
        self.max_frames_to_skip = max_frames_to_skip
        self.id_count = 0
        self.tracks = []
        self.iter = 0
        self.det = []
        self.jointnum = jointsnum

    def cost_metric(self, detections, N, M):
        cost_matrix = np.zeros(shape=(N, M))  # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                z = convert_bbox_to_z(detections[j])[:2]
                gating_dist = self.tracks[i].KF.mahalanobis_dist(z)
                cost_matrix[i][j] = gating_dist
        return cost_matrix

    def track(self, bbs):
        self.iter += 1
        detection_bbs = bbs

        if len(self.tracks) == 0:
            for i in range(len(detection_bbs)):
                track = Track(detection_bbs[i], self.id_count)
                self.id_count += 1
                self.tracks.append(track)

        N = len(self.tracks)
        M = len(detection_bbs)
        cost_matrix = self.cost_metric(detection_bbs, N, M)
        rows, cols = linear_sum_assignment(cost_matrix)
        matches = self._matching(rows, cols, bbs, N)
        # either predict or use detection bbs to track
        for i in range(len(matches)):
            # would be better if delete from the matches upon deleting tracks
            if i < len(self.tracks):
                if matches[i] is not None:
                    self.tracks[i].skipped_frames = 0
                    self.tracks[i].predict(self.iter, detection_bbs[matches[i]])
                else:
                    self.tracks[i].predict(it=self.iter)
        states = []
        for t, track in enumerate(self.tracks):
            if t < len(matches) and matches[t] is not None:
                # new assigned tracks in the n frames will be fist tracked in the n+1 frame - probably need fixing
                states.append(np.concatenate((track.prediction, [track.track_id, matches[t]])).reshape(1, -1)[0])
        if len(states) > 0:
            return np.stack(states)
        else:
            return np.empty((0, self.jointnum * 2 + 2))

    def _matching(self, rows, cols, detection_bbs, N):

        # assign detections as matches according to the linear assignment solution
        matches = [None] * N
        for i in range(len(rows)):
            matches[rows[i]] = cols[i]

        # validate matches :  if the iou are unreasonable unmatch, if linear assignment didnt find
        # a solution increase the number of skipped frames
        for i in range(len(matches)):
            if matches[i] is not None:
                _iou = iou(self.tracks[i].prediction, detection_bbs[matches[i]])
                if _iou > 9:
                    print("unmatched found")
                    matches[i] = None
            else:
                self.tracks[i].skipped_frames += 1
        # handle tracks with skipped frame > max_frame_to_skip -> assign death
        del_tracks = []
        for i in range(len(self.tracks)):
            if self.tracks[i].skipped_frames > self.max_frames_to_skip:
                del_tracks.append(i)

        if len(del_tracks) > 0:
            for i in del_tracks:
                del self.tracks[i]

        # for detections with no matches will be considered as births - new tracks
        for i in range(len(detection_bbs)):
            if i not in matches:
                track = Track(detection_bbs[i], self.id_count)
                self.id_count += 1
                self.tracks.append(track)
        return matches


class Track:
    def __init__(self, corr_bb, trackId):
        self.track_id = trackId
        self.KF = KalmanBoxTracker(corr_bb)
        self.prediction = corr_bb
        self.trace = dict()  # trace path
        self.skipped_frames = 0

    def predict(self, it, detection=None):
        if detection is not None:
            self.KF.update(detection)
            self.KF.predict()
            self.prediction = self.KF.get_state()[0]
        self.trace[it] = self.prediction # probably will need only the center
        self.KF.predict()
        return self.KF.get_state()[0]
