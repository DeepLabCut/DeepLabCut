# Copyright 2019 by
# Tabet Ehsainieh, ehsainit@informatik.uni-freiburg.de
# All rights reserved.


import numpy as np

from deeplabcut.lmot.extract_local_maxima import extract_locmaxima
from deeplabcut.lmot.linear_sum_assignment import assignment_problem
from deeplabcut.lmot.track import Track


class Tracker:
    def __init__(self, max_frames_to_skip):
        self.max_frames_to_skip = max_frames_to_skip
        self.id_count = 1
        self.tracks = []
        self.iter = 0
        self.det = []

    def cost_metric(self, detections, N, M):
        cost_matrix = np.zeros(shape=(N, M))  # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                gating_dist = self.tracks[i].KF.mahalanobis_dist(detections[j].coordinates)
                cost_matrix[i][j] = gating_dist
        return cost_matrix

    def track(self, scmap, locref):
        print("ITERATION NUMBER : " + str(self.iter))
        self.iter += 1
        scmap = np.squeeze(scmap)
        detections = extract_locmaxima(scmap, locref)
        self.det = detections

        if len(self.tracks) == 0:
            # print("# No Tracks Found, Creating New Tracks")
            for i in range(len(detections)):
                track = Track(detections[i], self.id_count)
                self.id_count += 1
                self.tracks.append(track)
            # print("# number of created tracks: " + str(len(self.tracks)))
        N = len(self.tracks)
        M = len(detections)
        print(detections, N, M)
        cost_matrix = self.cost_metric(detections, N, M)
        rows, cols = assignment_problem(cost_matrix)
        matches = self._matching(rows, cols, cost_matrix, detections, N)
        for i in range(len(matches)):
            if matches[i] is not None:
                self.tracks[i].skipped_frames = 0
                self.tracks[i].predict(self.iter, detections[matches[i]])
            else:
                self.tracks[i].predict(it=self.iter)
        print("tracks at this iteration", self.tracks, "counting", len(self.tracks))

    def _matching(self, rows, cols, costmatrix, detections, N):
        c = 0
        matches = [None] * N
        for i in range(len(rows)):
            matches[rows[i]] = cols[i]
            c += 1
        unmatching = []
        for i in range(len(matches)):
            if matches[i] is not None:
                if costmatrix[i][matches[i]] > 150:
                    print("unmatched found")
                    matches[i] = None
                    unmatching.append(i)
            else:
                self.tracks[i].skipped_frames += 1
        del_tracks = []
        for i in range(len(self.tracks)):
            if self.tracks[i].skipped_frames > self.max_frames_to_skip:
                del_tracks.append(self.tracks[i])

        if len(del_tracks) > 0:
            for i in range(len(del_tracks)):
                del self.tracks[i]
                del matches[i]

        for i in range(len(detections)):
            if i not in matches:
                track = Track(detections[i], self.id_count)
                self.id_count += 1
                self.tracks.append(track)
        return matches
