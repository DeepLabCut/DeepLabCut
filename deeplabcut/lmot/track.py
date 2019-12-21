# Copyright 2019 by
# Tabet Ehsainieh, ehsainit@informatik.uni-freiburg.de
# All rights reserved.

from deeplabcut.lmot.assets import get_random_color
from deeplabcut.lmot.kalman_filter import KalmanFilter


class Track:
    def __init__(self, detection, trackId):
        self.track_id = trackId
        self.KF = KalmanFilter(detection.coordinates)
        self.prediction = detection.coordinates
        self.height = detection.height
        self.width = detection.width
        self.trace = dict()  # trace path
        self.skipped_frames = 0
        self.color = get_random_color(self.track_id)

    def predict(self, it, detection=None):
        self.KF.predict()
        if detection is not None:
            self.KF.update(detection.coordinates)
            self.prediction = detection.coordinates
        self.trace[it] = self.prediction
