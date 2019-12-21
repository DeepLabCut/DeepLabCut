# Copyright 2019 by Tabet Ehsainieh.
# All rights reserved.
# Please see the LICENSE file that should have been included as part of this package.
# Copyright 2019 by
# Tabet Ehsainieh, ehsainit@informatik.uni-freiburg.de
# All rights reserved.

from copy import deepcopy

import numpy as np


class KalmanFilter:

    def __init__(self, x):
        self.dt = 0.005  # delta time

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])  # matrix in observation equations

        self.x = x  # initial state estimate
        self.Hmd = np.array([[1, 0],
                             [0, 1]])
        self.P = np.array([[3.0, 0, 0, 0],
                           [0, 3.0, 0, 0],
                           [0, 0, 3.0, 0],
                           [0, 0, 0, 3.0]])  # covariance matrix
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])  # state transition mat
        self.Q = 10000 * np.array(
            [[(self.dt ** 4) / 4, 0, (self.dt ** 3) / 2, 0],
             [0, (self.dt ** 4) / 4, 0, (self.dt ** 3) / 2],
             [(self.dt ** 3) / 2, 0, self.dt ** 2, 0],
             [0, (self.dt ** 3) / 2, 0, self.dt ** 2]])
        # process noise matrix
        self.R = 0.1 * np.eye(self.x.shape[0])  # observation noise matrix
        self.G = np.array([[self.dt ** 2 / 2], [self.dt ** 2 / 2], [self.dt], [self.dt]])

        self.z = np.array([[None] * 4]).T
        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        # read only - to keep track and further calculations
        self.S = None  # innovation (pre-fit residual) covariance
        self.y = None  # measurement pre-fit residual
        self.K = None  # Optimal Gain
        self.Inv_S = None

    def predict(self):
        # Predicted state estimate
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        return np.round(self.x)

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.identity(4) - np.dot(K, self.H)), self.P)
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        # read only - to keep track for further calculations
        self.y = y
        self.S = S
        self.Inv_S = np.linalg.inv(S)
        self.K = K
        return np.round(self.x)

    def mahalanobis_dist(self, z):
        """
           Compute the mahalanobis distance
        """
        x_hat = np.dot(self.Hmd, self.x[:2])
        y = z[:2] - x_hat
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        S = S[: 2:]
        S = S[:, :2]
        Inv_S = np.linalg.inv(S)
        d = np.dot(y.T, np.dot(Inv_S, y))
        return np.sqrt(d)

    def GetPreRes(self):
        if self.y is not None:
            return self.y
        else:
            return 0

    def estimateCov(self):
        pass

    def get_xPrior(self):
        return self.x_prior

    def _update(self):
        pass
