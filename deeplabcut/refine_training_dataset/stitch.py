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
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import re
import scipy.linalg.interpolative as sli
import shelve
import warnings
from collections import defaultdict

import deeplabcut
from deeplabcut.utils.auxfun_videos import VideoWriter
from functools import partial
from deeplabcut.pose_estimation_tensorflow.lib.trackingutils import (
    calc_iou,
    TRACK_METHODS,
)
from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal
from itertools import combinations, cycle
from networkx.algorithms.flow import preflow_push
from pathlib import Path
from scipy.linalg import hankel
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import mode
from tqdm import trange


class Tracklet:
    def __init__(self, data, inds):
        """
        Create a Tracklet object.

        Parameters
        ----------
        data : ndarray
            3D array of shape (nframes, nbodyparts, 3 or 4), where the last
            dimension is for x, y, likelihood and, optionally, identity.
        inds : array-like
            Corresponding time frame indices.
        """
        if data.ndim != 3 or data.shape[-1] not in (3, 4):
            raise ValueError("Data must of shape (nframes, nbodyparts, 3 or 4)")

        if data.shape[0] != len(inds):
            raise ValueError(
                "Data and corresponding indices must have the same length."
            )

        self.data = data.astype(np.float64)
        self.inds = np.array(inds)
        monotonically_increasing = all(a < b for a, b in zip(inds, inds[1:]))
        if not monotonically_increasing:
            idx = np.argsort(inds, kind="mergesort")  # For stable sort with duplicates
            self.inds = self.inds[idx]
            self.data = self.data[idx]
        self._centroid = None

    def __len__(self):
        return self.inds.size

    def __add__(self, other):
        """Join this tracklet to another one."""
        data = np.concatenate((self.data, other.data))
        inds = np.concatenate((self.inds, other.inds))
        return Tracklet(data, inds)

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other):
        mask = np.isin(self.inds, other.inds, assume_unique=True)
        if mask.all():
            return None
        return Tracklet(self.data[~mask], self.inds[~mask])

    def __lt__(self, other):
        """Test whether this tracklet precedes the other one."""
        return self.end < other.start

    def __gt__(self, other):
        """Test whether this tracklet follows the other one."""
        return self.start > other.end

    def __contains__(self, other_tracklet):
        """Test whether tracklets temporally overlap."""
        return np.isin(self.inds, other_tracklet.inds, assume_unique=True).any()

    def __repr__(self):
        return (
            f"Tracklet of length {len(self)} from {self.start} to {self.end} "
            f"with reliability {self.likelihood:.3f}"
        )

    @property
    def xy(self):
        """Return the x and y coordinates."""
        return self.data[..., :2]

    @property
    def centroid(self):
        """
        Return the instantaneous 2D position of the Tracklet centroid.
        For Tracklets longer than 10 frames, the centroid is automatically
        smoothed using an exponential moving average.
        The result is cached for efficiency.
        """
        if self._centroid is None:
            self._update_centroid()
        return self._centroid

    def _update_centroid(self):
        like = self.data[..., 2:3]
        self._centroid = np.nansum(self.xy * like, axis=1) / np.nansum(like, axis=1)

    @property
    def likelihood(self):
        """Return the average likelihood of all Tracklet detections."""
        return np.nanmean(self.data[..., 2])

    @property
    def identity(self):
        """Return the average predicted identity of all Tracklet detections."""
        try:
            return mode(self.data[..., 3], axis=None, nan_policy="omit")[0][0]
        except IndexError:
            return -1

    @property
    def start(self):
        """Return the time at which the tracklet starts."""
        return self.inds[0]

    @property
    def end(self):
        """Return the time at which the tracklet ends."""
        return self.inds[-1]

    @property
    def flat_data(self):
        return self.data[..., :3].reshape((len(self)), -1)

    def get_data_at(self, ind):
        return self.data[np.searchsorted(self.inds, ind)]

    def set_data_at(self, ind, data):
        self.data[np.searchsorted(self.inds, ind)] = data

    def del_data_at(self, ind):
        idx = np.searchsorted(self.inds, ind)
        self.inds = np.delete(self.inds, idx)
        self.data = np.delete(self.data, idx, axis=0)
        self._update_centroid()

    def interpolate(self, max_gap=1):
        if max_gap < 1:
            raise ValueError("Gap should be a strictly positive integer.")

        gaps = np.diff(self.inds) - 1
        valid_gaps = (0 < gaps) & (gaps <= max_gap)
        fills = []
        for i in np.flatnonzero(valid_gaps):
            s, e = self.inds[[i, i + 1]]
            data1, data2 = self.data[[i, i + 1]]
            diff = (data2 - data1) / (e - s)
            diff[np.isnan(diff)] = 0
            interp = diff[..., np.newaxis] * np.arange(1, e - s)
            data = data1 + np.rollaxis(interp, axis=2)
            data[..., 2] = 0.5  # Chance detections
            if data.shape[1] == 4:
                data[:, 3] = self.identity
            fills.append(Tracklet(data, np.arange(s + 1, e)))
        if not fills:
            return self
        return self + sum(fills)

    def contains_duplicates(self, return_indices=False):
        """
        Evaluate whether the Tracklet contains duplicate time indices.
        If `return_indices`, also return the indices of the duplicates.
        """
        has_duplicates = len(set(self.inds)) != len(self.inds)
        if not return_indices:
            return has_duplicates
        return has_duplicates, np.flatnonzero(np.diff(self.inds) == 0)

    def calc_velocity(self, where="head", norm=True):
        """
        Calculate the linear velocity of either the `head`
        or `tail` of the Tracklet, computed over the last or first
        three frames, respectively. If `norm`, return the absolute
        speed rather than a 2D vector.
        """
        if where == "tail":
            vel = (
                np.diff(self.centroid[:3], axis=0)
                / np.diff(self.inds[:3])[:, np.newaxis]
            )
        elif where == "head":
            vel = (
                np.diff(self.centroid[-3:], axis=0)
                / np.diff(self.inds[-3:])[:, np.newaxis]
            )
        else:
            raise ValueError(f"Unknown where={where}")
        if norm:
            return np.sqrt(np.sum(vel**2, axis=1)).mean()
        return vel.mean(axis=0)

    @property
    def maximal_velocity(self):
        vel = np.diff(self.centroid, axis=0) / np.diff(self.inds)[:, np.newaxis]
        return np.sqrt(np.max(np.sum(vel**2, axis=1)))

    def calc_rate_of_turn(self, where="head"):
        """
        Calculate the rate of turn (or angular velocity) of
        either the `head` or `tail` of the Tracklet, computed over
        the last or first three frames, respectively.
        """
        if where == "tail":
            v = np.diff(self.centroid[:3], axis=0)
        else:
            v = np.diff(self.centroid[-3:], axis=0)
        theta = np.arctan2(v[:, 1], v[:, 0])
        return (theta[1] - theta[0]) / (self.inds[1] - self.inds[0])

    @property
    def is_continuous(self):
        """Test whether there are gaps in the time indices."""
        return self.end - self.start + 1 == len(self)

    def immediately_follows(self, other_tracklet, max_gap=1):
        """
        Test whether this Tracklet follows another within
        a tolerance of`max_gap` frames.
        """
        return 0 < self.start - other_tracklet.end <= max_gap

    def distance_to(self, other_tracklet):
        """
        Calculate the Euclidean distance between this Tracklet and another.
        If the Tracklets overlap in time, this is the mean distance over
        those frames. Otherwise, it is the distance between the head/tail
        of one to the tail/head of the other.
        """
        if self in other_tracklet:
            dist = (
                self.centroid[np.isin(self.inds, other_tracklet.inds)]
                - other_tracklet.centroid[np.isin(other_tracklet.inds, self.inds)]
            )
            return np.sqrt(np.sum(dist**2, axis=1)).mean()
        elif self < other_tracklet:
            return np.sqrt(
                np.sum((self.centroid[-1] - other_tracklet.centroid[0]) ** 2)
            )
        else:
            return np.sqrt(
                np.sum((self.centroid[0] - other_tracklet.centroid[-1]) ** 2)
            )

    def motion_affinity_with(self, other_tracklet):
        """
        Evaluate the motion affinity of this Tracklet' with another one.
        This evaluates whether the Tracklets could realistically be reached
        by one another, knowing the time separating them and their velocities.
        Return 0 if the Tracklets overlap.
        """
        time_gap = self.time_gap_to(other_tracklet)
        if time_gap > 0:
            if self < other_tracklet:
                d1 = self.centroid[-1] + time_gap * self.calc_velocity(norm=False)
                d2 = other_tracklet.centroid[
                    0
                ] - time_gap * other_tracklet.calc_velocity("tail", False)
                delta1 = other_tracklet.centroid[0] - d1
                delta2 = self.centroid[-1] - d2
            else:
                d1 = other_tracklet.centroid[
                    -1
                ] + time_gap * other_tracklet.calc_velocity(norm=False)
                d2 = self.centroid[0] - time_gap * self.calc_velocity("tail", False)
                delta1 = self.centroid[0] - d1
                delta2 = other_tracklet.centroid[-1] - d2
            return (np.sqrt(np.sum(delta1**2)) + np.sqrt(np.sum(delta2**2))) / 2
        return 0

    def time_gap_to(self, other_tracklet):
        """Return the time gap separating this Tracklet to another."""
        if self in other_tracklet:
            t = 0
        elif self < other_tracklet:
            t = other_tracklet.start - self.end
        else:
            t = self.start - other_tracklet.end
        return t

    def shape_dissimilarity_with(self, other_tracklet):
        """Calculate the dissimilarity in shape between this Tracklet and another."""
        if self in other_tracklet:
            dist = np.inf
        elif self < other_tracklet:
            dist = self.undirected_hausdorff(self.xy[-1], other_tracklet.xy[0])
        else:
            dist = self.undirected_hausdorff(self.xy[0], other_tracklet.xy[-1])
        return dist

    def box_overlap_with(self, other_tracklet):
        """Calculate the overlap between each Tracklet's bounding box."""
        if self in other_tracklet:
            overlap = 0
        else:
            if self < other_tracklet:
                bbox1 = self.calc_bbox(-1)
                bbox2 = other_tracklet.calc_bbox(0)
            else:
                bbox1 = self.calc_bbox(0)
                bbox2 = other_tracklet.calc_bbox(-1)
            overlap = calc_iou(bbox1, bbox2)
        return overlap

    @staticmethod
    def undirected_hausdorff(u, v):
        return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])

    def calc_bbox(self, ind):
        xy = self.xy[ind]
        bbox = np.empty(4)
        bbox[:2] = np.nanmin(xy, axis=0)
        bbox[2:] = np.nanmax(xy, axis=0)
        return bbox

    @staticmethod
    def hankelize(xy):
        ncols = int(np.ceil(len(xy) * 2 / 3))
        nrows = len(xy) - ncols + 1
        mat = np.empty((2 * nrows, ncols))
        mat[::2] = hankel(xy[:nrows, 0], xy[-ncols:, 0])
        mat[1::2] = hankel(xy[:nrows, 1], xy[-ncols:, 1])
        return mat

    def to_hankelet(self):
        # See Li et al., 2012. Cross-view Activity Recognition using Hankelets.
        # As proposed in the paper, the Hankel matrix can either be formed from
        # the tracklet's centroid or its normalized velocity.
        # vel = np.diff(self.centroid, axis=0)
        # vel /= np.linalg.norm(vel, axis=1, keepdims=True)
        # return self.hankelize(vel)
        return self.hankelize(self.centroid)

    def dynamic_dissimilarity_with(self, other_tracklet):
        """
        Compute a dissimilarity score between Hankelets.
        This metric efficiently captures the degree of alignment of
        the subspaces spanned by the columns of both matrices.

        See Li et al., 2012.
            Cross-view Activity Recognition using Hankelets.
        """
        hk1 = self.to_hankelet()
        hk1 /= np.linalg.norm(hk1)
        hk2 = other_tracklet.to_hankelet()
        hk2 /= np.linalg.norm(hk2)
        min_shape = min(hk1.shape + hk2.shape)
        temp1 = (hk1 @ hk1.T)[:min_shape, :min_shape]
        temp2 = (hk2 @ hk2.T)[:min_shape, :min_shape]
        return 2 - np.linalg.norm(temp1 + temp2)

    def dynamic_similarity_with(self, other_tracklet, tol=0.01):
        """
        Evaluate the complexity of the tracklets' underlying dynamics
        from the rank of their Hankel matrices, and assess whether
        they originate from the same track. The idea is that if two
        tracklets are part of the same track, they can be approximated
        by a low order regressor. Conversely, tracklets belonging to
        different tracks will require a higher order regressor.

        See Dicle et al., 2013.
            The Way They Move: Tracking Multiple Targets with Similar Appearance.
        """
        # TODO Add missing data imputation
        joint_tracklet = self + other_tracklet
        joint_rank = joint_tracklet.estimate_rank(tol)
        rank1 = self.estimate_rank(tol)
        rank2 = other_tracklet.estimate_rank(tol)
        return (rank1 + rank2) / joint_rank - 1

    def estimate_rank(self, tol):
        """
        Estimate the (low) rank of a noisy matrix via
        hard thresholding of singular values.

        See Gavish & Donoho, 2013.
            The optimal hard threshold for singular values is 4/sqrt(3)
        """
        mat = self.to_hankelet()
        # nrows, ncols = mat.shape
        # beta = nrows / ncols
        # omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
        _, s, _ = sli.svd(mat, min(10, min(mat.shape)))
        # return np.argmin(s > omega * np.median(s))
        eigen = s**2
        diff = np.abs(np.diff(eigen / eigen[0]))
        return np.argmin(diff > tol)

    def plot(self, centroid_only=True, color=None, ax=None, interactive=False):
        if ax is None:
            fig, ax = plt.subplots()
        centroid = np.full((self.end + 1, 2), np.nan)
        centroid[self.inds] = self.centroid
        lines = ax.plot(centroid, c=color, lw=2, picker=interactive)
        if not centroid_only:
            xy = np.full((self.end + 1, self.xy.shape[1], 2), np.nan)
            xy[self.inds] = self.xy
            ax.plot(xy[..., 0], c=color, lw=1)
            ax.plot(xy[..., 1], c=color, lw=1)
        return lines


class TrackletStitcher:
    def __init__(
        self,
        tracklets,
        n_tracks,
        min_length=10,
        split_tracklets=True,
        prestitch_residuals=True,
    ):
        if n_tracks < 1:
            raise ValueError("There must at least be one track to reconstruct.")

        if min_length < 3:
            raise ValueError("A tracklet must have a minimal length of 3.")

        self.min_length = min_length
        self.filename = ""
        self.header = None
        self.single = None
        self.n_tracks = n_tracks
        self.G = None
        self.paths = None
        self.tracks = None

        self.tracklets = []
        self.residuals = []
        for unpure_tracklet in tracklets:
            tracklet = self.purify_tracklet(unpure_tracklet)
            if tracklet is None:
                continue
            if not tracklet.is_continuous and split_tracklets:
                idx = np.flatnonzero(np.diff(tracklet.inds) != 1) + 1
                tracklet = self.split_tracklet(tracklet, tracklet.inds[idx])
            if not isinstance(tracklet, list):
                tracklet = [tracklet]
            for t in tracklet:
                if len(t) >= min_length:
                    self.tracklets.append(t)
                elif len(t) < min_length:
                    self.residuals.append(t)

        if not len(self.tracklets):
            raise IOError("Tracklets are empty.")

        if prestitch_residuals:
            self._prestitch_residuals(5)  # Hard-coded but found to work very well
        self.tracklets = sorted(self.tracklets, key=lambda t: t.start)
        self._first_frame = self.tracklets[0].start
        self._last_frame = max(self.tracklets, key=lambda t: t.end).end

        # Note that if tracklets are very short, some may actually be part of the same track
        # and thus incorrectly reflect separate track endpoints...
        self._first_tracklets = sorted(self, key=lambda t: t.start)[: self.n_tracks]
        self._last_tracklets = sorted(self, key=lambda t: t.end)[-self.n_tracks :]

        # Map each Tracklet to an entry and output nodes and vice versa,
        # which is convenient once the tracklets are stitched.
        self._mapping = {
            tracklet: {"in": f"{i}in", "out": f"{i}out"}
            for i, tracklet in enumerate(self)
        }
        self._mapping_inv = {
            label: k for k, v in self._mapping.items() for label in v.values()
        }

        # Store tracklets and corresponding negatives (those that overlap in time)
        self._lu_overlap = defaultdict(list)
        for tracklet1, tracklet2 in combinations(self, 2):
            if tracklet2 in tracklet1:
                self._lu_overlap[tracklet1].append(tracklet2)
                self._lu_overlap[tracklet2].append(tracklet1)

    def __getitem__(self, item):
        return self.tracklets[item]

    def __len__(self):
        return len(self.tracklets)

    @classmethod
    def from_pickle(
        cls,
        pickle_file,
        n_tracks,
        min_length=10,
        split_tracklets=True,
        prestitch_residuals=True,
    ):
        with open(pickle_file, "rb") as file:
            tracklets = pickle.load(file)
        class_ = cls.from_dict_of_dict(
            tracklets, n_tracks, min_length, split_tracklets, prestitch_residuals
        )
        class_.filename = pickle_file
        return class_

    @classmethod
    def from_dict_of_dict(
        cls,
        dict_of_dict,
        n_tracks,
        min_length=10,
        split_tracklets=True,
        prestitch_residuals=True,
    ):
        tracklets = []
        header = dict_of_dict.pop("header", None)
        single = None
        for k, dict_ in dict_of_dict.items():
            inds, data = zip(*[(cls.get_frame_ind(k), v) for k, v in dict_.items()])
            inds = np.asarray(inds)
            data = np.asarray(data)
            try:
                nrows, ncols = data.shape
                data = data.reshape((nrows, ncols // 3, 3))
            except ValueError:
                pass
            tracklet = Tracklet(data, inds)
            if k == "single":
                single = tracklet
            else:
                tracklets.append(Tracklet(data, inds))
        class_ = cls(
            tracklets, n_tracks, min_length, split_tracklets, prestitch_residuals
        )
        class_.header = header
        class_.single = single
        return class_

    @staticmethod
    def get_frame_ind(s):
        if isinstance(s, str):
            return int(re.findall(r"\d+", s)[0])
        return s

    @staticmethod
    def purify_tracklet(tracklet):
        valid = ~np.isnan(tracklet.xy).all(axis=(1, 2))
        if not np.any(valid):
            return None
        return Tracklet(tracklet.data[valid], tracklet.inds[valid])

    @staticmethod
    def split_tracklet(tracklet, inds):
        idx = sorted(set(np.searchsorted(tracklet.inds, inds)))
        inds_new = np.split(tracklet.inds, idx)
        data_new = np.split(tracklet.data, idx)
        return [Tracklet(data, inds) for data, inds in zip(data_new, inds_new)]

    @property
    def n_frames(self):
        return self._last_frame - self._first_frame + 1

    # TODO Avoid looping over all pairs of tracklets
    @staticmethod
    def compute_max_gap(tracklets):
        gap = defaultdict(list)
        for tracklet1, tracklet2 in combinations(tracklets, 2):
            gap[tracklet1].append(tracklet1.time_gap_to(tracklet2))
        max_gap = 0
        for vals in gap.values():
            for val in sorted(vals):
                if val > 0:
                    if val > max_gap:
                        max_gap = val
                    break
        return max_gap

    def mine(self, n_samples):
        p = np.asarray([t.likelihood for t in self])
        p /= p.sum()
        triplets = []
        while len(triplets) != n_samples:
            tracklet = np.random.choice(self, p=p)
            overlapping_tracklets = self._lu_overlap[tracklet]
            if not overlapping_tracklets:
                continue
            # Pick the closest (spatially) overlapping tracklet
            ind_min = np.argmin(
                [tracklet.distance_to(t) for t in overlapping_tracklets]
            )
            overlapping_tracklet = overlapping_tracklets[ind_min]
            common_inds = set(tracklet.inds).intersection(overlapping_tracklet.inds)
            ind_anchor = np.random.choice(list(common_inds))
            anchor = tracklet.get_data_at(ind_anchor)[:, :2].astype(int)
            neg = overlapping_tracklet.get_data_at(ind_anchor)[:, :2].astype(int)
            ind_pos = np.random.choice(tracklet.inds[tracklet.inds != ind_anchor])
            pos = tracklet.get_data_at(ind_pos)[:, :2].astype(int)
            triplet = ((anchor, ind_anchor), (pos, ind_pos), (neg, ind_anchor))
            triplets.append(triplet)
        return triplets

    def build_graph(
        self,
        nodes=None,
        max_gap=None,
        weight_func=None,
    ):
        if nodes is None:
            nodes = self.tracklets
        nodes = sorted(nodes, key=lambda t: t.start)
        n_nodes = len(nodes)

        if not max_gap:
            max_gap = int(1.5 * self.compute_max_gap(nodes))

        self.G = nx.DiGraph()
        self.G.add_node("source", demand=-self.n_tracks)
        self.G.add_node("sink", demand=self.n_tracks)
        nodes_in, nodes_out = zip(
            *[v.values() for k, v in self._mapping.items() if k in nodes]
        )
        self.G.add_nodes_from(nodes_in, demand=1)
        self.G.add_nodes_from(nodes_out, demand=-1)
        self.G.add_edges_from(zip(nodes_in, nodes_out), capacity=1)
        self.G.add_edges_from(zip(["source"] * n_nodes, nodes_in), capacity=1)
        self.G.add_edges_from(zip(nodes_out, ["sink"] * n_nodes), capacity=1)
        if weight_func is None:
            weight_func = self.calculate_edge_weight
        for i in trange(n_nodes):
            node_i = nodes[i]
            end = node_i.end
            for j in range(i + 1, n_nodes):
                node_j = nodes[j]
                start = node_j.start
                gap = start - end
                if gap > max_gap:
                    break
                elif gap > 0:
                    # The algorithm works better with integer weights
                    w = int(100 * weight_func(node_i, node_j))
                    self.G.add_edge(
                        self._mapping[node_i]["out"],
                        self._mapping[node_j]["in"],
                        weight=w,
                        capacity=1,
                    )

    def _update_edge_weights(self, weight_func):
        if self.G is None:
            raise ValueError("Inexistent graph. Call `build_graph` first")

        for node1, node2, weight in self.G.edges.data("weight"):
            if weight is not None:
                w = weight_func(self._mapping_inv[node1], self._mapping_inv[node2])
                self.G.edges[(node1, node2)]["weight"] = w

    def stitch(self, add_back_residuals=True):
        if self.G is None:
            raise ValueError("Inexistent graph. Call `build_graph` first")

        try:
            _, self.flow = nx.capacity_scaling(self.G)
            self.paths = self.reconstruct_paths()
        except nx.exception.NetworkXUnfeasible:
            warnings.warn("No optimal solution found. Employing black magic...")
            # Let us prune the graph by removing all source and sink edges
            # but those connecting the `n_tracks` first and last tracklets.
            in_to_keep = [
                self._mapping[first_tracklet]["in"]
                for first_tracklet in self._first_tracklets
            ]
            out_to_keep = [
                self._mapping[last_tracklet]["out"]
                for last_tracklet in self._last_tracklets
            ]
            in_to_remove = set(
                node for _, node in self.G.out_edges("source")
            ).difference(in_to_keep)
            out_to_remove = set(node for node, _ in self.G.in_edges("sink")).difference(
                out_to_keep
            )
            self.G.remove_edges_from(zip(["source"] * len(in_to_remove), in_to_remove))
            self.G.remove_edges_from(zip(out_to_remove, ["sink"] * len(out_to_remove)))
            # Preflow push seems to work slightly better than shortest
            # augmentation path..., and is more computationally efficient.
            paths = []
            for path in nx.node_disjoint_paths(
                self.G, "source", "sink", preflow_push, self.n_tracks
            ):
                temp = set()
                for node in path[1:-1]:
                    self.G.remove_node(node)
                    temp.add(self._mapping_inv[node])
                paths.append(list(temp))
            incomplete_tracks = self.n_tracks - len(paths)
            remaining_nodes = set(
                self._mapping_inv[node]
                for node in self.G
                if node not in ("source", "sink")
            )
            if (
                incomplete_tracks == 1
            ):  # All remaining nodes must belong to the same track
                # Verify whether there are overlapping tracklets
                for t1, t2 in combinations(remaining_nodes, 2):
                    if t1 in t2:
                        # Pick the segment that minimizes "smoothness", computed here
                        # with the coefficient of variation of the differences.
                        if t1 in remaining_nodes:
                            remaining_nodes.remove(t1)
                        if t2 in remaining_nodes:
                            remaining_nodes.remove(t2)
                        track = sum(remaining_nodes)
                        hyp1 = track + t1
                        hyp2 = track + t2
                        dx1 = np.diff(hyp1.centroid, axis=0)
                        cv1 = dx1.std() / np.abs(dx1).mean()
                        dx2 = np.diff(hyp2.centroid, axis=0)
                        cv2 = dx2.std() / np.abs(dx2).mean()
                        if cv1 < cv2:
                            remaining_nodes.add(t1)
                            self.residuals.append(t2)
                        else:
                            remaining_nodes.add(t2)
                            self.residuals.append(t1)
                paths.append(list(remaining_nodes))
            elif incomplete_tracks > 1:
                # Rebuild a full graph from the remaining nodes without
                # temporal constraint on what tracklets can be stitched together.
                self.build_graph(list(remaining_nodes), max_gap=np.inf)
                self.G.nodes["source"]["demand"] = -incomplete_tracks
                self.G.nodes["sink"]["demand"] = incomplete_tracks
                _, self.flow = nx.capacity_scaling(self.G)
                paths += self.reconstruct_paths()
            self.paths = paths
            if len(self.paths) != self.n_tracks:
                warnings.warn(f"Only {len(self.paths)} tracks could be reconstructed.")

        finally:
            if self.paths is None:
                raise ValueError(
                    f"Could not reconstruct {self.n_tracks} tracks from the tracklets given."
                )

            self.tracks = np.asarray([sum(path) for path in self.paths if path])
            if add_back_residuals:
                _ = self._finalize_tracks()

    def _finalize_tracks(self):
        residuals = [res for res in sorted(self.residuals, key=len) if len(res) > 1]
        # Cycle through the residuals and incorporate back those
        # that only fit in a single tracklet.
        n_attemps = 0
        n_max = len(residuals)
        while n_attemps < n_max and residuals:
            for res in residuals[::-1]:
                easy_fit = [
                    i for i, track in enumerate(self.tracks) if res not in track
                ]
                if not easy_fit:
                    residuals.remove(res)
                    continue
                if len(easy_fit) == 1:
                    self.tracks[easy_fit[0]] += res
                    residuals.remove(res)
                    n_attemps = 0
                else:
                    n_attemps += 1

        # Greedily add the remaining residuals
        for res in residuals[::-1]:
            c1 = res.centroid[[0, -1]]
            easy_fit = [i for i, track in enumerate(self.tracks) if res not in track]
            dists = []
            for n, track in enumerate(self.tracks[easy_fit]):
                e = np.searchsorted(track.inds, res.end)
                s = e - 1
                try:
                    t = track.inds[[s, e]]
                except IndexError:
                    continue
                left_gap = res.start - t[0]
                right_gap = t[1] - res.end
                if not left_gap > 0 and right_gap > 0:
                    continue
                if left_gap <= 3:
                    dist = np.linalg.norm(track.centroid[s] - c1[0])
                elif right_gap <= 3:
                    dist = np.linalg.norm(track.centroid[e] - c1[1])
                else:
                    dist = np.linalg.norm(track.centroid[s] - c1[0]) + np.linalg.norm(
                        track.centroid[e] - c1[1]
                    )
                dists.append((n, dist))
            if not dists:
                continue
            if len(dists) == 1:
                ind = easy_fit[dists[0][0]]
            else:
                ind = sorted(dists, key=lambda x: x[1])[0][0]
            self.tracks[ind] += res
            residuals.remove(res)
        return residuals

    def _prestitch_residuals(self, max_gap=5):
        G = nx.DiGraph()
        residuals = sorted(self.residuals, key=lambda x: x.start)
        for i in range(len(residuals)):
            e = residuals[i].end
            for j in range(i + 1, len(residuals)):
                s = residuals[j].start
                gap = s - e
                if gap < 1:
                    continue
                if gap < max_gap:
                    w = 1 - residuals[i].box_overlap_with(residuals[j])
                    G.add_edge(i, j, weight=w)
                else:
                    break
        mini_tracks = []
        to_remove = []
        for comp in nx.connected_components(G.to_undirected()):
            sub_ = G.subgraph(comp)
            inds = nx.dag_longest_path(sub_)
            to_remove.extend(inds)
            mini_tracks.append(sum(residuals[ind] for ind in inds))
        for ind in sorted(to_remove, reverse=True):
            self.residuals.pop(ind)
        self.residuals.extend(mini_tracks)

    def concatenate_data(self):
        if self.tracks is None:
            raise ValueError("No tracks were found. Call `stitch` first")

        # Refresh temporal bounds
        self._first_frame = min(self.tracks, key=lambda t: t.start).start
        self._last_frame = max(self.tracks, key=lambda t: t.end).end
        data = []
        for track in self.tracks:
            flat_data = track.flat_data
            temp = np.full((self.n_frames, flat_data.shape[1]), np.nan)
            temp[track.inds - self._first_frame] = flat_data
            data.append(temp)
        return np.hstack(data)

    def format_df(self, animal_names=None):
        data = self.concatenate_data()
        if not animal_names or len(animal_names) != self.n_tracks:
            animal_names = [f"ind{i}" for i in range(1, self.n_tracks + 1)]
        coords = ["x", "y", "likelihood"]
        n_multi_bpts = data.shape[1] // (len(animal_names) * len(coords))
        n_unique_bpts = 0 if self.single is None else self.single.data.shape[1]

        if self.header is not None:
            scorer = self.header.get_level_values("scorer").unique().to_list()
            bpts = self.header.get_level_values("bodyparts").unique().to_list()
        else:
            scorer = ["scorer"]
            bpts = [f"bpt{i}" for i in range(1, n_multi_bpts + 1)]
            bpts += [f"bpt_unique{i}" for i in range(1, n_unique_bpts + 1)]

        columns = pd.MultiIndex.from_product(
            [scorer, animal_names, bpts[:n_multi_bpts], coords],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )
        inds = range(self._first_frame, self._last_frame + 1)
        df = pd.DataFrame(data, columns=columns, index=inds)
        df = df.reindex(range(self._last_frame + 1))
        if self.single is not None:
            columns = pd.MultiIndex.from_product(
                [scorer, ["single"], bpts[-n_unique_bpts:], coords],
                names=["scorer", "individuals", "bodyparts", "coords"],
            )
            df2 = pd.DataFrame(
                self.single.flat_data, columns=columns, index=self.single.inds
            )
            df = df.join(df2, how="outer")
        return df

    def write_tracks(
        self, output_name="", suffix="", animal_names=None, save_as_csv=False
    ):
        df = self.format_df(animal_names)
        if not output_name:
            if suffix:
                suffix = "_" + suffix
            output_name = self.filename.replace(".pickle", f"{suffix}.h5")
        df.to_hdf(output_name, "tracks", format="table", mode="w")
        if save_as_csv:
            df.to_csv(output_name.replace(".h5", ".csv"))

    @staticmethod
    def calculate_edge_weight(tracklet1, tracklet2):
        # Default to the distance cost function
        return tracklet1.distance_to(tracklet2)

    @property
    def weights(self):
        if self.G is None:
            raise ValueError("Inexistent graph. Call `build_graph` first")

        return nx.get_edge_attributes(self.G, "weight")

    def draw_graph(self, with_weights=False):
        if self.G is None:
            raise ValueError("Inexistent graph. Call `build_graph` first")

        pos = nx.spring_layout(self.G)
        nx.draw_networkx(self.G, pos)
        if with_weights:
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=self.weights)

    def plot_paths(self, colormap="Set2"):
        if self.paths is None:
            raise ValueError("No paths were found. Call `stitch` first")

        fig, ax = plt.subplots()
        ax.set_yticks([])
        for loc, spine in ax.spines.items():
            if loc != "bottom":
                spine.set_visible(False)
        for path in self.paths:
            length = len(path)
            colors = plt.get_cmap(colormap, length)(range(length))
            for tracklet, color in zip(path, colors):
                tracklet.plot(color=color, ax=ax)

    def plot_tracks(self, colormap="viridis"):
        if self.tracks is None:
            raise ValueError("No tracks were found. Call `stitch` first")

        fig, ax = plt.subplots()
        ax.set_yticks([])
        for loc, spine in ax.spines.items():
            if loc != "bottom":
                spine.set_visible(False)
        colors = plt.get_cmap(colormap, self.n_tracks)(range(self.n_tracks))
        for track, color in zip(self.tracks, colors):
            track.plot(color=color, ax=ax)

    def plot_tracklets(self, colormap="Paired"):
        fig, axes = plt.subplots(ncols=2, figsize=(14, 4))
        axes[0].set_yticks([])
        for loc, spine in axes[0].spines.items():
            if loc != "bottom":
                spine.set_visible(False)
        axes[1].axis("off")

        cmap = plt.get_cmap(colormap)
        colors = cycle(cmap.colors)
        line2tracklet = dict()
        tracklet2lines = dict()
        all_points = defaultdict(dict)
        for tracklet in self:
            color = next(colors)
            lines = tracklet.plot(ax=axes[0], color=color)
            tracklet2lines[tracklet] = lines
            for line in lines:
                line2tracklet[line] = tracklet
            for i, (x, y) in zip(tracklet.inds, tracklet.centroid):
                all_points[i][(x, y)] = color

    def reconstruct_paths(self):
        paths = []
        for node, flow in self.flow["source"].items():
            if flow == 1:
                path = self.reconstruct_path(node.replace("in", "out"))
                paths.append([self._mapping_inv[tracklet] for tracklet in path])
        return paths

    def reconstruct_path(self, source):
        path = [source]
        for node, flow in self.flow[source].items():
            if flow == 1:
                if node != "sink":
                    self.flow[source][node] -= 1
                    path.extend(self.reconstruct_path(node.replace("in", "out")))
                return path


def stitch_tracklets(
    config_path,
    videos,
    videotype="",
    shuffle=1,
    trainingsetindex=0,
    n_tracks=None,
    min_length=10,
    split_tracklets=True,
    prestitch_residuals=True,
    max_gap=None,
    weight_func=None,
    destfolder=None,
    modelprefix="",
    track_method="",
    output_name="",
    transformer_checkpoint="",
    save_as_csv=False,
):
    """
    Stitch sparse tracklets into full tracks via a graph-based,
    minimum-cost flow optimization problem.

    Parameters
    ----------
    config_path : str
        Path to the main project config.yaml file.

    videos : list
        A list of strings containing the full paths to videos for analysis or a path to the directory, where all the videos with same extension are stored.

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed.
        If left unspecified, videos with common extensions ('avi', 'mp4', 'mov', 'mpeg', 'mkv') are kept.

    shuffle: int, optional
        An integer specifying the shuffle index of the training dataset used for training the network. The default is 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    n_tracks : int, optional
        Number of tracks to reconstruct. By default, taken as the number
        of individuals defined in the config.yaml. Another number can be
        passed if the number of animals in the video is different from
        the number of animals the model was trained on.

    min_length : int, optional
        Tracklets less than `min_length` frames of length
        are considered to be residuals; i.e., they do not participate
        in building the graph and finding the solution to the
        optimization problem, but are rather added last after
        "almost-complete" tracks are formed. The higher the value,
        the lesser the computational cost, but the higher the chance of
        discarding relatively long and reliable tracklets that are
        essential to solving the stitching task.
        Default is 10, and must be 3 at least.

    split_tracklets : bool, optional
        By default, tracklets whose time indices are not consecutive integers
        are split in shorter tracklets whose time continuity is guaranteed.
        This is for example very powerful to get rid of tracking errors
        (e.g., identity switches) which are often signaled by a missing
        time frame at the moment they occur. Note though that for long
        occlusions where tracker re-identification capability can be trusted,
        setting `split_tracklets` to False is preferable.

    prestitch_residuals : bool, optional
        Residuals will by default be grouped together according to their
        temporal proximity prior to being added back to the tracks.
        This is done to improve robustness and simultaneously reduce complexity.

    max_gap : int, optional
        Maximal temporal gap to allow between a pair of tracklets.
        This is automatically determined by the TrackletStitcher by default.

    weight_func : callable, optional
        Function accepting two tracklets as arguments and returning a scalar
        that must be inversely proportional to the likelihood that the tracklets
        belong to the same track; i.e., the higher the confidence that the
        tracklets should be stitched together, the lower the returned value.

    destfolder: string, optional
        Specifies the destination folder for analysis data (default is the path of the video). Note that for subsequent analysis this
        folder also needs to be passed.

    track_method: string, optional
         Specifies the tracker used to generate the pose estimation data.
         For multiple animals, must be either 'box', 'skeleton', or 'ellipse'
         and will be taken from the config.yaml file if none is given.

    output_name : str, optional
        Name of the output h5 file.
        By default, tracks are automatically stored into the same directory
        as the pickle file and with its name.

    save_as_csv: bool, optional
        Whether to write the tracks to a CSV file too (False by default).

    Returns
    -------
    A TrackletStitcher object
    """
    vids = deeplabcut.utils.auxiliaryfunctions.get_list_of_videos(videos, videotype)
    if not vids:
        print("No video(s) found. Please check your path!")
        return

    cfg = auxiliaryfunctions.read_config(config_path)
    track_method = auxfun_multianimal.get_track_method(cfg, track_method=track_method)

    animal_names = cfg["individuals"]
    if n_tracks is None:
        n_tracks = len(animal_names)

    DLCscorer, _ = deeplabcut.utils.auxiliaryfunctions.GetScorerName(
        cfg,
        shuffle,
        cfg["TrainingFraction"][trainingsetindex],
        modelprefix=modelprefix,
    )

    if transformer_checkpoint:
        from deeplabcut.pose_tracking_pytorch import inference

        dlctrans = inference.DLCTrans(checkpoint=transformer_checkpoint)

    def trans_weight_func(tracklet1, tracklet2, nframe, feature_dict):
        zfill_width = int(np.ceil(np.log10(nframe)))
        if tracklet1 < tracklet2:
            ind_img1 = tracklet1.inds[-1]
            coord1 = tracklet1.data[-1][:, :2]
            ind_img2 = tracklet2.inds[0]
            coord2 = tracklet2.data[0][:, :2]
        else:
            ind_img2 = tracklet2.inds[-1]
            ind_img1 = tracklet1.inds[0]
            coord2 = tracklet2.data[-1][:, :2]
            coord1 = tracklet1.data[0][:, :2]
        t1 = (coord1, ind_img1)
        t2 = (coord2, ind_img2)

        dist = dlctrans(t1, t2, zfill_width, feature_dict)
        dist = (dist + 1) / 2

        return -dist

    for video in vids:
        print("Processing... ", video)
        nframe = len(VideoWriter(video))
        videofolder = str(Path(video).parents[0])
        dest = destfolder or videofolder
        deeplabcut.utils.auxiliaryfunctions.attempttomakefolder(dest)
        vname = Path(video).stem

        feature_dict_path = os.path.join(
            dest, vname + DLCscorer + "_bpt_features.pickle"
        )
        # should only exist one
        if transformer_checkpoint:
            import dbm

            try:
                feature_dict = shelve.open(feature_dict_path, flag="r")
            except dbm.error:
                raise FileNotFoundError(
                    f"{feature_dict_path} does not exist. Did you run transformer_reID()?"
                )

        dataname = os.path.join(dest, vname + DLCscorer + ".h5")

        method = TRACK_METHODS[track_method]
        pickle_file = dataname.split(".h5")[0] + f"{method}.pickle"
        try:
            stitcher = TrackletStitcher.from_pickle(
                pickle_file, n_tracks, min_length, split_tracklets, prestitch_residuals
            )
            with_id = any(tracklet.identity != -1 for tracklet in stitcher)
            if with_id and weight_func is None:
                # Add in identity weighing before building the graph
                def weight_func(t1, t2):
                    w = 0.01 if t1.identity == t2.identity else 1
                    return w * stitcher.calculate_edge_weight(t1, t2)

            if transformer_checkpoint:
                stitcher.build_graph(
                    max_gap=max_gap,
                    weight_func=partial(
                        trans_weight_func, nframe=nframe, feature_dict=feature_dict
                    ),
                )
            else:
                stitcher.build_graph(max_gap=max_gap, weight_func=weight_func)

            stitcher.stitch()
            if transformer_checkpoint:
                stitcher.write_tracks(
                    output_name=output_name,
                    animal_names=animal_names,
                    suffix="tr",
                    save_as_csv=save_as_csv,
                )
            else:
                stitcher.write_tracks(
                    output_name=output_name,
                    animal_names=animal_names,
                    suffix="",
                    save_as_csv=save_as_csv,
                )
        except FileNotFoundError as e:
            print(e, "\nSkipping...")
