import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
import re
import scipy.linalg.interpolative as sli
from itertools import combinations
from math import factorial
from networkx.algorithms.flow import preflow_push
from scipy.linalg import hankel
from scipy.spatial.distance import directed_hausdorff
from statsmodels.tsa.api import SimpleExpSmoothing
from tqdm import tqdm


pickle_file = ('/Users/Jessy/Downloads/MultiMouse-Daniel-2019-12-16/videos/'
               'videocompressed11DLC_resnet50_MultiMouseDec16shuffle1_50000_bx.pickle')
# pickle_file = ('/Users/Jessy/Downloads/Marmoset-Mackenzie-2019-05-29/videos/'
#                'short_videoDLC_resnet50_MarmosetMay29shuffle0_20000_bx.pickle')
# pickle_file = ('/Users/Jessy/Downloads/two_white_mice_052820-SN-2020-05-28/videos/'
#                'White_mice_togetherDLC_resnet50_two_white_mice_052820May28shuffle1_200000_el.pickle')


# TODO Heading over last couple of frames (circular average statistics)
# TODO Smoothness

class Tracklet:
    def __init__(self, data, inds):
        self.data = data
        self.inds = np.array(inds)
        monotonically_increasing = all(a < b for a, b in zip(inds, inds[1:]))
        if not monotonically_increasing:
            idx = np.argsort(inds)
            self.inds = self.inds[idx]
            self.data = self.data[idx]
        self._centroid = None

    def __len__(self):
        return len(self.inds)

    def __add__(self, other):
        """Join this tracklet to another one."""
        data = np.concatenate((self.data, other.data))
        inds = np.concatenate((self.inds, other.inds))
        return Tracklet(data, inds)

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
        return f'Tracklet of length {len(self)}, ' \
               f'from {self.start} to {self.end}'

    @property
    def xy(self):
        return self.data[..., :2]

    @property
    def centroid(self):
        if self._centroid is None:
            centroid = np.nanmean(self.xy, axis=1)
            fit_x = (SimpleExpSmoothing(centroid[:, 0])
                     .fit(smoothing_level=0.5, optimized=False)
                     .fittedvalues)
            fit_y = (SimpleExpSmoothing(centroid[:, 1])
                     .fit(smoothing_level=0.5, optimized=False)
                     .fittedvalues)
            self._centroid = np.c_[fit_x, fit_y]
        return self._centroid

    @property
    def likelihood(self):
        return self.data[..., 2]

    @property
    def start(self):
        return self.inds[0]

    @property
    def end(self):
        return self.inds[-1]

    def calc_velocity(self, where='head', norm=True):
        if where == 'tail':
            vel = (np.diff(self.centroid[:3], axis=0)
                   / np.diff(self.inds[:3])[:, np.newaxis])
        elif where == 'head':
            vel = (np.diff(self.centroid[-3:], axis=0)
                   / np.diff(self.inds[-3:])[:, np.newaxis])
        else:
            raise ValueError(f'Unknown where={where}')
        if norm:
            return np.sqrt(np.sum(vel ** 2, axis=1)).mean()
        return vel.mean(axis=0)

    def calc_rate_of_turn(self, where='head'):
        if where == 'tail':
            v = np.diff(self.centroid[:3], axis=0)
        else:
            v = np.diff(self.centroid[-3:], axis=0)
        theta = np.arctan2(v[:, 1], v[:, 0])
        return (theta[1] - theta[0]) / (self.inds[1] - self.inds[0])

    @property
    def is_continuous(self):
        return self.end - self.start + 1 == len(self)

    def immediately_follows(self, other_tracklet, max_gap=1):
        return 0 < self.start - other_tracklet.end <= max_gap

    def distance_to(self, other_tracklet):
        if self in other_tracklet:
            dist = (self.centroid[np.isin(self.inds, other_tracklet.inds)]
                    - other_tracklet.centroid[np.isin(other_tracklet.inds, self.inds)])
            return np.sqrt(np.sum(dist ** 2, axis=1)).mean()
        elif self < other_tracklet:
            return np.sqrt(np.sum((self.centroid[-1] - other_tracklet.centroid[0]) ** 2))
        else:
            return np.sqrt(np.sum((self.centroid[0] - other_tracklet.centroid[-1]) ** 2))

    def motion_affinity_with(self, other_tracklet):
        time_gap = self.time_gap_to(other_tracklet)
        if time_gap > 0:
            if self < other_tracklet:
                d1 = self.centroid[-1] + time_gap * self.calc_velocity(norm=False)
                d2 = other_tracklet.centroid[0] - time_gap * other_tracklet.calc_velocity('tail', False)
                delta1 = other_tracklet.centroid[0] - d1
                delta2 = self.centroid[-1] - d2
            else:
                d1 = other_tracklet.centroid[-1] + time_gap * other_tracklet.calc_velocity(norm=False)
                d2 = self.centroid[0] - time_gap * self.calc_velocity('tail', False)
                delta1 = self.centroid[0] - d1
                delta2 = other_tracklet.centroid[-1] - d2
            return (np.sqrt(np.sum(delta1 ** 2)) + np.sqrt(np.sum(delta2 ** 2))) / 2
        return 0

    def time_gap_to(self, other_tracklet):
        if self in other_tracklet:
            t = 0
        elif self < other_tracklet:
            t = other_tracklet.start - self.end
        else:
            t = self.start - other_tracklet.end
        return t

    def shape_dissimilarity_with(self, other_tracklet):
        if self in other_tracklet:
            dist = np.inf
        elif self < other_tracklet:
            dist = self.undirected_hausdorff(self.xy[-1], other_tracklet.xy[0])
        else:
            dist = self.undirected_hausdorff(self.xy[0], other_tracklet.xy[-1])
        return dist

    def box_overlap_with(self, other_tracklet):
        if self in other_tracklet:
            overlap = 0
        else:
            if self < other_tracklet:
                bbox1 = self.calc_bbox(-1)
                bbox2 = other_tracklet.calc_bbox(0)
            else:
                bbox1 = self.calc_bbox(0)
                bbox2 = other_tracklet.calc_bbox(-1)
            overlap = self.iou(bbox1, bbox2)
        return overlap

    @staticmethod
    def undirected_hausdorff(u, v):
        return max(directed_hausdorff(u, v)[0],
                   directed_hausdorff(v, u)[0])

    @staticmethod
    def iou(bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        wh = w * h
        return wh / ((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                     + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                     - wh)

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
        # vel = np.diff(self.centroid, axis=0)
        # vel /= np.linalg.norm(vel, axis=1, keepdims=True)
        # return self.hankelize(vel)
        return self.hankelize(self.centroid)

    def dynamic_dissimilarity_with(self, other_tracklet):
        """See Cross-view Activity Recognition using Hankelets."""
        hk1 = self.to_hankelet()
        hk1 /= np.linalg.norm(hk1)
        hk2 = other_tracklet.to_hankelet()
        hk2 /= np.linalg.norm(hk2)
        min_shape = min(hk1.shape + hk2.shape)
        temp1 = (hk1 @ hk1.T)[:min_shape, :min_shape]
        temp2 = (hk2 @ hk2.T)[:min_shape, :min_shape]
        return 2 - np.linalg.norm(temp1 + temp2)

    def dynamic_similarity_with(self, other_tracklet, tol=0.01):
        """See Fast Track Matching and Event Detection; The Way They Move: Tracking Multiple Targets with Similar Appearance"""
        joint_tracklet = self + other_tracklet
        joint_rank = joint_tracklet.estimate_rank(tol)
        rank1 = self.estimate_rank(tol)
        rank2 = other_tracklet.estimate_rank(tol)
        return (rank1 + rank2) / joint_rank - 1

    def estimate_rank(self, tol):
        """The optimal hard threshold for singular values is 4/np.sqrt(3)"""
        mat = self.to_hankelet()
        # nrows, ncols = mat.shape
        # beta = nrows / ncols
        # omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
        _, s, _ = sli.svd(mat, min(10, min(mat.shape)))
        # return np.argmin(s > omega * np.median(s))
        eigen = s ** 2
        diff = np.abs(np.diff(eigen / eigen[0]))
        return np.argmin(diff > tol)

    def plot(self, centroid_only=True, color='r'):
        plt.plot(self.inds, self.centroid, c=color, lw=2)
        if not centroid_only:
            plt.plot(self.inds, self.xy[..., 0], c=color, lw=1)
            plt.plot(self.inds, self.xy[..., 1], c=color, lw=1)


class TrackletStitcher:
    def __init__(self, pickle_file, n_tracks, min_length=10, split_discontinuous=True):
        if min_length <= 3:
            raise ValueError('A tracklet must have a minimal length of 3.')

        self.n_tracks = n_tracks
        self.G = None
        self.paths = None

        with open(pickle_file, 'rb') as file:
            tracklets = pickle.load(file)
        self.header = tracklets.pop("header")
        if not len(tracklets):
            raise IOError("Tracklets are empty.")

        self.tracklets = []
        self.residuals = []
        last_frames = []
        for dict_ in tracklets.values():
            inds, data = zip(*[(self.get_frame_ind(k), v) for k, v in dict_.items()])
            last_frames.append(inds[-1])
            inds = np.asarray(inds)
            data = np.asarray(data)
            # Input data as (nframes, nbodyparts, 3)
            nrows, ncols = data.shape
            temp = data.reshape((nrows, ncols // 3, 3))
            all_nans = np.isnan(temp).all(axis=(1, 2))
            if all_nans.any():
                temp = temp[~all_nans]
                inds = inds[~all_nans]
            tracklet = Tracklet(temp, inds)
            if not tracklet.is_continuous and split_discontinuous:
                tracklet = self.split_tracklet(tracklet)
            if not isinstance(tracklet, list):
                tracklet = [tracklet]
            for t in tracklet:
                if len(t) >= min_length:
                    self.tracklets.append(t)
                else:
                    self.residuals.append(t)
        self.n_frames = max(last_frames) + 1

        self._tracklets_start = sorted(self, key=lambda t: t.start)[:self.n_tracks]
        self._tracklets_end = sorted(self, key=lambda t: t.end)[-self.n_tracks:]

    def __getitem__(self, item):
        return self.tracklets[item]

    def __len__(self):
        return len(self.tracklets)

    @staticmethod
    def get_frame_ind(s):
        return int(re.findall(r"\d+", s)[0])

    @staticmethod
    def split_tracklet(tracklet):
        idx = np.flatnonzero(np.diff(tracklet.inds) != 1) + 1
        inds_new = np.split(tracklet.inds, idx)
        data_new = np.split(tracklet.data, idx)
        return (Tracklet(data, inds) for data, inds in zip(data_new, inds_new))

    def build_graph(self, max_gap=100):
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self)
        self.G.add_node('source', demand=-self.n_tracks)
        self.G.add_node('sink', demand=self.n_tracks)
        for tracklet_start in self._tracklets_start:
            self.G.add_edge('source', tracklet_start, capacity=1)
        for tracklet_end in self._tracklets_end:
            self.G.add_edge(tracklet_end, 'sink', capacity=1)
        n_combinations = int(factorial(len(self)) / (2 * factorial(len(self) - 2)))
        for tracklet1, tracklet2 in tqdm(combinations(self, 2), total=n_combinations):
            time_gap = tracklet1.time_gap_to(tracklet2)
            if 0 < time_gap <= max_gap:
                w = int(100 * self.calculate_weight(tracklet1, tracklet2))
                if tracklet2 > tracklet1:
                    self.G.add_edge(tracklet1, tracklet2,
                                    weight=w, capacity=1)
                else:
                    self.G.add_edge(tracklet2, tracklet1,
                                    weight=w, capacity=1)

    def build_graph2(self, max_gap=100):
        # Equivalent of a set cover problem... Need for more documentation here.
        self._mapping = {tracklet: {'in': f'{i}in', 'out': f'{i}out'}
                         for i, tracklet in enumerate(self)}
        self._mapping_inv = {label: k for k, v in self._mapping.items()
                             for label in v.values()}

        self.G = nx.DiGraph()
        self.G.add_node('source', demand=-self.n_tracks)
        self.G.add_node('sink', demand=self.n_tracks)
        nodes_in, nodes_out = zip(*[v.values() for v in self._mapping.values()])
        self.G.add_nodes_from(nodes_in, demand=1)
        self.G.add_nodes_from(nodes_out, demand=-1)
        for node_in, node_out in zip(nodes_in, nodes_out):
            self.G.add_edge(node_in, node_out, capacity=1)
        for tracklet_start in self._tracklets_start:
            self.G.add_edge('source', self._mapping[tracklet_start]['in'], capacity=1)
        for tracklet_end in self._tracklets_end:
            self.G.add_edge(self._mapping[tracklet_end]['out'], 'sink', capacity=1)
        n_combinations = int(factorial(len(self)) / (2 * factorial(len(self) - 2)))
        for tracklet1, tracklet2 in tqdm(combinations(self, 2), total=n_combinations):
            time_gap = tracklet1.time_gap_to(tracklet2)
            if 0 < time_gap <= max_gap:
                w = int(100 * self.calculate_weight(tracklet1, tracklet2))
                if tracklet2 > tracklet1:
                    self.G.add_edge(self._mapping[tracklet1]['out'],
                                    self._mapping[tracklet2]['in'],
                                    weight=w, capacity=1)
                else:
                    self.G.add_edge(self._mapping[tracklet2]['out'],
                                    self._mapping[tracklet1]['in'],
                                    weight=w, capacity=1)

    def stitch(self):
        if self.G is None:
            raise ValueError('Inexistent graph. Call `build_graph` first')

        try:
            # Preflow push seems to work slightly better than shortest
            # augmentation path..., and is more computationally efficient.
            self.paths = [path[1:-1] for path in nx.node_disjoint_paths(self.G, 'source', 'sink',
                                                                        preflow_push)]
            if len(self.paths) != self.n_tracks:
                ...
        except nx.exception.NetworkXNoPath:
            print('No disjoint paths were found.')

    @staticmethod
    def calculate_weight(tracklet1, tracklet2):
        return (
                -np.log(tracklet1.box_overlap_with(tracklet2) + np.finfo(float).eps)
                + tracklet1.distance_to(tracklet2)
                + tracklet1.shape_dissimilarity_with(tracklet2)
                + tracklet1.dynamic_dissimilarity_with(tracklet2)
        )

    @property
    def weights(self):
        if self.G is None:
            raise ValueError('Inexistent graph. Call `build_graph` first')

        return nx.get_edge_attributes(self.G, 'weight')

    def visualize_graph(self, with_weights=False):
        if self.G is None:
            raise ValueError('Inexistent graph. Call `build_graph` first')

        pos = nx.spring_layout(self.G)
        nx.draw_networkx(self.G, pos)
        if with_weights:
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=self.weights)

    def plot_path(self, path):
        for i in path:
            self._mapping_inv[i].plot()

    def reconstruct_paths(self, edges=None):
        if edges is None:
            edges = []
            for k, v in self.flow.items():
                if k not in ('source', 'sink'):
                    for i, j in v.items():
                        if i != 'sink' and j == 1:
                            edges.append((k, i))
                            break
        G = nx.Graph(edges)
        return list(map(tuple, nx.connected_components(G)))
        # return [sorted(tracklets, key=lambda t: t.start)
        #         for tracklets in nx.connected_components(G)]

    @classmethod
    def reconstruct_path(cls, flow, source):
        path = [source]
        for k, v in flow[source].items():
            if k == 'sink':
                break
            if v == 1:
                path.extend(cls.reconstruct_path(flow, k))
        return path


stitcher = TrackletStitcher(pickle_file, 3, 10)
stitcher.build_graph(100)
# stitcher.build_graph_set_cover(300)

# _, stitcher.flow = nx.capacity_scaling(stitcher.G)
# paths = stitcher.reconstruct_paths()

# for nodes, color in zip(meh2, ['darkgray', 'orange']):
#     for node in nodes[1:-1]:
#         node.plot(color=color)
#
# nodes = meh[0][1:-1]
# stitcher.G.remove_nodes_from(nodes)


def test_graphs():
    G = nx.DiGraph()
    G.add_node('source', demand=-2)
    G.add_node('sink', demand=2)
    G.add_nodes_from(range(5))
    G.add_edge('source', 0, capacity=1)
    G.add_edge('source', 1, capacity=1)
    G.add_edge(4, 'sink', capacity=1)
    G.add_edge(3, 'sink', capacity=1)
    G.add_edge(0, 2, capacity=1, weight=2)
    G.add_edge(0, 3, capacity=1, weight=5)
    G.add_edge(2, 4, capacity=1, weight=1)
    G.add_edge(1, 3, capacity=1, weight=1)
    G.add_edge(1, 2, capacity=1, weight=1)

    G = nx.DiGraph()
    G.add_node('source', demand=-2)
    G.add_node('sink', demand=2)
    G.add_nodes_from(range(5), demand=1)
    G.add_nodes_from(range(5, 10), demand=-1)
    G.add_edge('source', 0, capacity=1)
    G.add_edge('source', 1, capacity=1)
    G.add_edge(9, 'sink', capacity=1)
    G.add_edge(8, 'sink', capacity=1)
    G.add_edge(5, 2, capacity=1, weight=2)
    G.add_edge(5, 3, capacity=1, weight=5)
    G.add_edge(7, 4, capacity=1, weight=1)
    G.add_edge(6, 3, capacity=1, weight=1)
    G.add_edge(6, 2, capacity=1, weight=1)
    G.add_edge(0, 5)
    G.add_edge(1, 6)
    G.add_edge(2, 7)
    G.add_edge(3, 8)
    G.add_edge(4, 9)
