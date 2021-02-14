import networkx as nx
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, cdist
from sklearn.cluster._kmeans import kmeans_plusplus
from tqdm import tqdm


def _find_closest_neighbors(query, ref, k=3):
    n_preds = ref.shape[0]
    tree = cKDTree(ref)
    dist, inds = tree.query(query, k=k)
    idx = np.argsort(dist[:, 0])
    neighbors = np.full(len(query), -1, dtype=int)
    picked = set()
    for i, ind in enumerate(inds[idx]):
        for j in ind:
            if j not in picked:
                picked.add(j)
                neighbors[idx[i]] = j
                break
        if len(picked) == n_preds:
            break
    return neighbors


class Assembler:
    def __init__(
        self,
        pickle_file,
        *,
        max_n_individuals,
        graph=None,
        paf_inds=None,
        greedy=False,
        pcutoff=0.1,
        paf_threshold=0.05,
        sort_by='degree',
        method='m1',
    ):
        if sort_by not in ('affinity', 'degree'):
            raise ValueError("`sort_by` must either be 'affinity' or 'degree'.")

        with open(pickle_file, 'rb') as file:
            self.data = pickle.load(file)
        self.max_n_individuals = max_n_individuals
        self.greedy = greedy
        self.pcutoff = pcutoff
        self.paf_threshold = paf_threshold
        self.sort_by = sort_by
        self.method = method
        self.metadata = self.parse_metadata(self.data)
        self.graph = graph or self.metadata['paf_graph']
        self.paf_inds = paf_inds or self.metadata['paf']

        self.assemblies = np.full(
            (len(self.metadata['imnames']), self.max_n_individuals, self.n_keypoints, 4),
            fill_value=np.nan,
        )
        self._trees = dict()

    def __getitem__(self, item):
        return self.data[self.metadata['imnames'][item]]

    @property
    def n_keypoints(self):
        return self.metadata['num_joints']

    def rank_frames_by_crowdedness(self, pcutoff=0.8, seed=None):
        if seed is None:
            seed = np.random.RandomState(69)
        dists = []
        for data in tqdm(self):
            detections = self.flatten_detections(data)
            xy = detections[detections[:, 2] >= pcutoff, :2]
            centroids = kmeans_plusplus(xy, self.max_n_individuals, random_state=seed)[0]
            dists.append(pdist(centroids).min())
        return np.argsort(dists)[::-1]

    @staticmethod
    def flatten_detections(data_dict):
        xy = []
        for i, (coords, conf) in enumerate(zip(data_dict['coordinates'][0],
                                               data_dict['confidence'])):
            if not np.any(coords):
                continue
            xy.append(np.c_[(coords, conf, [i] * len(coords))])
        if not xy:
            return None
        xy = np.concatenate(xy)
        ids = data_dict.get('identity', None)
        if ids is not None:
            data = np.empty((xy.shape[0], 6))
            data[:, 3] = np.concatenate(ids).argmax(axis=1)
        else:
            data = np.empty((xy.shape[0], 5))
        data[:, [0, 1, 2, -2]] = xy
        data[:, -1] = np.arange(data.shape[0])
        return data

    def extract_best_edges(
        self,
        detections,
        costs,
        trees=None,
    ):
        edges = []
        for (s, t), ind in zip(self.graph, self.paf_inds):
            mask_s = detections[:, -2] == s
            mask_t = detections[:, -2] == t
            if not (np.any(mask_s) and np.any(mask_t)):
                continue
            dets_s = detections[mask_s]
            dets_t = detections[mask_t]
            aff = costs[ind][self.method].copy()
            aff[np.isnan(aff)] = 0  # FIXME Why is it even NaN??

            if any(trees):
                vecs = np.vstack(
                    [[*xy_s, *xy_t] for xy_s in dets_s[:, :2] for xy_t in dets_t[:, :2]]
                )
                dists = []
                for n, tree in enumerate(trees, start=1):
                    if tree is None:
                        break
                    d, _ = tree.query(vecs)
                    dists.append(np.exp(-0.01 * n * d))
                w = np.sum(dists, axis=0) / np.sum(np.exp(-0.01 * np.arange(1, len(dists) + 1)))
                aff += w.reshape(aff.shape)

            if self.greedy:
                conf = np.outer(dets_s[:, 2], dets_t[:, 2])
                rows, cols = np.where(
                    (conf >= self.pcutoff * self.pcutoff)
                    & (aff >= self.paf_threshold)
                )
                candidates = sorted(
                    zip(rows, cols, aff[rows, cols]),
                    key=lambda x: x[2], reverse=True,
                )
                connections = []
                i_seen = set()
                j_seen = set()
                for i, j, w in candidates:
                    if i not in i_seen and j not in j_seen:
                        i_seen.add(i)
                        j_seen.add(j)
                        ii = int(dets_s[i, -1])
                        jj = int(dets_t[j, -1])
                        connections.append([ii, jj, w])
                        if len(connections) == self.max_n_individuals:
                            break
                edges.extend(connections)
            else:  # Optimal keypoint pairing
                inds_s = np.argsort(dets_s[:, 2])[::-1][:self.max_n_individuals]
                inds_t = np.argsort(dets_t[:, 2])[::-1][:self.max_n_individuals]
                keep_s = inds_s[dets_s[inds_s, 2] >= self.pcutoff]
                keep_t = inds_t[dets_t[inds_t, 2] >= self.pcutoff]
                dets_s = dets_s[keep_s]
                dets_t = dets_t[keep_t]
                aff = aff[np.ix_(keep_s, keep_t)]
                rows, cols = linear_sum_assignment(aff, maximize=True)
                for row, col in zip(rows, cols):
                    w = aff[row, col]
                    if w >= self.paf_threshold:
                        edges.append(
                            [int(dets_s[row, -1]), int(dets_t[col, -1]), w]
                        )
        return edges

    def form_individuals(
        self,
        detections,
        edges,
    ):
        n_missing = self.n_keypoints - len(np.unique(detections[:, -2]))
        all_inds = detections[:, -2].astype(int)
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        subsets0 = np.empty((0, self.n_keypoints))
        # Fill the subsets with unambiguous, complete individuals
        to_remove = []
        for chain in nx.connected_components(G):
            if len(chain) == self.n_keypoints - n_missing:
                row = -1 * np.ones(self.n_keypoints)
                nodes = list(chain)
                row[all_inds[nodes]] = nodes
                subsets0 = np.vstack((subsets0, row))
                to_remove.extend(nodes)
        G.remove_nodes_from(to_remove)
        if not len(G):
            return subsets0.astype(int)

        if self.sort_by == 'affinity':
            # Sort connections in descending order of part affinity
            edges_left = sorted(
                G.edges.data('weight'),
                key=lambda x: x[2],
                reverse=True
            )
        else:
            # Alternatively, sort edges in decreasing order
            # of their vertices' weighted degree. This is to encourage
            # dense, strong connections to come first.
            nodes_weight = defaultdict(int)
            for ind1, ind2, score in G.edges.data('weight'):
                nodes_weight[ind1] += score
                nodes_weight[ind2] += score
            for node in nodes_weight:
                nodes_weight[node] /= G.degree(node)
            edges_left = sorted(
                G.edges.data('weight'),
                key=lambda x: x[2] + nodes_weight[x[0]] + nodes_weight[x[1]],
                reverse=True
            )

        ambiguous = []
        subsets = np.empty((0, self.n_keypoints))
        for edge in edges_left:
            nodes = list(edge[:2])
            inds = all_inds[nodes]
            subset_inds = np.nonzero(subsets[:, inds] == nodes)[0]
            found = subset_inds.size
            if found == 1:
                subset = subsets[subset_inds[0]]
                if np.all(subset[inds] != -1):
                    ambiguous.append(edge)
                elif subset[inds[0]] == -1:
                    subset[inds[0]] = nodes[0]
                else:
                    subset[inds[1]] = nodes[1]
            elif found == 2:
                # Test whether nodes were found in the same subset
                if subset_inds[0] == subset_inds[1]:
                    continue
                membership = np.sum(subsets[subset_inds] >= 0, axis=0)
                if not np.any(membership == 2):
                    subsets = self._merge_disjoint_subsets(subsets, *subset_inds)
                else:
                    ambiguous.append(edge)
            else:
                row = -1 * np.ones(self.n_keypoints)
                row[inds] = nodes
                subsets = np.vstack((subsets, row))

        nrows = len(subsets)
        left = self.max_n_individuals - len(subsets0)
        if nrows > left > 0:
            subsets = subsets[np.argsort(np.sum(subsets == -1, axis=1))]
            for nrow in range(nrows - 1, left - 1, -1):
                mask = (subsets[:left] >= 0).astype(int)
                row = subsets[nrow]
                mask2 = (row >= 0).astype(int)
                temp = mask + mask2
                free_rows = np.flatnonzero(~np.any(temp == 2, axis=1))
                if not len(free_rows):
                    continue
                if free_rows.size == 1:
                    ind = free_rows[0]
                else:
                    xy = detections[row[row != -1].astype(int), :2]
                    dists = []
                    for free_row in free_rows:
                        sub_ = subsets[free_row]
                        xy_ref = detections[sub_[sub_ != -1].astype(int), :2]
                        dists.append(cdist(xy, xy_ref).min())
                    ind = free_rows[np.argmin(dists)]
                subsets = self._merge_disjoint_subsets(subsets, ind, nrow)

        subsets = np.vstack((subsets0, subsets)).astype(int)
        individuals, discarded = np.split(subsets, [self.max_n_individuals], axis=0)
        if np.any(discarded):
            # Add back the discarded groups of detections
            dists = np.full((len(individuals), len(discarded)), np.inf)
            # mask = (individuals >= 0).astype(int)
            # mask2 = (discarded >= 0).astype(int)
            # temp = mask[:, None] + mask2
            # free_rows = ~np.any(temp == 2, axis=2)
            # for i, row in enumerate(free_rows):
            #     if not np.any(row):
            #         continue
            #     sub_ = individuals[i]
            #     sub_ = sub_[sub_ != -1]
            #     xy_ref = detections[sub_, :2]
            #     for j, col in enumerate(row):
            #         if col:
            #             meh = discarded[j]
            #             meh = meh[meh != -1]
            #             xy = detections[meh, :2]
            #             dists[i, j] = cdist(xy_ref, xy).min()
            # rows, cols = linear_sum_assignment(dists)
            # for row, col in zip(rows, cols):
            #     individuals[row] += discarded[col] + 1
        return individuals

    def assemble(self, window_size=1):
        for i, data_dict in enumerate(tqdm(self)):
            detections = self.flatten_detections(data_dict)
            if detections is None:
                continue
            edges = self.extract_best_edges(
                detections,
                data_dict["costs"],
                [self._trees.get(i - j, None) for j in range(1, window_size + 1)]
            )
            # Store selected edges for subsequent frames
            vecs = np.vstack([detections[edge[:2], :2] for edge in edges])
            self._trees[i] = cKDTree(vecs.reshape((-1, 4)))

            subsets = self.form_individuals(
                detections,
                edges,
            )
            for assembly, subset in zip(self.assemblies[i], subsets):
                mask = subset != -1
                assembly[mask, :3] = detections[subset[mask], :-2]

    @staticmethod
    def _merge_disjoint_subsets(subsets, row1, row2):
        subsets[row1] += subsets[row2] + 1
        return np.delete(subsets, row2, axis=0)

    def propagate_assemblies(self, ind):
        ref = np.concatenate(self.assemblies[ind])
        valid = ~np.isnan(ref).any(axis=1)
        ref = ref[valid]
        xy_ref = ref[:, :2]
        ids_ref = np.array([i for i in range(self.max_n_individuals)
                            for _ in range(self.n_keypoints)])[valid]
        dest = ind + 1
        candidates = self.flatten_detections(self[dest])
        neighbors = _find_closest_neighbors(xy_ref, candidates[:, :2], 2)
        mask = neighbors != -1
        inds = neighbors[mask]
        ids_ref = ids_ref[mask]
        for id_ in np.unique(ids_ref):
            temp = candidates[inds[ids_ref == id_]]
            self.assemblies[dest, id_, temp[:, -1].astype(int)] = temp[:, :3]

    @staticmethod
    def parse_metadata(data):
        params = dict()
        params["joint_names"] = data["metadata"]["all_joints_names"]
        params["num_joints"] = len(params["joint_names"])
        partaffinityfield_graph = data["metadata"]["PAFgraph"]
        params["paf"] = np.arange(len(partaffinityfield_graph))
        params["paf_graph"] = params["paf_links"] = [
            partaffinityfield_graph[l] for l in params["paf"]
        ]
        params["bpts"] = params["ibpts"] = range(params["num_joints"])
        params["imnames"] = [fn for fn in list(data) if fn != "metadata"]
        return params

    def to_h5(self, output_name):
        data = self.assemblies[..., :3].reshape((self.assemblies.shape[0], -1))
        index = pd.MultiIndex.from_product(
            [
                ['scorer'],
                map(str, range(self.max_n_individuals)),
                map(str, range(self.n_keypoints)),
                ['x', 'y', 'likelihood']
            ],
            names=['scorer', 'individuals', 'bodyparts', 'coords']
        )
        df = pd.DataFrame(data, columns=index)
        df.to_hdf(output_name, key='ass')


full_data_file = '/Users/Jessy/Desktop/MultiMouse-Daniel-2019-12-16/videos/videocompressed1DLC_resnet50_MultiMouseDec16shuffle0_60000_full.pickle'
ass = Assembler(
    full_data_file,
    max_n_individuals=3,
)
ass.assemble(window_size=5)


def flag_anomalous_assemblies(assemblies):
    bboxes = np.full((assemblies.shape[0], assemblies.shape[1], 4), np.nan)
    bboxes[..., 0] = np.nanmin(assemblies[..., 0], axis=2)
    bboxes[..., 1] = np.nanmin(assemblies[..., 1], axis=2)
    bboxes[..., 2] = np.nanmax(assemblies[..., 0], axis=2)
    bboxes[..., 3] = np.nanmax(assemblies[..., 1], axis=2)
    area = (bboxes[..., 2] - bboxes[..., 0]) * (bboxes[..., 3] - bboxes[..., 1])
