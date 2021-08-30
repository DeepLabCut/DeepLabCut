"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
import heapq
import itertools
import multiprocessing
import networkx as nx
import numpy as np
import operator
import pandas as pd
import pickle
import warnings
from collections import defaultdict
from dataclasses import dataclass
from math import sqrt, erf
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, cdist
from scipy.special import softmax
from scipy.stats import gaussian_kde, chi2
from tqdm import tqdm
from typing import Tuple


def _conv_square_to_condensed_indices(ind_row, ind_col, n):
    if ind_row == ind_col:
        raise ValueError("There are no diagonal elements in condensed matrices.")

    if ind_row < ind_col:
        ind_row, ind_col = ind_col, ind_row
    return n * ind_col - ind_col * (ind_col + 1) // 2 + ind_row - 1 - ind_col


Position = Tuple[float, float]


@dataclass(frozen=True)
class Joint:
    pos: Position
    confidence: float = 1.0
    label: int = None
    idx: int = None
    group: int = -1


class Link:
    def __init__(self, j1, j2, affinity=1):
        self.j1 = j1
        self.j2 = j2
        self.affinity = affinity
        self._length = sqrt((j1.pos[0] - j2.pos[0]) ** 2 + (j1.pos[1] - j2.pos[1]) ** 2)

    def __repr__(self):
        return (
            f"Link {self.idx}, affinity={self.affinity:.2f}, length={self.length:.2f}"
        )

    @property
    def confidence(self):
        return self.j1.confidence * self.j2.confidence

    @property
    def idx(self):
        return self.j1.idx, self.j2.idx

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, length):
        self._length = length

    def to_vector(self):
        return [*self.j1.pos, *self.j2.pos]


class Assembly:
    def __init__(self, size):
        self.data = np.full((size, 4), np.nan)
        self._affinity = 0
        self._links = []
        self._visible = set()
        self._idx = set()
        self._dict = dict()

    def __len__(self):
        return len(self._visible)

    def __contains__(self, assembly):
        return bool(self._visible.intersection(assembly._visible))

    def __add__(self, other):
        if other in self:
            raise ValueError("Assemblies contain shared joints.")

        assembly = Assembly(self.data.shape[0])
        for link in self._links + other._links:
            assembly.add_link(link)
        return assembly

    @property
    def xy(self):
        return self.data[:, :2]

    @property
    def extent(self):
        bbox = np.empty(4)
        bbox[:2] = np.nanmin(self.xy, axis=0)
        bbox[2:] = np.nanmax(self.xy, axis=0)
        return bbox

    @property
    def area(self):
        x1, y1, x2, y2 = self.extent
        return (x2 - x1) * (y2 - y1)

    @property
    def confidence(self):
        return np.nanmean(self.data[:, 2])

    @property
    def soft_identity(self):
        data = self.data[~np.isnan(self.data).any(axis=1)]
        unq, idx, cnt = np.unique(data[:, 3], return_inverse=True, return_counts=True)
        avg = np.bincount(idx, weights=data[:, 2]) / cnt
        soft = softmax(avg)
        return dict(zip(unq.astype(int), soft))

    @property
    def affinity(self):
        n_links = self.n_links
        if not n_links:
            return 0
        return self._affinity / n_links

    @property
    def n_links(self):
        return len(self._links)

    def intersection_with(self, other):
        x11, y11, x21, y21 = self.extent
        x12, y12, x22, y22 = other.extent
        x1 = max(x11, x12)
        y1 = max(y11, y12)
        x2 = min(x21, x22)
        y2 = min(y21, y22)
        if x2 < x1 or y2 < y1:
            return 0
        ll = np.array([x1, y1])
        ur = np.array([x2, y2])
        xy1 = self.xy[~np.isnan(self.xy).any(axis=1)]
        xy2 = other.xy[~np.isnan(other.xy).any(axis=1)]
        in1 = np.all((xy1 >= ll) & (xy1 <= ur), axis=1).sum()
        in2 = np.all((xy2 >= ll) & (xy2 <= ur), axis=1).sum()
        return min(in1 / len(self), in2 / len(other))

    def add_joint(self, joint):
        if joint.label in self._visible or joint.label is None:
            return False
        self.data[joint.label] = *joint.pos, joint.confidence, joint.group
        self._visible.add(joint.label)
        self._idx.add(joint.idx)
        return True

    def remove_joint(self, joint):
        if joint.label not in self._visible:
            return False
        self.data[joint.label] = np.nan
        self._visible.remove(joint.label)
        self._idx.remove(joint.idx)
        return True

    def add_link(self, link, store_dict=False):
        if store_dict:
            # Selective copy; deepcopy is >5x slower
            self._dict = {
                "data": self.data.copy(),
                "_affinity": self._affinity,
                "_links": self._links.copy(),
                "_visible": self._visible.copy(),
                "_idx": self._idx.copy(),
            }
        i1, i2 = link.idx
        if i1 in self._idx and i2 in self._idx:
            self._affinity += link.affinity
            self._links.append(link)
            return False
        if link.j1.label in self._visible and link.j2.label in self._visible:
            return False
        self.add_joint(link.j1)
        self.add_joint(link.j2)
        self._affinity += link.affinity
        self._links.append(link)
        return True

    def calc_pairwise_distances(self):
        return pdist(self.xy, metric="sqeuclidean")


class Assembler:
    def __init__(
        self,
        data,
        *,
        max_n_individuals,
        n_multibodyparts,
        graph=None,
        paf_inds=None,
        greedy=False,
        pcutoff=0.1,
        min_affinity=0.05,
        min_n_links=2,
        max_overlap=0.8,
        identity_only=False,
        nan_policy="little",
        force_fusion=False,
        add_discarded=False,
        window_size=0,
        method="m1",
    ):
        self.data = data
        self.metadata = self.parse_metadata(self.data)
        self.max_n_individuals = max_n_individuals
        self.n_multibodyparts = n_multibodyparts
        self.n_uniquebodyparts = self.n_keypoints - n_multibodyparts
        self.greedy = greedy
        self.pcutoff = pcutoff
        self.min_affinity = min_affinity
        self.min_n_links = min_n_links
        self.max_overlap = max_overlap
        self._has_identity = "identity" in self[0]
        if identity_only and not self._has_identity:
            warnings.warn(
                "The network was not trained with identity; setting `identity_only` to False."
            )
        self.identity_only = identity_only & self._has_identity
        self.nan_policy = nan_policy
        self.force_fusion = force_fusion
        self.add_discarded = add_discarded
        self.window_size = window_size
        self.method = method
        self.graph = graph or self.metadata["paf_graph"]
        self.paf_inds = paf_inds or self.metadata["paf"]
        self._gamma = 0.01
        self._trees = dict()
        self.safe_edge = False
        self._kde = None
        self.assemblies = dict()
        self.unique = dict()

    def __getitem__(self, item):
        return self.data[self.metadata["imnames"][item]]

    @property
    def n_keypoints(self):
        return self.metadata["num_joints"]

    def calibrate(self, train_data_file):
        df = pd.read_hdf(train_data_file)
        try:
            df.drop("single", level="individuals", axis=1, inplace=True)
        except KeyError:
            pass
        n_bpts = len(df.columns.get_level_values("bodyparts").unique())
        xy = df.to_numpy().reshape((-1, n_bpts, 2))
        frac_valid = np.mean(~np.isnan(xy), axis=(1, 2))
        # Only keeps skeletons that are more than 90% complete
        xy = xy[frac_valid >= 0.9]
        # TODO Normalize dists by longest length?
        # TODO Smarter imputation technique (Bayesian? Grassmann averages?)
        dists = np.vstack([pdist(data, "sqeuclidean") for data in xy])
        mu = np.nanmean(dists, axis=0)
        missing = np.isnan(dists)
        dists = np.where(missing, mu, dists)
        try:
            kde = gaussian_kde(dists.T)
            kde.mean = mu
            self._kde = kde
            self.safe_edge = True
        except np.linalg.LinAlgError:
            # Covariance matrix estimation fails due to numerical singularities
            warnings.warn(
                "The assembler could not be robustly calibrated. Continuing without it..."
            )

    def calc_assembly_mahalanobis_dist(
        self, assembly, return_proba=False, nan_policy="little"
    ):
        if self._kde is None:
            raise ValueError("Assembler should be calibrated first with training data.")

        dists = assembly.calc_pairwise_distances() - self._kde.mean
        mask = np.isnan(dists)
        if mask.any() and nan_policy == "little":
            inds = np.flatnonzero(~mask)
            dists = dists[inds]
            inv_cov = self._kde.inv_cov[np.ix_(inds, inds)]
            # Correct distance to account for missing observations
            factor = self._kde.d / len(inds)
        else:
            # Alternatively, reduce contribution of missing values to the Mahalanobis
            # distance to zero by substituting the corresponding means.
            dists[mask] = 0
            mask.fill(False)
            inv_cov = self._kde.inv_cov
            factor = 1
        dot = dists @ inv_cov
        mahal = factor * sqrt(np.sum((dot * dists), axis=-1))
        if return_proba:
            proba = 1 - chi2.cdf(mahal, np.sum(~mask))
            return mahal, proba
        return mahal

    def calc_link_probability(self, link):
        if self._kde is None:
            raise ValueError("Assembler should be calibrated first with training data.")

        i = link.j1.label
        j = link.j2.label
        ind = _conv_square_to_condensed_indices(i, j, self.n_multibodyparts)
        mu = self._kde.mean[ind]
        sigma = self._kde.covariance[i, j]
        z = (link.length ** 2 - mu) / sigma
        return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))

    @staticmethod
    def _flatten_detections(data_dict):
        ind = 0
        coordinates = data_dict["coordinates"][0]
        confidence = data_dict["confidence"]
        ids = data_dict.get("identity", None)
        if ids is None:
            ids = [np.ones(len(arr), dtype=int) * -1 for arr in confidence]
        else:
            ids = [arr.argmax(axis=1) for arr in ids]
        for i, (coords, conf, id_) in enumerate(zip(coordinates, confidence, ids)):
            if not np.any(coords):
                continue
            for xy, p, g in zip(coords, conf, id_):
                joint = Joint(tuple(xy), p.item(), i, ind, g)
                ind += 1
                yield joint

    def extract_best_links(self, joints_dict, costs, trees=None):
        links = []
        for ind in self.paf_inds:
            s, t = self.graph[ind]
            dets_s = joints_dict.get(s, None)
            dets_t = joints_dict.get(t, None)
            if dets_s is None or dets_t is None:
                continue
            lengths = costs[ind]["distance"]
            if np.isinf(lengths).all():
                continue
            aff = costs[ind][self.method].copy()
            aff[np.isnan(aff)] = 0

            if trees:
                vecs = np.vstack(
                    [[*det_s.pos, *det_t.pos] for det_s in dets_s for det_t in dets_t]
                )
                dists = []
                for n, tree in enumerate(trees, start=1):
                    d, _ = tree.query(vecs)
                    dists.append(np.exp(-self._gamma * n * d))
                w = np.mean(dists, axis=0)
                aff *= w.reshape(aff.shape)

            if self.greedy:
                conf = np.asarray(
                    [
                        [det_s.confidence * det_t.confidence for det_t in dets_t]
                        for det_s in dets_s
                    ]
                )
                rows, cols = np.where(
                    (conf >= self.pcutoff * self.pcutoff) & (aff >= self.min_affinity)
                )
                candidates = sorted(
                    zip(rows, cols, aff[rows, cols], lengths[rows, cols]),
                    key=lambda x: x[2],
                    reverse=True,
                )
                i_seen = set()
                j_seen = set()
                for i, j, w, l in candidates:
                    if i not in i_seen and j not in j_seen:
                        i_seen.add(i)
                        j_seen.add(j)
                        links.append(Link(dets_s[i], dets_t[j], w))
                        if len(i_seen) == self.max_n_individuals:
                            break
            else:  # Optimal keypoint pairing
                inds_s = sorted(
                    range(len(dets_s)), key=lambda x: dets_s[x].confidence, reverse=True
                )[: self.max_n_individuals]
                inds_t = sorted(
                    range(len(dets_t)), key=lambda x: dets_t[x].confidence, reverse=True
                )[: self.max_n_individuals]
                keep_s = [
                    ind for ind in inds_s if dets_s[ind].confidence >= self.pcutoff
                ]
                keep_t = [
                    ind for ind in inds_t if dets_t[ind].confidence >= self.pcutoff
                ]
                aff = aff[np.ix_(keep_s, keep_t)]
                rows, cols = linear_sum_assignment(aff, maximize=True)
                for row, col in zip(rows, cols):
                    w = aff[row, col]
                    if w >= self.min_affinity:
                        links.append(Link(dets_s[keep_s[row]], dets_t[keep_t[col]], w))
        return links

    def _fill_assembly(self, assembly, lookup, assembled, safe_edge, nan_policy):
        stack = []
        visited = set()
        tabu = []
        counter = itertools.count()

        def push_to_stack(i):
            for j, link in lookup[i].items():
                if j in assembly._idx:
                    continue
                if link.idx in visited:
                    continue
                heapq.heappush(stack, (-link.affinity, next(counter), link))
                visited.add(link.idx)

        for idx in assembly._idx:
            push_to_stack(idx)

        while stack and len(assembly) < self.n_multibodyparts:
            _, _, best = heapq.heappop(stack)
            i, j = best.idx
            if i in assembly._idx:
                new_ind = j
            elif j in assembly._idx:
                new_ind = i
            else:
                continue
            if new_ind in assembled:
                continue
            if safe_edge:
                d_old = self.calc_assembly_mahalanobis_dist(
                    assembly, nan_policy=nan_policy
                )
                success = assembly.add_link(best, store_dict=True)
                if not success:
                    assembly._dict = dict()
                    continue
                d = self.calc_assembly_mahalanobis_dist(assembly, nan_policy=nan_policy)
                if d < d_old:
                    push_to_stack(new_ind)
                    try:
                        _, _, link = heapq.heappop(tabu)
                        heapq.heappush(stack, (-link.affinity, next(counter), link))
                    except IndexError:
                        pass
                else:
                    heapq.heappush(tabu, (d - d_old, next(counter), best))
                    assembly.__dict__.update(assembly._dict)
                assembly._dict = dict()
            else:
                assembly.add_link(best)
                push_to_stack(new_ind)

    def build_assemblies(self, links):
        lookup = defaultdict(dict)
        for link in links:
            i, j = link.idx
            lookup[i][j] = link
            lookup[j][i] = link

        assemblies = []
        assembled = set()

        # Fill the subsets with unambiguous, complete individuals
        G = nx.Graph([link.idx for link in links])
        for chain in nx.connected_components(G):
            if len(chain) == self.n_multibodyparts:
                edges = [tuple(sorted(edge)) for edge in G.edges(chain)]
                assembly = Assembly(self.n_multibodyparts)
                for link in links:
                    i, j = link.idx
                    if (i, j) in edges:
                        success = assembly.add_link(link)
                        if success:
                            lookup[i].pop(j)
                            lookup[j].pop(i)
                assembled.update(assembly._idx)
                assemblies.append(assembly)

        if len(assemblies) == self.max_n_individuals:
            return assemblies, assembled

        for link in sorted(links, key=lambda x: x.affinity, reverse=True):
            if any(i in assembled for i in link.idx):
                continue
            assembly = Assembly(self.n_multibodyparts)
            assembly.add_link(link)
            self._fill_assembly(
                assembly, lookup, assembled, self.safe_edge, self.nan_policy
            )
            for link in assembly._links:
                i, j = link.idx
                lookup[i].pop(j)
                lookup[j].pop(i)
            assembled.update(assembly._idx)
            assemblies.append(assembly)

        # Fuse superfluous assemblies
        n_extra = len(assemblies) - self.max_n_individuals
        if n_extra > 0:
            if self.safe_edge:
                ds_old = [
                    self.calc_assembly_mahalanobis_dist(assembly)
                    for assembly in assemblies
                ]
                while len(assemblies) > self.max_n_individuals:
                    ds = []
                    for i, j in itertools.combinations(range(len(assemblies)), 2):
                        if assemblies[j] not in assemblies[i]:
                            temp = assemblies[i] + assemblies[j]
                            d = self.calc_assembly_mahalanobis_dist(temp)
                            delta = d - max(ds_old[i], ds_old[j])
                            ds.append((i, j, delta, d, temp))
                    if not ds:
                        break
                    min_ = sorted(ds, key=lambda x: x[2])
                    i, j, delta, d, new = min_[0]
                    if delta < 0 or len(min_) == 1:
                        assemblies[i] = new
                        assemblies.pop(j)
                        ds_old[i] = d
                        ds_old.pop(j)
                    else:
                        break
            elif self.force_fusion:
                assemblies = sorted(assemblies, key=len)
                for nrow in range(n_extra):
                    assembly = assemblies[nrow]
                    candidates = [a for a in assemblies[nrow:] if assembly not in a]
                    if not candidates:
                        continue
                    if len(candidates) == 1:
                        candidate = candidates[0]
                    else:
                        dists = []
                        for cand in candidates:
                            d = cdist(assembly.xy, cand.xy)
                            dists.append(np.nanmin(d))
                        candidate = candidates[np.argmin(dists)]
                    ind = assemblies.index(candidate)
                    assemblies[ind] += assembly
            else:
                store = dict()
                for assembly in assemblies:
                    if len(assembly) != self.n_multibodyparts:
                        for i in assembly._idx:
                            store[i] = assembly
                used = [link for assembly in assemblies for link in assembly._links]
                unconnected = [link for link in links if link not in used]
                for link in unconnected:
                    i, j = link.idx
                    try:
                        if store[j] not in store[i]:
                            temp = store[i] + store[j]
                            store[i].__dict__.update(temp.__dict__)
                            assemblies.remove(store[j])
                            for idx in store[j]._idx:
                                store[idx] = store[i]
                    except KeyError:
                        pass

        # Second pass without edge safety
        for assembly in assemblies:
            if len(assembly) != self.n_multibodyparts:
                self._fill_assembly(assembly, lookup, assembled, False, "")
                assembled.update(assembly._idx)

        return assemblies, assembled

    def _assemble(self, data_dict, ind_frame):
        joints = list(self._flatten_detections(data_dict))
        if not joints:
            return None, None

        bag = defaultdict(list)
        for joint in joints:
            bag[joint.label].append(joint)

        assembled = set()

        if self.n_uniquebodyparts:
            unique = np.full((self.n_uniquebodyparts, 3), np.nan)
            for n, ind in enumerate(range(self.n_multibodyparts, self.n_keypoints)):
                dets = bag[ind]
                if not dets:
                    continue
                if len(dets) > 1:
                    det = max(dets, key=lambda x: x.confidence)
                else:
                    det = dets[0]
                # Mark the unique body parts as assembled anyway so
                # they are not used later on to fill assemblies.
                assembled.update(d.idx for d in dets)
                if det.confidence <= self.pcutoff and not self.add_discarded:
                    continue
                unique[n] = *det.pos, det.confidence
            if np.isnan(unique).all():
                unique = None
        else:
            unique = None

        if not any(i in bag for i in range(self.n_multibodyparts)):
            return None, unique

        if self.identity_only:
            assemblies = []
            get_attr = operator.attrgetter("group")
            temp = sorted(
                (joint for joint in joints if np.isfinite(joint.confidence)),
                key=get_attr,
            )
            groups = itertools.groupby(temp, get_attr)
            for _, group in groups:
                ass = Assembly(self.n_multibodyparts)
                for joint in sorted(group, key=lambda x: x.confidence, reverse=True):
                    if (
                        joint.confidence >= self.pcutoff
                        and joint.label < self.n_multibodyparts
                    ):
                        ass.add_joint(joint)
                if len(ass):
                    assemblies.append(ass)
                    assembled.update(ass._idx)
        else:
            trees = []
            for j in range(1, self.window_size + 1):
                tree = self._trees.get(ind_frame - j, None)
                if tree is not None:
                    trees.append(tree)

            links = self.extract_best_links(bag, data_dict["costs"], trees)
            if self._kde:
                for link in links[::-1]:
                    p = max(self.calc_link_probability(link), 0.001)
                    link.affinity *= p
                    if link.affinity < self.min_affinity:
                        links.remove(link)

            if self.window_size >= 1 and links:
                # Store selected edges for subsequent frames
                vecs = np.vstack([link.to_vector() for link in links])
                self._trees[ind_frame] = cKDTree(vecs)

            assemblies, assembled_ = self.build_assemblies(links)
            assembled.update(assembled_)

        # Remove invalid assemblies
        discarded = set(
            joint
            for joint in joints
            if joint.idx not in assembled and np.isfinite(joint.confidence)
        )
        for assembly in assemblies[::-1]:
            if 0 < assembly.n_links < self.min_n_links:
                for link in assembly._links:
                    discarded.update((link.j1, link.j2))
                assemblies.remove(assembly)
        if 0 < self.max_overlap < 1:  # Non-maximum pose suppression
            if self._kde is not None:
                scores = [
                    -self.calc_assembly_mahalanobis_dist(ass) for ass in assemblies
                ]
            else:
                scores = [ass._affinity for ass in assemblies]
            lst = list(zip(scores, assemblies))
            assemblies = []
            while lst:
                temp = max(lst, key=lambda x: x[0])
                lst.remove(temp)
                assemblies.append(temp[1])
                for pair in lst[::-1]:
                    if temp[1].intersection_with(pair[1]) >= self.max_overlap:
                        lst.remove(pair)
        if len(assemblies) > self.max_n_individuals:
            assemblies = sorted(assemblies, key=len, reverse=True)
            for assembly in assemblies[self.max_n_individuals :]:
                for link in assembly._links:
                    discarded.update((link.j1, link.j2))
            assemblies = assemblies[: self.max_n_individuals]

        if self.add_discarded and discarded:
            # Fill assemblies with unconnected body parts
            for joint in sorted(discarded, key=lambda x: x.confidence, reverse=True):
                if self.safe_edge:
                    for assembly in assemblies:
                        if joint.label in assembly._visible:
                            continue
                        d_old = self.calc_assembly_mahalanobis_dist(assembly)
                        assembly.add_joint(joint)
                        d = self.calc_assembly_mahalanobis_dist(assembly)
                        if d < d_old:
                            break
                        assembly.remove_joint(joint)
                else:
                    dists = []
                    for i, assembly in enumerate(assemblies):
                        if joint.label in assembly._visible:
                            continue
                        d = cdist(assembly.xy, np.atleast_2d(joint.pos))
                        dists.append((i, np.nanmin(d)))
                    if not dists:
                        continue
                    min_ = sorted(dists, key=lambda x: x[1])
                    ind, _ = min_[0]
                    assemblies[ind].add_joint(joint)

        return assemblies, unique

    def assemble(self, chunk_size=1, n_processes=None):
        self.assemblies = dict()
        self.unique = dict()
        # Spawning (rather than forking) multiple processes does not
        # work nicely with the GUI or interactive sessions.
        # In that case, we fall back to the serial assembly.
        if chunk_size == 0 or multiprocessing.get_start_method() == "spawn":
            for i, data_dict in enumerate(tqdm(self)):
                assemblies, unique = self._assemble(data_dict, i)
                if assemblies:
                    self.assemblies[i] = assemblies
                if unique is not None:
                    self.unique[i] = unique
        else:
            global wrapped  # Hack to make the function pickable

            def wrapped(i):
                return i, self._assemble(self[i], i)

            n_frames = len(self.metadata["imnames"])
            with multiprocessing.Pool(n_processes) as p:
                with tqdm(total=n_frames) as pbar:
                    for i, (assemblies, unique) in p.imap_unordered(
                        wrapped, range(n_frames), chunksize=chunk_size
                    ):
                        if assemblies:
                            self.assemblies[i] = assemblies
                        if unique is not None:
                            self.unique[i] = unique
                        pbar.update()

    @staticmethod
    def parse_metadata(data):
        params = dict()
        params["joint_names"] = data["metadata"]["all_joints_names"]
        params["num_joints"] = len(params["joint_names"])
        params["paf_graph"] = data["metadata"]["PAFgraph"]
        params["paf"] = data["metadata"].get(
            "PAFinds", np.arange(len(params["joint_names"]))
        )
        params["bpts"] = params["ibpts"] = range(params["num_joints"])
        params["imnames"] = [fn for fn in list(data) if fn != "metadata"]
        return params

    def to_h5(self, output_name):
        data = np.full(
            (
                len(self.metadata["imnames"]),
                self.max_n_individuals,
                self.n_multibodyparts,
                4,
            ),
            fill_value=np.nan,
        )
        for ind, assemblies in self.assemblies.items():
            for n, assembly in enumerate(assemblies):
                data[ind, n] = assembly.data
        index = pd.MultiIndex.from_product(
            [
                ["scorer"],
                map(str, range(self.max_n_individuals)),
                map(str, range(self.n_multibodyparts)),
                ["x", "y", "likelihood"],
            ],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )
        temp = data[..., :3].reshape((data.shape[0], -1))
        df = pd.DataFrame(temp, columns=index)
        df.to_hdf(output_name, key="ass")

    def to_pickle(self, output_name):
        data = dict()
        for ind, assemblies in self.assemblies.items():
            data[ind] = [ass.data for ass in assemblies]
        if self.unique:
            data["single"] = self.unique
        with open(output_name, "wb") as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


def calc_object_keypoint_similarity(xy_pred, xy_true, sigma):
    visible_gt = ~np.isnan(xy_true).all(axis=1)
    if visible_gt.sum() < 2:  # At least 2 points needed to calculate scale
        return np.nan
    true = xy_true[visible_gt]
    pred = xy_pred[visible_gt]
    pred[np.isnan(pred)] = np.inf
    dist_squared = np.sum((pred - true) ** 2, axis=1)
    scale_squared = np.product(np.ptp(true, axis=0) + np.spacing(1))
    if np.isclose(scale_squared, 0):
        return np.nan
    k_squared = (2 * sigma) ** 2
    oks = np.exp(-dist_squared / (2 * scale_squared * k_squared))
    return np.mean(oks)


def match_assemblies(ass_pred, ass_true, sigma):
    inds_true = list(range(len(ass_true)))
    inds_pred = np.argsort(
        [ins.affinity if ins.n_links else ins.confidence for ins in ass_pred]
    )[::-1]
    matched = []
    for ind_pred in inds_pred:
        xy_pred = ass_pred[ind_pred].xy
        oks = []
        for ind_true in inds_true:
            xy_true = ass_true[ind_true].xy
            oks.append(calc_object_keypoint_similarity(xy_pred, xy_true, sigma))
        if np.all(np.isnan(oks)):
            continue
        ind_best = np.nanargmax(oks)
        ind_true_best = inds_true.pop(ind_best)
        matched.append((ass_pred[ind_pred], ass_true[ind_true_best], oks[ind_best]))
        if not inds_true:
            break
    unmatched = [ass_true[ind] for ind in inds_true]
    return matched, unmatched


def parse_ground_truth_data_file(h5_file):
    df = pd.read_hdf(h5_file)
    try:
        df.drop("single", axis=1, level="individuals", inplace=True)
    except KeyError:
        pass
    n_individuals = len(df.columns.get_level_values("individuals").unique())
    n_bodyparts = len(df.columns.get_level_values("bodyparts").unique())
    data = df.to_numpy().reshape((df.shape[0], n_individuals, n_bodyparts, 3))
    return _parse_ground_truth_data(data)


def _parse_ground_truth_data(data):
    gt = dict()
    for i, arr in enumerate(data):
        temp = []
        for row in arr:
            if np.isnan(row[:, :2]).all():
                continue
            ass = Assembly(row.shape[0])
            ass.data[:, : row.shape[1]] = row
            temp.append(ass)
        if not temp:
            continue
        gt[i] = temp
    return gt


def find_outlier_assemblies(dict_of_assemblies, criterion="area", qs=(5, 95)):
    if not hasattr(Assembly, criterion):
        raise ValueError(f"Invalid criterion {criterion}.")

    if len(qs) != 2:
        raise ValueError(
            "Two percentiles (for lower and upper bounds) should be given."
        )

    tuples = []
    for frame_ind, assemblies in dict_of_assemblies.items():
        for assembly in assemblies:
            tuples.append((frame_ind, getattr(assembly, criterion)))
    frame_inds, vals = zip(*tuples)
    vals = np.asarray(vals)
    lo, up = np.percentile(vals, qs, interpolation="nearest")
    inds = np.flatnonzero((vals < lo) | (vals > up)).tolist()
    return list(set(frame_inds[i] for i in inds))


def evaluate_assembly(
    ass_pred_dict,
    ass_true_dict,
    oks_sigma=0.072,
    oks_thresholds=np.linspace(0.5, 0.95, 10),
):
    # sigma is taken as the median of all COCO keypoint standard deviations
    all_matched = []
    all_unmatched = []
    for ind, ass_pred in tqdm(ass_pred_dict.items()):
        ass_true = ass_true_dict.get(ind)
        if ass_true is None:
            continue
        matched, unmatched = match_assemblies(ass_pred, ass_true, oks_sigma)
        all_matched.extend(matched)
        all_unmatched.extend(unmatched)
    if not all_matched:
        return {
            "precisions": np.array([]),
            "recalls": np.array([]),
            "mAP": 0.0,
            "mAR": 0.0,
        }

    conf_pred = np.asarray([match[0].affinity for match in all_matched])
    idx = np.argsort(-conf_pred, kind="mergesort")
    # Sort matching score (OKS) in descending order of assembly affinity
    oks = np.asarray([match[2] for match in all_matched])[idx]
    ntot = len(all_matched) + len(all_unmatched)
    recall_thresholds = np.linspace(0, 1, 101)
    precisions = []
    recalls = []
    for th in oks_thresholds:
        tp = np.cumsum(oks >= th)
        fp = np.cumsum(oks < th)
        rc = tp / ntot
        pr = tp / (fp + tp + np.spacing(1))
        recall = rc[-1]
        # Guarantee precision decreases monotonically
        # See https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)
        for i in range(len(pr) - 1, 0, -1):
            if pr[i] > pr[i - 1]:
                pr[i - 1] = pr[i]
        inds_rc = np.searchsorted(rc, recall_thresholds)
        precision = np.zeros(inds_rc.shape)
        valid = inds_rc < len(pr)
        precision[valid] = pr[inds_rc[valid]]
        precisions.append(precision)
        recalls.append(recall)
    precisions = np.asarray(precisions)
    recalls = np.asarray(recalls)
    return {
        "precisions": precisions,
        "recalls": recalls,
        "mAP": precisions.mean(),
        "mAR": recalls.mean(),
    }
