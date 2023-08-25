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
import numpy as np
import pandas as pd
import pickle
import re
from deeplabcut.post_processing import columnwise_spline_interp
from deeplabcut.utils import auxiliaryfunctions
from tqdm import trange


class TrackletManager:
    def __init__(self, config, min_swap_len=2, min_tracklet_len=2, max_gap=0):
        """

        Parameters
        ----------
        config : str
            Path to a configuration file.
        min_swap_len : float, optional (default=2)
            Minimum swap length.
            Swaps shorter than 2 frames are discarded by default.
        min_tracklet_len : float, optional (default=2)
            Minimum tracklet length.
            Tracklets shorter than 2 frames are discarded by default.
        max_gap : int, optional (default = 0).
            Number of frames to consider when filling in missing data.

        Examples
        --------

        manager = TrackletManager(config_path, min_swap_frac=0, min_tracklet_frac=0)

        manager.load_tracklets_from_pickle(filename)
        # Alternatively
        manager.load_tracklets_from_h5(filename)

        manager.find_swapping_bodypart_pairs()
        """
        self.config = config
        self.cfg = auxiliaryfunctions.read_config(config)
        self.min_swap_len = min_swap_len
        self.min_tracklet_len = min_tracklet_len
        self.max_gap = max_gap

        self.filename = ""
        self.data = None
        self.xy = None
        self._xy = None
        self.prob = None
        self.nframes = 0
        self.times = []
        self.scorer = None
        self.bodyparts = []
        self.nindividuals = len(self.cfg["individuals"])
        self.individuals = []
        self.tracklet2id = []
        self.tracklet2bp = []
        self.swapping_pairs = []
        self.swapping_bodyparts = []
        self._label_pairs = None

    def _load_tracklets(self, tracklets, auto_fill):
        header = tracklets.pop("header")
        self.scorer = header.get_level_values("scorer").unique().to_list()
        bodyparts = header.get_level_values("bodyparts")
        bodyparts_multi = [
            bp for bp in self.cfg["multianimalbodyparts"] if bp in bodyparts
        ]
        bodyparts_single = self.cfg["uniquebodyparts"]
        mask_multi = bodyparts.isin(bodyparts_multi)
        mask_single = bodyparts.isin(bodyparts_single)
        self.bodyparts = list(bodyparts[mask_multi]) * self.nindividuals + list(
            bodyparts[mask_single]
        )

        # Sort tracklets by length to prioritize greater continuity
        temp = sorted(tracklets.values(), key=len)
        if not len(temp):
            raise IOError("Tracklets are empty.")

        get_frame_ind = lambda s: int(re.findall(r"\d+", s)[0])

        # Drop tracklets that are too short
        tracklets_sorted = []
        last_frames = []
        for tracklet in temp:
            last_frames.append(get_frame_ind(list(tracklet)[-1]))
            if len(tracklet) > self.min_tracklet_len:
                tracklets_sorted.append(tracklet)
        self.nframes = max(last_frames) + 1
        self.times = np.arange(self.nframes)

        if auto_fill:  # Recursively fill the data containers
            tracklets_multi = np.full(
                (self.nindividuals, self.nframes, len(bodyparts_multi) * 3),
                np.nan,
                np.float16,
            )
            tracklets_single = np.full(
                (self.nframes, len(bodyparts_single) * 3), np.nan, np.float16
            )
            for _ in trange(len(tracklets_sorted)):
                tracklet = tracklets_sorted.pop()
                inds, temp = zip(*[(get_frame_ind(k), v) for k, v in tracklet.items()])
                inds = np.asarray(inds)
                data = np.asarray(temp, dtype=np.float16)
                data_single = data[:, mask_single]
                is_multi = np.isnan(data_single).all()
                if not is_multi:
                    # Where slots are available, copy the data over
                    is_free = np.isnan(tracklets_single[inds])
                    has_data = ~np.isnan(data_single)
                    mask = has_data & is_free
                    rows, cols = np.nonzero(mask)
                    tracklets_single[inds[rows], cols] = data_single[mask]
                    # If about to overwrite data, keep tracklets with highest confidence
                    overwrite = has_data & ~is_free
                    if overwrite.any():
                        rows, cols = np.nonzero(overwrite)
                        more_confident = (
                            data_single[overwrite] > tracklets_single[inds[rows], cols]
                        )[2::3]
                        idx = np.flatnonzero(more_confident)
                        for i in idx:
                            sl = slice(i * 3, i * 3 + 3)
                            tracklets_single[inds[rows[sl]], cols[sl]] = data_single[
                                rows[sl], cols[sl]
                            ]
                else:
                    is_free = np.isnan(tracklets_multi[:, inds])
                    data_multi = data[:, mask_multi]
                    has_data = ~np.isnan(data_multi)
                    overwrite = has_data & ~is_free
                    overwrite_risk = np.any(overwrite, axis=(1, 2))
                    if overwrite_risk.all():
                        # Squeeze some data into empty slots
                        n_empty = is_free.all(axis=2).sum(axis=1)
                        for ind in np.argsort(n_empty)[::-1]:
                            mask = has_data & is_free
                            current_mask = mask[ind]
                            rows, cols = np.nonzero(current_mask)
                            if rows.size:
                                tracklets_multi[ind, inds[rows], cols] = data_multi[
                                    current_mask
                                ]
                                is_free[ind, current_mask] = False
                                has_data[current_mask] = False
                        if has_data.any():
                            # For the remaining data, overwrite where we are least confident
                            remaining = data_multi[has_data].reshape((-1, 3))
                            mask3d = np.broadcast_to(
                                has_data, (self.nindividuals,) + has_data.shape
                            )
                            dims, rows, cols = np.nonzero(mask3d)
                            temp = tracklets_multi[dims, inds[rows], cols].reshape(
                                (self.nindividuals, -1, 3)
                            )
                            diff = remaining - temp
                            # Find keypoints closest to the remaining data
                            # Use Manhattan distance to avoid overflow
                            dist = np.abs(diff[:, :, 0]) + np.abs(diff[:, :, 1])
                            closest = np.argmin(dist, axis=0)
                            # Only overwrite if improving confidence
                            prob = diff[closest, range(len(closest)), 2]
                            better = np.flatnonzero(prob > 0)
                            idx = closest[better]
                            rows, cols = np.nonzero(has_data)
                            for i, j in zip(idx, better):
                                sl = slice(j * 3, j * 3 + 3)
                                tracklets_multi[
                                    i, inds[rows[sl]], cols[sl]
                                ] = remaining.flat[sl]
                    else:
                        rows, cols = np.nonzero(has_data)
                        n = np.argmin(overwrite_risk)
                        tracklets_multi[n, inds[rows], cols] = data_multi[has_data]

            multi = tracklets_multi.swapaxes(0, 1).reshape((self.nframes, -1))
            data = np.c_[multi, tracklets_single].reshape((self.nframes, -1, 3))
            xy = data[:, :, :2].reshape((self.nframes, -1))
            prob = data[:, :, 2].reshape((self.nframes, -1))

            # Fill existing gaps
            missing = np.isnan(xy)
            xy_filled = columnwise_spline_interp(xy, self.max_gap)
            filled = ~np.isnan(xy_filled)
            xy[filled] = xy_filled[filled]
            inds = np.argwhere(missing & filled)
            if inds.size:
                # Retrieve original individual label indices
                inds[:, 1] //= 2
                inds = np.unique(inds, axis=0)
                prob[inds[:, 0], inds[:, 1]] = 0.01
            data[:, :, :2] = xy.reshape((self.nframes, -1, 2))
            data[:, :, 2] = prob
            self.data = data.swapaxes(0, 1)
            self.xy = self.data[:, :, :2]
            self.prob = self.data[:, :, 2]

            # Map a tracklet # to the animal ID it belongs to or the bodypart # it corresponds to.
            self.individuals = self.cfg["individuals"] + (
                ["single"] if len(self.cfg["uniquebodyparts"]) else []
            )
            self.tracklet2id = [
                i for i in range(0, self.nindividuals) for _ in bodyparts_multi
            ] + [self.nindividuals] * len(bodyparts_single)
            bps = bodyparts_multi + bodyparts_single
            map_ = dict(zip(bps, range(len(bps))))
            self.tracklet2bp = [map_[bp] for bp in self.bodyparts[::3]]
            self._label_pairs = self.get_label_pairs()
        else:
            tracklets_raw = np.full(
                (len(tracklets_sorted), self.nframes, len(bodyparts)),
                np.nan,
                np.float16,
            )
            for n, tracklet in enumerate(tracklets_sorted[::-1]):
                for frame, data in tracklet.items():
                    i = get_frame_ind(frame)
                    tracklets_raw[n, i] = data
            self.data = (
                tracklets_raw.swapaxes(0, 1)
                .reshape((self.nframes, -1, 3))
                .swapaxes(0, 1)
            )
            self.xy = self.data[:, :, :2]
            self.prob = self.data[:, :, 2]
            self.tracklet2id = self.tracklet2bp = [0] * self.data.shape[0]

    def load_tracklets_from_pickle(self, filename, auto_fill=True):
        self.filename = filename
        with open(filename, "rb") as file:
            tracklets = pickle.load(file)
        self._load_tracklets(tracklets, auto_fill)
        self._xy = self.xy.copy()

    def load_tracklets_from_hdf(self, filename):
        self.filename = filename
        df = pd.read_hdf(filename)

        # Fill existing gaps
        data = df.to_numpy()
        mask = ~df.columns.get_level_values(level="coords").str.contains("likelihood")
        xy = data[:, mask]
        prob = data[:, ~mask]
        missing = np.isnan(xy)
        xy_filled = columnwise_spline_interp(xy, self.max_gap)
        filled = ~np.isnan(xy_filled)
        xy[filled] = xy_filled[filled]
        inds = np.argwhere(missing & filled)
        if inds.size:
            # Retrieve original individual label indices
            inds[:, 1] //= 2
            inds = np.unique(inds, axis=0)
            prob[inds[:, 0], inds[:, 1]] = 0.01
        data[:, mask] = xy
        data[:, ~mask] = prob
        df = pd.DataFrame(data, index=df.index, columns=df.columns)

        idx = df.columns
        self.scorer = idx.get_level_values("scorer").unique().to_list()
        self.bodyparts = idx.get_level_values("bodyparts")
        self.nframes = len(df)
        self.times = np.arange(self.nframes)
        self.data = df.values.reshape((self.nframes, -1, 3)).swapaxes(0, 1)
        self.xy = self.data[:, :, :2]
        self.prob = self.data[:, :, 2]
        individuals = idx.get_level_values("individuals")
        self.individuals = individuals.unique().to_list()
        self.tracklet2id = individuals.map(
            dict(zip(self.individuals, range(len(self.individuals))))
        ).tolist()[::3]
        bodyparts = self.bodyparts.unique()
        self.tracklet2bp = self.bodyparts.map(
            dict(zip(bodyparts, range(len(bodyparts))))
        ).tolist()[::3]
        self._label_pairs = list(idx.droplevel(["scorer", "coords"]).unique())
        self._xy = self.xy.copy()

    def calc_completeness(self, xy, by_individual=False):
        comp = np.sum(~np.isnan(xy).any(axis=2), axis=1)
        if by_individual:
            inds = np.insert(np.diff(self.tracklet2id), 0, 1)
            comp = np.add.reduceat(comp, np.flatnonzero(inds))
        return comp

    def to_num_bodypart(self, ind):
        return self.tracklet2bp[ind]

    def to_num_individual(self, ind):
        return self.tracklet2id[ind]

    def get_non_nan_elements(self, at):
        data = self.xy[:, at]
        mask = ~np.isnan(data).any(axis=1)
        return data[mask], mask, np.flatnonzero(mask)

    def swap_tracklets(self, track1, track2, inds):
        self.xy[np.ix_([track1, track2], inds)] = self.xy[
            np.ix_([track2, track1], inds)
        ]
        self.prob[np.ix_([track1, track2], inds)] = self.prob[
            np.ix_([track2, track1], inds)
        ]
        self.tracklet2bp[track1], self.tracklet2bp[track2] = (
            self.tracklet2bp[track2],
            self.tracklet2bp[track1],
        )

    def find_swapping_bodypart_pairs(self, force_find=False):
        if not self.swapping_pairs or force_find:
            sub = (
                self.xy[:, np.newaxis] - self.xy
            )  # Broadcasting for efficient subtraction of X and Y coordinates
            with np.errstate(
                invalid="ignore"
            ):  # Get rid of annoying warnings when comparing with NaNs
                pos = sub > 0
                neg = sub <= 0
                down = neg[:, :, 1:] & pos[:, :, :-1]
                up = pos[:, :, 1:] & neg[:, :, :-1]
                zero_crossings = down | up
            # ID swaps occur when X and Y simultaneously intersect each other.
            self.tracklet_swaps = zero_crossings.all(axis=3)
            cross = self.tracklet_swaps.sum(axis=2) > self.min_swap_len
            mat = np.tril(cross)
            temp_pairs = np.where(mat)
            # Get only those bodypart pairs that belong to different individuals
            pairs = []
            for a, b in zip(*temp_pairs):
                if self.tracklet2id[a] != self.tracklet2id[b]:
                    pairs.append((a, b))
            self.swapping_pairs = pairs
            self.swapping_bodyparts = np.unique(pairs).tolist()

    def get_swap_indices(self, tracklet1, tracklet2):
        return np.flatnonzero(self.tracklet_swaps[tracklet1, tracklet2])

    def get_nonoverlapping_segments(self, tracklet1, tracklet2):
        swap_inds = self.get_swap_indices(tracklet1, tracklet2)
        inds = np.insert(swap_inds, [0, len(swap_inds)], [0, self.nframes])
        mask = np.ones_like(self.times, dtype=bool)
        for i, j in zip(inds[::2], inds[1::2]):
            mask[i:j] = False
        return mask

    def flatten_data(self):
        data = np.concatenate((self.xy, np.expand_dims(self.prob, axis=2)), axis=2)
        return data.swapaxes(0, 1).reshape((self.nframes, -1))

    def format_multiindex(self):
        scorer = self.scorer * len(self.bodyparts)
        map_ = dict(zip(range(len(self.individuals)), self.individuals))
        individuals = [map_[ind] for ind in self.tracklet2id for _ in range(3)]
        coords = ["x", "y", "likelihood"] * len(self.tracklet2id)
        return pd.MultiIndex.from_arrays(
            [scorer, individuals, self.bodyparts, coords],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )

    def get_label_pairs(self):
        return list(self.format_multiindex().droplevel(["scorer", "coords"]).unique())

    def format_data(self):
        columns = self.format_multiindex()
        return pd.DataFrame(self.flatten_data(), columns=columns, index=self.times)

    def find_edited_frames(self):
        mask = np.isclose(self.xy, self._xy, equal_nan=True).all(axis=(0, 2))
        return np.flatnonzero(~mask)

    def save(self, output_name="", *args):
        df = self.format_data()
        if not output_name:
            output_name = self.filename.replace("pickle", "h5")
        df.to_hdf(output_name, "df_with_missing", format="table", mode="w")
