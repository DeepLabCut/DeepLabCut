import pickle
import re
from threading import Event, Thread

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib.widgets import Slider, LassoSelector, Button, CheckButtons

from deeplabcut import generate_training_dataset
from deeplabcut.post_processing import columnwise_spline_interp
from deeplabcut.utils.auxiliaryfunctions import read_config, attempttomakefolder


class BackgroundPlayer:
    def __init__(self, viz):
        self.viz = viz
        self.can_run = Event()
        self.can_run.clear()
        self.running = True
        self.paused = True
        self.speed = ""

    def run(self):
        while self.running:
            self.can_run.wait()
            i = self.viz.curr_frame + 1
            if "F" in self.speed:
                i += 2 * len(self.speed)
            elif "R" in self.speed:
                i -= 2 * len(self.speed)
            if i > self.viz.manager.nframes:
                i = 0
            elif i < 0:
                i = self.viz.manager.nframes
            self.viz.slider.set_val(i)

    def pause(self):
        self.can_run.clear()
        self.paused = True

    def resume(self):
        self.can_run.set()
        self.paused = False

    def toggle(self):
        if self.paused:
            self.resume()
        else:
            self.pause()

    def forward(self):
        speed = self.speed
        if "R" in speed:
            speed = ""
        if len(speed) < 4:
            speed += "F"
        self.speed = speed
        self.resume()

    def rewind(self):
        speed = self.speed
        if "F" in speed:
            speed = ""
        if len(speed) < 4:
            speed += "R"
        self.speed = speed
        self.resume()

    def terminate(self, *args):
        self.running = False


class PointSelector:
    def __init__(self, tracker, ax, collection, alpha, alpha_other=0.2):
        self.tracker = tracker
        self.ax = ax
        self.collection = collection
        self.fc = collection.get_facecolors()
        self.alpha = alpha
        self.alpha_other = alpha_other
        self.lasso = LassoSelector(ax, onselect=self.on_select)
        self.is_connected = True
        self.toggle()

    def on_select(self, verts):
        path = Path(verts)
        xy = self.collection.get_offsets()
        self.tracker.picked = list(np.nonzero(path.contains_points(xy))[0])
        self.fc[:, -1] = self.alpha_other
        self.fc[self.tracker.picked, -1] = self.alpha
        self.collection.set_color(self.fc)
        self.tracker.display_traces()
        self.tracker.fig.canvas.draw_idle()  # Force wx backend to redraw the figure

    def toggle(self, *args):
        if self.is_connected:
            self.disconnect()
        else:
            self.reconnect()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.is_connected = False
        self.tracker.picked = []
        self.tracker.picked_pair = []
        self.fc[:, -1] = self.alpha
        self.collection.set_color(self.fc)
        self.tracker.display_traces(only_picked=False)
        self.tracker.fig.canvas.draw_idle()  # Force wx backend to redraw the figure

    def reconnect(self):
        self.lasso.connect_default_events()
        self.is_connected = True


class TrackletManager:
    def __init__(self, config, min_swap_frac=0.01, min_tracklet_frac=0.01, max_gap=0):
        """

        Parameters
        ----------
        config : str
            Path to a configuration file.
        min_swap_frac : float, optional (default=0.01)
            Relative fraction of the data below which bodypart swaps are ignored.
            By default, swaps representing less than 1% of the total number of frames are discarded.
        min_tracklet_frac : float, optional (default=0.01)
            Relative fraction of the data below which tracklets are ignored.
            By default, tracklets shorter than 1% of the total number of frames are discarded.
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
        self.cfg = read_config(config)
        self.min_swap_frac = min_swap_frac
        self.min_tracklet_frac = min_tracklet_frac
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
        frames = sorted(
            set([frame for tracklet in tracklets.values() for frame in tracklet])
        )
        if not len(frames):
            raise IOError("Tracklets are empty.")

        self.nframes = int(re.findall(r"\d+", frames[-1])[0]) + 1
        self.times = np.arange(self.nframes)
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

        # Store tracklets, such that we later manipulate long chains
        # rather than data of individual frames, yielding greater continuity.
        tracklets_unsorted = dict()
        for num_tracklet in sorted(tracklets):
            to_fill = np.full((self.nframes, len(bodyparts)), np.nan)
            for frame_name, data in tracklets[num_tracklet].items():
                ind_frame = int(re.findall(r"\d+", frame_name)[0])
                to_fill[ind_frame] = data
            nonempty = np.any(~np.isnan(to_fill), axis=1)
            completeness = nonempty.sum() / self.nframes
            if completeness >= self.min_tracklet_frac:
                is_single = np.isnan(to_fill[:, mask_multi]).all()
                if is_single:
                    to_fill = to_fill[:, mask_single]
                else:
                    to_fill = to_fill[:, mask_multi]
                if to_fill.size:
                    tracklets_unsorted[num_tracklet] = to_fill, completeness, is_single
        tracklets_sorted = sorted(tracklets_unsorted.items(), key=lambda kv: kv[1][1])

        if auto_fill:
            # Recursively fill the data containers
            tracklets_multi = np.full(
                (self.nindividuals, self.nframes, len(bodyparts_multi) * 3), np.nan
            )
            tracklets_single = np.full(
                (self.nframes, len(bodyparts_single) * 3), np.nan
            )
            while tracklets_sorted:
                _, (data, _, is_single) = tracklets_sorted.pop()
                has_data = ~np.isnan(data)
                if is_single:
                    # Where slots are available, copy the data over
                    is_free = np.isnan(tracklets_single)
                    mask = has_data & is_free
                    tracklets_single[mask] = data[mask]
                    # If about to overwrite data, keep tracklets with highest confidence
                    overwrite = has_data & ~is_free
                    if overwrite.any():
                        rows, cols = np.nonzero(overwrite)
                        more_confident = (
                            data[overwrite] > tracklets_single[overwrite]
                        )[2::3]
                        inds = np.flatnonzero(more_confident)
                        for ind in inds:
                            sl = slice(ind * 3, ind * 3 + 3)
                            inds = rows[sl], cols[sl]
                            tracklets_single[inds] = data[inds]
                else:
                    is_free = np.isnan(tracklets_multi)
                    overwrite = has_data & ~is_free
                    overwrite_risk = np.any(overwrite, axis=(1, 2))
                    if overwrite_risk.all():
                        # Squeeze some data into empty slots
                        mask = has_data & is_free
                        space_left = mask.any(axis=(1, 2))
                        for ind in np.flatnonzero(space_left):
                            current_mask = mask[ind]
                            tracklets_multi[ind, current_mask] = data[current_mask]
                            has_data[current_mask] = False
                        # For the remaining data, overwrite where we are least confident
                        remaining = data[has_data].reshape((-1, 3))
                        mask3d = np.broadcast_to(
                            has_data, (self.nindividuals,) + has_data.shape
                        )
                        temp = tracklets_multi[mask3d].reshape(
                            (self.nindividuals, -1, 3)
                        )
                        diff = remaining - temp
                        # Find keypoints closest to the remaining data
                        dist = np.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2)
                        closest = np.argmin(dist, axis=0)
                        # Only overwrite if improving confidence
                        prob = diff[closest, range(len(closest)), 2]
                        better = np.flatnonzero(prob > 0)
                        inds = closest[better]
                        rows, cols = np.nonzero(has_data)
                        for i, j in zip(inds, better):
                            sl = slice(j * 3, j * 3 + 3)
                            tracklets_multi[i, rows[sl], cols[sl]] = remaining.flat[sl]
                    else:
                        tracklets_multi[np.argmin(overwrite_risk), has_data] = data[
                            has_data
                        ]

            multi = tracklets_multi.swapaxes(0, 1).reshape((self.nframes, -1))
            data = np.c_[multi, tracklets_single].reshape((self.nframes, -1, 3))
            xy = data[:, :, :2].reshape((self.nframes, -1))
            prob = data[:, :, 2].reshape((self.nframes, -1))

            # Fill existing gaps and slightly smooth the tracklets
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
                (len(tracklets_sorted), self.nframes, len(bodyparts)), np.nan
            )
            for n, data in enumerate(tracklets_sorted[::-1]):
                xy = data[1][0]
                tracklets_raw[n, :, : xy.shape[1]] = xy
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
            cross = self.tracklet_swaps.sum(axis=2) > self.min_swap_frac * self.nframes
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


class TrackletVisualizer:
    def __init__(self, manager, videoname, trail_len=50):
        self.manager = manager
        self.cmap = plt.cm.get_cmap(
            manager.cfg["colormap"], len(set(manager.tracklet2id))
        )
        self.videoname = videoname
        self.video = cv2.VideoCapture(videoname)
        if not self.video.isOpened():
            raise IOError("Video could not be opened.")
        self.nframes = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        # Take into consideration imprecise OpenCV estimation of total number of frames
        if abs(self.nframes - manager.nframes) >= 0.05 * manager.nframes:
            print(
                "Video duration and data length do not match. Continuing nonetheless..."
            )
        self.trail_len = trail_len
        self.help_text = ""
        self.draggable = False
        self._curr_frame = 0
        self.curr_frame = 0

        self.picked = []
        self.picked_pair = []
        self.cuts = []

        self.player = BackgroundPlayer(self)
        self.thread_player = Thread(target=self.player.run, daemon=True)
        self.thread_player.start()

        self.dps = []

    def _prepare_canvas(self, manager, fig):
        params = {
            "keymap.save": "s",
            "keymap.back": "left",
            "keymap.forward": "right",
            "keymap.yscale": "l",
        }
        for k, v in params.items():
            if v in plt.rcParams[k]:
                plt.rcParams[k].remove(v)

        self.dotsize = manager.cfg["dotsize"]
        self.alpha = manager.cfg["alphavalue"]

        if fig is None:
            self.fig = plt.figure(figsize=(13, 8))
        else:
            self.fig = fig
        gs = self.fig.add_gridspec(2, 2)
        self.ax1 = self.fig.add_subplot(gs[:, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[1, 1], sharex=self.ax2)
        plt.subplots_adjust(bottom=0.2)
        for ax in self.ax1, self.ax2, self.ax3:
            ax.axis("off")

        self.colors = self.cmap(manager.tracklet2id)
        self.colors[:, -1] = self.alpha

        img = self._read_frame()
        self.im = self.ax1.imshow(img)
        self.scat = self.ax1.scatter([], [], s=self.dotsize ** 2, picker=True)
        self.scat.set_offsets(manager.xy[:, 0])
        self.scat.set_color(self.colors)
        self.trails = sum(
            [self.ax1.plot([], [], "-", lw=2, c=c) for c in self.colors], []
        )
        self.lines_x = sum(
            [self.ax2.plot([], [], "-", lw=1, c=c, picker=5) for c in self.colors], []
        )
        self.lines_y = sum(
            [self.ax3.plot([], [], "-", lw=1, c=c, picker=5) for c in self.colors], []
        )
        self.vline_x = self.ax2.axvline(0, 0, 1, c="k", ls=":")
        self.vline_y = self.ax3.axvline(0, 0, 1, c="k", ls=":")
        custom_lines = [
            plt.Line2D([0], [0], color=self.cmap(i), lw=4)
            for i in range(len(manager.individuals))
        ]
        self.leg = self.fig.legend(
            custom_lines,
            manager.individuals,
            frameon=False,
            fancybox=None,
            ncol=len(manager.individuals),
            fontsize="small",
            bbox_to_anchor=(0, 0.9, 1, 0.1),
            loc="center",
        )
        for line in self.leg.get_lines():
            line.set_picker(5)

        self.ax_slider = self.fig.add_axes([0.1, 0.1, 0.5, 0.03], facecolor="lightgray")
        self.ax_slider2 = self.fig.add_axes(
            [0.1, 0.05, 0.3, 0.03], facecolor="darkorange"
        )
        self.slider = Slider(
            self.ax_slider,
            "# Frame",
            self.curr_frame,
            manager.nframes - 1,
            valinit=0,
            valstep=1,
            valfmt="%i",
        )
        self.slider.on_changed(self.on_change)
        self.slider2 = Slider(
            self.ax_slider2,
            "Marker size",
            1,
            30,
            valinit=self.dotsize,
            valstep=1,
            valfmt="%i",
        )
        self.slider2.on_changed(self.update_dotsize)
        self.ax_drag = self.fig.add_axes([0.65, 0.1, 0.05, 0.03])
        self.ax_lasso = self.fig.add_axes([0.7, 0.1, 0.05, 0.03])
        self.ax_flag = self.fig.add_axes([0.75, 0.1, 0.05, 0.03])
        self.ax_save = self.fig.add_axes([0.80, 0.1, 0.05, 0.03])
        self.ax_help = self.fig.add_axes([0.85, 0.1, 0.05, 0.03])
        self.save_button = Button(self.ax_save, "Save", color="darkorange")
        self.save_button.on_clicked(self.save)
        self.help_button = Button(self.ax_help, "Help")
        self.help_button.on_clicked(self.display_help)
        self.drag_toggle = CheckButtons(self.ax_drag, ["Drag"])
        self.drag_toggle.on_clicked(self.toggle_draggable_points)
        self.flag_button = Button(self.ax_flag, "Flag")
        self.flag_button.on_clicked(self.flag_frame)

        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.fig.canvas.mpl_connect("key_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("close_event", self.player.terminate)

        self.selector = PointSelector(self, self.ax1, self.scat, self.alpha)
        self.lasso_toggle = CheckButtons(self.ax_lasso, ["Lasso"])
        self.lasso_toggle.on_clicked(self.selector.toggle)
        self.display_traces(only_picked=False)
        self.ax1_background = self.fig.canvas.copy_from_bbox(self.ax1.bbox)
        plt.show()

    def show(self, fig=None):
        self._prepare_canvas(self.manager, fig)

    def _read_frame(self):
        frame = self.video.read()[1]
        if frame is None:
            return
        return frame[:, :, ::-1]

    def fill_shaded_areas(self):
        self.clean_collections()
        if self.picked_pair:
            mask = self.manager.get_nonoverlapping_segments(*self.picked_pair)
            for ax in self.ax2, self.ax3:
                ax.fill_between(
                    self.manager.times,
                    *ax.dataLim.intervaly,
                    mask,
                    facecolor="darkgray",
                    alpha=0.2,
                )
            trans = mtransforms.blended_transform_factory(
                self.ax_slider.transData, self.ax_slider.transAxes
            )
            self.ax_slider.vlines(
                np.flatnonzero(mask), 0, 0.5, color="darkorange", transform=trans
            )

    def toggle_draggable_points(self, *args):
        self.draggable = not self.draggable
        if self.draggable:
            self._curr_frame = self.curr_frame
            self.scat.set_offsets([])
            self.add_draggable_points()
        else:
            self.save_coords()
            self.clean_points()
            self.display_points(self._curr_frame)
        self.fig.canvas.draw_idle()

    def add_point(self, center, animal, bodypart, **kwargs):
        circle = patches.Circle(center, **kwargs)
        self.ax1.add_patch(circle)
        dp = generate_training_dataset.auxfun_drag_label_multiple_individuals.DraggablePoint(
            circle, animal, bodypart
        )
        dp.connect()
        self.dps.append(dp)

    def clean_points(self):
        for dp in self.dps:
            dp.annot.set_visible(False)
            dp.disconnect()
        self.dps = []
        for patch in self.ax1.patches[::-1]:
            patch.remove()

    def add_draggable_points(self):
        self.clean_points()
        xy, _, inds = self.manager.get_non_nan_elements(self.curr_frame)
        for i, (animal, bodypart) in enumerate(self.manager._label_pairs):
            if i in inds:
                coords = xy[inds == i].squeeze()
                self.add_point(
                    coords,
                    animal,
                    bodypart,
                    radius=self.dotsize,
                    fc=self.colors[i],
                    alpha=self.alpha,
                )

    def save_coords(self):
        coords, nonempty, inds = self.manager.get_non_nan_elements(self._curr_frame)
        prob = self.manager.prob[:, self._curr_frame]
        for dp in self.dps:
            label = dp.individual_names, dp.bodyParts
            ind = self.manager._label_pairs.index(label)
            nrow = np.flatnonzero(inds == ind)[0]
            if not np.array_equal(coords[nrow], dp.point.center):  # Keypoint has been displaced
                coords[nrow] = dp.point.center
                prob[ind] = 1
        self.manager.xy[nonempty, self._curr_frame] = coords

    def flag_frame(self, *args):
        self.cuts.append(self.curr_frame)
        self.ax_slider.axvline(self.curr_frame, color="r")
        if len(self.cuts) == 2:
            self.cuts.sort()
            mask = np.zeros_like(self.manager.times, dtype=bool)
            mask[self.cuts[0] : self.cuts[1] + 1] = True
            for ax in self.ax2, self.ax3:
                ax.fill_between(
                    self.manager.times,
                    *ax.dataLim.intervaly,
                    mask,
                    facecolor="darkgray",
                    alpha=0.2,
                )
            trans = mtransforms.blended_transform_factory(
                self.ax_slider.transData, self.ax_slider.transAxes
            )
            self.ax_slider.vlines(
                np.flatnonzero(mask), 0, 0.5, color="darkorange", transform=trans
            )
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        cur_xlim = self.ax1.get_xlim()
        cur_ylim = self.ax1.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        if event.button == "up":
            scale_factor = 0.5
        elif event.button == "down":
            scale_factor = 2
        else:  # This should never happen anyway
            scale_factor = 1

        self.ax1.set_xlim(
            [
                xdata - (xdata - cur_xlim[0]) / scale_factor,
                xdata + (cur_xlim[1] - xdata) / scale_factor,
            ]
        )
        self.ax1.set_ylim(
            [
                ydata - (ydata - cur_ylim[0]) / scale_factor,
                ydata + (cur_ylim[1] - ydata) / scale_factor,
            ]
        )
        self.fig.canvas.draw()

    def on_press(self, event):
        if event.key == "right":
            self.move_forward()
        elif event.key == "left":
            self.move_backward()
        elif event.key == "s":
            self.swap()
        elif event.key == "i":
            self.invert()
        elif event.key == "x":
            self.flag_frame()
            if len(self.cuts) > 1:
                self.cuts.sort()
                if self.picked_pair:
                    self.manager.tracklet_swaps[self.picked_pair][
                        self.cuts
                    ] = ~self.manager.tracklet_swaps[self.picked_pair][self.cuts]
                    self.fill_shaded_areas()
                    self.cuts = []
                    self.ax_slider.lines = []
        elif event.key == "backspace":
            if not self.dps:  # Last flag deletion
                try:
                    self.cuts.pop()
                    self.ax_slider.lines.pop()
                    if not len(self.cuts) == 2:
                        self.clean_collections()
                except IndexError:
                    pass
            else:  # Smart point removal
                i = np.nanargmin(
                    [
                        self.calc_distance(*dp.point.center, event.xdata, event.ydata)
                        for dp in self.dps
                    ]
                )
                closest_dp = self.dps[i]
                label = closest_dp.individual_names, closest_dp.bodyParts
                closest_dp.disconnect()
                closest_dp.point.remove()
                self.dps.remove(closest_dp)
                ind = self.manager._label_pairs.index(label)
                self.manager.xy[ind, self._curr_frame] = np.nan
                self.manager.prob[ind, self._curr_frame] = np.nan
            self.fig.canvas.draw_idle()
        elif event.key == "l":
            self.lasso_toggle.set_active(not self.lasso_toggle.get_active)
        elif event.key == "d":
            self.drag_toggle.set_active(not self.drag_toggle.get_active)
        elif event.key == "alt+right":
            self.player.forward()
        elif event.key == "alt+left":
            self.player.rewind()
        elif event.key == "tab":
            self.player.toggle()

    def move_forward(self):
        if self.curr_frame < self.manager.nframes - 1:
            self.curr_frame += 1
            self.slider.set_val(self.curr_frame)

    def move_backward(self):
        if self.curr_frame > 0:
            self.curr_frame -= 1
            self.slider.set_val(self.curr_frame)

    def swap(self):
        if self.picked_pair:
            swap_inds = self.manager.get_swap_indices(*self.picked_pair)
            inds = np.insert(
                swap_inds, [0, len(swap_inds)], [0, self.manager.nframes - 1]
            )
            if len(inds):
                ind = np.argmax(inds > self.curr_frame)
                self.manager.swap_tracklets(
                    *self.picked_pair, range(inds[ind - 1], inds[ind] + 1)
                )
                self.display_traces()
                self.slider.set_val(self.curr_frame)

    def invert(self):
        if not self.picked_pair and len(self.picked) == 2:
            self.picked_pair = self.picked
        if self.picked_pair:
            self.manager.swap_tracklets(*self.picked_pair, [self.curr_frame])
            self.display_traces()
            self.slider.set_val(self.curr_frame)

    def on_pick(self, event):
        artist = event.artist
        if artist.axes == self.ax1:
            self.picked = list(event.ind)
        elif artist.axes == self.ax2:
            if isinstance(artist, plt.Line2D):
                self.picked = [self.lines_x.index(artist)]
        elif artist.axes == self.ax3:
            if isinstance(artist, plt.Line2D):
                self.picked = [self.lines_y.index(artist)]
        else:  # Click on the legend lines
            if self.picked:
                num_individual = self.leg.get_lines().index(artist)
                nrow = self.manager.tracklet2id.index(num_individual)
                inds = [
                    nrow + self.manager.to_num_bodypart(pick) for pick in self.picked
                ]
                xy = self.manager.xy[self.picked]
                p = self.manager.prob[self.picked]
                mask = np.zeros(xy.shape[1], dtype=bool)
                if len(self.cuts) > 1:
                    mask[self.cuts[-2] : self.cuts[-1] + 1] = True
                    self.cuts = []
                    self.ax_slider.lines = []
                    self.clean_collections()
                else:
                    return
                sl_inds = np.ix_(inds, mask)
                sl_picks = np.ix_(self.picked, mask)
                old_xy = self.manager.xy[sl_inds].copy()
                old_prob = self.manager.prob[sl_inds].copy()
                self.manager.xy[sl_inds] = xy[:, mask]
                self.manager.prob[sl_inds] = p[:, mask]
                self.manager.xy[sl_picks] = old_xy
                self.manager.prob[sl_picks] = old_prob
        self.picked_pair = []
        if len(self.picked) == 1:
            for pair in self.manager.swapping_pairs:
                if self.picked[0] in pair:
                    self.picked_pair = pair
                    break
        self.clean_collections()
        self.display_traces()
        if self.picked_pair:
            self.fill_shaded_areas()
        self.slider.set_val(self.curr_frame)

    def on_click(self, event):
        if (
            event.inaxes in (self.ax2, self.ax3)
            and event.button == 1
            and not any(line.contains(event)[0] for line in self.lines_x + self.lines_y)
        ):
            x = max(0, min(event.xdata, self.manager.nframes - 1))
            self.update_vlines(x)
            self.slider.set_val(x)
        elif event.inaxes == self.ax1 and not self.scat.contains(event)[0]:
            self.display_traces(only_picked=False)
            self.clean_collections()

    def clean_collections(self):
        for coll in (
            self.ax2.collections + self.ax3.collections + self.ax_slider.collections
        ):
            coll.remove()

    def display_points(self, val):
        data = self.manager.xy[:, val]
        self.scat.set_offsets(data)

    def display_trails(self, val):
        sl = slice(val - self.trail_len // 2, val + self.trail_len // 2)
        for n, trail in enumerate(self.trails):
            if n in self.picked:
                xy = self.manager.xy[n, sl]
                trail.set_data(*xy.T)
            else:
                trail.set_data([], [])

    def display_traces(self, only_picked=True):
        if only_picked:
            inds = self.picked + list(self.picked_pair)
        else:
            inds = self.manager.swapping_bodyparts
        for n, (line_x, line_y) in enumerate(zip(self.lines_x, self.lines_y)):
            if n in inds:
                line_x.set_data(self.manager.times, self.manager.xy[n, :, 0])
                line_y.set_data(self.manager.times, self.manager.xy[n, :, 1])
            else:
                line_x.set_data([], [])
                line_y.set_data([], [])
        for ax in self.ax2, self.ax3:
            ax.relim()
            ax.autoscale_view()

    def display_help(self, event):
        if not self.help_text:
            self.help_text = """
            Key D: activate "drag" so you can adjust bodyparts in that particular frame
            Key I: invert the position of a pair of bodyparts
            Key L: toggle the lasso selector
            Key S: swap two tracklets
            Key X: cut swapping tracklets
            Left/Right arrow: navigate through the video
            Tab: play/pause the video
            Alt+Right/Left: fast forward/rewind
            """
            self.text = self.fig.text(
                0.5,
                0.5,
                self.help_text,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12,
                color="red",
            )
        else:
            self.help_text = ""
            self.text.remove()

    def update_vlines(self, val):
        self.vline_x.set_xdata([val, val])
        self.vline_y.set_xdata([val, val])

    def on_change(self, val):
        self.curr_frame = int(val)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.curr_frame)
        img = self._read_frame()
        if img is not None:
            # Automatically disable the draggable points
            if self.draggable:
                self.drag_toggle.set_active(False)

            self.im.set_array(img)
            self.display_points(self.curr_frame)
            self.display_trails(self.curr_frame)
            self.update_vlines(self.curr_frame)

    def update_dotsize(self, val):
        self.dotsize = val
        self.scat.set_sizes([self.dotsize ** 2])

    @staticmethod
    def calc_distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def save(self, *args):
        self.save_coords()
        self.manager.save()

    def export_to_training_data(self, pcutoff=0.1):
        import os
        from skimage import io

        inds = self.manager.find_edited_frames()
        if not len(inds):
            print("No frames have been manually edited.")
            return

        # Save additional frames to the labeled-data directory
        strwidth = int(np.ceil(np.log10(self.nframes)))
        vname = os.path.splitext(os.path.basename(self.videoname))[0]
        tmpfolder = os.path.join(
            self.manager.cfg["project_path"], "labeled-data", vname
        )
        if os.path.isdir(tmpfolder):
            print(
                "Frames from video", vname, " already extracted (more will be added)!"
            )
        else:
            attempttomakefolder(tmpfolder)
        index = []
        for ind in inds:
            imagename = os.path.join(
                tmpfolder, "img" + str(ind).zfill(strwidth) + ".png"
            )
            index.append(os.path.join(*imagename.rsplit(os.path.sep, 3)[-3:]))
            if not os.path.isfile(imagename):
                self.video.set(cv2.CAP_PROP_POS_FRAMES, ind)
                frame = self._read_frame()
                if frame is None:
                    print("Frame could not be read. Skipping...")
                    continue
                frame = frame.astype(np.ubyte)
                if self.manager.cfg["cropping"]:
                    x1, x2, y1, y2 = [
                        int(self.manager.cfg[key]) for key in ("x1", "x2", "y1", "y2")
                    ]
                    frame = frame[y1:y2, x1:x2]
                io.imsave(imagename, frame)

        # Store the newly-refined data
        data = self.manager.format_data()
        df = data.iloc[inds]

        # Uncertain keypoints are ignored
        def filter_low_prob(cols, prob):
            mask = cols.iloc[:, 2] < prob
            cols.loc[mask] = np.nan
            return cols

        df = df.groupby(level='bodyparts', axis=1).apply(filter_low_prob, prob=pcutoff)
        df.index = index
        machinefile = os.path.join(
            tmpfolder, "machinelabels-iter" + str(self.manager.cfg["iteration"]) + ".h5"
        )
        if os.path.isfile(machinefile):
            df_old = pd.read_hdf(machinefile, "df_with_missing")
            df_joint = pd.concat([df_old, df])
            df_joint = df_joint[~df_joint.index.duplicated(keep="first")]
            df_joint.to_hdf(machinefile, key="df_with_missing", mode="w")
            df_joint.to_csv(os.path.join(tmpfolder, "machinelabels.csv"))
        else:
            df.to_hdf(machinefile, key="df_with_missing", mode="w")
            df.to_csv(os.path.join(tmpfolder, "machinelabels.csv"))

        # Merge with the already existing annotated data
        df.columns.set_levels(
            [self.manager.cfg["scorer"]], level="scorer", inplace=True
        )
        df.drop("likelihood", level="coords", axis=1, inplace=True)
        output_path = os.path.join(
            tmpfolder, f'CollectedData_{self.manager.cfg["scorer"]}.h5'
        )
        if os.path.isfile(output_path):
            print(
                "A training dataset file is already found for this video. The refined machine labels are merged to this data!"
            )
            df_orig = pd.read_hdf(output_path, "df_with_missing")
            df_joint = pd.concat([df, df_orig])
            # Now drop redundant ones keeping the first one [this will make sure that the refined machine file gets preference]
            df_joint = df_joint[~df_joint.index.duplicated(keep="first")]
            df_joint.sort_index(inplace=True)
            df_joint.to_hdf(output_path, key="df_with_missing", mode="w")
            df_joint.to_csv(output_path.replace("h5", "csv"))
        else:
            df.sort_index(inplace=True)
            df.to_hdf(output_path, key="df_with_missing", mode="w")
            df.to_csv(output_path.replace("h5", "csv"))


def refine_tracklets(
    config,
    pickle_or_h5_file,
    video,
    min_swap_frac=0.01,
    min_tracklet_frac=0,
    max_gap=0,
    trail_len=50,
):
    manager = TrackletManager(config, min_swap_frac, min_tracklet_frac, max_gap)
    if pickle_or_h5_file.endswith("pickle"):
        manager.load_tracklets_from_pickle(pickle_or_h5_file)
    else:
        manager.load_tracklets_from_hdf(pickle_or_h5_file)
    manager.find_swapping_bodypart_pairs()
    viz = TrackletVisualizer(manager, video, trail_len)
    viz.show()
    return manager, viz


def convert_raw_tracks_to_h5(config, tracks_pickle, output_name=""):
    manager = TrackletManager(config, 0, 0)
    manager.load_tracklets_from_pickle(tracks_pickle)
    manager.save(output_name)
