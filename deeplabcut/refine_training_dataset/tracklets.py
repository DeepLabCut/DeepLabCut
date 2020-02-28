import cv2
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import os
import pandas as pd
import pickle
from matplotlib.path import Path
from matplotlib.widgets import Slider, LassoSelector
from ruamel.yaml import YAML


def read_config(configname):
    if not os.path.exists(configname):
        raise FileNotFoundError(
            'Config file is not found. Please make sure that the file exists and/or '
            'there are no unnecessary spaces in the path of the config file!')
    with open(configname) as file:
        yaml = YAML()
        return yaml.load(file)


class PointSelector:
    def __init__(self, tracker, ax, collection, alpha, alpha_other=0.2):
        self.tracker = tracker
        self.ax = ax
        self.collection = collection
        self.fc = collection.get_facecolors()
        self.alpha = alpha
        self.alpha_other = alpha_other
        self.lasso = LassoSelector(ax, onselect=self.on_select)
        self._is_connected = True
        self.toggle_callback()

    def on_select(self, verts):
        path = Path(verts)
        xy = self.collection.get_offsets()
        self.tracker.picked = list(np.nonzero(path.contains_points(xy))[0])
        self.fc[:, -1] = self.alpha_other
        self.fc[self.tracker.picked, -1] = self.alpha
        self.collection.set_color(self.fc)

    def toggle_callback(self):
        if self._is_connected:
            self.disconnect()
        else:
            self.reconnect()

    def disconnect(self):
        self.lasso.disconnect_events()
        self._is_connected = False
        self.fc[:, -1] = self.alpha
        self.collection.set_color(self.fc)

    def reconnect(self):
        self.lasso.connect_default_events()
        self._is_connected = True


class TrackletManager:
    def __init__(self, config, min_swap_frac=0.01, min_tracklet_frac=0.01):
        """

        Parameters
        ----------
        config : str
            Path to a configuration file.
        min_swap_frac : float, optional (default=0.01)
            Relative fraction of the data below which bodypart swaps are ignored.
            By default, swaps representing less than 1% of the total number of frames are discarded.
        min_tracklet_frac : float, optional (default=0.01)
            Relative fraction of the data below which unidentified tracklets are ignored.
            By default, unidentified tracklets shorter than 1% of the total number of frames are discarded.
        """

        self.cfg = read_config(config)
        self.min_swap_frac = min_swap_frac
        self.min_tracklet_frac = min_tracklet_frac

        self.data = None
        self.xy = None
        self.prob = None
        self.nframes = 0
        self.times = []
        self.scorer = None
        self.nindividuals = len(self.cfg['individuals'])
        self.tracklet2id = []
        self.unidentified_tracklets = set()
        self.swapping_pairs = []
        self.swapping_bodyparts = []

    def load_from_pickle(self, filename):
        with open(filename, 'rb') as file:
            tracklets = pickle.load(file)
        header = tracklets.pop('header')
        self.scorer = header.get_level_values('scorer').unique().to_list()
        self.bodyparts = header.get_level_values('bodyparts').unique().to_list()
        self.nbodyparts = len(self.bodyparts)
        num_individuals = sorted(list(tracklets))
        nindividuals = len(num_individuals)
        tracklet2id = [i for i in range(nindividuals) for _ in range(self.nbodyparts)]  # Map a bodypart # to the animal ID it belongs to
        frames = sorted(set([frame for tracklet in tracklets.values() for frame in tracklet]))
        self.nframes = len(frames)
        self.times = np.arange(self.nframes)

        # TODO Perhaps improve data storage as num_indiv, num_bodypart, nframes, 3
        all_data = np.full((self.nframes, nindividuals, self.nbodyparts * 3), np.nan)
        for i, num in enumerate(num_individuals):
            for k, v in tracklets[num].items():
                all_data[frames.index(k), i] = v
        self.data = all_data.reshape(self.nframes, -1, 3)

        # Remove the tracklets that contained too little information.
        xy = self.data[:, :, :2]
        full_frames = np.sum(~np.isnan(xy).any(axis=2), axis=0)
        to_keep = full_frames > self.min_tracklet_frac * self.nframes
        self.xy = xy[:, to_keep]
        self.prob = self.data[:, to_keep, 2]
        self.tracklet2id = [item for item, keep in zip(tracklet2id, to_keep) if keep]
        self.unidentified_tracklets = set(range(self.nindividuals * self.nbodyparts, len(self.tracklet2id)))
        self.find_swapping_bodypart_pairs()

    def load_from_hdf(self, filename):
        # Only used for now to validate the data post refinement;
        # therefore we assume data are complete.
        df = pd.read_hdf(filename)
        self.scorer = df.columns.get_level_values('scorer').unique().to_list()
        individuals = df.columns.get_level_values('individuals').unique().to_list()
        self.bodyparts = df.columns.get_level_values('bodyparts').unique().to_list()
        self.nbodyparts = len(self.bodyparts)
        self.nframes = len(df)
        self.data = df.values.reshape(self.nframes, -1, 3)
        self.xy = self.data[:, :, :2]
        self.prob = self.data[:, :, 2]
        self.tracklet2id = [i for i in range(len(individuals)) for _ in range(self.nbodyparts)]
        self.find_swapping_bodypart_pairs()

    def find_swapping_bodypart_pairs(self):
        if not self.swapping_pairs:
            temp = np.swapaxes(self.xy, 0, 2)
            # Broadcasting makes subtraction of X and Y coordinates very efficient
            sub = temp[:, :, np.newaxis] - temp[:, np.newaxis]
            with np.errstate(invalid='ignore'):  # Get rid of annoying warnings when comparing with NaNs
                down = (sub[:, :, :, 1:] <= 0) & (sub[:, :, :, :-1] > 0)
                up = (sub[:, :, :, 1:] > 0) & (sub[:, :, :, :-1] <= 0)
                zero_crossings = down | up
            # ID swaps occur when X and Y simultaneously intersect each other
            self.tracklet_swaps = zero_crossings.all(axis=0)
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

    # TODO
    def attach_tracklet_to_id(self, picked, id):
        pass

    def get_swap_indices(self, tracklet1, tracklet2):
        return np.flatnonzero(self.tracklet_swaps[tracklet1, tracklet2])

    def get_nonoverlapping_segments(self, tracklet1, tracklet2):
        swap_inds = self.get_swap_indices(tracklet1, tracklet2)
        inds = np.insert(swap_inds, [0, len(swap_inds)], [0, self.nframes])
        mask = np.ones_like(self.times, dtype=bool)
        for i, j in zip(inds[::2], inds[1::2]):
            mask[i:j] = False
        return mask

    def save(self, output_name):
        columns = pd.MultiIndex.from_product([self.scorer,
                                              self.cfg['individuals'],
                                              self.bodyparts,
                                              ['x', 'y', 'likelihood']],
                                             names=['scorer', 'individuals', 'bodyparts', 'coords'])
        data = np.concatenate((self.xy, np.expand_dims(self.prob, axis=2)), axis=2)
        # Trim off the then-unidentified tracklets
        data = data[:, :self.nindividuals * self.nbodyparts]
        df = pd.DataFrame(data.reshape(self.nframes, -1), columns=columns, index=self.times)
        df.to_hdf(output_name, 'df_with_missing', format='table', mode='w')


class TrackletVisualizer:
    def __init__(self, manager, videoname, trail_len=50):
        self.manager = manager
        self.cmap = plt.cm.get_cmap(manager.cfg['colormap'], manager.nindividuals)
        self.video = cv2.VideoCapture(videoname)
        if not self.video.isOpened():
            raise IOError('Video could not be opened.')
        nframes = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        if nframes != manager.nframes:
            print('Video duration and data length do not match. Continuing nonetheless...')
        self.trail_len = trail_len
        self.picked = []
        self.picked_pair = []
        self.cuts = []

    def _prepare_canvas(self, manager):
        params = {'keymap.save': 's',
                  'keymap.back': 'left',
                  'keymap.forward': 'right',
                  'keymap.yscale': 'l'}
        for k, v in params.items():
            if v in plt.rcParams[k]:
                plt.rcParams[k].remove(v)

        self.fig = plt.figure(figsize=(13, 8))
        gs = self.fig.add_gridspec(2, 2)
        self.ax1 = self.fig.add_subplot(gs[:, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[1, 1], sharex=self.ax2)
        plt.subplots_adjust(bottom=0.2)
        for ax in self.ax1, self.ax2, self.ax3:
            ax.axis('off')

        colors = self.cmap(manager.tracklet2id)
        # Color in black the unidentified tracklets
        colors[manager.nindividuals * manager.nbodyparts:] = 0, 0, 0, 1
        colors[:, -1] = manager.cfg['alphavalue']

        img = self._read_frame()
        self.im = self.ax1.imshow(img)
        self.scat = self.ax1.scatter([], [], s=manager.cfg['dotsize'] ** 2, picker=True)
        self.scat.set_offsets(manager.xy[0])
        self.scat.set_color(colors)
        self.selector = PointSelector(self, self.ax1, self.scat, manager.cfg['alphavalue'])
        self.trails = sum([self.ax1.plot([], [], '-', lw=2, c=c) for c in colors], [])
        self.lines_x = sum([self.ax2.plot([], [], '-', lw=1, c=c, picker=5) for c in colors], [])
        self.lines_y = sum([self.ax3.plot([], [], '-', lw=1, c=c, picker=5) for c in colors], [])
        self.vline_x = self.ax2.axvline(0, 0, 1, c='k', ls=':')
        self.vline_y = self.ax3.axvline(0, 0, 1, c='k', ls=':')
        custom_lines = [plt.Line2D([0], [0], color=self.cmap(i), lw=4) for i in range(manager.nindividuals)]
        self.leg = self.fig.legend(custom_lines, manager.cfg['individuals'], frameon=False, fancybox=None,
                                   ncol=manager.nindividuals, fontsize='small',
                                   bbox_to_anchor=(0, 0.9, 1, 0.1), loc='center')
        for line in self.leg.get_lines():
            line.set_picker(5)

        self.display_traces()
        self.ax_slider = self.fig.add_axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgray')
        self.slider = Slider(self.ax_slider, '# Frame', 0, manager.nframes - 1, valinit=0, valstep=1, valfmt='%i')
        self.slider.on_changed(self.on_change)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def show(self):
        self._prepare_canvas(self.manager)

    def _read_frame(self):
        frame = self.video.read()[1]
        if frame is None:
            return
        return frame[:, :, ::-1]

    def fill_shaded_areas(self):
        mask = self.manager.get_nonoverlapping_segments(*self.picked_pair)
        for coll in self.ax2.collections + self.ax3.collections + self.ax_slider.collections:
            coll.remove()
        if self.picked_pair:
            for ax in self.ax2, self.ax3:
                ax.fill_between(self.manager.times, *self.ax2.dataLim.intervaly, mask,
                                facecolor='darkgray', alpha=0.2)
            trans = mtransforms.blended_transform_factory(self.ax_slider.transData, self.ax_slider.transAxes)
            self.ax_slider.vlines(np.flatnonzero(mask), 0, 0.5, color='darkorange', transform=trans)

    def on_press(self, event):
        i = int(self.slider.val)
        if event.key == 'right':
            if i < self.manager.nframes - 1:
                self.slider.set_val(i + 1)
        elif event.key == 'left':
            if i > 0:
                self.slider.set_val(i - 1)
        elif event.key == 's':
            swap_inds = self.manager.get_swap_indices(*self.picked_pair)
            if len(swap_inds):
                ind = np.argmax(swap_inds > i)
                sl = slice(swap_inds[ind - 1], swap_inds[ind])
                self.manager.xy[sl, [self.picked_pair]] = self.manager.xy[sl, [self.picked_pair[::-1]]]
                self.manager.prob[sl, [self.picked_pair]] = self.manager.prob[sl, [self.picked_pair[::-1]]]
                self.display_traces()
                self.slider.set_val(int(self.slider.val))
        elif event.key == 'i':
            self.manager.xy[i, [self.picked_pair]] = self.manager.xy[i, [self.picked_pair[::-1]]]
            self.manager.prob[i, [self.picked_pair]] = self.manager.prob[i, [self.picked_pair[::-1]]]
            self.display_traces()
            self.slider.set_val(int(self.slider.val))
        elif event.key == 'x':
            self.cuts.append(i)
            if len(self.cuts) > 1:
                self.manager.tracklet_swaps[self.picked_pair][self.cuts] = ~self.manager.tracklet_swaps[self.picked_pair][self.cuts]
                self.fill_shaded_areas()
                self.cuts = []
        elif event.key == 'l':
            self.selector.toggle_callback()

    def on_pick(self, event):
        artist = event.artist
        if artist.axes == self.ax1:
            self.picked = event.ind
        elif artist.axes == self.ax2:
            if isinstance(artist, plt.Line2D):
                self.picked = [self.lines_x.index(artist)]
        elif artist.axes == self.ax3:
            if isinstance(artist, plt.Line2D):
                self.picked = [self.lines_y.index(artist)]
        else:  # Click on the legend lines
            # Avoid accidental reassignment of already identified tracklets
            # valid_picks = [ind for ind in self.picked if ind in self.unidentified_tracks]
            valid_picks = self.picked
            if valid_picks:
                num_individual = self.leg.get_lines().index(artist)
                nrow = num_individual * self.manager.nbodyparts
                inds = [nrow + pick % self.manager.nbodyparts for pick in valid_picks]
                xy = self.manager.xy[:, valid_picks]
                p = self.manager.prob[:, valid_picks]
                mask = ~np.isnan(xy).any(axis=(1, 2))
                sl_inds = np.ix_(mask, inds)
                sl_picks = np.ix_(mask, valid_picks)
                # Ensure that we do not overwrite identified tracklets
                # if not np.all(np.isnan(self.xy[sl])):
                #     return
                old_xy = self.manager.xy[sl_inds].copy()
                old_prob = self.manager.prob[sl_inds].copy()
                self.manager.xy[sl_inds] = xy[mask]
                self.manager.prob[sl_inds] = p[mask]
                self.manager.xy[sl_picks] = old_xy
                self.manager.prob[sl_picks] = old_prob
                for pick in valid_picks:
                    try:
                        self.manager.unidentified_tracklets.remove(pick)
                    except KeyError:
                        pass
                self.display_traces()
        self.picked_pair = []
        if len(self.picked) == 1:
            for pair in self.manager.swapping_pairs:
                if self.picked[0] in pair:
                    self.picked_pair = pair
                    break
        if self.picked_pair:
            self.fill_shaded_areas()
        self.update_traces()
        self.slider.set_val(int(self.slider.val))

    def on_click(self, event):
        if event.inaxes in (self.ax2, self.ax3) and event.button == 1 \
                and not any(line.contains(event)[0] for line in self.lines_x + self.lines_y):
            x = event.xdata
            self.update_vlines(x)
            self.slider.set_val(x)

    def display_points(self, val):
        self.scat.set_offsets(self.manager.xy[val])

    def display_trails(self, val):
        for n, trail in enumerate(self.trails):
            if n in self.picked:
                xy = self.manager.xy[max(0, val - self.trail_len // 2):val + self.trail_len // 2, n]
                trail.set_data(*xy.T)
            else:
                trail.set_data([], [])

    def display_traces(self):
        for n, (line_x, line_y) in enumerate(zip(self.lines_x, self.lines_y)):
            if n in self.manager.swapping_bodyparts or n in self.manager.unidentified_tracklets:
                line_x.set_data(self.manager.times, self.manager.xy[:, n, 0])
                line_y.set_data(self.manager.times, self.manager.xy[:, n, 1])
            else:
                line_x.set_data([], [])
                line_y.set_data([], [])
        for ax in self.ax2, self.ax3:
            ax.relim()
            ax.autoscale_view()

    def update_traces(self):
        for n, (line_x, line_y) in enumerate(zip(self.lines_x, self.lines_y)):
            if n in self.picked_pair or n in self.picked:
                line_x.set_lw(3)
                line_y.set_lw(3)
                line_x.set_alpha(1)
                line_y.set_alpha(1)
            else:
                line_x.set_lw(1)
                line_y.set_lw(1)
                line_x.set_alpha(0.3)
                line_y.set_alpha(0.3)

    def update_vlines(self, val):
        self.vline_x.set_xdata([val, val])
        self.vline_y.set_xdata([val, val])

    def on_change(self, val):
        val = int(val)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, val)
        img = self._read_frame()
        if img is not None:
            self.im.set_array(img)
            self.display_points(val)
            self.display_trails(val)
            self.update_vlines(val)


class IDTracker:
    def __init__(self, config, min_swap_frac=0.01, min_tracklet_frac=0.00):
        """

        Parameters
        ----------
        config : str
            Path to a configuration file.
        min_swap_frac : float, optional (default=0.01)
            Relative fraction of the data below which bodypart swaps are ignored.
            By default, swaps representing less than 1% of the total number of frames are discarded.
        min_tracklet_frac : float, optional (default=0.01)
            Relative fraction of the data below which unidentified tracklets are ignored.
            By default, unidentified tracklets shorter than 1% of the total number of frames are discarded.
        """

        self.cfg = read_config(config)
        self.min_swap_frac = min_swap_frac
        self.min_tracklet_frac = min_tracklet_frac
        self._xy = None
        self.xy = None
        self.prob = None
        self.nframes = 0
        self.picked = []
        self.picked_pair = []
        self.cuts = []
        self.lag = 100
        self._swapping_pairs = None
        self.currently_selecting = False

    def fill_unidentified_tracklets(self):
        # TODO Find # of identified tracklets
        #  If # == nindividuals - 1, then only one possibility
        pass

    def load_tracklets_from_pickle(self, filename):
        with open(filename, 'rb') as file:
            tracks = pickle.load(file)
        header = tracks.pop('header')
        self.scorer = header.get_level_values('scorer').unique().to_list()
        self.bodyparts = header.get_level_values('bodyparts').unique().to_list()
        num_tracks = sorted(list(tracks))
        individuals = self.cfg['individuals']
        self.nbodyparts = len(self.bodyparts)

        all_frames = sorted(set([frame for track in tracks.values() for frame in track]))
        self.nframes = len(all_frames)
        self.times = np.arange(self.nframes)
        # TODO Perhaps improve data storage as num_indiv, num_bodypart, nframes, 3
        all_data = np.full((self.nframes, len(num_tracks), self.nbodyparts * 3), np.nan)
        for i, num in enumerate(num_tracks):
            for k, v in tracks[num].items():
                all_data[all_frames.index(k), i] = v
        data = all_data.reshape(self.nframes, -1, 3)
        xy = data[:, :, :2]

        # Remove the tracklets that contained too little information.
        mask = np.sum(~np.isnan(xy).any(axis=2), axis=0)
        to_keep = mask > self.min_tracklet_frac * self.nframes
        self._xy = xy[:, to_keep]
        self.xy = self._xy.copy()
        self.prob = data[:, to_keep, 2]
        ntracklets = self.xy.shape[1]
        mapping = [i for i in range(len(num_tracks)) for _ in range(self.nbodyparts)]  # Map a bodypart # to the animal ID it belongs to
        self.mapping = [m for m, keep in zip(mapping, to_keep) if keep]
        self.unidentified_tracks = set(range(len(individuals) * self.nbodyparts, len(self.mapping)))
        self.cmap = plt.cm.get_cmap(self.cfg['colormap'], len(individuals))
        self.find_swapping_bodypart_pairs()

    def _prepare_canvas(self, img):
        params = {'keymap.save': 's',
                  'keymap.back': 'left',
                  'keymap.forward': 'right',
                  'keymap.yscale': 'l'}
        for k, v in params.items():
            if v in plt.rcParams[k]:
                plt.rcParams[k].remove(v)

        self.fig = plt.figure(figsize=(13, 8))
        gs = self.fig.add_gridspec(2, 2)
        self.ax1 = self.fig.add_subplot(gs[:, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[1, 1], sharex=self.ax2)
        plt.subplots_adjust(bottom=0.2)
        for ax in self.ax1, self.ax2, self.ax3:
            ax.axis('off')
        colors = self.cmap(self.mapping)
        # Color in black the unidentified tracklets
        colors[len(self.cfg['individuals']) * self.nbodyparts:] = 0, 0, 0, 1
        colors[:, -1] = self.cfg['alphavalue']
        self.im = self.ax1.imshow(img)
        self.scat = self.ax1.scatter([], [], s=200, picker=True)
        self.scat.set_offsets(self.xy[0])
        self.scat.set_color(colors)
        self.selector = PointSelector(self, self.ax1, self.scat, self.cfg['alphavalue'])
        self.trails = sum([self.ax1.plot([], [], '-', lw=2, c=c) for c in colors], [])
        self.lines_x = sum([self.ax2.plot([], [], '-', lw=1, c=c, picker=5) for c in colors], [])
        self.lines_y = sum([self.ax3.plot([], [], '-', lw=1, c=c, picker=5) for c in colors], [])
        self.vline_x = self.ax2.axvline(0, 0, 1, c='k', ls=':')
        self.vline_y = self.ax3.axvline(0, 0, 1, c='k', ls=':')
        custom_lines = [plt.Line2D([0], [0], color=self.cmap(i), lw=4) for i in range(len(self.cfg['individuals']))]
        self.leg = self.fig.legend(custom_lines, self.cfg['individuals'], frameon=False, fancybox=None,
                                   ncol=len(self.cfg['individuals']), fontsize='small',
                                   bbox_to_anchor=(0, 0.9, 1, 0.1), loc='center')
        for line in self.leg.get_lines():
            line.set_picker(5)

        self.display_traces()
        self.newax = self.fig.add_axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgray')
        self.slider = Slider(self.newax, '# Frame', 0, self.nframes - 1, valinit=0, valstep=1, valfmt='%i')
        self.slider.on_changed(self.on_change)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def visualize(self, videoname):
        self.video = cv2.VideoCapture(videoname)
        img = self._read_frame()
        self._prepare_canvas(img)

    def _read_frame(self):
        frame = self.video.read()[1]
        if frame is None:
            return
        return frame[:, :, ::-1]

    def find_swapping_bodypart_pairs(self):
        if self._swapping_pairs is None:
            temp = np.swapaxes(self.xy, 0, 2)
            sub = temp[:, :, np.newaxis] - temp[:, np.newaxis]
            zero_crossings = self.is_crossing_zero(sub)
            self.track_crossings = zero_crossings.all(axis=0)
            cross = self.track_crossings.sum(axis=2) > self.min_swap_frac * self.nframes
            mat = np.tril(cross)
            temp_pairs = np.where(mat)
            # Get only those bodypart pairs that belong to different individuals
            pairs = []
            for a, b in zip(*temp_pairs):
                if self.mapping[a] != self.mapping[b]:
                    pairs.append((a, b))
            self._swapping_pairs = pairs
            self._swapping_bodyparts = np.unique(pairs)
        return self._swapping_pairs

    def attach_tracklet_to_id(self, picked, id):
        pass

    def is_crossing_zero(self, data):
        with np.errstate(invalid='ignore'):  # Get rid of annoying warnings when comparing with NaNs
            down = (data[:, :, :, 1:] <= 0) & (data[:, :, :, :-1] > 0)
            up = (data[:, :, :, 1:] > 0) & (data[:, :, :, :-1] <= 0)
            return down | up

    def get_crossing_indices(self):
        inds = np.flatnonzero(self.track_crossings[self.picked_pair])
        return np.insert(inds, [0, len(inds)], [0, self.nframes])

    def get_nonoverlapping_segments(self):
        actual_inds = self.get_crossing_indices()
        m = np.ones_like(self.times, dtype=bool)
        for i, j in zip(actual_inds[::2], actual_inds[1::2]):
            m[i:j] = False
        return m

    def fill_shaded_areas(self):
        m = self.get_nonoverlapping_segments()
        for coll in self.ax2.collections + self.ax3.collections + self.newax.collections:
            coll.remove()
        if self.picked_pair:
            self.ax2.fill_between(self.times, *self.ax2.dataLim.intervaly, m, facecolor='darkgray', alpha=0.2)
            self.ax3.fill_between(self.times, *self.ax3.dataLim.intervaly, m, facecolor='darkgray', alpha=0.2)
            trans = mtransforms.blended_transform_factory(self.newax.transData, self.newax.transAxes)
            self.newax.vlines(np.flatnonzero(m), 0, 0.5, color='darkorange', transform=trans)

    def on_press(self, event):
        i = int(self.slider.val)
        if event.key == 'right':
            if i < self.nframes - 1:
                self.slider.set_val(i + 1)
        elif event.key == 'left':
            if i > 0:
                self.slider.set_val(i - 1)
        elif event.key == 's':
            actual_inds = self.get_crossing_indices()
            if len(actual_inds):
                ind = np.argmax(actual_inds > i)
                sl = slice(actual_inds[ind - 1], actual_inds[ind])
                self.xy[sl, [self.picked_pair]] = self.xy[sl, [self.picked_pair[::-1]]]
                self.prob[sl, [self.picked_pair]] = self.prob[sl, [self.picked_pair[::-1]]]
                self.display_traces()
                self.slider.set_val(int(self.slider.val))
        elif event.key == 'i':
            self.xy[i, [self.picked_pair]] = self.xy[i, [self.picked_pair[::-1]]]
            self.prob[i, [self.picked_pair]] = self.prob[i, [self.picked_pair[::-1]]]
            self.display_traces()
            self.slider.set_val(int(self.slider.val))
        elif event.key == 'x':
            self.cuts.append(i)
            if len(self.cuts) > 1:
                self.track_crossings[self.picked_pair][self.cuts] = ~self.track_crossings[self.picked_pair][self.cuts]
                self.fill_shaded_areas()
                self.cuts = []
        elif event.key == 'l':
            self.selector.toggle_callback()

    def on_pick(self, event):
        artist = event.artist
        if artist.axes == self.ax1:
            self.picked = event.ind
        elif artist.axes == self.ax2:
            if isinstance(artist, plt.Line2D):
                self.picked = [self.lines_x.index(artist)]
        elif artist.axes == self.ax3:
            if isinstance(artist, plt.Line2D):
                self.picked = [self.lines_y.index(artist)]
        else:  # Click on the legend lines
            # Avoid accidental reassignment of already identified tracklets
            # valid_picks = [ind for ind in self.picked if ind in self.unidentified_tracks]
            valid_picks = self.picked
            if valid_picks:
                num_individual = self.leg.get_lines().index(artist)
                nrow = num_individual * self.nbodyparts
                inds = [nrow + pick % self.nbodyparts for pick in valid_picks]
                xy = self.xy[:, valid_picks]
                p = self.prob[:, valid_picks]
                mask = ~np.isnan(xy).any(axis=(1, 2))
                sl_inds = np.ix_(mask, inds)
                sl_picks = np.ix_(mask, valid_picks)
                # Ensure that we do not overwrite identified tracklets
                # if not np.all(np.isnan(self.xy[sl])):
                #     return
                old_xy = self.xy[sl_inds].copy()
                old_prob = self.prob[sl_inds].copy()
                self.xy[sl_inds] = xy[mask]
                self.prob[sl_inds] = p[mask]
                self.xy[sl_picks] = old_xy
                self.prob[sl_picks] = old_prob
                for pick in valid_picks:
                    try:
                        self.unidentified_tracks.remove(pick)
                    except KeyError:
                        pass
                self.display_traces()
        self.picked_pair = []
        if len(self.picked) == 1:
            for pair in self.find_swapping_bodypart_pairs():
                if self.picked[0] in pair:
                    self.picked_pair = pair
                    break
        if self.picked_pair:
            self.fill_shaded_areas()
        self.update_traces()
        self.slider.set_val(int(self.slider.val))

    def on_click(self, event):
        if event.inaxes in (self.ax2, self.ax3) and event.button == 1 \
                and not any(line.contains(event)[0] for line in self.lines_x + self.lines_y):
            x = event.xdata
            self.update_vlines(x)
            self.slider.set_val(x)

    def display_points(self, val):
        self.scat.set_offsets(self.xy[val])

    def display_trails(self, val):
        for n, trail in enumerate(self.trails):
            if n in self.picked:
                xy = self.xy[max(0, val - self.lag):val + self.lag, n]
                trail.set_data(*xy.T)
            else:
                trail.set_data([], [])

    def display_traces(self):
        for n, (line_x, line_y) in enumerate(zip(self.lines_x, self.lines_y)):
            if n in self._swapping_bodyparts or n in self.unidentified_tracks:
                line_x.set_data(self.times, self.xy[:, n, 0])
                line_y.set_data(self.times, self.xy[:, n, 1])
            else:
                line_x.set_data([], [])
                line_y.set_data([], [])
        for ax in self.ax2, self.ax3:
            ax.relim()
            ax.autoscale_view()

    def update_traces(self):
        for n, (line_x, line_y) in enumerate(zip(self.lines_x, self.lines_y)):
            if n in self.picked_pair or n in self.picked:
                line_x.set_lw(3)
                line_y.set_lw(3)
                line_x.set_alpha(1)
                line_y.set_alpha(1)
            else:
                line_x.set_lw(1)
                line_y.set_lw(1)
                line_x.set_alpha(0.3)
                line_y.set_alpha(0.3)

    def update_vlines(self, val):
        self.vline_x.set_xdata([val, val])
        self.vline_y.set_xdata([val, val])

    def on_change(self, val):
        val = int(val)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, val)
        img = self._read_frame()
        if img is not None:
            self.im.set_array(img)
            self.display_points(val)
            self.display_trails(val)
            self.update_vlines(val)

    def save(self, output_name):
        columns = pd.MultiIndex.from_product([self.scorer,
                                              self.cfg['individuals'],
                                              self.bodyparts,
                                              ['x', 'y', 'likelihood']],
                                             names=['scorer', 'individuals', 'bodyparts', 'coords'])
        data = np.concatenate((self.xy, np.expand_dims(self.prob, axis=2)), axis=2)
        # Trim off the then-unidentified tracklets
        data = data[:, :len(self.cfg['individuals']) * self.nbodyparts]
        df = pd.DataFrame(data.reshape(self.nframes, -1), columns=columns, index=self.times)
        df.to_hdf(output_name, 'df_with_missing', format='table', mode='w')


tracker = IDTracker('/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/silversideschooling-Valentina-2019-07-14/config.yaml')
tracker.load_tracklets_from_pickle('/Users/Jessy/Downloads/deeplc.menidia.school4.59rpm.S11.D.shortDLC_resnet50_silversideschoolingJul14shuffle0_30000tracks.pickle')
tracker.visualize('/Users/Jessy/Downloads/deeplc.menidia.school4.59rpm.S11.D.short.avi')
