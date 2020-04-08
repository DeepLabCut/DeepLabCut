import cv2
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import os
import pandas as pd
import pickle
import re
from functools import partial
from matplotlib.path import Path
from matplotlib.widgets import Slider, LassoSelector, Button, CheckButtons
from ruamel.yaml import YAML
from threading import Event, Thread


def read_config(configname):
    if not os.path.exists(configname):
        raise FileNotFoundError(
            'Config file is not found. Please make sure that the file exists and/or '
            'there are no unnecessary spaces in the path of the config file!')
    with open(configname) as file:
        yaml = YAML()
        return yaml.load(file)


class BackgroundPlayer:
    def __init__(self, viz):
        self.viz = viz
        self.can_run = Event()
        self.can_run.clear()
        self.running = True
        self.paused = True
        self.speed = ''

    def run(self):
        while self.running:
            self.can_run.wait()
            i = self.viz.slider.val + 1
            if 'F' in self.speed:
                i += 2 * len(self.speed)
            elif 'R' in self.speed:
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
        if 'R' in speed:
            speed = ''
        if len(speed) < 4:
            speed += 'F'
        self.speed = speed
        self.resume()

    def rewind(self):
        speed = self.speed
        if 'F' in speed:
            speed = ''
        if len(speed) < 4:
            speed += 'R'
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

    def toggle(self):
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

    def reconnect(self):
        self.lasso.connect_default_events()
        self.is_connected = True


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
            Relative fraction of the data below which tracklets are ignored.
            By default, tracklets shorter than 1% of the total number of frames are discarded.


        Examples
        --------

        manager = TrackletManager(config_path, min_swap_frac=0, min_tracklet_frac=0)

        manager.load_tracklets_from_pickle(filename)
        # Alternatively
        manager.load_tracklets_from_h5(filename)

        manager.find_swapping_bodypart_pairs()
        """

        self.cfg = read_config(config)
        self.min_swap_frac = min_swap_frac
        self.min_tracklet_frac = min_tracklet_frac

        self.filename = ''
        self.data = None
        self.xy = None
        self.prob = None
        self.nframes = 0
        self.times = []
        self.scorer = None
        self.bodyparts = []
        self.nindividuals = len(self.cfg['individuals'])
        self.individuals = []
        self.tracklet2id = []
        self.tracklet2bp = []
        self.swapping_pairs = []
        self.swapping_bodyparts = []

    def _load_tracklets(self, tracklets, auto_fill):
        header = tracklets.pop('header')
        self.scorer = header.get_level_values('scorer').unique().to_list()
        frames = sorted(set([frame for tracklet in tracklets.values() for frame in tracklet]))
        self.nframes = int(re.findall(r'\d+', frames[-1])[0]) + 1
        self.times = np.arange(self.nframes)
        bodyparts = header.get_level_values('bodyparts')
        bodyparts_multi = [bp for bp in self.cfg['multianimalbodyparts'] if bp in bodyparts]
        bodyparts_single = self.cfg['uniquebodyparts']
        mask_multi = bodyparts.isin(bodyparts_multi)
        mask_single = bodyparts.isin(bodyparts_single)
        self.bodyparts = list(bodyparts[mask_multi]) * self.nindividuals + list(bodyparts[mask_single])

        # Store tracklets, such that we later manipulate long chains
        # rather than data of individual frames, yielding greater continuity.
        tracklets_unsorted = dict()
        for num_tracklet in sorted(tracklets):
            to_fill = np.full((self.nframes, len(bodyparts)), np.nan)
            for frame_name, data in tracklets[num_tracklet].items():
                ind_frame = int(re.findall(r'\d+', frame_name)[0])
                to_fill[ind_frame] = data
            nonempty = np.any(~np.isnan(to_fill), axis=1)
            completeness = nonempty.sum() / self.nframes
            if completeness >= self.min_tracklet_frac:
                is_single = np.isnan(to_fill[:, mask_multi]).all()
                if is_single:
                    to_fill = to_fill[:, mask_single]
                else:
                    to_fill = to_fill[:, mask_multi]
                tracklets_unsorted[num_tracklet] = to_fill, completeness, is_single
        tracklets_sorted = sorted(tracklets_unsorted.items(), key=lambda kv: kv[1][1])

        if auto_fill:
            # Recursively fill the data containers
            tracklets_multi = np.full((self.nindividuals, self.nframes, len(bodyparts_multi) * 3), np.nan)
            tracklets_single = np.full((self.nframes, len(bodyparts_single) * 3), np.nan)
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
                        more_confident = (data[overwrite] > tracklets_single[overwrite])[2::3]
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
                        mask3d = np.broadcast_to(has_data, (self.nindividuals,) + has_data.shape)
                        temp = tracklets_multi[mask3d].reshape((self.nindividuals, -1, 3))
                        diff = remaining - temp
                        largest_diff = np.argmax(diff[:, :, 2], axis=0)
                        prob = diff[largest_diff, range(len(largest_diff)), 2]
                        better = np.flatnonzero(prob > 0)
                        inds = largest_diff[better]
                        rows, cols = np.nonzero(has_data)
                        for i, j in zip(inds, better):
                            sl = slice(j * 3, j * 3 + 3)
                            tracklets_multi[i, rows[sl], cols[sl]] = remaining.flat[sl]
                    else:
                        tracklets_multi[np.argmin(overwrite_risk), has_data] = data[has_data]

            multi = tracklets_multi.swapaxes(0, 1).reshape((self.nframes, -1))
            self.data = np.c_[multi, tracklets_single].reshape((self.nframes, -1, 3)).swapaxes(0, 1)
            self.xy = self.data[:, :, :2]
            self.prob = self.data[:, :, 2]

            # Map a tracklet # to the animal ID it belongs to or the bodypart # it corresponds to.
            self.individuals = self.cfg['individuals'] + (['single'] if len(self.cfg['uniquebodyparts']) else [])
            self.tracklet2id = [i for i in range(0, self.nindividuals) for _ in bodyparts_multi] + \
                               [self.nindividuals] * len(bodyparts_single)
            bps = bodyparts_multi + bodyparts_single
            map_ = dict(zip(bps, range(len(bps))))
            self.tracklet2bp = [map_[bp] for bp in self.bodyparts[::3]]
        else:
            tracklets_raw = np.full((len(tracklets_sorted), self.nframes, len(bodyparts)), np.nan)
            for n, data in enumerate(tracklets_sorted[::-1]):
                tracklets_raw[n] = data[1][0]
            self.data = tracklets_raw.swapaxes(0, 1).reshape((self.nframes, -1, 3)).swapaxes(0, 1)
            self.xy = self.data[:, :, :2]
            self.prob = self.data[:, :, 2]
            self.tracklet2id = self.tracklet2bp = [0] * self.data.shape[0]

    def load_tracklets_from_pickle(self, filename, auto_fill=True):
        self.filename = filename
        with open(filename, 'rb') as file:
            tracklets = pickle.load(file)
        self._load_tracklets(tracklets, auto_fill)

    def load_tracklets_from_hdf(self, filename):
        self.filename = filename
        df = pd.read_hdf(filename)
        idx = df.columns
        self.scorer = idx.get_level_values('scorer').unique().to_list()
        self.bodyparts = idx.get_level_values('bodyparts')
        self.nframes = len(df)
        self.times = np.arange(self.nframes)
        self.data = df.values.reshape((self.nframes, -1, 3)).swapaxes(0, 1)
        self.xy = self.data[:, :, :2]
        self.prob = self.data[:, :, 2]
        individuals = idx.get_level_values('individuals')
        self.individuals = individuals.unique().to_list()
        self.tracklet2id = individuals.map(dict(zip(self.individuals, range(len(self.individuals))))).tolist()[::3]
        bodyparts = self.bodyparts.unique()
        self.tracklet2bp = self.bodyparts.map(dict(zip(bodyparts, range(len(bodyparts))))).tolist()[::3]

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
        self.xy[np.ix_([track1, track2], inds)] = self.xy[np.ix_([track2, track1], inds)]
        self.prob[np.ix_([track1, track2], inds)] = self.prob[np.ix_([track2, track1], inds)]
        self.tracklet2bp[track1], self.tracklet2bp[track2] = self.tracklet2bp[track2], self.tracklet2bp[track1]

    def find_swapping_bodypart_pairs(self, force_find=False):
        if not self.swapping_pairs or force_find:
            sub = self.xy[:, np.newaxis] - self.xy  # Broadcasting for efficient subtraction of X and Y coordinates
            with np.errstate(invalid='ignore'):  # Get rid of annoying warnings when comparing with NaNs
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

    def format_data(self):
        scorer = self.scorer * len(self.bodyparts)
        map_ = dict(zip(range(len(self.individuals)), self.individuals))
        individuals = [map_[ind] for ind in self.tracklet2id for _ in range(3)]
        coords = ['x', 'y', 'likelihood'] * len(self.tracklet2id)
        columns = pd.MultiIndex.from_arrays([scorer, individuals, self.bodyparts, coords],
                                            names=['scorer', 'individuals', 'bodyparts', 'coords'])
        return pd.DataFrame(self.flatten_data(), columns=columns, index=self.times)

    def save(self, output_name='', *args):
        df = self.format_data()
        if not output_name:
            output_name = self.filename.replace('pickle', 'h5')
        df.to_hdf(output_name, 'df_with_missing', format='table', mode='w')


class TrackletVisualizer:
    def __init__(self, manager, videoname, trail_len=50):
        self.manager = manager
        self.cmap = plt.cm.get_cmap(manager.cfg['colormap'], len(set(manager.tracklet2id)))
        self.video = cv2.VideoCapture(videoname)
        if not self.video.isOpened():
            raise IOError('Video could not be opened.')
        nframes = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        if nframes != manager.nframes:
            print('Video duration and data length do not match. Continuing nonetheless...')
        self.trail_len = trail_len
        self.help_text = ''
        self.single = False

        self.picked = []
        self.picked_pair = []
        self.cuts = []

        self.background = BackgroundPlayer(self)
        self.thread_background = Thread(target=self.background.run, daemon=True)
        self.thread_background.start()

    def _prepare_canvas(self, manager, fig):
        params = {'keymap.save': 's',
                  'keymap.back': 'left',
                  'keymap.forward': 'right',
                  'keymap.yscale': 'l'}
        for k, v in params.items():
            if v in plt.rcParams[k]:
                plt.rcParams[k].remove(v)

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
            ax.axis('off')

        self.colors = self.cmap(manager.tracklet2id)
        self.colors[:, -1] = manager.cfg['alphavalue']

        img = self._read_frame()
        self.im = self.ax1.imshow(img)
        self.scat = self.ax1.scatter([], [], s=manager.cfg['dotsize'] ** 2, picker=True)
        self.scat.set_offsets(manager.xy[:, 0])
        self.scat.set_color(self.colors)
        self.trails = sum([self.ax1.plot([], [], '-', lw=2, c=c) for c in self.colors], [])
        self.lines_x = sum([self.ax2.plot([], [], '-', lw=1, c=c, picker=5) for c in self.colors], [])
        self.lines_y = sum([self.ax3.plot([], [], '-', lw=1, c=c, picker=5) for c in self.colors], [])
        self.vline_x = self.ax2.axvline(0, 0, 1, c='k', ls=':')
        self.vline_y = self.ax3.axvline(0, 0, 1, c='k', ls=':')
        custom_lines = [plt.Line2D([0], [0], color=self.cmap(i), lw=4)
                        for i in range(len(manager.individuals))]
        self.leg = self.fig.legend(custom_lines, manager.individuals, frameon=False, fancybox=None,
                                   ncol=len(manager.individuals), fontsize='small',
                                   bbox_to_anchor=(0, 0.9, 1, 0.1), loc='center')
        for line in self.leg.get_lines():
            line.set_picker(5)

        self.ax_slider = self.fig.add_axes([0.1, 0.1, 0.6, 0.03], facecolor='lightgray')
        self.slider = Slider(self.ax_slider, '# Frame', 0, manager.nframes - 1, valinit=0, valstep=1, valfmt='%i')
        self.slider.on_changed(self.on_change)
        self.ax_save = self.fig.add_axes([0.75, 0.1, 0.05, 0.03])
        self.ax_help = self.fig.add_axes([0.8, 0.1, 0.05, 0.03])
        self.ax_check = self.fig.add_axes([0.85, 0.1, 0.05, 0.03])
        self.save_button = Button(self.ax_save, 'Save')
        self.save_button.on_clicked(partial(self.manager.save, ''))
        self.help_button = Button(self.ax_help, 'Help')
        self.help_button.on_clicked(self.display_help)
        self.check_button = CheckButtons(self.ax_check, ['Single'])
        self.check_button.on_clicked(self.toggle_single_frame_edit)

        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('close_event', self.background.terminate)

        self.selector = PointSelector(self, self.ax1, self.scat, manager.cfg['alphavalue'])
        self.display_traces(only_picked=False)
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
                ax.fill_between(self.manager.times, *ax.dataLim.intervaly, mask,
                                facecolor='darkgray', alpha=0.2)
            trans = mtransforms.blended_transform_factory(self.ax_slider.transData, self.ax_slider.transAxes)
            self.ax_slider.vlines(np.flatnonzero(mask), 0, 0.5, color='darkorange', transform=trans)

    def toggle_single_frame_edit(self, event):
        self.single = not self.single

    def on_press(self, event):
        i = int(self.slider.val)
        if event.key == 'right':
            self.move_forward()
        elif event.key == 'left':
            self.move_backward()
        elif event.key == 's':
            self.swap()
        elif event.key == 'i':
            self.invert()
        elif event.key == 'x':
            self.cuts.append(i)
            self.ax_slider.axvline(i, color='r')
            if len(self.cuts) > 1:
                self.cuts.sort()
                if self.picked_pair:
                    self.manager.tracklet_swaps[self.picked_pair][self.cuts] = ~self.manager.tracklet_swaps[self.picked_pair][self.cuts]
                    self.fill_shaded_areas()
                    self.cuts = []
                    self.ax_slider.lines = []
        elif event.key == 'l':
            self.selector.toggle()
        elif event.key == 'alt+right':
            self.background.forward()
        elif event.key == 'alt+left':
            self.background.rewind()
        elif event.key == 'tab':
            self.background.toggle()

    def move_forward(self):
        i = int(self.slider.val)
        if i < self.manager.nframes - 1:
            self.slider.set_val(i + 1)

    def move_backward(self):
        i = int(self.slider.val)
        if i > 0:
            self.slider.set_val(i - 1)

    def swap(self):
        i = int(self.slider.val)
        if self.picked_pair:
            swap_inds = self.manager.get_swap_indices(*self.picked_pair)
            inds = np.insert(swap_inds, [0, len(swap_inds)], [0, self.manager.nframes - 1])
            if len(inds):
                ind = np.argmax(inds > i)
                self.manager.swap_tracklets(*self.picked_pair, range(inds[ind - 1], inds[ind] + 1))
                self.display_traces()
                self.slider.set_val(int(self.slider.val))

    def invert(self):
        if not self.picked_pair and len(self.picked) == 2:
            self.picked_pair = self.picked
        if self.picked_pair:
            i = int(self.slider.val)
            self.manager.swap_tracklets(*self.picked_pair, [i])
            self.display_traces()
            self.slider.set_val(int(self.slider.val))

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
                inds = [nrow + self.manager.to_num_bodypart(pick) for pick in self.picked]
                xy = self.manager.xy[self.picked]
                p = self.manager.prob[self.picked]
                mask = np.zeros(xy.shape[1], dtype=bool)
                if self.single:
                    mask[int(self.slider.val)] = True
                elif self.cuts:
                    mask[self.cuts[0]:self.cuts[1] + 1] = True
                    self.cuts = []
                    self.ax_slider.lines = []
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
        self.slider.set_val(int(self.slider.val))

    def on_click(self, event):
        if event.inaxes in (self.ax2, self.ax3) and event.button == 1 \
                and not any(line.contains(event)[0] for line in self.lines_x + self.lines_y):
            x = event.xdata
            self.update_vlines(x)
            self.slider.set_val(x)
        elif event.inaxes == self.ax1 and not self.scat.contains(event)[0]:
            self.display_traces(only_picked=False)
            self.clean_collections()

    def clean_collections(self):
        for coll in self.ax2.collections + self.ax3.collections + self.ax_slider.collections:
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
            self.help_text = '''
            Key I: invert the position of a pair of bodyparts
            Key L: toggle the lasso selector
            Key S: swap two tracklets
            Key X: cut swapping tracklets
            Left/Right arrow: navigate through the video
            Tab: play/pause the video
            Alt+Right/Left: fast forward/rewind
            '''
            self.text = self.fig.text(0.5, 0.5, self.help_text,
                                      horizontalalignment='center',
                                      verticalalignment='center',
                                      fontsize=20, color='red')
        else:
            self.help_text = ''
            self.text.remove()

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


def refine_tracklets(config, picklefile, video,
                     min_swap_frac=0.01, min_tracklet_frac=0.01, trail_len=50):
    manager = TrackletManager(config, min_swap_frac, min_tracklet_frac)
    manager.load_tracklets_from_pickle(picklefile)
    manager.find_swapping_bodypart_pairs()
    viz = TrackletVisualizer(manager, video, trail_len)
    viz.show()
    return manager


def convert_raw_tracks_to_h5(config, tracks_pickle, output_name=''):
    manager = TrackletManager(config, 0, 0)
    manager.load_tracklets_from_pickle(tracks_pickle)
    manager.save(output_name)
