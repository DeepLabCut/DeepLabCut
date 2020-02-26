import cv2
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import os
import pandas as pd
import pickle
from matplotlib.widgets import Slider
from ruamel.yaml import YAML


def read_config(configname):
    if not os.path.exists(configname):
        raise FileNotFoundError(
            'Config file is not found. Please make sure that the file exists and/or '
            'there are no unnecessary spaces in the path of the config file!')
    with open(configname) as file:
        yaml = YAML()
        return yaml.load(file)


class IDTracker:
    def __init__(self, config):
        self.cfg = read_config(config)
        self._xy = None
        self.xy = None
        self.prob = None
        self.nframes = 0
        self.picked = 0
        self.picked_pair = []
        self.cuts = []
        self.lag = 40
        self._swapping_pairs = None

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
        for num in num_tracks:
            for k, v in tracks[num].items():
                all_data[all_frames.index(k), num_tracks.index(num)] = v
        data = all_data.reshape(self.nframes, -1, 3)
        self._xy = data[:, :, :2]
        self.xy = self._xy.copy()
        self.prob = data[:, :, 2]
        self.mapping = [i for i in range(len(num_tracks)) for _ in range(self.nbodyparts)]  # Map a bodypart # to the animal ID it belongs to
        self.unidentified_tracks = list(range(len(individuals) * self.nbodyparts, len(num_tracks) * self.nbodyparts))
        self.cmap = plt.cm.get_cmap(self.cfg['colormap'], len(individuals))

    def _prepare_canvas(self, img):
        params = {'keymap.save': 's',
                  'keymap.back': 'left',
                  'keymap.forward': 'right'}
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
        self.im = self.ax1.imshow(img)
        self.scat = self.ax1.scatter([], [], s=self.cfg['dotsize'] ** 2, alpha=self.cfg['alphavalue'], picker=True, zorder=2)
        self.scat.set_offsets(self.xy[0])
        self.scat.set_color(colors)
        self.trails = sum([self.ax1.plot([], [], '-', lw=2, c=c) for c in colors], [])
        self.lines_x = sum([self.ax2.plot([], [], '-', lw=1, c=c, picker=5) for c in colors], [])
        self.lines_y = sum([self.ax3.plot([], [], '-', lw=1, c=c, picker=5) for c in colors], [])
        self.vline_x = self.ax2.axvline(0, 0, 1, c='k', ls=':')
        self.vline_y = self.ax3.axvline(0, 0, 1, c='k', ls=':')
        custom_lines = [plt.Line2D([0], [0], color=self.cmap(i), lw=4) for i in range(len(self.cfg['individuals']))]
        self.leg = self.fig.legend(custom_lines, self.cfg['individuals'], frameon=False, fancybox=None,
                                   ncol=len(self.cfg['individuals']), fontsize='small',
                                   bbox_to_anchor=(0, 0.9, 1, 0.1), loc='center')  # , bbox_transform=fig.transFigure)
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
            cross = self.track_crossings.any(axis=2)
            mat = np.tril(cross)
            temp_pairs = np.where(mat)
            # Get only those bodypart pairs that belong to different individuals
            pairs = []
            for a, b in zip(*temp_pairs):
                if self.mapping[a] != self.mapping[b]:
                    pairs.append((a, b))
            self._swapping_pairs = pairs
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

    def on_pick(self, event):
        artist = event.artist
        if artist.axes == self.ax1:
            self.picked = event.ind[0]
        elif artist.axes == self.ax2:
            if isinstance(artist, plt.Line2D):
                self.picked = self.lines_x.index(artist)
        elif artist.axes == self.ax3:
            if isinstance(artist, plt.Line2D):
                self.picked = self.lines_y.index(artist)
        else:  # Click on the legend lines
            if self.picked in self.unidentified_tracks:  # Avoid accidental reassignment of already identified tracklets
                num_individual = self.leg.get_lines().index(artist)
                num_bodypart = self.picked % self.nbodyparts
                xy = self.xy[:, self.picked]
                p = self.prob[:, self.picked]
                mask = ~np.isnan(xy).any(axis=1)
                ind = num_individual * self.nbodyparts + num_bodypart
                # Ensure that we do not overwrite an identified tracklet
                if not np.all(np.isnan(self.xy[mask, ind])):
                    return
                self.xy[mask, ind] = xy[mask]
                self.prob[mask, ind] = p[mask]
                self.unidentified_tracks.remove(self.picked)
                self.xy[:, self.picked] = np.nan
                self.display_traces()
        self.picked_pair = []
        for pair in self.find_swapping_bodypart_pairs():
            if self.picked in pair:
                self.picked_pair = pair
                break
        self.fill_shaded_areas()
        self.update_traces()
        for trail in self.trails:
            trail.set_data([], [])
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
        x = self.xy[max(0, val - self.lag):val + self.lag, self.picked, 0].T
        y = self.xy[max(0, val - self.lag):val + self.lag, self.picked, 1].T
        self.trails[self.picked].set_data(x, y)

    def display_traces(self):
        for n, (line_x, line_y) in enumerate(zip(self.lines_x, self.lines_y)):
            if n in np.unique(self.find_swapping_bodypart_pairs()) or n in self.unidentified_tracks:
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
            if n in self.picked_pair or n == self.picked:
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
