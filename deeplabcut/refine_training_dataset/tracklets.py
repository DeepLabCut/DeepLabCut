import cv2
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import os
import pandas as pd
import pickle
from functools import partial
from matplotlib.path import Path
from matplotlib.widgets import Slider, LassoSelector, Button
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

    def toggle(self):
        if self.is_connected:
            self.disconnect()
        else:
            self.reconnect()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.is_connected = False
        self.fc[:, -1] = self.alpha
        self.collection.set_color(self.fc)

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
            Relative fraction of the data below which unidentified tracklets are ignored.
            By default, unidentified tracklets shorter than 1% of the total number of frames are discarded.
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
        self.nindividuals = len(self.cfg['individuals'])
        self.tracklet2id = []
        self.tracklet2bp = []
        self.unidentified_tracklets = []
        self.empty_tracklets = []
        self.swapping_pairs = []
        self.swapping_bodyparts = []

    def load_tracklets_from_pickle(self, filename):
        self.filename = filename
        with open(filename, 'rb') as file:
            tracklets = pickle.load(file)
        header = tracklets.pop('header')
        self.scorer = header.get_level_values('scorer').unique().to_list()
        self.bodyparts = header.get_level_values('bodyparts').unique().to_list()
        self.nbodyparts = len(self.bodyparts)
        num_individuals = sorted(list(tracklets))
        nindividuals = len(num_individuals)
        frames = sorted(set([frame for tracklet in tracklets.values() for frame in tracklet]))
        self.nframes = len(frames)
        self.times = np.arange(self.nframes)

        data = np.full((nindividuals, self.nbodyparts, self.nframes, 3), np.nan)
        for i, num in enumerate(num_individuals):
            for k, v in tracklets[num].items():
                data[i, :, frames.index(k)] = v.reshape(-1, 3)
        data = data.reshape((-1, self.nframes, 3))

        # Sort data by completeness so that identified tracklets are always the longest
        xy = data[:, :, :2]
        comp_per_ind = self.calc_completeness(xy, by_individual=True)
        inds = np.argsort(comp_per_ind)[::-1]
        inds_flat = [ind * self.nbodyparts + j for ind in inds for j in range(self.nbodyparts)]
        self.data = data[inds_flat]
        self.xy = self.data[:, :, :2]
        self.prob = self.data[:, :, 2]
        # Map a tracklet # to the animal ID it belongs to or the bodypart # it corresponds to.
        self.tracklet2id, self.tracklet2bp = zip(*[(i, j) for i in range(nindividuals)
                                                   for j in range(self.nbodyparts)])

        # Identify the tracklets that contained too little information.
        comp = self.calc_completeness(self.xy)
        self.empty_tracklets = comp <= self.min_tracklet_frac * self.nframes
        unidentified = np.zeros_like(self.empty_tracklets, dtype=bool)
        unidentified[self.nindividuals * self.nbodyparts:] = True
        self.unidentified_tracklets = unidentified & np.logical_not(self.empty_tracklets)
        self.find_swapping_bodypart_pairs()

    def load_tracklets_from_hdf(self, filename):
        # Only used for now to validate the data post refinement;
        # therefore we assume data are complete.
        df = pd.read_hdf(filename)
        self.scorer = df.columns.get_level_values('scorer').unique().to_list()
        individuals = df.columns.get_level_values('individuals').unique().to_list()
        self.bodyparts = df.columns.get_level_values('bodyparts').unique().to_list()
        self.nbodyparts = len(self.bodyparts)
        self.nframes = len(df)
        self.data = df.values.reshape((self.nframes, -1, 3))
        self.xy = self.data[:, :, :2]
        self.prob = self.data[:, :, 2]
        self.tracklet2id = [i for i in range(len(individuals)) for _ in range(self.nbodyparts)]
        self.find_swapping_bodypart_pairs()

    def calc_completeness(self, xy, by_individual=False):
        comp = np.sum(~np.isnan(xy).any(axis=2), axis=1)
        if by_individual:
            return comp.reshape(-1, self.nbodyparts).sum(axis=1)
        return comp

    def to_num_bodypart(self, ind):
        return self.tracklet2bp[ind]

    def to_num_individual(self, ind):
        return self.tracklet2id[ind]

    def get_nonempty(self, at):
        data = self.xy[:, at]
        mask = ~np.isnan(data).any(axis=1)
        return data[mask], mask

    def swap_tracklets(self, tracklet1, tracklet2, inds):
        self.xy[np.ix_([tracklet1, tracklet2], inds)] = self.xy[np.ix_([tracklet2, tracklet1], inds)]
        self.prob[np.ix_([tracklet1, tracklet2], inds)] = self.prob[np.ix_([tracklet2, tracklet1], inds)]

    def cut_tracklet(self, num_tracklet, inds):
        if num_tracklet in np.flatnonzero(np.logical_not(self.unidentified_tracklets)):
            ind_empty = np.argmax(self.empty_tracklets)
            self.swap_tracklets(num_tracklet, ind_empty, inds)
            self.unidentified_tracklets[ind_empty] = True
            self.empty_tracklets[ind_empty] = False

    def find_swapping_bodypart_pairs(self, force_find=False):
        if not self.swapping_pairs or force_find:
            # Only keep the non-empty tracklets to accelerate computation
            nonempty = np.flatnonzero(np.logical_not(self.empty_tracklets))
            xy = self.xy[nonempty]
            sub = xy[:, np.newaxis] - xy  # Broadcasting makes subtraction of X and Y coordinates very efficient
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
            temp_inds = np.where(mat)
            # Convert back into original indices
            temp_pairs = [nonempty[inds] for inds in temp_inds]
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

    def save(self, output_name='', *args):
        columns = pd.MultiIndex.from_product([self.scorer,
                                              self.cfg['individuals'],
                                              self.bodyparts,
                                              ['x', 'y', 'likelihood']],
                                             names=['scorer', 'individuals', 'bodyparts', 'coords'])
        data = np.concatenate((self.xy, np.expand_dims(self.prob, axis=2)), axis=2)
        # Trim off the then-unidentified tracklets
        data = data[np.logical_not(self.unidentified_tracklets | self.empty_tracklets)]
        df = pd.DataFrame(data.reshape((self.nframes, -1)), columns=columns, index=self.times)
        if not output_name:
            output_name = self.filename.replace('pickle', 'h5')
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
        self.help_text = ''

        self.picked = []
        self.picked_pair = []
        self.cuts = []
        self.current_mask = []

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
        # Color in black the unidentified tracklets
        self.colors[manager.unidentified_tracklets | manager.empty_tracklets] = 0, 0, 0, 1
        self.colors[:, -1] = manager.cfg['alphavalue']

        img = self._read_frame()
        self.im = self.ax1.imshow(img)
        self.scat = self.ax1.scatter([], [], s=manager.cfg['dotsize'] ** 2, picker=True)
        self.scat.set_offsets(manager.xy[:, 0])
        self.scat.set_color(self.colors)
        self.selector = PointSelector(self, self.ax1, self.scat, manager.cfg['alphavalue'])
        self.trails = sum([self.ax1.plot([], [], '-', lw=2, c=c) for c in self.colors], [])
        self.lines_x = sum([self.ax2.plot([], [], '-', lw=1, c=c, picker=5) for c in self.colors], [])
        self.lines_y = sum([self.ax3.plot([], [], '-', lw=1, c=c, picker=5) for c in self.colors], [])
        self.vline_x = self.ax2.axvline(0, 0, 1, c='k', ls=':')
        self.vline_y = self.ax3.axvline(0, 0, 1, c='k', ls=':')
        custom_lines = [plt.Line2D([0], [0], color=self.cmap(i), lw=4) for i in range(manager.nindividuals)]
        self.leg = self.fig.legend(custom_lines, manager.cfg['individuals'], frameon=False, fancybox=None,
                                   ncol=manager.nindividuals, fontsize='small',
                                   bbox_to_anchor=(0, 0.9, 1, 0.1), loc='center')
        for line in self.leg.get_lines():
            line.set_picker(5)

        self.display_traces()
        self.ax_slider = self.fig.add_axes([0.1, 0.1, 0.6, 0.03], facecolor='lightgray')
        self.slider = Slider(self.ax_slider, '# Frame', 0, manager.nframes - 1, valinit=0, valstep=1, valfmt='%i')
        self.slider.on_changed(self.on_change)
        self.ax_save = self.fig.add_axes([0.75, 0.1, 0.05, 0.03])
        self.ax_help = self.fig.add_axes([0.8, 0.1, 0.05, 0.03])
        self.save_button = Button(self.ax_save, 'Save')
        self.save_button.on_clicked(partial(self.manager.save, ''))
        self.help_button = Button(self.ax_help, 'Help')
        self.help_button.on_clicked(self.display_help)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('close_event', self.background.terminate)

    def show(self, fig=None):
        self._prepare_canvas(self.manager, fig)

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
                ax.fill_between(self.manager.times, *ax.dataLim.intervaly, mask,
                                facecolor='darkgray', alpha=0.2)
            trans = mtransforms.blended_transform_factory(self.ax_slider.transData, self.ax_slider.transAxes)
            self.ax_slider.vlines(np.flatnonzero(mask), 0, 0.5, color='darkorange', transform=trans)

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
            if len(self.cuts) > 1:
                # self.manager.tracklet_swaps[self.picked_pair][self.cuts] = ~self.manager.tracklet_swaps[self.picked_pair][self.cuts]
                # self.fill_shaded_areas()
                self.manager.cut_tracklet(self.picked[0], slice(*self.cuts))
                self.cuts = []
                self.display_traces()
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
            inds = np.insert(swap_inds, [0, len(swap_inds)], [0, self.manager.nframes])
            if len(inds):
                ind = np.argmax(inds > i)
                sl = slice(inds[ind - 1], inds[ind])
                self.manager.swap_tracklets(*self.picked_pair, sl)
                self.display_traces()
                self.slider.set_val(int(self.slider.val))

    def invert(self):
        i = int(self.slider.val)
        self.manager.swap_tracklets(*self.picked_pair, i)
        self.display_traces()
        self.slider.set_val(int(self.slider.val))

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
        data = self.manager.xy[val]
        self.scat.set_offsets(data)
        # mask = ~np.isnan(data).any(axis=1)
        # self.scat.set_offsets(data[mask])
        # self.scat.set_color(self.colors[mask])

    def display_trails(self, val):
        sl = slice(val - self.trail_len // 2, val + self.trail_len // 2)
        for n, trail in enumerate(self.trails):
            if n in self.picked:
                xy = self.manager.xy[n, sl]
                trail.set_data(*xy.T)
            else:
                trail.set_data([], [])

    def display_traces(self):
        for n, (line_x, line_y) in enumerate(zip(self.lines_x, self.lines_y)):
            if n in self.manager.swapping_bodyparts or n in np.flatnonzero(self.manager.unidentified_tracklets):
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
            self.help_text = 'help needed'
            self.text = self.fig.text(0.5, 0.5, self.help_text,
                                      horizontalalignment='center',
                                      verticalalignment='center',
                                      fontsize=20, color='red')
        else:
            self.help_text = ''
            self.text.remove()

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


def refine_tracklets(config, picklefile, video,
                     min_swap_frac=0.01, min_tracklet_frac=0.01, trail_len=50):
    manager = TrackletManager(config, min_swap_frac, min_tracklet_frac)
    manager.load_tracklets_from_pickle(picklefile)
    viz = TrackletVisualizer(manager, video, trail_len)
    viz.show()
    return manager
