import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from threading import Event, Thread
from deeplabcut.refine_training_dataset.tracklets import TrackletManager
from deeplabcut.utils.auxfun_videos import VideoReader
from deeplabcut.utils.auxiliaryfunctions import attempttomakefolder
from matplotlib.path import Path
from matplotlib.widgets import Slider, LassoSelector, Button, CheckButtons
from PySide2.QtWidgets import QMessageBox


class DraggablePoint:
    lock = None  # only one can be animated at a time

    def __init__(self, point, bodyParts, individual_names=None, likelihood=None):
        self.point = point
        self.bodyParts = bodyParts
        self.individual_names = individual_names
        self.likelihood = likelihood
        self.press = None
        self.background = None
        self.final_point = (0.0, 0.0)
        self.annot = self.point.axes.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        self.annot.set_visible(False)
        self.coords = []

    def connect(self):
        "connect to all the events we need"

        self.cidpress = self.point.figure.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.cidrelease = self.point.figure.canvas.mpl_connect(
            "button_release_event", self.on_release
        )
        self.cidmotion = self.point.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )
        self.cidhover = self.point.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_hover
        )

    def on_press(self, event):
        """
        Define the event for the button press!
        """
        if event.inaxes != self.point.axes:
            return
        if DraggablePoint.lock is not None:
            return
        contains, attrd = self.point.contains(event)
        if not contains:
            return
        if event.button == 1:
            """
            This button press corresponds to the left click
            """
            self.press = (self.point.center), event.xdata, event.ydata
            DraggablePoint.lock = self
            canvas = self.point.figure.canvas
            axes = self.point.axes
            self.point.set_animated(True)
            canvas.draw()
            self.background = canvas.copy_from_bbox(self.point.axes.bbox)
            axes.draw_artist(self.point)
            canvas.blit(axes.bbox)
        elif event.button == 2:
            """
            To remove a predicted label. Internally, the coordinates of the selected predicted label is replaced with nan. The user needs to middle click for the event. After right
            click the data point is removed from the plot.
            """
            message = f"Do you want to remove the label {self.bodyParts}?"
            if self.likelihood is not None:
                message += " You cannot undo this step!"
            msg = QMessageBox(
                title="Remove!",
                text=message,
            )
            msg.setStandardButtons(msg.Yes | msg.No)
            if msg == 2:
                self.delete_data()

    def delete_data(self):
        self.press = None
        DraggablePoint.lock = None
        self.point.set_animated(False)
        self.background = None
        self.final_point = (np.nan, np.nan, self.individual_names, self.bodyParts)
        self.point.center = (np.nan, np.nan)
        self.coords.append(self.final_point)
        self.point.figure.canvas.draw()

    def on_motion(self, event):
        """
        During the drag!
        """
        if DraggablePoint.lock is not self:
            return
        if event.inaxes != self.point.axes:
            return

        if event.button == 1:
            self.point.center, xpress, ypress = self.press
            dx = event.xdata - xpress
            dy = event.ydata - ypress
            self.point.center = (self.point.center[0] + dx, self.point.center[1] + dy)
            canvas = self.point.figure.canvas
            axes = self.point.axes
            # restore the background region
            canvas.restore_region(self.background)
            axes.draw_artist(self.point)
            canvas.blit(axes.bbox)

    def on_release(self, event):
        "on release we reset the press data"
        if DraggablePoint.lock is not self:
            return
        if event.button == 1:
            self.press = None
            DraggablePoint.lock = None
            self.point.set_animated(False)
            self.background = None
            self.point.figure.canvas.draw()
            self.final_point = (
                self.point.center[0],
                self.point.center[1],
                self.individual_names,
                self.bodyParts,
            )
            self.coords.append(self.final_point)

    def on_hover(self, event):
        """
        Annotate the labels and likelihood when the user hovers over the data points.
        """
        vis = self.annot.get_visible()

        if event.inaxes == self.point.axes:
            contains, attrd = self.point.contains(event)
            if contains:
                self.annot.xy = (self.point.center[0], self.point.center[1])
                text = str(self.bodyParts)
                if self.individual_names is not None:
                    text = f"{self.individual_names},{text}"
                if self.likelihood is not None:
                    text += f",p={self.likelihood:.2f}"
                self.annot.set_text(text)
                self.annot.get_bbox_patch().set_alpha(0.4)
                self.annot.set_visible(True)
                self.point.figure.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)

    def disconnect(self):
        "disconnect all the stored connection ids"
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)
        self.point.figure.canvas.mpl_disconnect(self.cidhover)


class BackgroundPlayer:
    def __init__(self, viz):
        self.viz = viz
        self.can_run = Event()
        self.can_run.clear()
        self.running = True
        self.paused = True
        self.speed = "F"

    def run(self):
        while self.running:
            self.can_run.wait()
            i = self.viz.curr_frame
            if "F" in self.speed:
                if len(self.speed) == 1:
                    i += 1
                else:
                    i += 2 * (len(self.speed) - 1)
            elif "R" in self.speed:
                if len(self.speed) == 1:
                    i -= 1
                else:
                    i -= 2 * (len(self.speed) - 1)
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
            speed = "F"
        elif len(speed) < 5:
            speed += "F"
        elif len(speed) == 5:
            speed = "F"
        print(speed)
        self.speed = speed
        self.resume()

    def rewind(self):
        speed = self.speed
        if "F" in speed:
            speed = "R"
        elif len(speed) < 5:
            speed += "R"
        elif len(speed) == 5:
            speed = "R"
        print(speed)
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


class TrackletVisualizer:
    def __init__(self, manager, videoname, trail_len=50):
        self.manager = manager
        self.cmap = plt.cm.get_cmap(
            manager.cfg["colormap"], len(set(manager.tracklet2id))
        )
        self.videoname = videoname
        self.video = VideoReader(videoname)
        self.nframes = len(self.video)
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

        img = self.video.read_frame()
        self.im = self.ax1.imshow(img)
        self.scat = self.ax1.scatter([], [], s=self.dotsize ** 2, picker=True)
        self.scat.set_offsets(manager.xy[:, 0])
        self.scat.set_color(self.colors)
        self.trails = sum(
            [self.ax1.plot([], [], "-", lw=2, c=c) for c in self.colors], []
        )
        self.lines_x = sum(
            [self.ax2.plot([], [], "-", lw=1, c=c, pickradius=5) for c in self.colors],
            [],
        )
        self.lines_y = sum(
            [self.ax3.plot([], [], "-", lw=1, c=c, pickradius=5) for c in self.colors],
            [],
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
            self.scat.set_offsets(np.empty((0, 2)))
            self.add_draggable_points()
        else:
            self.save_coords()
            self.clean_points()
            self.display_points(self._curr_frame)
        self.fig.canvas.draw_idle()

    def add_point(self, center, animal, bodypart, **kwargs):
        circle = patches.Circle(center, **kwargs)
        self.ax1.add_patch(circle)
        dp = DraggablePoint(circle, bodypart, animal)
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
        if not inds.size:
            return
        prob = self.manager.prob[:, self._curr_frame]
        for dp in self.dps:
            label = dp.individual_names, dp.bodyParts
            ind = self.manager._label_pairs.index(label)
            nrow = np.flatnonzero(inds == ind)
            if not nrow.size:
                return
            nrow = nrow[0]
            if not np.array_equal(
                coords[nrow], dp.point.center
            ):  # Keypoint has been displaced
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
        if event.key == "n" or event.key == "right":
            self.move_forward()
        elif event.key == "b" or event.key == "left":
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
                    self.ax_slider.lines.clear()
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
        elif event.key == " " or event.key == "tab":
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
                    self.ax_slider.lines.clear()
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
            Left/Right arrow OR Key B/Key N: navigate through the video (back/next)
            Tab or SPACE: play/pause the video
            Alt+Right/Left: fast forward/rewind - toggles through 5 speed levels
            Backspace: deletes last flag (if set) or deletes point
            Key P: toggles on pan/zoom tool - left button and drag to pan, right button and drag to zoom
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
        self.video.set_to_frame(self.curr_frame)
        img = self.video.read_frame()
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
        tmpfolder = os.path.join(
            self.manager.cfg["project_path"], "labeled-data", self.video.name
        )
        if os.path.isdir(tmpfolder):
            print(
                "Frames from video",
                self.video.name,
                " already extracted (more will be added)!",
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
                self.video.set_to_frame(ind)
                frame = self.video.read_frame()
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

        df = df.groupby(level="bodyparts", axis=1).apply(filter_low_prob, prob=pcutoff)
        df.index = index
        machinefile = os.path.join(
            tmpfolder, "machinelabels-iter" + str(self.manager.cfg["iteration"]) + ".h5"
        )
        if os.path.isfile(machinefile):
            df_old = pd.read_hdf(machinefile)
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
            df_orig = pd.read_hdf(output_path)
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
    min_swap_len=2,
    min_tracklet_len=2,
    max_gap=2,
    trail_len=0,
):
    """
    Refine tracklets stored either in pickle or h5 format.
    The procedure is done in two stages:
    (i) freshly-converted detections are read by the TrackletManager,
    which automatically attempts to optimize tracklet continuity by
    assigning higher priority to long tracks while maximizing
    keypoint likelihood;
    (ii) loaded tracklets are displayed into the TrackletVisualizer
    for manual editing. Individual labels can be dragged around
    like in the labeling toolbox; several of them can also be simultaneously
    selected using the Lasso tool in order to re-assign multiple tracks
    to another identity at once.

    Parameters
    ----------
    config: str
        Full path of the config.yaml file.

    pickle_or_h5_file: str
        Full path of either the pickle file obtained after calling
        deeplabcut.convert_detections2tracklets, or the h5 file written after
        refining the tracklets a first time. Note that refined tracklets are
        always stored in the h5 format.

    video: str
        Full path of the corresponding video.
        If the video duration and the total length of the tracklets disagree
        by more than 5%, a message is printed indicating that the selected
        video may not be the right one.

    min_swap_len : float, optional (default=2)
        Minimum swap length.
        Set to 2 by default. Retained swaps appear in the right panel in
        shaded regions.

    min_tracklet_len : float, optional (default=2)
        Minimum tracklet length.
        By default, tracklets shorter than 2 frames are discarded,
        leaving missing data instead. If set to 0, all tracklets are kept.

    max_gap : int, optional (default=2).
        Maximal gap size (in number of frames) of missing data to be filled.
        The procedure fits a cubic spline over all individual trajectories,
        and fills all gaps smaller than or equal to 2 frames by default.

    trail_len : int, optional (default=0)
        Number of trailing points. None by default, to accelerate visualization.
    """
    manager = TrackletManager(config, min_swap_len, min_tracklet_len, max_gap)
    if pickle_or_h5_file.endswith("pickle"):
        manager.load_tracklets_from_pickle(pickle_or_h5_file)
    else:
        manager.load_tracklets_from_hdf(pickle_or_h5_file)
    manager.find_swapping_bodypart_pairs()
    viz = TrackletVisualizer(manager, video, trail_len)
    viz.show()
    return manager, viz
