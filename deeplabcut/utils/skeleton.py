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
"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import logging
import os
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, LassoSelector
from scipy.spatial import KDTree
from skimage import io

from deeplabcut.core.config import read_config_as_dict, write_config
from deeplabcut.generate_training_dataset.trainingsetmanipulation import drop_likelihood_columns

logger = logging.getLogger(__name__)


class SkeletonBuilder:
    ### Usage parameters
    lasso_select_size = 10
    clear_button_axes = [0.85, 0.55, 0.1, 0.1]
    clear_button_text = "Clear"
    export_button_axes = [0.85, 0.45, 0.1, 0.1]
    export_button_text = "Save"
    ampl = 1.3  # Amplification factor for the zoomed-in view of the animal

    def __init__(self, config_path):
        self.config_path = config_path
        self.cfg = read_config_as_dict(config_path)
        # Find uncropped labeled data
        self._ax = None
        self.df = None
        found = False
        root = os.path.join(self.cfg["project_path"], "labeled-data")
        for dir_ in os.listdir(root):
            folder = os.path.join(root, dir_)
            if os.path.isdir(folder) and not any(folder.endswith(s) for s in ("cropped", "labeled")):
                self.df = pd.read_hdf(os.path.join(folder, f"CollectedData_{self.cfg['scorer']}.h5"))
                self.df = drop_likelihood_columns(self.df)
                row, col = self.pick_labeled_frame()
                if "individuals" in self.df.columns.names:
                    self.df = self.df.xs(col, axis=1, level="individuals")
                self.xy = self.df.loc[row].values.reshape((-1, 2))
                missing = np.flatnonzero(np.isnan(self.xy).all(axis=1))
                if not missing.size:
                    found = True
                    break
        if self.df is None:
            raise OSError("No labeled data were found.")

        self.bpts = self.df.columns.get_level_values("bodyparts").unique()
        if not found:
            warnings.warn(
                f"A fully labeled animal could not be found. "
                f"{', '.join(self.bpts[missing])} will need to be manually connected in the config.yaml.",
                stacklevel=2,
            )
        self.tree = KDTree(self.xy)
        # Handle image previously annotated on a different platform
        if isinstance(row, str):
            sep = "/" if "/" in row else "\\"
            row = row.split(sep)
        self.image = io.imread(os.path.join(self.cfg["project_path"], *row))
        self.inds = set()
        self.segs = set()
        # Draw the skeleton if already existent
        skeleton = self.cfg.get("skeleton", [])
        if skeleton:
            for bone in skeleton:
                pair = np.flatnonzero(self.bpts.isin(bone))
                if len(pair) != 2:
                    continue
                pair_sorted = tuple(sorted(pair))
                self.inds.add(pair_sorted)
                self.segs.add(tuple(map(tuple, self.xy[pair_sorted, :])))
        self.lines = LineCollection(self.segs, colors=mcolors.to_rgba(self.cfg["skeleton_color"]))
        self.lines.set_picker(True)
        self.build_ui()
        self.display()

    def build_ui(self):
        self.fig = plt.figure()
        self._ax = self.fig.add_subplot(111)
        self._ax.axis("off")
        lo = np.nanmin(self.xy, axis=0)
        hi = np.nanmax(self.xy, axis=0)
        center = (hi + lo) / 2
        w, h = hi - lo
        w *= self.ampl
        h *= self.ampl
        self._ax.set_xlim(center[0] - w / 2, center[0] + w / 2)
        self._ax.set_ylim(center[1] - h / 2, center[1] + h / 2)
        self._ax.imshow(self.image)
        self._ax.scatter(*self.xy.T, s=self.cfg["dotsize"] ** 2)
        self._ax.add_collection(self.lines)
        self._ax.invert_yaxis()

        self.lasso = LassoSelector(self._ax, onselect=self.on_select)
        ax_clear = self.fig.add_axes(self.clear_button_axes)
        ax_export = self.fig.add_axes(self.export_button_axes)
        self.clear_button = Button(ax_clear, self.clear_button_text)
        self.clear_button.on_clicked(self.clear)
        self.export_button = Button(ax_export, self.export_button_text)
        self.export_button.on_clicked(self.export)
        self.fig.canvas.mpl_connect("pick_event", self.on_pick)

    def display(self):
        plt.show()

    def pick_labeled_frame(self):
        # Find the most 'complete' animal
        if "individuals" in self.df.columns.names:
            count = self.df.T.groupby(level="individuals").count().T
            if "single" in count.columns:
                count = count.drop(columns="single")
        else:
            count = self.df.count(axis=1).to_frame()
        mask = count.where(count == count.to_numpy().max())
        kept = mask.stack().index.to_list()
        np.random.shuffle(kept)
        picked = kept.pop()
        row = picked[:-1]
        col = picked[-1]
        return row, col

    def clear(self, *args):
        self.inds.clear()
        self.segs.clear()
        self.lines.set_segments([])
        self.fig.canvas.draw_idle()

    def read_config(self, config_path):
        return read_config_as_dict(config_path)

    def write_config(self, config_path, cfg):
        write_config(config_path, cfg)

    def _show_export_feedback(self):
        if not hasattr(self, "export_button"):
            return

        button = self.export_button
        canvas = self.fig.canvas

        original_text = button.label.get_text()
        original_color = button.ax.get_facecolor()

        n_edges = len(self.cfg.get("skeleton") or [])
        button.label.set_text(f"Saved {n_edges}")
        button.ax.set_facecolor("#c8e6c9")  # light green
        canvas.draw_idle()

        def reset_button():
            button.label.set_text(original_text)
            button.ax.set_facecolor(original_color)
            canvas.draw_idle()
            return False  # stop Matplotlib timer

        timer = canvas.new_timer(interval=1200)
        timer.add_callback(reset_button)

        # Keep a reference so the timer is not garbage-collected.
        self._export_feedback_timer = timer
        timer.start()

    def export(self, *args) -> bool:
        try:
            inds_flat = set(ind for pair in self.inds for ind in pair)
            unconnected = [i for i in range(len(self.xy)) if i not in inds_flat]
            # if empty, mention we are saving an empty skeleton
            if not self.inds:
                logger.warning("No bodyparts are connected. Saving an empty skeleton.")
            elif len(unconnected):
                logger.warning(
                    "Not all bodyparts are connected. Note that connecting all bodyparts is not necessary.",
                )
            # sort to ensure consistent order in config.yaml
            self.cfg["skeleton"] = [tuple(self.bpts[list(pair)]) for pair in sorted(self.inds)]
            self.write_config(self.config_path, self.cfg)
            self._show_export_feedback()
            return True
        except Exception as e:
            logger.warning(f"Failed to export skeleton: {e}", stacklevel=2)
            return False

    def on_pick(self, event):
        if event.mouseevent.button == 3:
            seg = tuple(map(tuple, event.artist.get_segments()[event.ind[0]]))
            self.segs.discard(seg)

            pair = tuple(sorted(self.tree.query(np.asarray(seg))[1]))
            self.inds.discard(pair)

            self.lines.set_segments(list(self.segs))
            self.fig.canvas.draw_idle()

    def on_select(self, verts):
        # Transform keypoints and lasso vertices from image/data coordinates
        # into display coordinates. This makes the grab radius independent of
        # the image resolution and current zoom level.
        xy_display = self._ax.transData.transform(self.xy)
        verts_display = self._ax.transData.transform(np.asarray(verts))

        tree_display = KDTree(xy_display)
        inds = tree_display.query_ball_point(
            verts_display,
            self.lasso_select_size,
        )

        inds_unique = []
        for lst in inds:
            if len(lst) and lst[0] not in inds_unique:
                inds_unique.append(lst[0])

        for pair in zip(inds_unique, inds_unique[1:], strict=False):
            pair_sorted = tuple(sorted(pair))
            self.inds.add(pair_sorted)
            self.segs.add(tuple(map(tuple, self.xy[pair_sorted, :])))

        self.lines.set_segments(list(self.segs))
        self.fig.canvas.draw_idle()
