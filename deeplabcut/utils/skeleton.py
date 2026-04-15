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

import os
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, LassoSelector
from ruamel.yaml import YAML
from scipy.spatial import KDTree
from skimage import io


# NOTE @C-Achard 2026-03-26 duplicate config read/write functions
# should be addressed in config refactor
def read_config(configname):
    if not os.path.exists(configname):
        raise FileNotFoundError(f"Config {configname} is not found. Please make sure that the file exists.")
    yaml = YAML(typ="rt")
    with open(configname, encoding="utf-8") as file:
        return yaml.load(file)


def write_config(configname, cfg):
    yaml = YAML(typ="rt")
    with open(configname, "w", encoding="utf-8") as file:
        yaml.dump(cfg, file)


class SkeletonBuilder:
    def __init__(self, config_path):
        self.config_path = config_path
        self.cfg = read_config(config_path)
        # Find uncropped labeled data
        self.df = None
        found = False
        root = os.path.join(self.cfg["project_path"], "labeled-data")
        for dir_ in os.listdir(root):
            folder = os.path.join(root, dir_)
            if os.path.isdir(folder) and not any(folder.endswith(s) for s in ("cropped", "labeled")):
                self.df = pd.read_hdf(os.path.join(folder, f"CollectedData_{self.cfg['scorer']}.h5"))
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
        if self.cfg["skeleton"]:
            for bone in self.cfg["skeleton"]:
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
        ax = self.fig.add_subplot(111)
        ax.axis("off")
        lo = np.nanmin(self.xy, axis=0)
        hi = np.nanmax(self.xy, axis=0)
        center = (hi + lo) / 2
        w, h = hi - lo
        ampl = 1.3
        w *= ampl
        h *= ampl
        ax.set_xlim(center[0] - w / 2, center[0] + w / 2)
        ax.set_ylim(center[1] - h / 2, center[1] + h / 2)
        ax.imshow(self.image)
        ax.scatter(*self.xy.T, s=self.cfg["dotsize"] ** 2)
        ax.add_collection(self.lines)
        ax.invert_yaxis()

        self.lasso = LassoSelector(ax, onselect=self.on_select)
        ax_clear = self.fig.add_axes([0.85, 0.55, 0.1, 0.1])
        ax_export = self.fig.add_axes([0.85, 0.45, 0.1, 0.1])
        self.clear_button = Button(ax_clear, "Clear")
        self.clear_button.on_clicked(self.clear)
        self.export_button = Button(ax_export, "Export")
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
        return read_config(config_path)

    def write_config(self, config_path, cfg):
        write_config(config_path, cfg)

    def export(self, *args):
        inds_flat = set(ind for pair in self.inds for ind in pair)
        unconnected = [i for i in range(len(self.xy)) if i not in inds_flat]
        if len(unconnected):
            warnings.warn(
                "You didn't connect all the bodyparts (which is fine!). This is just a note to let you know.",
                stacklevel=2,
            )
        # sort to ensure consistent order in config.yaml
        self.cfg["skeleton"] = [tuple(self.bpts[list(pair)]) for pair in sorted(self.inds)]
        self.write_config(self.config_path, self.cfg)

    def on_pick(self, event):
        if event.mouseevent.button == 3:
            seg = tuple(map(tuple, event.artist.get_segments()[event.ind[0]]))
            self.segs.discard(seg)

            pair = tuple(sorted(self.tree.query(np.asarray(seg))[1]))
            self.inds.discard(pair)

            self.lines.set_segments(list(self.segs))
            self.fig.canvas.draw_idle()

    def on_select(self, verts):
        # self.path = Path(verts)
        # self.verts = verts
        inds = self.tree.query_ball_point(verts, 5)
        inds_unique = []
        for lst in inds:
            if len(lst) and lst[0] not in inds_unique:
                inds_unique.append(lst[0])
        for pair in zip(inds_unique, inds_unique[1:], strict=False):
            pair_sorted = tuple(sorted(pair))
            self.inds.add(pair_sorted)
            self.segs.add(tuple(map(tuple, self.xy[pair_sorted, :])))
        self.lines.set_segments(self.segs)
        self.fig.canvas.draw_idle()
