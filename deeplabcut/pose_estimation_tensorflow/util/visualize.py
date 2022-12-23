#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Adapted from DeeperCut by Eldar Insafutdinov
# https://github.com/eldar/pose-tensorflow
#
# Licensed under GNU Lesser General Public License v3.0
#
"""
Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from deeplabcut.utils.auxfun_videos import imresize


def _npcircle(image, cx, cy, radius, color, transparency=0.0):
    """Draw a circle on an image using only numpy methods."""
    radius = int(radius)
    cx = int(cx)
    cy = int(cy)
    y, x = np.ogrid[-radius:radius, -radius:radius]
    index = x**2 + y**2 <= radius**2
    image[cy - radius : cy + radius, cx - radius : cx + radius][index] = (
        image[cy - radius : cy + radius, cx - radius : cx + radius][index].astype(
            "float32"
        )
        * transparency
        + np.array(color).astype("float32") * (1.0 - transparency)
    ).astype("uint8")


def check_point(cur_x, cur_y, minx, miny, maxx, maxy):
    return minx < cur_x < maxx and miny < cur_y < maxy


def visualize_joints(image, pose):
    marker_size = 8
    minx = 2 * marker_size
    miny = 2 * marker_size
    maxx = image.shape[1] - 2 * marker_size
    maxy = image.shape[0] - 2 * marker_size
    num_joints = pose.shape[0]

    visim = image.copy()
    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [0, 245, 255],
        [255, 131, 250],
        [255, 255, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [0, 245, 255],
        [255, 131, 250],
        [255, 255, 0],
        [0, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
    ]
    for p_idx in range(num_joints):
        cur_x = pose[p_idx, 0]
        cur_y = pose[p_idx, 1]
        if check_point(cur_x, cur_y, minx, miny, maxx, maxy):
            _npcircle(visim, cur_x, cur_y, marker_size, colors[p_idx], 0.0)
    return visim


def show_heatmaps(cfg, img, scmap, pose, cmap="jet"):
    interp = "bilinear"
    all_joints = cfg["all_joints"]
    all_joints_names = cfg["all_joints_names"]
    subplot_width = 3
    subplot_height = math.ceil((len(all_joints) + 1) / subplot_width)
    f, axarr = plt.subplots(subplot_height, subplot_width)
    for pidx, part in enumerate(all_joints):
        plot_j = (pidx + 1) // subplot_width
        plot_i = (pidx + 1) % subplot_width
        scmap_part = np.sum(scmap[:, :, part], axis=2)
        scmap_part = imresize(scmap_part, 8.0, interp="bicubic")
        scmap_part = np.lib.pad(scmap_part, ((4, 0), (4, 0)), "minimum")
        curr_plot = axarr[plot_j, plot_i]
        curr_plot.set_title(all_joints_names[pidx])
        curr_plot.axis("off")
        curr_plot.imshow(img, interpolation=interp)
        curr_plot.imshow(scmap_part, alpha=0.5, cmap=cmap, interpolation=interp)

    curr_plot = axarr[0, 0]
    curr_plot.set_title("Pose")
    curr_plot.axis("off")
    curr_plot.imshow(visualize_joints(img, pose))

    plt.show()


def waitforbuttonpress():
    plt.waitforbuttonpress(timeout=1)
