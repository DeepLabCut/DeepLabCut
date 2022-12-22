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


import logging

import matplotlib.pyplot as plt
import numpy as np

from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.datasets import (
    Batch,
    PoseDatasetFactory,
)
from deeplabcut.utils.auxfun_videos import imresize


def display_dataset():
    logging.basicConfig(level=logging.DEBUG)

    cfg = load_config()
    dataset = PoseDatasetFactory.create(cfg)
    dataset.set_shuffle(False)

    while True:
        batch = dataset.next_batch()

        for frame_id in range(1):
            img = batch[Batch.inputs][frame_id, :, :, :]
            img = np.squeeze(img).astype("uint8")

            scmap = batch[Batch.part_score_targets][frame_id, :, :, :]
            scmap = np.squeeze(scmap)

            # scmask = batch[Batch.part_score_weights]
            # if scmask.size > 1:
            #     scmask = np.squeeze(scmask).astype('uint8')
            # else:
            #     scmask = np.zeros(img.shape)

            subplot_height = 4
            subplot_width = 5
            num_plots = subplot_width * subplot_height
            f, axarr = plt.subplots(subplot_height, subplot_width)

            for j in range(num_plots):
                plot_j = j // subplot_width
                plot_i = j % subplot_width

                curr_plot = axarr[plot_j, plot_i]
                curr_plot.axis("off")

                if j >= cfg["num_joints"]:
                    continue

                scmap_part = scmap[:, :, j]
                scmap_part = imresize(scmap_part, 8.0, interp="nearest")
                scmap_part = np.lib.pad(scmap_part, ((4, 0), (4, 0)), "minimum")

                curr_plot.set_title("{}".format(j + 1))
                curr_plot.imshow(img)
                curr_plot.hold(True)
                curr_plot.imshow(scmap_part, alpha=0.5)

        # figure(0)
        # plt.imshow(np.sum(scmap, axis=2))
        # plt.figure(100)
        # plt.imshow(img)
        # plt.figure(2)
        # plt.imshow(scmask)
        plt.show()
        plt.waitforbuttonpress()


if __name__ == "__main__":
    display_dataset()
