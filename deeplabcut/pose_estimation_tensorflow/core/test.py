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

import argparse
import logging
import os

import numpy as np
import scipy.io
import scipy.ndimage

from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.datasets.factory import PoseDatasetFactory
from deeplabcut.pose_estimation_tensorflow.datasets import Batch
from .predict import (
    setup_pose_prediction,
    extract_cnn_output,
    argmax_pose_predict,
)
from deeplabcut.pose_estimation_tensorflow.util import visualize


def test_net(visualise, cache_scoremaps):
    logging.basicConfig(level=logging.INFO)

    cfg = load_config()
    dataset = PoseDatasetFactory.create(cfg)
    dataset.set_shuffle(False)
    dataset.set_test_mode(True)

    sess, inputs, outputs = setup_pose_prediction(cfg)

    if cache_scoremaps:
        out_dir = cfg["scoremap_dir"]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    num_images = dataset.num_images
    predictions = np.zeros((num_images,), dtype=np.object)

    for k in range(num_images):
        print("processing image {}/{}".format(k, num_images - 1))

        batch = dataset.next_batch()

        outputs_np = sess.run(outputs, feed_dict={inputs: batch[Batch.inputs]})

        scmap, locref = extract_cnn_output(outputs_np, cfg)

        pose = argmax_pose_predict(scmap, locref, cfg["stride"])

        pose_refscale = np.copy(pose)
        pose_refscale[:, 0:2] /= cfg["global_scale"]
        predictions[k] = pose_refscale

        if visualise:
            img = np.squeeze(batch[Batch.inputs]).astype("uint8")
            visualize.show_heatmaps(cfg, img, scmap, pose)
            visualize.waitforbuttonpress()

        if cache_scoremaps:
            base = os.path.basename(batch[Batch.data_item].im_path)
            raw_name = os.path.splitext(base)[0]
            out_fn = os.path.join(out_dir, raw_name + ".mat")
            scipy.io.savemat(out_fn, mdict={"scoremaps": scmap.astype("float32")})

            out_fn = os.path.join(out_dir, raw_name + "_locreg" + ".mat")
            if cfg["location_refinement"]:
                scipy.io.savemat(
                    out_fn, mdict={"locreg_pred": locref.astype("float32")}
                )

    scipy.io.savemat("predictions.mat", mdict={"joints": predictions})

    sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--novis", default=False, action="store_true")
    parser.add_argument("--cache", default=False, action="store_true")
    args, unparsed = parser.parse_known_args()

    test_net(not args.novis, args.cache)
