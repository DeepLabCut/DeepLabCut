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

import argparse
import logging
import os
from pathlib import Path

import tensorflow as tf
import tf_slim as slim

from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.datasets import PoseDatasetFactory
from deeplabcut.pose_estimation_tensorflow.nnets import PoseNetFactory
from deeplabcut.pose_estimation_tensorflow.nnets.utils import get_batch_spec
from deeplabcut.pose_estimation_tensorflow.util.logging import setup_logging
from deeplabcut.pose_estimation_tensorflow.core.train import (
    setup_preloading,
    start_preloading,
    get_optimizer,
    LearningRate,
)


def train(
    config_yaml,
    displayiters,
    saveiters,
    maxiters,
    max_to_keep=5,
    keepdeconvweights=True,
    allow_growth=True,
):
    start_path = os.getcwd()
    os.chdir(
        str(Path(config_yaml).parents[0])
    )  # switch to folder of config_yaml (for logging)

    setup_logging()

    cfg = load_config(config_yaml)
    if cfg["optimizer"] != "adam":
        print(
            "Setting batchsize to 1! Larger batchsize not supported for this loader:",
            cfg["dataset_type"],
        )
        cfg["batch_size"] = 1

    if (
        cfg["partaffinityfield_predict"] and "multi-animal" in cfg["dataset_type"]
    ):  # the PAF code currently just hijacks the pairwise net stuff (for the batch feeding via Batch.pairwise_targets: 5)
        print("Activating limb prediction...")
        cfg["pairwise_predict"] = True

    dataset = PoseDatasetFactory.create(cfg)
    batch_spec = get_batch_spec(cfg)
    batch, enqueue_op, placeholders = setup_preloading(batch_spec)

    losses = PoseNetFactory.create(cfg).train(batch)
    total_loss = losses["total_loss"]

    for k, t in losses.items():
        tf.compat.v1.summary.scalar(k, t)
    merged_summaries = tf.compat.v1.summary.merge_all()
    net_type = cfg["net_type"]

    stem = Path(cfg["init_weights"]).stem
    if "snapshot" in stem and keepdeconvweights:
        print("Loading already trained DLC with backbone:", net_type)
        variables_to_restore = slim.get_variables_to_restore()
        start_iter = int(stem.split("-")[1])
    else:
        print("Loading ImageNet-pretrained", net_type)
        # loading backbone from ResNet, MobileNet etc.
        if "resnet" in net_type:
            variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])
        elif "mobilenet" in net_type:
            variables_to_restore = slim.get_variables_to_restore(
                include=["MobilenetV2"]
            )
        elif "efficientnet" in net_type:
            variables_to_restore = slim.get_variables_to_restore(
                include=["efficientnet"]
            )
            variables_to_restore = {
                var.op.name.replace("efficientnet/", "")
                + "/ExponentialMovingAverage": var
                for var in variables_to_restore
            }
        else:
            print("Wait for DLC 2.3.")
        start_iter = 0

    restorer = tf.compat.v1.train.Saver(variables_to_restore)
    saver = tf.compat.v1.train.Saver(
        max_to_keep=max_to_keep
    )  # selects how many snapshots are stored, see https://github.com/DeepLabCut/DeepLabCut/issues/8#issuecomment-387404835

    if allow_growth:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
    else:
        sess = tf.compat.v1.Session()

    coord, thread = start_preloading(sess, enqueue_op, dataset, placeholders)
    train_writer = tf.compat.v1.summary.FileWriter(cfg["log_dir"], sess.graph)
    learning_rate, train_op, tstep = get_optimizer(total_loss, cfg)

    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())

    restorer.restore(sess, cfg["init_weights"])
    if maxiters is None:
        max_iter = int(cfg["multi_step"][-1][1])
    else:
        max_iter = min(int(cfg["multi_step"][-1][1]), int(maxiters))
        # display_iters = max(1,int(displayiters))
        print("Max_iters overwritten as", max_iter)

    if displayiters is None:
        display_iters = max(1, int(cfg["display_iters"]))
    else:
        display_iters = max(1, int(displayiters))
        print("Display_iters overwritten as", display_iters)

    if saveiters is None:
        save_iters = max(1, int(cfg["save_iters"]))

    else:
        save_iters = max(1, int(saveiters))
        print("Save_iters overwritten as", save_iters)

    cumloss, partloss, locrefloss, pwloss = 0.0, 0.0, 0.0, 0.0
    lr_gen = LearningRate(cfg)
    stats_path = Path(config_yaml).with_name("learning_stats.csv")
    lrf = open(str(stats_path), "w")

    print("Training parameters:")
    print(cfg)
    print("Starting multi-animal training....")
    max_iter += start_iter  # max_iter is relative to start_iter
    for it in range(start_iter, max_iter + 1):
        if "efficientnet" in net_type:
            lr_dict = {tstep: it - start_iter}
            current_lr = sess.run(learning_rate, feed_dict=lr_dict)
        else:
            current_lr = lr_gen.get_lr(it - start_iter)
            lr_dict = {learning_rate: current_lr}

        # [_, loss_val, summary] = sess.run([train_op, total_loss, merged_summaries],feed_dict={learning_rate: current_lr})
        [_, alllosses, loss_val, summary] = sess.run(
            [train_op, losses, total_loss, merged_summaries], feed_dict=lr_dict
        )

        partloss += alllosses["part_loss"]  # scoremap loss
        if cfg["location_refinement"]:
            locrefloss += alllosses["locref_loss"]
        if cfg["pairwise_predict"]:  # paf loss
            pwloss += alllosses["pairwise_loss"]

        cumloss += loss_val
        train_writer.add_summary(summary, it)

        if it % display_iters == 0 and it > start_iter:
            logging.info(
                "iteration: {} loss: {} scmap loss: {} locref loss: {} limb loss: {} lr: {}".format(
                    it,
                    "{0:.4f}".format(cumloss / display_iters),
                    "{0:.4f}".format(partloss / display_iters),
                    "{0:.4f}".format(locrefloss / display_iters),
                    "{0:.4f}".format(pwloss / display_iters),
                    current_lr,
                )
            )

            lrf.write(
                "iteration: {}, loss: {}, scmap loss: {}, locref loss: {}, limb loss: {}, lr: {}\n".format(
                    it,
                    "{0:.4f}".format(cumloss / display_iters),
                    "{0:.4f}".format(partloss / display_iters),
                    "{0:.4f}".format(locrefloss / display_iters),
                    "{0:.4f}".format(pwloss / display_iters),
                    current_lr,
                )
            )

            cumloss, partloss, locrefloss, pwloss = 0.0, 0.0, 0.0, 0.0
            lrf.flush()

        # Save snapshot
        if (it % save_iters == 0 and it != start_iter) or it == max_iter:
            model_name = cfg["snapshot_prefix"]
            saver.save(sess, model_name, global_step=it)

    lrf.close()

    sess.close()
    coord.request_stop()
    coord.join([thread])

    # return to original path.
    os.chdir(str(start_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to yaml configuration file.")
    cli_args = parser.parse_args()

    train(Path(cli_args.config).resolve())
