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
import threading
import warnings
from pathlib import Path

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
import tf_slim as slim

from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.datasets import (
    Batch,
    PoseDatasetFactory,
)
from deeplabcut.pose_estimation_tensorflow.nnets import PoseNetFactory
from deeplabcut.pose_estimation_tensorflow.util.logging import setup_logging


class LearningRate(object):
    def __init__(self, cfg):
        self.steps = cfg["multi_step"]
        self.current_step = 0

    def get_lr(self, iteration):
        lr = self.steps[self.current_step][0]
        if iteration == self.steps[self.current_step][1]:
            self.current_step += 1

        return lr


def get_batch_spec(cfg):
    num_joints = cfg["num_joints"]
    batch_size = cfg["batch_size"]
    return {
        Batch.inputs: [batch_size, None, None, 3],
        Batch.part_score_targets: [batch_size, None, None, num_joints],
        Batch.part_score_weights: [batch_size, None, None, num_joints],
        Batch.locref_targets: [batch_size, None, None, num_joints * 2],
        Batch.locref_mask: [batch_size, None, None, num_joints * 2],
    }


def setup_preloading(batch_spec):
    placeholders = {
        name: tf.compat.v1.placeholder(tf.float32, shape=spec)
        for (name, spec) in batch_spec.items()
    }
    names = placeholders.keys()
    placeholders_list = list(placeholders.values())

    QUEUE_SIZE = 20
    q = tf.queue.FIFOQueue(QUEUE_SIZE, [tf.float32] * len(batch_spec))
    enqueue_op = q.enqueue(placeholders_list)
    batch_list = q.dequeue()

    batch = {}
    for idx, name in enumerate(names):
        batch[name] = batch_list[idx]
        batch[name].set_shape(batch_spec[name])
    return batch, enqueue_op, placeholders


def load_and_enqueue(sess, enqueue_op, coord, dataset, placeholders):
    while not coord.should_stop():
        batch_np = dataset.next_batch()
        food = {pl: batch_np[name] for (name, pl) in placeholders.items()}
        sess.run(enqueue_op, feed_dict=food)


def start_preloading(sess, enqueue_op, dataset, placeholders):
    coord = tf.compat.v1.train.Coordinator()
    t = threading.Thread(
        target=load_and_enqueue,
        args=(sess, enqueue_op, coord, dataset, placeholders),
    )
    t.start()
    return coord, t


def get_optimizer(loss_op, cfg):
    tstep = tf.compat.v1.placeholder(tf.int32, shape=[], name="tstep")
    if "efficientnet" in cfg["net_type"]:
        print("Switching to cosine decay schedule with adam!")
        cfg["optimizer"] = "adam"
        learning_rate = tf.compat.v1.train.cosine_decay(
            cfg["lr_init"], tstep, cfg["decay_steps"], alpha=cfg["alpha_r"]
        )
    else:
        learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[])

    if cfg["optimizer"] == "sgd":
        optimizer = tf.compat.v1.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=0.9
        )
    elif cfg["optimizer"] == "adam":
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    else:
        raise ValueError("unknown optimizer {}".format(cfg["optimizer"]))
    train_op = slim.learning.create_train_op(loss_op, optimizer)

    return learning_rate, train_op, tstep


def get_optimizer_with_freeze(loss_op, cfg):
    learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[])

    if cfg["optimizer"] == "sgd":
        optimizer = tf.compat.v1.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=0.9
        )
    elif cfg["optimizer"] == "adam":
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    else:
        raise ValueError("unknown optimizer {}".format(cfg["optimizer"]))

    train_unfrozen_op = slim.learning.create_train_op(loss_op, optimizer)
    variables_unfrozen = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "pose"
    )

    train_frozen_op = slim.learning.create_train_op(
        loss_op, optimizer, variables_to_train=variables_unfrozen
    )

    return learning_rate, train_unfrozen_op, train_frozen_op


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
    net_type = cfg["net_type"]
    if cfg["dataset_type"] in ("scalecrop", "tensorpack", "deterministic"):
        print(
            "Switching batchsize to 1, as tensorpack/scalecrop/deterministic loaders do not support batches >1. Use imgaug/default loader."
        )
        cfg["batch_size"] = 1  # in case this was edited for analysis.-

    dataset = PoseDatasetFactory.create(cfg)
    batch_spec = get_batch_spec(cfg)
    batch, enqueue_op, placeholders = setup_preloading(batch_spec)

    losses = PoseNetFactory.create(cfg).train(batch)
    total_loss = losses["total_loss"]

    for k, t in losses.items():
        tf.compat.v1.summary.scalar(k, t)
    merged_summaries = tf.compat.v1.summary.merge_all()

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

    # Auto-switch to Adam on M1/M2 chips, as the momentum optimizer crashes
    from tensorflow.python.platform import build_info

    info = build_info.build_info
    if not info["is_cuda_build"]:  # Apple Silicon is not built with CUDA
        warnings.warn("Switching to Adam, as SGD crashes on Apple Silicon.")
        cfg["optimizer"] = "adam"

    if cfg.get("freezeencoder", False):
        if "efficientnet" in net_type:
            print("Freezing ONLY supported MobileNet/ResNet currently!!")
            learning_rate, train_op, tstep = get_optimizer(total_loss, cfg)

        print("Freezing encoder...")
        learning_rate, _, train_op = get_optimizer_with_freeze(total_loss, cfg)
    else:
        learning_rate, train_op, tstep = get_optimizer(total_loss, cfg)

    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())

    # Restore variables from disk.
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

    cum_loss = 0.0
    lr_gen = LearningRate(cfg)

    stats_path = Path(config_yaml).with_name("learning_stats.csv")
    lrf = open(str(stats_path), "w")

    print("Training parameter:")
    print(cfg)
    print("Starting training....")
    max_iter += start_iter  # max_iter is relative to start_iter
    for it in range(start_iter, max_iter + 1):
        if "efficientnet" in net_type:
            lr_dict = {tstep: it - start_iter}
            current_lr = sess.run(learning_rate, feed_dict=lr_dict)
        else:
            current_lr = lr_gen.get_lr(it - start_iter)
            lr_dict = {learning_rate: current_lr}

        [_, loss_val, summary] = sess.run(
            [train_op, total_loss, merged_summaries], feed_dict=lr_dict
        )
        cum_loss += loss_val
        train_writer.add_summary(summary, it)

        if it % display_iters == 0 and it > start_iter:
            average_loss = cum_loss / display_iters
            cum_loss = 0.0
            logging.info(
                "iteration: {} loss: {} lr: {}".format(
                    it, "{0:.4f}".format(average_loss), current_lr
                )
            )
            lrf.write("{}, {:.5f}, {}\n".format(it, average_loss, current_lr))
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
