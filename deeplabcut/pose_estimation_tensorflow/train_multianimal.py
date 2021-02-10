"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
"""
import argparse
import logging
import os
import threading
from pathlib import Path

import tensorflow as tf

vers = (tf.__version__).split(".")
if int(vers[0]) == 1 and int(vers[1]) > 12:
    TF = tf.compat.v1
else:
    TF = tf
import tensorflow.contrib.slim as slim

from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.dataset.factory import (
    create as create_dataset,
)
from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net
from deeplabcut.pose_estimation_tensorflow.nnet.pose_net import get_batch_spec
from deeplabcut.pose_estimation_tensorflow.util.logging import setup_logging


class LearningRate(object):
    def __init__(self, cfg):
        self.steps = cfg.multi_step
        self.current_step = 0

    def get_lr(self, iteration):
        lr = self.steps[self.current_step][0]
        if iteration == self.steps[self.current_step][1]:
            self.current_step += 1

        return lr


def setup_preloading(batch_spec):
    placeholders = {
        name: TF.placeholder(tf.float32, shape=spec)
        for (name, spec) in batch_spec.items()
    }
    names = placeholders.keys()
    placeholders_list = list(placeholders.values())

    QUEUE_SIZE = 20
    vers = (tf.__version__).split(".")
    if int(vers[0]) == 1 and int(vers[1]) > 12:
        q = tf.queue.FIFOQueue(QUEUE_SIZE, [tf.float32] * len(batch_spec))
    else:
        q = tf.FIFOQueue(QUEUE_SIZE, [tf.float32] * len(batch_spec))
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
    coord = TF.train.Coordinator()
    t = threading.Thread(
        target=load_and_enqueue, args=(sess, enqueue_op, coord, dataset, placeholders)
    )
    t.start()

    return coord, t


def get_optimizer(loss_op, cfg):
    tstep = tf.placeholder(tf.int32,shape=[],name='tstep')
    if 'efficientnet' in cfg.net_type:
        print("Switching to cosine decay schedule with adam!")
        cfg.optimizer == "adam"
        learning_rate = tf.train.cosine_decay(cfg.lr_init,
                                              tstep,
                                              cfg.decay_steps,
                                              alpha=cfg.alpha_r)
    else:
        learning_rate = tf.placeholder(tf.float32, shape=[])

    if cfg.optimizer == "sgd":
        optimizer = TF.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=0.9
        )
    elif cfg.optimizer == "adam":
        optimizer = TF.train.AdamOptimizer(learning_rate)
    else:
        raise ValueError("unknown optimizer {}".format(cfg.optimizer))
    train_op = slim.learning.create_train_op(loss_op, optimizer)

    return learning_rate, train_op, tstep


def train(
    config_yaml,
    displayiters,
    saveiters,
    maxiters,
    max_to_keep=5,
    keepdeconvweights=True,
    allow_growth=False,
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

    dataset = create_dataset(cfg)
    batch_spec = get_batch_spec(cfg)
    batch, enqueue_op, placeholders = setup_preloading(batch_spec)

    losses = pose_net(cfg).train(batch)
    total_loss = losses["total_loss"]

    for k, t in losses.items():
        TF.summary.scalar(k, t)
    merged_summaries = TF.summary.merge_all()

    if "snapshot" in Path(cfg.init_weights).stem and keepdeconvweights:
        print("Loading already trained DLC with backbone:", cfg.net_type)
        variables_to_restore = slim.get_variables_to_restore()
    else:
        print("Loading ImageNet-pretrained", cfg.net_type)
        # loading backbone from ResNet, MobileNet etc.
        if "resnet" in cfg.net_type:
            variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])
        elif "mobilenet" in cfg.net_type:
            variables_to_restore = slim.get_variables_to_restore(
                include=["MobilenetV2"]
            )
        elif 'efficientnet' in cfg.net_type:
            variables_to_restore = slim.get_variables_to_restore(include=["efficientnet"])
            variables_to_restore = {
                    var.op.name.replace("efficientnet/", "")
                    + '/ExponentialMovingAverage':var for var in variables_to_restore}
        else:
            print("Wait for DLC 2.3.")

    restorer = TF.train.Saver(variables_to_restore)
    saver = TF.train.Saver(
        max_to_keep=max_to_keep
    )  # selects how many snapshots are stored, see https://github.com/AlexEMG/DeepLabCut/issues/8#issuecomment-387404835

    if allow_growth:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = TF.Session(config=config)
    else:
        sess = TF.Session()

    coord, thread = start_preloading(sess, enqueue_op, dataset, placeholders)
    train_writer = TF.summary.FileWriter(cfg.log_dir, sess.graph)
    learning_rate, train_op, tstep = get_optimizer(total_loss, cfg)

    sess.run(TF.global_variables_initializer())
    sess.run(TF.local_variables_initializer())

    # Restore variables from disk.
    if 'efficientnet' in cfg.net_type:
        init_weights = os.path.join(cfg.init_weights,"model.ckpt")
    else:
        init_weights = cfg.init_weights

    restorer.restore(sess, init_weights)
    if maxiters == None:
        max_iter = int(cfg.multi_step[-1][1])
    else:
        max_iter = min(int(cfg.multi_step[-1][1]), int(maxiters))
        # display_iters = max(1,int(displayiters))
        print("Max_iters overwritten as", max_iter)

    if displayiters == None:
        display_iters = max(1, int(cfg.display_iters))
    else:
        display_iters = max(1, int(displayiters))
        print("Display_iters overwritten as", display_iters)

    if saveiters == None:
        save_iters = max(1, int(cfg.save_iters))

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
    for it in range(max_iter + 1):
        if 'efficientnet' in cfg.net_type:
            dict={tstep: it}
            current_lr = sess.run(learning_rate,feed_dict=dict)
        else:
            current_lr = lr_gen.get_lr(it)
            dict={learning_rate: current_lr}

        # [_, loss_val, summary] = sess.run([train_op, total_loss, merged_summaries],feed_dict={learning_rate: current_lr})
        [_, alllosses, loss_val, summary] = sess.run(
            [train_op, losses, total_loss, merged_summaries],
            feed_dict=dict,
        )

        partloss += alllosses["part_loss"]  # scoremap loss
        if cfg.location_refinement:
            locrefloss += alllosses["locref_loss"]
        if cfg.pairwise_predict:  # paf loss
            pwloss += alllosses["pairwise_loss"]

        cumloss += loss_val
        train_writer.add_summary(summary, it)

        if it % display_iters == 0 and it > 0:
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
        if (it % save_iters == 0 and it != 0) or it == max_iter:
            model_name = cfg.snapshot_prefix
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
