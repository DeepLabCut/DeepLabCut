"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow

"""
import logging, os
import threading
import argparse
from pathlib import Path
import tensorflow as tf
vers = (tf.__version__).split('.')
if int(vers[0])==1 and int(vers[1])>12:
    TF=tf.compat.v1
else:
    TF=tf
import tensorflow.contrib.slim as slim

from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import Batch
from deeplabcut.pose_estimation_tensorflow.dataset.factory import create as create_dataset
from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net
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

def get_batch_spec(cfg):
    num_joints = cfg.num_joints
    batch_size = cfg.batch_size
    return {
        Batch.inputs: [batch_size, None, None, 3],
        Batch.part_score_targets: [batch_size, None, None, num_joints],
        Batch.part_score_weights: [batch_size, None, None, num_joints],
        Batch.locref_targets: [batch_size, None, None, num_joints * 2],
        Batch.locref_mask: [batch_size, None, None, num_joints * 2]
    }

def setup_preloading(batch_spec):
    placeholders = {name: TF.placeholder(tf.float32, shape=spec) for (name, spec) in batch_spec.items()}
    names = placeholders.keys()
    placeholders_list = list(placeholders.values())

    QUEUE_SIZE = 20
    vers = (tf.__version__).split('.')
    if int(vers[0])==1 and int(vers[1])>12:
        q = tf.queue.FIFOQueue(QUEUE_SIZE, [tf.float32]*len(batch_spec))
    else:
        q = tf.FIFOQueue(QUEUE_SIZE, [tf.float32]*len(batch_spec))
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

    t = threading.Thread(target=load_and_enqueue,
                         args=(sess, enqueue_op, coord, dataset, placeholders))
    t.start()

    return coord, t

def get_optimizer(loss_op, cfg):
    learning_rate = TF.placeholder(tf.float32, shape=[])

    if cfg.optimizer == "sgd":
        optimizer = TF.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    elif cfg.optimizer == "adam":
        optimizer = TF.train.AdamOptimizer(learning_rate)
    else:
        raise ValueError('unknown optimizer {}'.format(cfg.optimizer))
    train_op = slim.learning.create_train_op(loss_op, optimizer)

    return learning_rate, train_op

def train(config_yaml,displayiters,saveiters,maxiters,max_to_keep=5,keepdeconvweights=True,allow_growth=False):
    start_path=os.getcwd()
    os.chdir(str(Path(config_yaml).parents[0])) #switch to folder of config_yaml (for logging)
    setup_logging()

    cfg = load_config(config_yaml)
    if cfg.dataset_type=='default' or cfg.dataset_type=='tensorpack' or cfg.dataset_type=='deterministic':
        print("Switching batchsize to 1, as default/tensorpack/deterministic loaders do not support batches >1. Use imgaug loader.")
        cfg['batch_size']=1 #in case this was edited for analysis.-

    dataset = create_dataset(cfg)
    batch_spec = get_batch_spec(cfg)
    batch, enqueue_op, placeholders = setup_preloading(batch_spec)
    losses = pose_net(cfg).train(batch)
    total_loss = losses['total_loss']

    for k, t in losses.items():
        TF.summary.scalar(k, t)
    merged_summaries = TF.summary.merge_all()

    if 'snapshot' in Path(cfg.init_weights).stem and keepdeconvweights:
        print("Loading already trained DLC with backbone:", cfg.net_type)
        variables_to_restore = slim.get_variables_to_restore()
    else:
        print("Loading ImageNet-pretrained", cfg.net_type)
        #loading backbone from ResNet, MobileNet etc.
        if 'resnet' in cfg.net_type:
            variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])
        elif 'mobilenet' in cfg.net_type:
            variables_to_restore = slim.get_variables_to_restore(include=["MobilenetV2"])
        else:
            print("Wait for DLC 2.3.")

    restorer = TF.train.Saver(variables_to_restore)
    saver = TF.train.Saver(max_to_keep=max_to_keep) # selects how many snapshots are stored, see https://github.com/AlexEMG/DeepLabCut/issues/8#issuecomment-387404835

    if allow_growth==True:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = TF.Session(config=config)
    else:
        sess = TF.Session()

    coord, thread = start_preloading(sess, enqueue_op, dataset, placeholders)
    train_writer = TF.summary.FileWriter(cfg.log_dir, sess.graph)
    learning_rate, train_op = get_optimizer(total_loss, cfg)

    sess.run(TF.global_variables_initializer())
    sess.run(TF.local_variables_initializer())

    # Restore variables from disk.
    restorer.restore(sess, cfg.init_weights)
    if maxiters==None:
        max_iter = int(cfg.multi_step[-1][1])
    else:
        max_iter = min(int(cfg.multi_step[-1][1]),int(maxiters))
        #display_iters = max(1,int(displayiters))
        print("Max_iters overwritten as",max_iter)

    if displayiters==None:
        display_iters = max(1,int(cfg.display_iters))
    else:
        display_iters = max(1,int(displayiters))
        print("Display_iters overwritten as",display_iters)

    if saveiters==None:
        save_iters=max(1,int(cfg.save_iters))

    else:
        save_iters=max(1,int(saveiters))
        print("Save_iters overwritten as",save_iters)

    cum_loss = 0.0
    lr_gen = LearningRate(cfg)

    stats_path = Path(config_yaml).with_name('learning_stats.csv')
    lrf = open(str(stats_path), 'w')

    print("Training parameter:")
    print(cfg)
    print("Starting training....")
    for it in range(max_iter+1):
        current_lr = lr_gen.get_lr(it)
        [_, loss_val, summary] = sess.run([train_op, total_loss, merged_summaries],
                                          feed_dict={learning_rate: current_lr})
        cum_loss += loss_val
        train_writer.add_summary(summary, it)

        if it % display_iters == 0 and it>0:
            average_loss = cum_loss / display_iters
            cum_loss = 0.0
            logging.info("iteration: {} loss: {} lr: {}"
                         .format(it, "{0:.4f}".format(average_loss), current_lr))
            lrf.write("{}, {:.5f}, {}\n".format(it, average_loss, current_lr))
            lrf.flush()

        # Save snapshot
        if (it % save_iters == 0 and it != 0) or it == max_iter:
            model_name = cfg.snapshot_prefix
            saver.save(sess, model_name, global_step=it)

    lrf.close()
    sess.close()
    coord.request_stop()
    coord.join([thread])
    #return to original path.
    os.chdir(str(start_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to yaml configuration file.')
    cli_args = parser.parse_args()

    train(Path(cli_args.config).resolve())
