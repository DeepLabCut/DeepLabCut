'''
Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow

'''

#import importlib
import logging, os
#importlib.reload(logging)

import numpy as np
import scipy.io

from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.dataset.factory import create as create_dataset
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import Batch

def remap_keys(mapping):
    return [{'key': k, 'value': v} for k, v in mapping.items()]

#TODO: update to pickle file!

def load_pairwise_stats(cfg):
    mat_stats = scipy.io.loadmat(cfg.pairwise_stats_fn)
    pairwise_stats = {}
    for id in range(len(mat_stats['graph'])):
        pair = tuple(mat_stats['graph'][id])
        pairwise_stats[pair] = {"mean": mat_stats['means'][id], "std": mat_stats['std_devs'][id]}
    for pair in pairwise_stats:
        pairwise_stats[pair]["mean"] *= cfg.global_scale
        pairwise_stats[pair]["std"] *= cfg.global_scale
    return pairwise_stats

def save_stats(stats, cfg):
    mat_stats = {}
    mat_stats["graph"] = []
    mat_stats["means"] = []
    mat_stats["std_devs"] = []
    for start in range(cfg.num_joints):
        for end in range(cfg.num_joints):
            if start != end:
                joint_pair = (start, end)
                mat_stats["graph"].append([start, end])
                mat_stats["means"].append(stats[joint_pair]["mean"])
                mat_stats["std_devs"].append(stats[joint_pair]["std"])
    print(mat_stats)
    scipy.io.savemat(cfg.pairwise_stats_fn, mat_stats)

# Compute pairwise statistics at reference scale
def pairwise_stats(path):
    cfg = load_config(path)

    dataset = create_dataset(cfg)
    dataset.set_shuffle(True)
    dataset.set_pairwise_stats_collect(True)

    num_images = dataset.num_images
    all_pairwise_differences = {}

    if cfg.mirror:
        num_images *= 2

    for k in range(num_images):
        print('processing image {}/{}'.format(k, num_images-1))
        batch = dataset.next_batch()
        batch_stats = batch[Batch.data_item].pairwise_stats
        for joint_pair in batch_stats:
            if joint_pair not in all_pairwise_differences:
                all_pairwise_differences[joint_pair] = []
            all_pairwise_differences[joint_pair] += batch_stats[joint_pair]

    stats = {}
    for joint_pair in all_pairwise_differences:
        stats[joint_pair] = {}
        stats[joint_pair]["mean"] = np.mean(all_pairwise_differences[joint_pair], axis=0)
        stats[joint_pair]["std"] = np.std(all_pairwise_differences[joint_pair], axis=0)

    save_stats(stats, cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to yaml configuration file.')
    cli_args = parser.parse_args()

    pairwise_stats(Path(cli_args.config).resolve())
