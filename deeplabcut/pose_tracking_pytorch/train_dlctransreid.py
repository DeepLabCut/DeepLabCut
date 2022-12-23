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

import random

try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Unsupervised identity learning requires PyTorch. Please run `pip install torch`."
    )
import numpy as np
import os
import glob
from deeplabcut.utils import auxiliaryfunctions
from pathlib import Path
from .config import cfg
from .datasets import make_dlc_dataloader
from .model import make_dlc_model
from .solver import make_easy_optimizer
from .solver.scheduler_factory import create_scheduler
from .loss import easy_triplet_loss
from .processor import do_dlc_train


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def split_train_test(npy_list, train_frac):
    # with npy list form videos, split each to train and test

    x_list = []
    train_list = []
    test_list = []

    for npy in npy_list:
        vectors = np.load(npy)
        n_samples = vectors.shape[0]
        indices = np.random.permutation(n_samples)
        num_train = int(n_samples * train_frac)
        vectors = vectors[indices]
        train = vectors[:num_train]
        test = vectors[num_train:]
        train_list.append(train)
        test_list.append(test)

    train_list = np.concatenate(train_list, axis=0)
    test_list = np.concatenate(test_list, axis=0)

    return train_list, test_list


def train_tracking_transformer(
    path_config_file,
    dlcscorer,
    videos,
    videotype="",
    train_frac=0.8,
    modelprefix="",
    train_epochs=100,
    batch_size=64,
    ckpt_folder="",
    destfolder=None,
):
    npy_list = []
    videos = auxiliaryfunctions.get_list_of_videos(videos, videotype)
    for video in videos:
        videofolder = str(Path(video).parents[0])
        if destfolder is None:
            destfolder = videofolder
        video_name = Path(video).stem
        # video_name = '.'.join(video.split("/")[-1].split(".")[:-1])
        files = glob.glob(os.path.join(destfolder, video_name + dlcscorer + "*.npy"))

        # assuming there is only one match
        npy_list.append(files[0])

    train_list, test_list = split_train_test(npy_list, train_frac)

    train_loader, val_loader = make_dlc_dataloader(train_list, test_list, batch_size)

    # make my own model factory
    num_kpts = train_list.shape[2]
    feature_dim = train_list.shape[-1]
    model = make_dlc_model(cfg, feature_dim, num_kpts)

    # make my own loss factory
    triplet_loss = easy_triplet_loss()

    optimizer = make_easy_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)

    num_query = 1

    do_dlc_train(
        cfg,
        model,
        triplet_loss,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        num_kpts,
        feature_dim,
        num_query,
        total_epochs=train_epochs,
        ckpt_folder=ckpt_folder,
    )
