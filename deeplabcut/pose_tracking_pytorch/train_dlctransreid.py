from .tracking_utils.logger import setup_logger
from .datasets import  make_dlc_dataloader
from .model import  make_dlc_model
from .solver import make_optimizer, make_easy_optimizer
from .solver.scheduler_factory import create_scheduler
from .loss import make_loss
from .loss import easy_triplet_loss
from .processor import  do_dlc_train
import random
import torch
import numpy as np
import os
import sys
import argparse
import glob
from pathlib import Path

# from timm.scheduler import create_scheduler
from .config import cfg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def split_train_test(npy_list):
    # with npy list form videos, split each to train and test

    x_list = []
    train_list = []
    test_list = []
    
    for npy in npy_list:
        vectors = np.load(npy)
        n_samples = vectors.shape[0]
        indices = np.random.permutation(n_samples)
        num_train = int(n_samples * 0.8)
        vectors = vectors[indices]
        train = vectors[:num_train]
        test = vectors[num_train:]
        train_list.append(train)
        test_list.append(test)

    train_list = np.concatenate(train_list, axis = 0)
    test_list = np.concatenate(test_list, axis = 0)

    
    
    return train_list, test_list
    
    
def train_tracking_transformer(videos):

    npy_list = []
    for video in videos:
        videofolder = str(Path(video).parents[0])
        video_name = video.split('/')[-1].split('.')[0]
        files = glob.glob(os.path.join(videofolder,video_name+'*.npy'))
        # assuming there is only one match
        npy_list.append(files[0])        

    train_list, test_list = split_train_test(npy_list)
        
    train_loader, val_loader = make_dlc_dataloader(train_list, test_list)


    # make my own model factory 

    num_kpts = train_list.shape[2]
    
    model = make_dlc_model(cfg, num_kpts)
    
    # make my own loss factory

    triplet_loss= easy_triplet_loss()

    # ok

    optimizer = make_easy_optimizer(cfg,model)
    
    #optimizer, optimizer_center = make_optimizer(cfg, model, triplet_loss)

    # probably need to change optimizer too
    
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
        num_query, 0,  total_epochs = 100
    )


    
