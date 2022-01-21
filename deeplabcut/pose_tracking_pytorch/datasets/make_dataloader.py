import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .dlc_vec import TripletDataset, PairDataset

__factory = {"dlc_triplet": TripletDataset, "dlc_pair": PairDataset}


def train_collate_fn(batch):
    """ """
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return (
        torch.stack(imgs, dim=0),
        pids,
        camids,
        viewids,
    )


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths




def make_dlc_dataloader(train_list, test_list):

    # normalizing the features in the dataset getitem function

    train_transform = None
    val_transform = None

    num_workers = 2

    # use my own vec dataset

    print(train_list.shape)

    train_dataset = TripletDataset(train_list)

    # use triplet loss

    batch_size = 64


    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataset = TripletDataset(test_list)

    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
