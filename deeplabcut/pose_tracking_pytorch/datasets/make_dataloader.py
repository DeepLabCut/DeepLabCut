"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
from torch.utils.data import DataLoader
from .dlc_vec import TripletDataset


def make_dlc_dataloader(train_list, test_list, batch_size=64):
    train_dataset = TripletDataset(train_list)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataset = TripletDataset(test_list)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader
