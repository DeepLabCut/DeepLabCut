import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
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


def make_dataloader(cfg):
    train_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(
                probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"
            ),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ]
    )

    val_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if "triplet" in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print("DIST_TRAIN START")
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(
                dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE
            )
            batch_sampler = torch.utils.data.sampler.BatchSampler(
                data_sampler, mini_batch_size, True
            )
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set,
                batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(
                    dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE
                ),
                num_workers=num_workers,
                collate_fn=train_collate_fn,
            )
    elif cfg.DATALOADER.SAMPLER == "softmax":
        print("using softmax sampler")
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collate_fn,
        )
    else:
        print(
            "unsupported sampler! expected softmax or triplet but got {}".format(
                cfg.SAMPLER
            )
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
    )
    train_loader_normal = DataLoader(
        train_set_normal,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
    )
    return (
        train_loader,
        train_loader_normal,
        val_loader,
        len(dataset.query),
        num_classes,
        cam_num,
        view_num,
    )


def make_dlc_pair_dataloader():

    test_dataset = PairDataset(f"{animal}_pair_test.pickle")

    test_loader = DataLoader(test_dataset, batch_size=32)

    return test_loader


def make_dlc_dataloader(train_list, test_list):

    # normalizing the features in the dataset getitem function

    train_transform = None
    val_transform = None

    num_workers = 2

    # use my own vec dataset

    print(train_list.shape)

    train_dataset = TripletDataset(train_list)

    # use triplet loss

    train_loader = DataLoader(train_dataset, batch_size=128)
    val_dataset = TripletDataset(test_list)

    val_loader = DataLoader(val_dataset, batch_size=128)

    return train_loader, val_loader
