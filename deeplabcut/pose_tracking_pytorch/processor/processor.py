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

import logging
import os
import time
import torch
import torch.nn as nn
from ..tracking_utils.meter import AverageMeter
from ..tracking_utils.metrics import R1_mAP_eval
import torch.distributed as dist
import pickle
import numpy as np


def dist(a, b):
    return torch.sqrt(torch.sum((a - b) ** 2, dim=1))


def calc_correct(anchor, pos, neg):
    # cos = torch.cdist
    ap_dist = dist(anchor, pos)
    an_dist = dist(anchor, neg)
    indices = ap_dist < an_dist

    return torch.sum(indices)


def calc_cos_correct(vec1, gt1, vec2, gt2, threshold=0.5):

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    confidence = cos(vec1, vec2)

    pred_mask = confidence > threshold

    gt_mask = gt1 == gt2
    n_correct = torch.sum(torch.eq(pred_mask, gt_mask))
    return n_correct


# TODO: maybe find a better spot for this.
def default_device(device="cuda"):  # setting CPU, if no GPU available
    # dev =  device if torch.cuda.is_available() else "cpu"
    dev = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
    return dev


def do_dlc_train(
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
    total_epochs=300,
    ckpt_folder="",
):

    log_period = cfg["log_period"]
    checkpoint_period = cfg["checkpoint_period"]
    eval_period = 10

    device = default_device(cfg["device"])

    logger = logging.getLogger("transreid.train")
    logger.info("start training")
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(device)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg["feat_norm"])

    # train
    epoch_list = []
    train_acc_list = []
    test_acc_list = []
    plot_dict = {}
    for epoch in range(1, total_epochs + 1):
        epoch_list.append(epoch)
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        total_n = 0.0
        total_correct = 0.0
        for n_iter, (anchor, pos, neg) in enumerate(train_loader):

            optimizer.zero_grad()

            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            anchor_feat = model(anchor)
            pos_feat = model(pos)
            neg_feat = model(neg)

            loss = triplet_loss(anchor_feat, pos_feat, neg_feat)

            loss.backward()

            optimizer.step()

            total_n += anchor_feat.shape[0]
            total_correct += calc_correct(anchor_feat, pos_feat, neg_feat)

            loss_meter.update(loss.item())  # , img.shape[0])

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, , Base Lr: {:.2e}".format(
                        epoch,
                        (n_iter + 1),
                        len(train_loader),
                        loss_meter.avg,
                        scheduler._get_lr(epoch)[0],
                    )
                )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        train_acc = total_correct / total_n
        train_acc_list.append(train_acc.item())

        if cfg["dist_train"]:
            pass
        else:
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch, time_per_batch, train_loader.batch_size / time_per_batch
                )
            )

        model_name = f"dlc_transreid"

        if epoch % checkpoint_period == 0:

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "num_kpts": num_kpts,
                    "feature_dim": feature_dim,
                },
                os.path.join(ckpt_folder, model_name + "_{}.pth".format(epoch)),
            )

        if epoch % eval_period == 0:
            model.eval()
            val_loss = 0.0
            total_n = 0.0
            total_correct = 0.0
            for n_iter, (anchor, pos, neg) in enumerate(val_loader):
                with torch.no_grad():
                    anchor = anchor.to(device)
                    pos = pos.to(device)
                    neg = neg.to(device)
                    anchor_feat = model(anchor)
                    pos_feat = model(pos)
                    neg_feat = model(neg)
                    loss = triplet_loss(anchor_feat, pos_feat, neg_feat)
                    val_loss += loss.item()

                    total_n += anchor_feat.shape[0]
                    total_correct += calc_correct(anchor_feat, pos_feat, neg_feat)

            logger.info("Validation Results - Epoch: {}".format(epoch))

            # print (f'validation loss {val_loss/len(val_loader)}')
            test_acc = total_correct / total_n
            test_acc_list.append(test_acc.item())
            print(f"Epoch {epoch}, train acc: {train_acc:.2f}")
            print(f"Epoch {epoch}, test acc {test_acc:.2f}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    plot_dict["train_acc"] = train_acc_list
    plot_dict["test_acc"] = test_acc_list
    plot_dict["epochs"] = epoch_list

    with open(
        os.path.join(ckpt_folder, "dlc_transreid_results.pickle"), "wb"
    ) as handle:
        pickle.dump(plot_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def do_dlc_inference(cfg, model, triplet_loss, val_loader, num_query):

    device = default_device(cfg["device"])
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg["feat_norm"])

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for inference".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    val_loss = 0.0

    features_list = []
    labels_list = []
    total_n = 0.0
    total_correct = 0.0
    for n_iter, (anchor, pos, neg) in enumerate(val_loader):
        with torch.no_grad():

            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            anchor_feat = model(anchor)
            pos_feat = model(pos)
            neg_feat = model(neg)

            features_list.append(pos_feat.cpu().detach().numpy())
            features_list.append(neg_feat.cpu().detach().numpy())
            for i in range(neg.shape[0]):
                labels_list.append(0)
                labels_list.append(1)
            total_n += anchor_feat.shape[0]
            total_correct += calc_correct(anchor_feat, pos_feat, neg_feat)

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_dist = cos(anchor_feat, pos_feat)
            print("cos_dist ap", cos_dist)
            cos_dist = cos(anchor_feat, neg_feat)
            print("cos_dist an", cos_dist)

            loss = triplet_loss(anchor_feat, pos_feat, neg_feat)
            val_loss += loss.item()

    features_list = np.vstack(features_list)
    with open("video_trans_features.npy", "wb") as f:
        np.save(f, features_list)
    with open("labels.npy", "wb") as f:
        np.save(f, labels_list)
    print(f"validation loss {val_loss/len(val_loader)}")
    print(f" acc {total_correct/total_n}")
    logger.info("Validation Results ")


def do_dlc_pair_inference(cfg, model, val_loader, num_query):

    device = default_device(cfg["device"])
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg["feat_norm"])

    evaluator.reset()

    if device and torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for inference".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    val_loss = 0.0

    total_n = 0.0
    total_correct = 0.0
    for n_iter, ((vec1, gt1), (vec2, gt2)) in enumerate(val_loader):
        with torch.no_grad():

            gt1 = gt1.to(device)
            gt2 = gt2.to(device)
            vec1 = vec1.to(device)
            vec2 = vec2.to(device)

            vec1_feat = model(vec1)
            vec2_feat = model(vec2)

            total_n += vec1_feat.shape[0]
            total_correct += calc_cos_correct(vec1_feat, gt1, vec2_feat, gt2)

    print(f" acc {total_correct/total_n}")
    logger.info("Validation Results ")
