import logging
import os
import time
import torch
import torch.nn as nn
from ..tracking_utils.meter import AverageMeter
from ..tracking_utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import matplotlib.pyplot as plt
import pickle
import numpy as np
def dist(a,b):
    return  torch.sqrt(torch.sum((a - b)**2, dim = 1))
def calc_correct(anchor,pos,neg):
    #cos = torch.cdist
    ap_dist = dist(anchor, pos)
    an_dist = dist(anchor, neg)
    indices = ap_dist < an_dist

    return torch.sum(indices)
def calc_cos_correct(vec1, gt1, vec2, gt2, threshold = 0.5):
    
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    confidence = cos(vec1, vec2)

    pred_mask = confidence> threshold

    gt_mask = gt1 == gt2
    n_correct = torch.sum(torch.eq(pred_mask,gt_mask))
    return n_correct
    



    


def do_dlc_train(cfg,
             model,
             triplet_loss,
             train_loader,
             val_loader,
             optimizer,
             scheduler,
             num_kpts,
                 num_query, local_rank,  total_epochs = 300, ckpt_folder = ''):
    
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    #eval_period = cfg.SOLVER.EVAL_PERIOD
    eval_period = 10

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
            model = torch.nn.parallel.DistributedDataParallel(model,  find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    #scaler = amp.GradScaler()
    # train
    epoch_list = []
    train_acc_list = []
    test_acc_list = []
    plot_dict = {}
    for epoch in range(1, total_epochs+1):
        epoch_list.append(epoch)
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        total_n = 0.0
        total_correct = 0.0        
        for n_iter, (anchor,pos,neg) in enumerate(train_loader):
            
            optimizer.zero_grad()
                        
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            
            #anchor_feat, pos_feat, neg_feat = model(anchor,pos,neg)
            anchor_feat = model(anchor)
            pos_feat = model(pos)
            neg_feat = model(neg)  

            loss = triplet_loss(anchor_feat, pos_feat, neg_feat)

            loss.backward()

            optimizer.step()

            total_n += anchor_feat.shape[0]
            total_correct += calc_correct(anchor_feat, pos_feat, neg_feat)            

            loss_meter.update(loss.item())#, img.shape[0])


            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, , Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg,  scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        train_acc = total_correct/total_n
        train_acc_list.append(train_acc.item())

        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        model_name = f'dlc_transreid'
        
        
        if epoch % checkpoint_period == 0:
        
            torch.save(
                       {'state_dict':model.state_dict(),
                        'num_kpts': num_kpts
                        },
                       os.path.join(ckpt_folder, model_name + '_{}.pth'.format(epoch)))


        if epoch % eval_period == 0:
            model.eval()
            val_loss = 0.0
            total_n = 0.0
            total_correct = 0.0
            for n_iter, (anchor,pos,neg) in enumerate(val_loader):
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
            
            #print (f'validation loss {val_loss/len(val_loader)}')
            test_acc = total_correct/total_n
            test_acc_list.append(test_acc.item())
            print (f'Epoch {epoch}, train acc: {train_acc:.2f}')               
            print (f'Epoch {epoch}, test acc {test_acc:.2f}')
            torch.cuda.empty_cache()
    plot_dict['train_acc'] = train_acc_list
    plot_dict['test_acc'] = test_acc_list
    plot_dict['epochs'] = epoch_list

    with open(f'plot.pickle', 'wb') as handle:
        pickle.dump(plot_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


def do_dlc_inference(cfg,
                     model,
                     triplet_loss,
                     val_loader,
                     num_query):
    
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
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
            print ('cos_dist ap', cos_dist)
            cos_dist = cos(anchor_feat, neg_feat)
            print ('cos_dist an', cos_dist)            
            
            
            loss = triplet_loss(anchor_feat, pos_feat, neg_feat)
            val_loss += loss.item()
            
    features_list = np.vstack(features_list)
    with open('video_trans_features.npy', 'wb') as f:
        np.save(f, features_list)
    with open('labels.npy', 'wb') as f:
        np.save(f, labels_list)            
    print (f'validation loss {val_loss/len(val_loader)}')
    print (f' acc {total_correct/total_n}')
    logger.info("Validation Results ")


def do_dlc_pair_inference(cfg,
                     model,                
                     val_loader,
                     num_query):
    
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    val_loss = 0.0

        
    total_n = 0.0
    total_correct = 0.0
    for n_iter, ((vec1,gt1), (vec2, gt2)) in enumerate(val_loader):
        with torch.no_grad():

            gt1 = gt1.to(device)
            gt2 = gt2.to(device)
            vec1 = vec1.to(device)
            vec2 = vec2.to(device)
            
            vec1_feat = model(vec1)
            vec2_feat = model(vec2)

            total_n += vec1_feat.shape[0]
            total_correct += calc_cos_correct(vec1_feat,gt1, vec2_feat, gt2)
            

    print (f' acc {total_correct/total_n}')
    logger.info("Validation Results ")
    
