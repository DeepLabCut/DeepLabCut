'''
Adapted from original predict.py by Eldar Insafutdinov's implementation of [DeeperCut](https://github.com/eldar/pose-tensorflow)
To do faster inference on videos. See https://www.biorxiv.org/content/early/2018/10/30/457242

Source: DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
'''

import math
import os
import sys
import numpy as np
from collections import namedtuple
import tensorflow as tf
import tensorflow.contrib.slim as slim
from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net

## Functions below implement are for batch sizes > 1:
def extract_cnn_output(outputs_np, cfg):
    ''' extract locref + scmap from network
    Dimensions: image batch x imagedim1 x imagedim2 x bodypart'''
    scmap = outputs_np[0]
    if cfg.location_refinement:
        locref =outputs_np[1]
        shape = locref.shape
        locref = np.reshape(locref, (shape[0], shape[1],shape[2], -1, 2))
        locref *= cfg.locref_stdev
    else:
        locref = None
    if cfg.pairwise_predict:
        pairwise_diff = outputs_np[2]
    else:
        paf=None

    if len(scmap.shape)==2: #for single body part!
        scmap=np.expand_dims(scmap,axis=2)
    return scmap, locref, pairwise_diff

def get_pairwise_index(j_id, j_id_end, num_joints):
    return (num_joints - 1) * j_id + j_id_end - int(j_id < j_id_end)

def getposewithpairwise(image, cfg, sess, inputs, outputs, pairwise_stats):
    ''' Adapted from DeeperCut, performs numpy-based faster inference on batches'''
    outputs_np = sess.run(outputs, feed_dict={inputs: image})

    scmap, locref, pairwise_diff = extract_cnn_output(outputs_np, cfg) #processes image batch.
    batchsize,ny,nx,num_joints = scmap.shape

    pairwise_diff = np.reshape(pairwise_diff, (batchsize,ny, nx, -1, 2))
    for pair in pairwise_stats: #rescaling
            #pair_id = (num_joints - 1) * pair[0] + pair[1] - int(pair[0] < pair[1])
            pair_id = get_pairwise_index(pair[0], pair[1], num_joints)
            pairwise_diff[:,:, :, pair_id, 0] *= pairwise_stats[pair]["std"][0] #x
            pairwise_diff[:,:, :, pair_id, 0] += pairwise_stats[pair]["mean"][0]
            pairwise_diff[:,:, :, pair_id, 1] *= pairwise_stats[pair]["std"][1] #y
            pairwise_diff[:,:, :, pair_id, 1] += pairwise_stats[pair]["mean"][1]

    num_pairwise_relations = pairwise_diff.shape[3]
    #pairwisepredictions=np.zeros((batchsize,num_pairwise_relations,2)) #[joint0 > 1 x,y >2 x,y > 3x,y etc.]

    pairwisepredictions=np.zeros((batchsize,num_joints,num_joints,2)) 

    #Combine scoremat and offsets to the final pose.
    LOCREF=locref.reshape(batchsize,nx*ny,num_joints,2)
    MAXLOC=np.argmax(scmap.reshape(batchsize,nx*ny,num_joints),axis=1)
    Y,X=np.unravel_index(MAXLOC,dims=(ny,nx))
    DZ=np.zeros((batchsize,num_joints,3))

    for l in range(batchsize):
        for joint_idx in range(num_joints):
            DZ[l,joint_idx,:2]=LOCREF[l,MAXLOC[l,joint_idx],joint_idx,:]
            DZ[l,joint_idx,2]=scmap[l,Y[l,joint_idx],X[l,joint_idx],joint_idx] #peak probability
            for joint_idx_end in range(num_joints):
                if joint_idx_end != joint_idx:
                    #pair_id = (num_joints - 1) * joint_idx + joint_idx_end - int(joint_idx < joint_idx_end)
                    pair_id = get_pairwise_index(joint_idx, joint_idx_end, num_joints)
                    #difference = np.array(pairwise_diff[maxloc][pair_id])[::-1] if pairwise_diff is not None else 0
                    #pairwisepredictions[l,pair_id,:]=pairwise_diff[l,Y[l,joint_idx],X[l,joint_idx],pair_id,:]
                    pairwisepredictions[l,joint_idx,joint_idx_end,:]=pairwise_diff[l,Y[l,joint_idx],X[l,joint_idx],pair_id,:]

    X=X.astype('float32')*cfg.stride+.5*cfg.stride+DZ[:,:,0]
    Y=Y.astype('float32')*cfg.stride+.5*cfg.stride+DZ[:,:,1]

    pose = np.empty((cfg['batch_size'], cfg['num_joints']*3), dtype=X.dtype)
    pose[:,0::3] = X
    pose[:,1::3] = Y
    pose[:,2::3] = DZ[:,:,2] #P

    return pose,pairwisepredictions
