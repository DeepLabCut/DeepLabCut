"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

####################################################
# Dependencies
####################################################

import os.path
from deeplabcut.pose_estimation_tensorflow.nnet import predict_multianimal as predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import data_to_input
import time
import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal
import cv2
from skimage.util import img_as_ubyte


def AnalyzeMultiAnimalVideo(video,DLCscorer,trainFraction,cfg,dlc_cfg,sess,inputs, outputs,pdindex,save_as_csv, destfolder=None):
    ''' Helper function for analyzing a video with multiple individuals'''
    print("Starting to analyze % ", video)
    vname = Path(video).stem
    if destfolder is None:
        destfolder = str(Path(video).parents[0])
    dataname = os.path.join(destfolder,vname + DLCscorer + '.h5')

    if os.path.isfile(dataname.split('.h5')[0]+'_full.pickle'):
            print("Video already analyzed!", dataname)
    else:
        print("Loading ", video)
        cap=cv2.VideoCapture(video)

        fps = cap.get(5) #https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
        nframes = int(cap.get(7))
        duration=nframes*1./fps
        size=(int(cap.get(4)),int(cap.get(3)))

        ny,nx=size
        print("Duration of video [s]: ", round(duration,2), ", recorded with ", round(fps,2),"fps!")
        print("Overall # of frames: ", nframes," found with (before cropping) frame dimensions: ", nx,ny)
        start = time.time()

        print("Starting to extract posture")
        if int(dlc_cfg["batch_size"])>1:
            PredicteData,nframes=GetPoseandCostsF(cfg,dlc_cfg, sess, inputs, outputs,cap,nframes,int(dlc_cfg["batch_size"]))
        else:
            PredicteData,nframes=GetPoseandCostsS(cfg,dlc_cfg, sess, inputs, outputs,cap,nframes)

        stop = time.time()

        if cfg['cropping']==True:
            coords=[cfg['x1'],cfg['x2'],cfg['y1'],cfg['y2']]
        else:
            coords=[0, nx, 0, ny]

        dictionary = {
            "start": start,
            "stop": stop,
            "run_duration": stop - start,
            "Scorer": DLCscorer,
            "DLC-model-config file": dlc_cfg,
            "fps": fps,
            "batch_size": dlc_cfg["batch_size"],
            "frame_dimensions": (ny, nx),
            "nframes": nframes,
            "iteration (active-learning)": cfg["iteration"],
            "training set fraction": trainFraction,
            "cropping": cfg['cropping'],
            "cropping_parameters": coords
        }
        metadata = {'data': dictionary}
        print("Saving results in %s..." %(destfolder))

        #auxiliaryfunctions.SaveData(PredicteData[:nframes,:], metadata, dataname, pdindex, range(nframes),save_as_csv)
        #TODO: check whether already analyze...
        auxfun_multianimal.SaveFullMultiAnimalData(PredicteData, metadata, dataname)

def GetPoseandCostsF(cfg,dlc_cfg, sess, inputs, outputs,cap,nframes,batchsize):
    ''' Batchwise prediction of pose '''
    strwidth = int(np.ceil(np.log10(nframes))) #width for strings
    batch_ind = 0 # keeps track of which image within a batch should be written to
    batch_num = 0 # keeps track of which batch you are at
    ny,nx=int(cap.get(4)),int(cap.get(3))
    if cfg['cropping']:
        print("Cropping based on the x1 = %s x2 = %s y1 = %s y2 = %s. You can adjust the cropping coordinates in the config.yaml file." %(cfg['x1'], cfg['x2'],cfg['y1'], cfg['y2']))
        nx=cfg['x2']-cfg['x1']
        ny=cfg['y2']-cfg['y1']
        if nx>0 and ny>0:
            pass
        else:
            raise Exception('Please check the order of cropping parameter!')
        if cfg['x1']>=0 and cfg['x2']<int(cap.get(3)+1) and cfg['y1']>=0 and cfg['y2']<int(cap.get(4)+1):
            pass #good cropping box
        else:
            raise Exception('Please check the boundary of cropping!')

    frames = np.empty((batchsize, ny, nx, 3), dtype='ubyte') # this keeps all frames in a batch
    pbar=tqdm(total=nframes)
    counter=0
    step=max(10,int(nframes/100))

    PredicteData ={}
    #initializing constants
    dist_grid = predict.make_nms_grid(dlc_cfg.nmsradius)
    stride,halfstride=dlc_cfg.stride, dlc_cfg.stride*.5
    num_joints = dlc_cfg.num_joints
    det_min_score=dlc_cfg.minconfidence
    num_idchannel=dlc_cfg.get('num_idchannel', 0)
    while(cap.isOpened()):
            if counter%step==0:
                pbar.update(step)
            ret, frame = cap.read()
            if ret:
                frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if cfg['cropping']:
                    frames[batch_ind] = img_as_ubyte(frame[cfg['y1']:cfg['y2'],cfg['x1']:cfg['x2']])
                else:
                    frames[batch_ind] = img_as_ubyte(frame)

                if batch_ind==batchsize-1:
                    #PredicteData['frame'+str(counter)]=predict.get_detectionswithcosts(frame, dlc_cfg, sess, inputs, outputs, outall=False,nms_radius=dlc_cfg.nmsradius,det_min_score=dlc_cfg.minconfidence)
                    D=predict.get_batchdetectionswithcosts(frames, dlc_cfg, dist_grid, batchsize,num_joints,num_idchannel, stride, halfstride, det_min_score, sess, inputs, outputs)
                    for l in range(batchsize):
                        #pose = predict.getposeNP(frames,dlc_cfg, sess, inputs, outputs)
                        #PredicteData[batch_num*batchsize:(batch_num+1)*batchsize, :] = pose
                        PredicteData['frame'+str(batch_num*batchsize+l).zfill(strwidth)]=D[l]

                    batch_ind = 0
                    batch_num += 1
                else:
                   batch_ind+=1
            else:
                nframes = counter
                print("Detected frames: ", nframes)
                if batch_ind>0:
                    #pose = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs) #process the whole batch (some frames might be from previous batch!)
                    #PredicteData[batch_num*batchsize:batch_num*batchsize+batch_ind, :] = pose[:batch_ind,:]
                    D=predict.get_batchdetectionswithcosts(frames, dlc_cfg, dist_grid, batchsize,num_joints,num_idchannel, stride, halfstride, det_min_score, sess, inputs, outputs)
                    for l in range(batch_ind):
                        #pose = predict.getposeNP(frames,dlc_cfg, sess, inputs, outputs)
                        #PredicteData[batch_num*batchsize:(batch_num+1)*batchsize, :] = pose
                        PredicteData['frame'+str(batch_num*batchsize+l).zfill(strwidth)]=D[l]
                break
            counter+=1

    pbar.close()
    PredicteData['metadata']={
        'nms radius': dlc_cfg.nmsradius,
        'minimal confidence': dlc_cfg.minconfidence,
        'PAFgraph': dlc_cfg.partaffinityfield_graph,
        "all_joints": [[i] for i in range(len(dlc_cfg.all_joints))],
        "all_joints_names": [dlc_cfg.all_joints_names[i] for i in range(len(dlc_cfg.all_joints))],
        "nframes": nframes
        }
    return PredicteData,nframes

def GetPoseandCostsS(cfg,dlc_cfg, sess, inputs, outputs,cap,nframes):
    ''' Non batch wise pose estimation for video cap.'''
    strwidth = int(np.ceil(np.log10(nframes))) #width for strings
    if cfg['cropping']:
        print("Cropping based on the x1 = %s x2 = %s y1 = %s y2 = %s. You can adjust the cropping coordinates in the config.yaml file." %(cfg['x1'], cfg['x2'],cfg['y1'], cfg['y2']))
        nx=cfg['x2']-cfg['x1']
        ny=cfg['y2']-cfg['y1']
        if nx>0 and ny>0:
            pass
        else:
            raise Exception('Please check the order of cropping parameter!')
        if cfg['x1']>=0 and cfg['x2']<int(cap.get(3)+1) and cfg['y1']>=0 and cfg['y2']<int(cap.get(4)+1):
            pass #good cropping box
        else:
            raise Exception('Please check the boundary of cropping!')

    PredicteData ={} #np.zeros((nframes, 3 * len(dlc_cfg['all_joints_names'])))
    pbar=tqdm(total=nframes)
    counter=0
    step=max(10,int(nframes/100))
    while(cap.isOpened()):
            if counter%step==0:
                pbar.update(step)

            ret, frame = cap.read()
            if ret:
                frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if cfg['cropping']:
                    frame= img_as_ubyte(frame[cfg['y1']:cfg['y2'],cfg['x1']:cfg['x2']])
                else:
                    frame = img_as_ubyte(frame)
                PredicteData['frame'+str(counter).zfill(strwidth)]=predict.get_detectionswithcosts(frame, dlc_cfg, sess, inputs, outputs, outall=False,nms_radius=dlc_cfg.nmsradius,det_min_score=dlc_cfg.minconfidence)
            else:
                nframes=counter
                break
            counter+=1

    pbar.close()

    #print("frame 0", PredicteData[0])
    #print("frame 1", PredicteData[1])
    PredicteData['metadata']={
        'nms radius': dlc_cfg.nmsradius,
        'minimal confidence': dlc_cfg.minconfidence,
        'PAFgraph': dlc_cfg.partaffinityfield_graph,
        "all_joints": [[i] for i in range(len(dlc_cfg.all_joints))],
        "all_joints_names": [dlc_cfg.all_joints_names[i] for i in range(len(dlc_cfg.all_joints))],
        "nframes": nframes
        }

    #print(PredicteData)
    return PredicteData,nframes
