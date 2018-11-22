"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu

"""

####################################################
# Dependencies
####################################################

import os.path
from deeplabcut.pose_estimation_tensorflow.nnet import predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import data_to_input

from random import sample
import time
import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from deeplabcut.utils import auxiliaryfunctions
import cv2
from skimage.util import img_as_ubyte

####################################################
# Loading data, and defining model folder
####################################################

def analyze_videos(config,videos,shuffle=1,trainingsetindex=0,videotype='avi',gputouse=None,save_as_csv=False):
    """
    Makes prediction based on a trained network. The index of the trained network is specified by parameters in the config file (in particular the variable 'snapshotindex')

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    videos : list
        A list of strings containing the full paths to videos for analysis or a path to the directory where all the videos with same extension are stored.

    shuffle: int, optional
        An integer specifying the shuffle index of the training dataset used for training the network. The default is 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).
        
    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\nOnly videos with this extension are analyzed. The default is ``.avi``

    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU put None.
    See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    save_as_csv: bool, optional
        Saves the predictions in a .csv file. The default is ``False``; if provided it must be either ``True`` or ``False``

    Examples
    --------
    If you want to analyze only 1 video
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/reachingvideo1.avi'])
    --------
    
    If you want to analyze all videos of type avi in a folder:
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos'],videotype='.avi')
    --------

    If you want to analyze multiple videos
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/reachingvideo1.avi','/analysis/project/videos/reachingvideo2.avi'])
    --------

    If you want to analyze multiple videos with shuffle = 2
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/reachingvideo1.avi','/analysis/project/videos/reachingvideo2.avi'], shuffle=2)

    --------
    If you want to analyze multiple videos with shuffle = 2 and save results as an additional csv file too
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/reachingvideo1.avi','/analysis/project/videos/reachingvideo2.avi'], shuffle=2,save_as_csv=True)
    --------

    """
    if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
        del os.environ['TF_CUDNN_USE_AUTOTUNE'] #was potentially set during training
    
    tf.reset_default_graph()
    
    cfg = auxiliaryfunctions.read_config(config)
    trainFraction = cfg['TrainingFraction'][trainingsetindex]
    modelfolder=os.path.join(cfg["project_path"],str(auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,cfg)))
    path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'
    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError("It seems the model for shuffle %s and trainFraction %s does not exist."%(shuffle,trainFraction))

    # Check which snapshots are available and sort them by # iterations
    try:
      Snapshots = np.array([fn.split('.')[0]for fn in os.listdir(os.path.join(modelfolder , 'train'))if "index" in fn])
    except FileNotFoundError:
      raise FileNotFoundError("Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."%(shuffle,shuffle))

    if cfg['snapshotindex'] == 'all':
        print("Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!")
        snapshotindex = -1
    else:
        snapshotindex=cfg['snapshotindex']
        
    increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]
    
    print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)

    ##################################################
    # Load and setup CNN part detector
    ##################################################

    # Check if data already was generated:
    dlc_cfg['init_weights'] = os.path.join(modelfolder , 'train', Snapshots[snapshotindex])
    trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1]
    
    #update batchsize (based on parameters in config.yaml)
    dlc_cfg['batch_size']=cfg['batch_size']
    # Name for scorer:
    DLCscorer = auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction,trainingsiterations=trainingsiterations)
    sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)
    pdindex = pd.MultiIndex.from_product([[DLCscorer], dlc_cfg['all_joints_names'], ['x', 'y', 'likelihood']],names=['scorer', 'bodyparts', 'coords'])

    if gputouse is not None: #gpu selectinon
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)
    
    ##################################################
    # Datafolder
    ##################################################
    #checks if input is a directory
    if [os.path.isdir(i) for i in videos] == [True]:#os.path.isdir(video)==True:
        """
        Analyzes all the videos in the directory.
        """
        print("Analyzing all the videos in the directory")
        videofolder= videos[0]
        os.chdir(videofolder)
        videolist=[fn for fn in os.listdir(os.curdir) if (videotype in fn) and ('_labeled.mp4' not in fn)] #exclude labeled-videos!
        Videos = sample(videolist,len(videolist)) # this is useful so multiple nets can be used to analzye simultanously
    else:
        if isinstance(videos,str):
            if os.path.isfile(videos): # #or just one direct path!
                Videos=[videos]
            else:
                Videos=[]
        else:
            Videos=[v for v in videos if os.path.isfile(v)]
    
    if len(Videos)>0:
        #looping over videos
        for video in Videos:
            AnalzyeVideo(video,DLCscorer,cfg,dlc_cfg,sess,inputs, outputs,pdindex,save_as_csv)

    print("The videos are analyzed. Now your research can truly start! \n You can create labeled videos with 'create_labeled_video'.")
    print("If the tracking is not satisfactory for some videos, consider expanding the training set. You can use the function 'extract_outlier_frames' to extract any outlier frames!")


def GetPoseF(cfg,dlc_cfg, sess, inputs, outputs,cap,nframes,batchsize):
    ''' Batchwise prediction of pose '''
    
    PredicteData = np.zeros((nframes, 3 * len(dlc_cfg['all_joints_names'])))
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
                    pose = predict.getposeNP(frames,dlc_cfg, sess, inputs, outputs)
                    PredicteData[batch_num*batchsize:(batch_num+1)*batchsize, :] = pose
                    batch_ind = 0
                    batch_num += 1
                else:
                   batch_ind+=1
            else:
                nframes = counter
                print("Detected frames: ", nframes)
                if batch_ind>0:
                    pose = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs) #process the whole batch (some frames might be from previous batch!)
                    PredicteData[batch_num*batchsize:batch_num*batchsize+batch_ind, :] = pose[:batch_ind,:]
                break
            counter+=1

    return PredicteData,nframes

def GetPoseS(cfg,dlc_cfg, sess, inputs, outputs,cap,nframes):
    ''' Non batch wise pose estimation for video cap.'''
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
    
    PredicteData = np.zeros((nframes, 3 * len(dlc_cfg['all_joints_names'])))
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
                pose = predict.getpose(frame, dlc_cfg, sess, inputs, outputs)
                PredicteData[counter, :] = pose.flatten()  # NOTE: thereby cfg['all_joints_names'] should be same order as bodyparts!
            else:
                nframes=counter
                break
            counter+=1
            
    pbar.close()
    return PredicteData,nframes


def AnalzyeVideo(video,DLCscorer,cfg,dlc_cfg,sess,inputs, outputs,pdindex,save_as_csv):
    #from moviepy.editor import VideoFileClip
    
    print(video)
    #videotype = Path(video).suffix
    print("Starting % ", video)
    vname = Path(video).stem
    dataname = os.path.join(str(Path(video).parents[0]),vname + DLCscorer + '.h5')
    try:
        # Attempt to load data...
        pd.read_hdf(dataname)
        print("Video already analyzed!", dataname)
    except FileNotFoundError:
        print("Loading ", video)
        cap=cv2.VideoCapture(video)
        
        fps = cap.get(5) #https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
        nframes = int(cap.get(7))
        duration=nframes*1./fps
        size=(int(cap.get(4)),int(cap.get(3)))
        
        ny,nx=size
        print("Duration of video [s]: ", round(duration,2), ", recorded with ", round(fps,2),"fps!")
        print("Overall # of frames: ", nframes,"without cropped frame dimensions: ", nx,ny)

        start = time.time()

        print("Starting to extract posture")
        if int(dlc_cfg["batch_size"])>1:
            PredicteData,nframes=GetPoseF(cfg,dlc_cfg, sess, inputs, outputs,cap,nframes,int(dlc_cfg["batch_size"]))
        else:
            PredicteData,nframes=GetPoseS(cfg,dlc_cfg, sess, inputs, outputs,cap,nframes)

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
            "config file": dlc_cfg,
            "fps": fps,
            "batch_size": dlc_cfg["batch_size"],
            "frame_dimensions": (ny, nx),
            "nframes": nframes,
            "cropping": cfg['cropping'],
            "cropping_parameters": coords
        }
        metadata = {'data': dictionary}

        print("Saving results in %s..." %(Path(video).parents[0]))
        auxiliaryfunctions.SaveData(PredicteData[:nframes,:], metadata, dataname, pdindex, range(nframes),save_as_csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('config')
    cli_args = parser.parse_args()
