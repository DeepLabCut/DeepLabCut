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
from deeplabcut.pose_estimation_tensorflow.nnet import processing
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

def analyze_videos(config,videos,videotype='avi',shuffle=1,trainingsetindex=0,gputouse=None,save_as_csv=False, destfolder=None, predictor = None):
    """
    Makes prediction based on a trained network. The index of the trained network is specified by parameters in the config file (in particular the variable 'snapshotindex')
    
    You can crop the video (before analysis), by changing 'cropping'=True and setting 'x1','x2','y1','y2' in the config file. The same cropping parameters will then be used for creating the video.
    
    Output: The labels are stored as MultiIndex Pandas Array, which contains the name of the network, body part name, (x, y) label position \n
            in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient Hierarchical Data Format (HDF) \n
            in the same directory, where the video is stored. However, if the flag save_as_csv is set to True, the data can also be exported in \n
            comma-separated values format (.csv), which in turn can be imported in many programs, such as MATLAB, R, Prism, etc.
    
    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    videos : list
        A list of strings containing the full paths to videos for analysis or a path to the directory, where all the videos with same extension are stored.
    
    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed. The default is ``.avi``

    shuffle: int, optional
        An integer specifying the shuffle index of the training dataset used for training the network. The default is 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).
    
    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU put None.
    See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    save_as_csv: bool, optional
        Saves the predictions in a .csv file. The default is ``False``; if provided it must be either ``True`` or ``False``

    destfolder: string, optional
        Specifies the destination folder for analysis data (default is the path of the video). Note that for subsequent analysis this 
        folder also needs to be passed.

    predictor: The prediction algorithm to use on the probability outputs of the deeplabcut neural net. Defaults to
               "singleargmax". The options available depends on the currently available Predictor plugins in the
               predictors folder.

    Examples
    --------
    
    Windows example for analyzing 1 video 
    >>> deeplabcut.analyze_videos('C:\\myproject\\reaching-task\\config.yaml',['C:\\yourusername\\rig-95\\Videos\\reachingvideo1.avi'])
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
    start_path=os.getcwd() #record cwd to return to this directory in the end
    
    cfg = auxiliaryfunctions.read_config(config)
    trainFraction =  cfg['TrainingFraction'][trainingsetindex]
    
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
    # Load selected or default predictor plugin
    ##################################################
    # If predictor is None, change to default
    if(predictor is None):
        predictor = "singleargmax"

    # Load plugin predictor class
    predictor_cls = processing.get_predictor(predictor)

    
    ##################################################
    # Datafolder
    ##################################################
    Videos=auxiliaryfunctions.Getlistofvideos(videos,videotype)
    if len(Videos)>0:
        #looping over videos
        for video in Videos:
            AnalyzeVideo(video,DLCscorer,trainFraction,cfg,dlc_cfg,sess,inputs, outputs,pdindex,save_as_csv, predictor_cls, destfolder)
    
    os.chdir(str(start_path))
    print("The videos are analyzed. Now your research can truly start! \n You can create labeled videos with 'create_labeled_video'.")
    print("If the tracking is not satisfactory for some videos, consider expanding the training set. You can use the function 'extract_outlier_frames' to extract any outlier frames!")


# Utility method used by GetPoseALL, gets a batch of frames, stores them in frame_store and returns the size of the batch
def GetVideoBatch(cap, batch_size, cfg, frame_store) -> int:
    """ Gets a batch size of frames, and returns them """
    current_frame = 0

    # While the cap is still going and the current frame is less then the batch size...
    while(cap.isOpened() and current_frame < batch_size):
        # Read a frame
        ret_val, frame = cap.read()

        # If we got an actual frame, store it in the frame store.
        if(ret_val):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if cfg['cropping']:
                frame_store[current_frame] = img_as_ubyte(frame[cfg['y1']:cfg['y2'], cfg['x1']:cfg['x2']])
            else:
                frame_store[current_frame] = img_as_ubyte(frame)
        else:
            # If we don't we have reached the end most likely return the amount of frames we managed to get
            return current_frame
        # Increment frame counter
        current_frame += 1

    # Return the frame counter, most likely the same as batch_size at this point....
    return current_frame


# Method Added by Isaac Robinson, replaces old system of getting poses and uses a new plugin system for predicting
# poses...
def GetPoseALL(cfg, dlc_cfg, sess, inputs, outputs, cap, nframes, batchsize, predictor):
    """ Gets the poses for any batch size, including batch size of only 1 """
    # Create a numpy array to hold all pose prediction data...
    pose_prediction_data = np.zeros((nframes, 3 * len(dlc_cfg["all_joints_names"])))

    # Create the progress bar
    pbar = tqdm(total=nframes)

    ny, nx = int(cap.get(4)), int(cap.get(3))
    # Same cropping code as other methods, no need to change
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

    # Create the temporary batch frame store...
    frame_store = np.empty((batchsize, ny, nx, 3), dtype='ubyte')

    # Create a counter to keep track of the progress bar
    counter = 0
    prog_step = max(10, int(nframes / 100))
    current_step = 0

    # Keeps track of data that has been processed
    frames_done = 0

    # Iterating until we get all video frames... (Check is done at the bottom...)
    while(True):
        size = GetVideoBatch(cap, batchsize, cfg, frame_store)
        counter += size

        # If we pass the current step or phase, update the progress bar
        if(counter // prog_step > current_step):
            pbar.update(prog_step)
            current_step += 1

        if(size > 0):
            # If we received any frames, process them...
            scmap, locref = predict.extract_cnn_outputmulti(sess.run(outputs, feed_dict={inputs: frame_store}), dlc_cfg)
            down_scale = dlc_cfg.stride

            if len(scmap.shape) == 2:  # If there is a single body part, add a dimension at the end
                scmap = np.expand_dims(scmap, axis=2)

            # Run the predictor providing it the scmap, locref, and scaling using a TrackingData object
            pose = predictor.on_frames(processing.TrackingData(scmap[:size], locref, down_scale))

            if(pose is not None):
                # If the predictor returned a pose, add it to the pose_prediction_data and increment the counter
                # That keeps track of fully finished frames.
                pose_prediction_data[frames_done:frames_done + pose.get_frame_count()] = pose.get_all()
                frames_done += pose.get_frame_count()

        if(size < batchsize):
            # If the output frames by the video capture were less then a full batch, we have reached the end of the
            # video, break the loop...
            break

    pbar.update(counter - (prog_step * current_step)) # Finish the progress bar
    pbar.close() # Close the progress bar

    # Phase 2: Post processing...

    # Get all of the final poses that are still held by the predictor
    post_pbar = tqdm(total = nframes - frames_done)
    final_poses = predictor.on_end(post_pbar)
    post_pbar.close()

    # Add any post-processed frames
    if(final_poses is not None):
        pose_prediction_data[frames_done:frames_done + final_poses.get_frame_count()] = final_poses.get_all()
        frames_done += final_poses.get_frame_count()

    # Check and make sure the predictor returned all frames, otherwise throw an error
    if(frames_done != nframes):
        raise ValueError(f"The predictor algorithm did not return the same amount of frames as are in the video. "
                         f"Expected Amount: {nframes}, Actual Amount Returned: {frames_done}")

    # We are good, return data
    return pose_prediction_data, nframes


# Arg h5_path added
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


# Utility method used by , gets the settings for the given predictor plugin
def GetPredictorSettings(cfg, predictor_cls):
    """ Get the predictor settings from deeplabcut config and return a dictionary for plugin to use... """
    # Grab setting blueprints for predictor plugins(list of tuples of name, desc, default val)....
    setting_info = predictor_cls.get_settings()
    name = predictor_cls.get_name()

    # If it is None, return none
    if(setting_info is None):
        return None

    # Pull out the name and default values into a dictionary...
    setting_info = {name: def_val for (name, desc, def_val) in predictor_cls.get_settings()}

    # If the dlc config contains a category predictors, and predictors contains a category named after the plugin, load
    # the user cfg for this plugin and merge it with default values
    if(("predictors" in cfg) and (cfg["predictors"]) and (name in cfg["predictors"])):
        setting_info.update(
            {key: cfg["predictors"][name][key] for key in (setting_info.keys() & cfg["predictors"][name].keys())}
        )

    return setting_info



def AnalyzeVideo(video,DLCscorer,trainFraction,cfg,dlc_cfg,sess,inputs, outputs,pdindex,save_as_csv, predictor, destfolder=None):
    ''' Helper function for analyzing a video '''
    # Note: predictor is not a string but rather the selected plugin's class
    print("Starting to analyze % ", video)
    vname = Path(video).stem
    if destfolder is None:
        destfolder = str(Path(video).parents[0])
    dataname = os.path.join(destfolder,vname + DLCscorer + '.h5')
    
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

        # Create a predictor plugin instance...
        predictor_settings = GetPredictorSettings(cfg, predictor) # Grab the plugin settings for this plugin...
        print(f"Plugin {predictor.get_name()} Settings: {predictor_settings}")
        predictor_inst = predictor(dlc_cfg['all_joints_names'], nframes, predictor_settings)
        
        ny,nx=size
        print("Duration of video [s]: ", round(duration,2), ", recorded with ", round(fps,2),"fps!")
        print("Overall # of frames: ", nframes," found with (before cropping) frame dimensions: ", nx,ny)
        start = time.time()

        print("Starting to extract posture")

        # cfg, dlc_cfg, sess, inputs, outputs, cap, nframes, batchsize, predictor
        PredicteData, nframes = GetPoseALL(cfg, dlc_cfg, sess, inputs, outputs, cap, nframes,
                                           int(dlc_cfg["batch_size"]), predictor_inst)

        # Old DLC Code
        """
        if int(dlc_cfg["batch_size"])>1:
            PredicteData,nframes=GetPoseF(cfg,dlc_cfg, sess, inputs, outputs,cap,nframes,int(dlc_cfg["batch_size"]), dataname)
        else:
            PredicteData,nframes=GetPoseS(cfg,dlc_cfg, sess, inputs, outputs,cap,nframes, dataname)
        """

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

        print("Saving results in %s..." %(Path(video).parents[0]))
        auxiliaryfunctions.SaveData(PredicteData[:nframes,:], metadata, dataname, pdindex, range(nframes),save_as_csv)


def GetPosesofFrames(cfg,dlc_cfg, sess, inputs, outputs,directory,framelist,nframes,batchsize,rgb):
    ''' Batchwise prediction of pose  for framelist in directory'''
    from skimage import io
    print("Starting to extract posture")
    if rgb:
        im=io.imread(os.path.join(directory,framelist[0]),mode='RGB')
    else:
        im=io.imread(os.path.join(directory,framelist[0]))
    
    ny,nx,nc=np.shape(im)
    print("Overall # of frames: ", nframes," found with (before cropping) frame dimensions: ", nx,ny)

    PredicteData = np.zeros((nframes, 3 * len(dlc_cfg['all_joints_names'])))
    batch_ind = 0 # keeps track of which image within a batch should be written to
    batch_num = 0 # keeps track of which batch you are at

    if cfg['cropping']:
        print("Cropping based on the x1 = %s x2 = %s y1 = %s y2 = %s. You can adjust the cropping coordinates in the config.yaml file." %(cfg['x1'], cfg['x2'],cfg['y1'], cfg['y2']))
        nx,ny=cfg['x2']-cfg['x1'],cfg['y2']-cfg['y1']
        if nx>0 and ny>0:
            pass
        else:
            raise Exception('Please check the order of cropping parameter!')
        if cfg['x1']>=0 and cfg['x2']<int(np.shape(im)[1]) and cfg['y1']>=0 and cfg['y2']<int(np.shape(im)[0]):
            pass #good cropping box
        else:
            raise Exception('Please check the boundary of cropping!')
    
    pbar=tqdm(total=nframes)
    counter=0
    step=max(10,int(nframes/100))
    
    if batchsize==1:
        for counter,framename in enumerate(framelist):
                #frame=io.imread(os.path.join(directory,framename),mode='RGB')
                if rgb:
                    im=io.imread(os.path.join(directory,framename),mode='RGB')
                else:
                    im=io.imread(os.path.join(directory,framename))
                    
                if counter%step==0:
                    pbar.update(step)

                if cfg['cropping']:
                    frame= img_as_ubyte(im[cfg['y1']:cfg['y2'],cfg['x1']:cfg['x2'],:])
                else:
                    frame = img_as_ubyte(im)
                    
                pose = predict.getpose(frame, dlc_cfg, sess, inputs, outputs)
                PredicteData[counter, :] = pose.flatten()
    else:
        frames = np.empty((batchsize, ny, nx, 3), dtype='ubyte') # this keeps all the frames of a batch
        for counter,framename in enumerate(framelist):
                if rgb:
                    im=io.imread(os.path.join(directory,framename),mode='RGB')
                else:
                    im=io.imread(os.path.join(directory,framename))
                
                if counter%step==0:
                    pbar.update(step)

                if cfg['cropping']:
                    frames[batch_ind] = img_as_ubyte(im[cfg['y1']:cfg['y2'],cfg['x1']:cfg['x2'],:])
                else:
                    frames[batch_ind] = img_as_ubyte(im)
                    
                if batch_ind==batchsize-1:
                    pose = predict.getposeNP(frames,dlc_cfg, sess, inputs, outputs)
                    PredicteData[batch_num*batchsize:(batch_num+1)*batchsize, :] = pose
                    batch_ind = 0
                    batch_num += 1
                else:
                   batch_ind+=1
            
        if batch_ind>0: #take care of the last frames (the batch that might have been processed)
            pose = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs) #process the whole batch (some frames might be from previous batch!)
            PredicteData[batch_num*batchsize:batch_num*batchsize+batch_ind, :] = pose[:batch_ind,:]

    pbar.close()
    return PredicteData,nframes,nx,ny


def analyze_time_lapse_frames(config,directory,frametype='.png',shuffle=1,trainingsetindex=0,gputouse=None,save_as_csv=False,rgb=True):
    """
    Analyzed all images (of type = frametype) in a folder and stores the output in one file. 
    
    You can crop the frames (before analysis), by changing 'cropping'=True and setting 'x1','x2','y1','y2' in the config file. 
    
    Output: The labels are stored as MultiIndex Pandas Array, which contains the name of the network, body part name, (x, y) label position \n
            in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient Hierarchical Data Format (HDF) \n
            in the same directory, where the video is stored. However, if the flag save_as_csv is set to True, the data can also be exported in \n
            comma-separated values format (.csv), which in turn can be imported in many programs, such as MATLAB, R, Prism, etc.
    
    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    directory: string
        Full path to directory containing the frames that shall be analyzed

    frametype: string, optional
        Checks for the file extension of the frames. Only images with this extension are analyzed. The default is ``.png``

    shuffle: int, optional
        An integer specifying the shuffle index of the training dataset used for training the network. The default is 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).
    
    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU put None.
    See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    save_as_csv: bool, optional
        Saves the predictions in a .csv file. The default is ``False``; if provided it must be either ``True`` or ``False``

    rbg: bool, optional.
        Whether to load image as rgb; Note e.g. some tiffs do not alow that option in io.imread, then just set this to false.

    Examples
    --------
    If you want to analyze all frames in /analysis/project/timelapseexperiment1
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml','/analysis/project/timelapseexperiment1')
    --------
    
    If you want to analyze all frames in /analysis/project/timelapseexperiment1
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml','/analysis/project/timelapseexperiment1', frametype='.bmp')
    --------
    
    Note: for test purposes one can extract all frames from a video with ffmeg, e.g. ffmpeg -i testvideo.avi thumb%04d.png 
    """
    if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
        del os.environ['TF_CUDNN_USE_AUTOTUNE'] #was potentially set during training
    
    tf.reset_default_graph()
    start_path=os.getcwd() #record cwd to return to this directory in the end
    
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
    # Loading the images
    ##################################################
    #checks if input is a directory
    if os.path.isdir(directory)==True:
        """
        Analyzes all the frames in the directory.
        """
        print("Analyzing all frames in the directory: ", directory)
        os.chdir(directory)
        framelist=np.sort([fn for fn in os.listdir(os.curdir) if (frametype in fn)])

        vname = Path(directory).stem
        dataname = os.path.join(directory,vname + DLCscorer + '.h5')
        try:
            # Attempt to load data...
            pd.read_hdf(dataname)
            print("Frames already analyzed!", dataname)
        except FileNotFoundError:
            nframes = len(framelist)
            if nframes>1:
                start = time.time()
                
                PredicteData,nframes,nx,ny=GetPosesofFrames(cfg,dlc_cfg, sess, inputs, outputs,directory,framelist,nframes,dlc_cfg['batch_size'],rgb)
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
                    "batch_size": dlc_cfg["batch_size"],
                    "frame_dimensions": (ny, nx),
                    "nframes": nframes,
                    "cropping": cfg['cropping'],
                    "cropping_parameters": coords
                }
                metadata = {'data': dictionary}
        
                print("Saving results in %s..." %(directory))
                
                auxiliaryfunctions.SaveData(PredicteData[:nframes,:], metadata, dataname, pdindex, framelist,save_as_csv)
                print("The folder was analyzed. Now your research can truly start!")
                print("If the tracking is not satisfactory for some frome, consider expanding the training set.")
            else:
                print("No frames were found. Consider changing the path or the frametype.")
    
    os.chdir(str(start_path))


def list_predictor_plugins():
    """
    Retrieve and print all currently installed and available predictor plugins that can be used with deeplabcut to
    the console...

    :return: Nothing, if one wants to get the plugins for other code look at processing.get_predictor_plugins().
    """
    # Load the plugins...
    predictors = processing.get_predictor_plugins()

    for predictor in predictors:
        print(f"Plugin Name: '{predictor.get_name()}'")
        print("Description: ")
        print(predictor.get_description())
        print()


def get_predictor_settings(predictor_name = None):
    """
    Gets the available/modifiable settings for a specified predictor plugin...

    :param predictor_name: The string or list of strings being the names of the predictor plugins to view customizable
                           settings for. If None, will print settings for all currently available predictors.
                           Defaults to None.

    :return: Nothing, prints to console....
    """
    from typing import Iterable

    # Convert whatever the predictor_name argument is to a list of predictor plugins
    if(predictor_name is None):
        predictors = processing.get_predictor_plugins()
    elif(isinstance(predictor_name, str)):
        predictors = [processing.get_predictor(predictor_name)]
    elif(isinstance(predictor_name, Iterable)):
        predictors = [processing.get_predictor(name) for name in predictor_name]
    else:
        raise ValueError("Argument 'predictor_name' not of type Iterable[str], string, or None!!!")

    # Print name, and settings for each plugin.
    for predictor in predictors:
        print(f"Plugin Name: {predictor.get_name()}")
        print("Arguments: ")
        if(predictor.get_settings() is None):
            print("None")
        else:
            for name, desc, def_val in predictor.get_settings():
                print(f"Name: '{name}'")
                print(f"Description: \n{desc}")
                print(f"Default Value: {def_val} \n")
        print()


def test_predictor_plugin(predictor_name = None, interactive = False):
    """
    Run the tests for a predictor plugin.

    :param predictor_name: The name of the predictor or to run tests for, or a list of names of the predictors to run.
                           If the predictor_name is not specified or set to None, then run tests for all of the
                           predictor plugins...
                           Note: names are strings...
    :param interactive: A boolean. If True, the program will wait for user input after every test, to allow the user
                        to easily read test one by one... If false, all tests will be run at once with no user
                        interaction. Defaults to false.


    :return: Nothing, prints test info to console...
    """
    from typing import Iterable
    import traceback

    # Convert whatever the predictor_name argument is to a list of predictor plugins
    if(predictor_name is None):
        predictors = processing.get_predictor_plugins()
    elif(isinstance(predictor_name, str)):
        predictors = [processing.get_predictor(predictor_name)]
    elif(isinstance(predictor_name, Iterable)):
        predictors = [processing.get_predictor(name) for name in predictor_name]
    else:
        raise ValueError("Argument 'predictor_name' not of type Iterable[str], string, or None!!!")

    # Test plugins by calling there tests...
    for predictor in predictors:
        print(f"Testing Plugin: {predictor.get_name()}")\
        # Get the tests...
        tests = predictor.get_tests()

        # If this test contains no test, let the user know and move to the next plugin.
        if(tests is None):
            print(f"Plugin {predictor.get_name()} has no tests...")
            print()
            continue

        # Iterate tests printing there results...
        for test_meth in tests:
            print(f"Running Test {test_meth.__name__}:")
            try:
                passed, expected, actual = test_meth()

                print(f"Results: {'Passed' if passed else 'Failed'}")
                if(not passed):
                    print(f"Expected Results: {expected}")
                    print(f"Actual Results: {actual}")

            except Exception as excep:
                print("Results: Failed With Exception")
                traceback.print_exception(excep, excep, excep.__traceback__)
            finally:
                print()
                if(interactive):
                    input("Press Enter To Continue: ")
                    print()
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('config')
    cli_args = parser.parse_args()
