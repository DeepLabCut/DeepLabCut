"""
DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu

"""
import os
from pathlib import Path

def train_network(config,shuffle=1,trainingsetindex=0,gputouse=None,max_snapshots_to_keep=5,autotune=False,displayiters=None,saveiters=None,maxiters=None):
    """Trains the network with the labels in the training dataset.

    Parameter
    ----------
    config : string
        Full path of the config.yaml file as a string.

    shuffle: int, optional
        Integer value specifying the shuffle index to select for training. Default is set to 1

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).
    
    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU put None.
    See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries
    
    Additional parameters:
    
    max_snapshots_to_keep: int, or None. Sets how many snapshots are kept, i.e. states of the trained network. Every savinginteration many times 
    a snapshot is stored, however only the last max_snapshots_to_keep many are kept! If you change this to None, then all are kept. 
    See: https://github.com/AlexEMG/DeepLabCut/issues/8#issuecomment-387404835
    
    autotune: property of TensorFlow, somehow faster if 'false' (as Eldar found out, see https://github.com/tensorflow/tensorflow/issues/13317). Default: False
    
    displayiters: this variable is actually set in pose_config.yaml. However, you can overwrite it with this hack. Don't use this regularly, just if you are too lazy to dig out 
    the pose_config.yaml file for the corresponding project. If None, the value from there is used, otherwise it is overwritten! Default: None
    
    saveiters: this variable is actually set in pose_config.yaml. However, you can overwrite it with this hack. Don't use this regularly, just if you are too lazy to dig out 
    the pose_config.yaml file for the corresponding project. If None, the value from there is used, otherwise it is overwritten! Default: None
    
    maxiters: this variable is actually set in pose_config.yaml. However, you can overwrite it with this hack. Don't use this regularly, just if you are too lazy to dig out 
    the pose_config.yaml file for the corresponding project. If None, the value from there is used, otherwise it is overwritten! Default: None
    
    Example
    --------
    for training the network for first shuffle of the training dataset.
    >>> deeplabcut.train_network('/analysis/project/reaching-task/config.yaml')
    --------

    for training the network for second shuffle of the training dataset.
    >>> deeplabcut.train_network('/analysis/project/reaching-task/config.yaml',shuffle=2)
    --------

    """
    import tensorflow as tf
    #reload logger.
    import importlib
    import logging
    importlib.reload(logging)
    logging.shutdown()
    
    from deeplabcut.pose_estimation_tensorflow.train import train
    from deeplabcut.utils import auxiliaryfunctions

    tf.reset_default_graph()
    start_path=os.getcwd()
    
    # Read file path for pose_config file. >> pass it on
    cfg = auxiliaryfunctions.read_config(config)
    modelfoldername=auxiliaryfunctions.GetModelFolder(cfg["TrainingFraction"][trainingsetindex],shuffle,cfg)
    poseconfigfile=Path(os.path.join(cfg['project_path'],str(modelfoldername),"train","pose_cfg.yaml"))
    if not poseconfigfile.is_file():
      print("The training datafile ", poseconfigfile, " is not present.")
      print("Probably, the training dataset for this secific shuffle index was not created.")
      print("Try with a different shuffle/trainingsetfraction or use function 'create_training_dataset' to create a new trainingdataset with this shuffle index." )
    else:
      # Set environment variables
      if autotune is not False: #see: https://github.com/tensorflow/tensorflow/issues/13317
          os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
      if gputouse is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)


      try:
          train(str(poseconfigfile),displayiters,saveiters,maxiters,max_to_keep=max_snapshots_to_keep) #pass on path and file name for pose_cfg.yaml!
      except BaseException as e:
          raise e
      finally:
          os.chdir(str(start_path))
      print("The network is now trained and ready to evaluate. Use the function 'evaluate_network' to evaluate the network.")
