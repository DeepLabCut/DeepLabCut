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

    Parameters
    ----------
    config : str
        Full path of the config.yaml file.
    shuffle : int, optional
        The shuffle index to select for training (default 1).
    trainingsetindex : int, optional
        Which TrainingsetFraction to use (default 1).
        Note that TrainingFraction is a list in config.yaml.
    gputouse : int or None, optional
        Number of your GPU (default None).
        nvidia-smi can be used to view the GPU number.
        See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    Other Parameters
    ________________
    max_snapshots_to_keep : int or None, optional
        How many snapshots are kept (default 5).
        A snapshot is stored every `save_iters` many times, however only the last `max_snapshots_to_keep` many are kept!
        If None, then all snapshots are kept. See: https://github.com/AlexEMG/DeepLabCut/issues/8#issuecomment-387404835
    autotune : bool, optional
        property of TensorFlow, somehow faster if 'false' (as Eldar found out, see https://github.com/tensorflow/tensorflow/issues/13317) (default False).
    displayiters : int or None, optional
        variable from pose_config.yaml (default None).
        This is a hack that should not be used regularly.
        If None, the value from pose_config.yaml is used.
    saveiters : int or None, optional
        variable from pose_config.yaml (default None).
        This is a hack that should not be used regularly.
        If None, the value from pose_config.yaml is used.
    maxiters : int or None, optional
        variable from pose_config.yaml (default None).
        This is a hack that should not be used regularly.
        If None, the value from pose_config.yaml is used.

    Examples
    --------
    Training the network for first shuffle of the training dataset:

    >>> deeplabcut.train_network('/analysis/project/reaching-task/config.yaml')

    Training the network for second shuffle of the training dataset:

    >>> deeplabcut.train_network('/analysis/project/reaching-task/config.yaml',shuffle=2)

    """
    import tensorflow as tf
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
