"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""


import os
import numpy as np
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.nnet import predict
import tensorflow as tf

tf_ver = [int(v) for v in tf.__version__.split('.')]
if tf_ver[1] < 14:
    from tensorflow.contrib.tensorrt import trt_convert as trt
else:
    from tensorflow.python.compiler.tensorrt import trt_convert as trt


output = 'pose/locref_pred/block4/BiasAdd' # should it be ['concat_1'] or ['pose/part_pred/block4/BiasAdd']?


def load_model(cfg, shuffle=1, trainingsetindex=0):
    '''

    Loads a tensorflow session with a DLC model from the associated configuration

    return a tensorflow session with DLC model given cfg and shuffle

    Parameters:
    -----------
    cfg : dict
        Configuration read from the project's cfg.yaml file

    shuffle : int, optional
        which shuffle to use

    trainingsetindex : int. optional
        which training fraction to use, identified by its index

    Returns:
    --------
    sess : tensorflow session
        tensorflow session with DLC model from the provided configuration, shuffle, and trainingsetindex

    checkpoint file path : string
        the path to the checkpoint file associated with the loaded model

    '''

    ########################
    ### find snapshot to use
    ########################

    train_fraction = cfg['TrainingFraction'][trainingsetindex]
    model_folder = os.path.join(cfg['project_path'], str(auxiliaryfunctions.GetModelFolder(train_fraction, shuffle, cfg)))
    path_test_config = os.path.normpath(model_folder + '/test/pose_cfg.yaml')

    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError("It seems the model for shuffle %s and trainFraction %s does not exist."%(shuffle, train_fraction))

    # Check which snapshots are available and sort them by # iterations
    try:
      Snapshots = np.array([fn.split('.')[0]for fn in os.listdir(os.path.join(model_folder , 'train'))if "index" in fn])
    except FileNotFoundError:
      raise FileNotFoundError("Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."%(shuffle,shuffle))

    if cfg['snapshotindex'] == 'all':
        print("Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!")
        snapshotindex = -1
    else:
        snapshotindex = cfg['snapshotindex']

    increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]

    ####################################
    ### Load and setup CNN part detector
    ####################################

    # Check if data already was generated:
    dlc_cfg['init_weights'] = os.path.join(model_folder, 'train', Snapshots[snapshotindex])
    trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1]
    dlc_cfg['num_outputs'] = cfg.get('num_outputs', dlc_cfg.get('num_outputs', 1))
    dlc_cfg['batch_size'] = cfg['batch_size']

    # load network
    sess, _, _ = predict.setup_pose_prediction(dlc_cfg)

    return sess, dlc_cfg['init_weights']


def tf_to_pb(sess, checkpoint, dir=None):
    '''

    Saves a frozen tensorflow graph (a protobuf file).
    See https://leimao.github.io/blog/Save-Load-Inference-From-TF-Frozen-Graph/

    Parameters
    ----------
    sess : tensorflow session
        session with graph to be saved

    checkpoint : string
        checkpoint of tensorflow model to be converted to protobuf (output will be <checkpoint>.pb)

    '''

    # export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        [output]
    )

    # serialize and dump the output graph
    if dir:
        output_graph = os.path.normpath(dir + '/' + os.path.basename(checkpoint) + '.pb')
    else:
        output_graph = checkpoint + '.pb'

    tf.gfile.GFile(output_graph, "wb").write(output_graph_def.SerializeToString())


def pb_to_lite(checkpoint, dir=None):
    '''

    converts a protobuf file to tensorflow lite format

    Parameters:
    -----------
    checkpoint : string
        the path to the checkpoint to be converted

    '''

    # model_dir = os.path.dirname(checkpoint)
    # cfg = auxiliaryfunctions.read_config(os.path.normpath(model_dir+'/pose_cfg.yaml'))

    pb_file = checkpoint+'.pb'
    if not os.path.isfile(pb_file):
        raise FileNotFoundError("The checkpoint has not been converted to a protobuf. You must first convert to protobuf (using tf_to_pb) before converting to tflite format.")

    # if 'resnet' in cfg['net_type']:
    #     input = 'Placeholder'
    # elif 'mobilenet' in cfg['net_type']:
    #     input = 'MobilenetV2/input'
    # else:
    #     raise Exception("Net type not found!")

    input = 'Placeholder'
    converter = tf.lite.TFLiteConverter.from_frozen_graph(pb_file, [input], [output], input_shapes={input : [None, 512, 512, 3]})
    tflite_model = converter.convert()

    if dir:
        lite_file = os.path.normpath(dir + '/' + os.path.basename(checkpoint) + '.tflite')
    else:
        lite_file = checkpoint + '.tflite'
    open(lite_file, 'wb').write(tflite_model)


def pb_to_trt(checkpoint, precision_mode='FP16', dir=None):
    '''

    converts a protobuf file to tensorrt optimized graph

    Parameters:
    -----------
    checkpoint : string
        the path to the checkpoint to be converted

    precision_mode : string, optional
        precision mode for tensorrt optimized graph (FP16 by default, accepts FP32, INT8)

    '''

    pb_file = checkpoint+'.pb'
    if not os.path.isfile(pb_file):
        raise FileNotFoundError("The checkpoint has not been converted to a protobuf. You must first convert to protobuf (using tf_to_pb) before converting to tflite format.")

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(tf.gfile.GFile(pb_file, 'rb').read())

    if tf_ver[1] >= 14:
        converter = trt.TrtGraphConverter(input_graph_def=graph_def, nodes_blacklist=[output])
        trt_graph = converter.convert()
    else:
        trt_graph = trt.create_inference_graph(input_graph_def=graph_def, outputs=[output], max_batch_size=64, precision_mode='FP16')

    if dir:
        trt_file = os.path.normpath(dir + '/' + os.path.basename(checkpoint) + '_trt.pb')
    else:
        trt_file = checkpoint + '_trt.pb'
    tf.gfile.GFile(trt_file, "wb").write(trt_graph.SerializeToString())


def export_models(cfg_path, iteration=None, shuffle=1, trainingsetindex=0, overwrite=False):

    try:
        cfg = auxiliaryfunctions.read_config(cfg_path)
    except FileNotFoundError:
        FileNotFoundError("The config.yaml file at %s does not exist." % cfg_path)

    cfg['project_path'] = os.path.dirname(cfg_path)
    cfg['iteration'] = iteration if iteration is not None else cfg['iteration']
    cfg['batch_size'] = cfg['batch_size'] if cfg['batch_size'] > 1 else 2

    sess, ckpt = load_model(cfg, shuffle, trainingsetindex)

    # set up export directory
    export_dir = os.path.normpath(cfg['project_path'] + '/exported_models')
    if not os.path.isdir(export_dir):
        os.mkdir(export_dir)

    sub_dir_name = 'iteration-%d_shuffle-%d_trainfraction-%s' % (cfg['iteration'], shuffle, cfg['TrainingFraction'][trainingsetindex])
    export_dir += '/' + sub_dir_name

    if os.path.isdir(export_dir):
        if not overwrite:
            raise FileExistsError("Export directory %s already exists. Terminating export..." % export_dir)
    else:
        os.mkdir(export_dir)

    tf_to_pb(sess, ckpt, dir=export_dir)
    pb_to_lite(ckpt, dir=export_dir)
    pb_to_trt(ckpt, dir=export_dir)
