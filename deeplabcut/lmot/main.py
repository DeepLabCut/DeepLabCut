# Copyright 2019 by
# Tabet Ehsainieh, ehsainit@informatik.uni-freiburg.de
# All rights reserved.

import os
import os.path
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from skimage.util import img_as_ubyte

from deeplabcut.lmot import mot
from deeplabcut.lmot.assets import bbox
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.nnet import predict
from deeplabcut.utils import auxiliaryfunctions


def analyze_image(config, vid, shuffle=1, trainingsetindex=0, gputouse=None, save_as_csv=False):
    if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
        del os.environ['TF_CUDNN_USE_AUTOTUNE']  # was potentially set during training

    tf.reset_default_graph()
    cfg = auxiliaryfunctions.read_config(config)
    trainFraction = cfg['TrainingFraction'][trainingsetindex]

    modelfolder = os.path.join(cfg["project_path"], str(auxiliaryfunctions.GetModelFolder(trainFraction, shuffle, cfg)))
    path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'
    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError(
            "It seems the model for shuffle %s and trainFraction %s does not exist." % (shuffle, trainFraction))

    # Check which snapshots are available and sort them by # iterations
    try:
        Snapshots = np.array(
            [fn.split('.')[0] for fn in os.listdir(os.path.join(modelfolder, 'train')) if "index" in fn])
    except FileNotFoundError:
        raise FileNotFoundError(
            "Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s." % (
                shuffle, shuffle))

    if cfg['snapshotindex'] == 'all':
        print(
            "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!")
        snapshotindex = -1
    else:
        snapshotindex = cfg['snapshotindex']

    increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]

    print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)

    ##################################################
    # Load and setup CNN part detector
    ##################################################

    # Check if data already was generated:
    dlc_cfg['init_weights'] = os.path.join(modelfolder, 'train', Snapshots[snapshotindex])

    # update batchsize (based on parameters in config.yaml)
    dlc_cfg['batch_size'] = 1

    sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)

    if gputouse is not None:  # gpu selectinon
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)
    #####################################################
    # Video analysis
    #####################################################
    print("Starting to analyze % ", vid)
    #####################################################
    # Read Video
    #####################################################
    tracker = mot.Tracker(100)
    vname = Path(vid).stem + '.avi'
    print("Loading ", vid)
    cap = cv2.VideoCapture(vid)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(vname, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (frame_width, frame_height))
    while cap.isOpened():
        ret, frame = cap.read()
        start = time.time()
        if ret:
            scrmap, locref = getPose(dlc_cfg, sess, inputs, outputs, frame)
            tracker.track(scrmap, locref)
            for obj in range(len(tracker.tracks)):
                bbox(frame, tracker.tracks[obj])
            out.write(frame)
        else:
            print("frame was analyzed in " + str(time.time() - start))
            print("")
            print("")
            break
            # cv2.waitKey(50)
    # do a bit of cleanup
    print("[INFO] cleaning up...")
    cv2.destroyAllWindows()
    out.release()


def getPose(dlc_cfg, sess, inputs, outputs, img):
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame = img_as_ubyte(frame)
    scmap, locref = predict.getpose(frame, dlc_cfg, sess, inputs, outputs, live=True)
    return scmap, locref


if __name__ == '__main__':
    if len(sys.argv) == 3:
        conf = sys.argv[1]
        vid = sys.argv[2]
        print('configuration :', conf)
        print('input vid :', vid)
        analyze_image(conf, vid)
    else:
        print('Usage: python3 main.py [config path] [image]\n'
              'config path: configuration path of the trained network\n'
              'video: input video')
