"""LightningPose Evaluation as used by Matthew R. Whiteway

Transmitted on January 3rd, 2024
Forwarded on January 5th, 2024
"""
import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm

import deeplabcut
from deeplabcut.pose_estimation_tensorflow import pairwisedistances
from deeplabcut.utils.auxfun_videos import imread, imresize
from deeplabcut.pose_estimation_tensorflow.core import predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.datasets.utils import data_to_input
from deeplabcut.utils import auxiliaryfunctions, conversioncode


DATA_DIR = "/home/niels/datasets/lightning-pose"
DISPLAY_ITERS = 500
SAVE_ITERS = 5000
MAX_ITERS = 50000


def pixel_error(keypoints_true: np.ndarray, keypoints_pred: np.ndarray) -> np.ndarray:
    """Root mean square error between true and predicted keypoints.

    Taken from https://github.com/danbider/lightning-pose/blob/main/lightning_pose/metrics.py

    Args:
        keypoints_true: shape (samples, n_keypoints, 2)
        keypoints_pred: shape (samples, n_keypoints, 2)

    Returns:
        shape (samples, n_keypoints)

    """
    error = np.linalg.norm(keypoints_true - keypoints_pred, axis=2)
    return error


def evaluate_network(
    config,
    csv_file,
    resultsfilename,
    shuffle=0,
    trainingsetindex=0,
    gputouse=None,
    modelprefix="",
    scale=1.0,
):
    tf.compat.v1.reset_default_graph()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  #
    #    tf.logging.set_verbosity(tf.logging.WARN)

    # Read file path for pose_config file. >> pass it on
    cfg = auxiliaryfunctions.read_config(config)
    if gputouse is not None:  # gpu selectinon
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gputouse)

    if trainingsetindex < len(cfg["TrainingFraction"]) and trainingsetindex >= 0:
        trainFraction = cfg["TrainingFraction"][int(trainingsetindex)]
    else:
        raise Exception(
            "Please check the trainingsetindex! ",
            trainingsetindex,
            " should be an integer from 0 .. ",
            int(len(cfg["TrainingFraction"]) - 1),
        )

    # Loading human annotatated data
    data = pd.read_csv(csv_file, index_col=0, header=[0, 1, 2])
    df_index = data.index.copy()

    ##################################################
    # Load and setup CNN part detector
    ##################################################
    modelfolder = os.path.join(
        cfg["project_path"],
        str(
            auxiliaryfunctions.get_model_folder(
                trainFraction, shuffle, cfg, modelprefix=modelprefix
            )
        ),
    )

    path_test_config = Path(modelfolder) / "test" / "pose_cfg.yaml"

    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError(
            "It seems the model for shuffle %s and trainFraction %s does not exist."
            % (shuffle, trainFraction)
        )

    # change batch size, if it was edited during analysis!
    dlc_cfg["batch_size"] = 1  # in case this was edited for analysis.

    # Check which snapshots are available and sort them by # iterations
    Snapshots = np.array(
        [
            fn.split(".")[0]
            for fn in os.listdir(os.path.join(str(modelfolder), "train"))
            if "index" in fn
        ]
    )
    try:  # check if any where found?
        Snapshots[0]
    except IndexError:
        raise FileNotFoundError(
            "Snapshots not found! It seems the dataset for shuffle %s and trainFraction %s is not trained.\nPlease train it before evaluating.\nUse the function 'train_network' to do so."
            % (shuffle, trainFraction)
        )

    increasing_indices = np.argsort([int(m.split("-")[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]
    snapindex = -1

    conversioncode.guarantee_multiindex_rows(data)
    ##################################################
    # Compute predictions over images
    ##################################################
    # setting weights to corresponding snapshot.
    dlc_cfg["init_weights"] = os.path.join(str(modelfolder), "train", Snapshots[snapindex])
    # read how many training siterations that corresponds to.
    trainingsiterations = (dlc_cfg["init_weights"].split(os.sep)[-1]).split("-")[-1]

    # Name for deeplabcut net (based on its parameters)
    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.get_scorer_name(
        cfg,
        shuffle,
        trainFraction,
        trainingsiterations,
        modelprefix=modelprefix,
    )
    print("Running ", DLCscorer, " with # of training iterations:", trainingsiterations)

    # Specifying state of model (snapshot / training state)
    sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)
    Numimages = len(df_index)
    PredicteData = np.zeros((Numimages, 3 * len(dlc_cfg["all_joints_names"])))
    print("Running evaluation ...")
    for imageindex, imagename in tqdm(enumerate(df_index)):
        image = imread(
            os.path.join(cfg["project_path"], imagename),
            mode="skimage",
        )
        if scale != 1:
            image = imresize(image, scale)

        image_batch = data_to_input(image)
        # Compute prediction with the CNN
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
        scmap, locref = predict.extract_cnn_output(outputs_np, dlc_cfg)

        # Extract maximum scoring location from the heatmap, assume 1 person
        pose = predict.argmax_pose_predict(scmap, locref, dlc_cfg["stride"])
        PredicteData[imageindex, :] = pose.flatten()
        # NOTE: thereby cfg_test['all_joints_names'] should be same order as bodyparts!

    sess.close()  # closes the current tf session

    index = pd.MultiIndex.from_product(
        [
            [DLCscorer],
            dlc_cfg["all_joints_names"],
            ["x", "y", "likelihood"],
        ],
        names=["scorer", "bodyparts", "coords"],
    )

    # rescale
    PredicteData[:, 0::3] /= scale
    PredicteData[:, 1::3] /= scale

    # Saving results
    DataMachine = pd.DataFrame(PredicteData, columns=index, index=df_index)
    # DataMachine.loc[:, ("set", "", "")] = "test"
    DataMachine.to_csv(resultsfilename)

    tf.compat.v1.reset_default_graph()

    # compute metrics
    conversioncode.guarantee_multiindex_rows(DataMachine)
    DataCombined = pd.concat(
        [data.T, DataMachine.T], axis=0, sort=False
    ).T

    rmse, rmse_pcutoff = pairwisedistances(
        DataCombined,
        cfg["scorer"],
        DLCscorer,
        cfg["pcutoff"],
        bodyparts=None,
    )
    test_error = np.nanmean(rmse.values.flatten())
    test_error_pcutoff = np.nanmean(rmse_pcutoff.values.flatten())
    print(f"Test error         {test_error:.2f}")
    print(f"Test error pcutoff {test_error_pcutoff:.2f}")

    pred_data = DataMachine.drop(labels="likelihood", level=2, axis=1)
    num_images = len(pred_data)
    lp_pixel_error = pixel_error(
        data.to_numpy().reshape((num_images, -1, 2)),
        pred_data.to_numpy().reshape((num_images, -1, 2))
    )
    print(f"Test error LP      {np.nanmean(lp_pixel_error)}")
    return np.nanmean(lp_pixel_error)


def run_main(args):
    batch_size = 8

    if args.dataset == 'mirror-mouse':
        scorer = 'rick'
        date = '2022-12-02'
        date_str = 'Dec2'
        global_scale = 0.64
        if args.train_frames == 75:
            shuffle_list = [750, 751, 752, 753, 754]
            trainingsetindex = 0
            trainingset = 49
        else:
            shuffle_list = [10, 11, 12, 13, 14]
            trainingsetindex = 1
            trainingset = 89
    elif args.dataset == 'mirror-fish':
        scorer = 'rick'
        date = '2023-10-26'
        date_str = 'Oct26'
        global_scale = 0.7
        if args.train_frames == 75:
            shuffle_list = [750, 751, 752, 753, 754]
            trainingsetindex = 0
            trainingset = 81
        else:
            shuffle_list = [10, 11, 12, 13, 14]
            trainingsetindex = 1
            trainingset = 95
    elif args.dataset == 'ibl-pupil':
        scorer = 'mic'
        date = '2022-12-06'
        date_str = 'Dec6'
        global_scale = 1.28
        if args.train_frames == 75:
            shuffle_list = [750, 751, 752, 753, 754]
            trainingsetindex = 0
            trainingset = 22
        else:
            shuffle_list = [10, 11, 12, 13, 14]
            trainingsetindex = 1
            trainingset = 89
    elif args.dataset == 'ibl-paw':
        scorer = 'mic'
        date = '2023-01-09'
        date_str = 'Jan9'
        global_scale = 1.28
        if args.train_frames == 75:
            shuffle_list = [750, 751, 752, 753, 754]
            trainingsetindex = 0
            trainingset = 11
        else:
            shuffle_list = [10, 11, 12, 13, 14]
            trainingsetindex = 1
            trainingset = 89
    else:
        raise NotImplementedError

    project_dir = os.path.join(DATA_DIR, '%s-%s-%s' % (args.dataset, scorer, date))
    config_path = os.path.join(project_dir, 'config.yaml')

    shuffle_results = []
    for shuffle in shuffle_list:
        model_folder = os.path.join(
            project_dir, 'dlc-models', 'iteration-0', '%s%s-trainset%ishuffle%i' % (
                args.dataset, date_str, trainingset, shuffle,
            )
        )

        # evaluate model on OOD data
        print(f"Shuffle {shuffle}")
        shuffle_results.append(
            evaluate_network(
                config_path,
                csv_file=os.path.join(project_dir, 'CollectedData_new.csv'),
                resultsfilename=os.path.join(model_folder, 'predictions_new.csv'),
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
                gputouse=args.gpu_id,
                scale=global_scale,
            )
        )

    print(f"Results on all shuffles")
    print(f"  Mean: {np.mean(shuffle_results):.2f}")
    print(f"  STD:  {np.std(shuffle_results):.2f}")


if __name__ == '__main__':
    """(dlc) python eval_lp_ood.py --dataset=mirror-fish --gpu_id=0 --train_frames=75"""
    """(dlc) python eval_lp_ood.py --dataset=mirror-mouse --gpu_id=0 --train_frames=75"""

    parser = argparse.ArgumentParser()

    # base params
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--train_frames', type=int)

    namespace, _ = parser.parse_known_args()
    run_main(namespace)
