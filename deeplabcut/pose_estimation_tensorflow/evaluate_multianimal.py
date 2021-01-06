"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""


import os
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.color
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage import io
from skimage.util import img_as_ubyte
from tqdm import tqdm

from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.utils import visualization


def _percentile(n):
    def percentile_(x):
        return x.quantile(n)

    percentile_.__name__ = f"percentile_{100 * n:.0f}"
    return percentile_


def _compute_stats(df):
    return df.agg(
        [
            "min",
            "max",
            "mean",
            np.std,
            _percentile(0.25),
            _percentile(0.50),
            _percentile(0.75),
        ]
    ).stack(level=1)


def evaluate_multianimal_full(
    config,
    Shuffles=[1],
    trainingsetindex=0,
    plotting=None,
    show_errors=True,
    comparisonbodyparts="all",
    gputouse=None,
    modelprefix="",
    c_engine=False,
):
    from deeplabcut.pose_estimation_tensorflow.nnet import predict
    from deeplabcut.pose_estimation_tensorflow.nnet import (
        predict_multianimal as predictma,
    )
    from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal

    import tensorflow as tf

    if "TF_CUDNN_USE_AUTOTUNE" in os.environ:
        del os.environ["TF_CUDNN_USE_AUTOTUNE"]  # was potentially set during training

    tf.reset_default_graph()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  #
    if gputouse is not None:  # gpu selectinon
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gputouse)

    start_path = os.getcwd()

    ##################################################
    # Load data...
    ##################################################
    cfg = auxiliaryfunctions.read_config(config)
    if trainingsetindex == "all":
        TrainingFractions = cfg["TrainingFraction"]
    else:
        TrainingFractions = [cfg["TrainingFraction"][trainingsetindex]]

    # Loading human annotatated data
    trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(cfg)
    Data = pd.read_hdf(
        os.path.join(
            cfg["project_path"],
            str(trainingsetfolder),
            "CollectedData_" + cfg["scorer"] + ".h5",
        ),
        "df_with_missing",
    )
    # Handle data previously annotated on a different platform
    sep = "/" if "/" in Data.index[0] else "\\"
    if sep != os.path.sep:
        Data.index = Data.index.str.replace(sep, os.path.sep)
    # Get list of body parts to evaluate network for
    comparisonbodyparts = auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(
        cfg, comparisonbodyparts
    )
    all_bpts = np.asarray(
        len(cfg["individuals"]) * cfg["multianimalbodyparts"] + cfg["uniquebodyparts"]
    )
    colors = visualization.get_cmap(len(comparisonbodyparts), name=cfg["colormap"])
    # Make folder for evaluation
    auxiliaryfunctions.attempttomakefolder(
        str(cfg["project_path"] + "/evaluation-results/")
    )
    for shuffle in Shuffles:
        for trainFraction in TrainingFractions:
            ##################################################
            # Load and setup CNN part detector
            ##################################################
            datafn, metadatafn = auxiliaryfunctions.GetDataandMetaDataFilenames(
                trainingsetfolder, trainFraction, shuffle, cfg
            )
            modelfolder = os.path.join(
                cfg["project_path"],
                str(
                    auxiliaryfunctions.GetModelFolder(
                        trainFraction, shuffle, cfg, modelprefix=modelprefix
                    )
                ),
            )
            path_test_config = Path(modelfolder) / "test" / "pose_cfg.yaml"

            # Load meta data
            (
                data,
                trainIndices,
                testIndices,
                trainFraction,
            ) = auxiliaryfunctions.LoadMetadata(
                os.path.join(cfg["project_path"], metadatafn)
            )

            try:
                dlc_cfg = load_config(str(path_test_config))
            except FileNotFoundError:
                raise FileNotFoundError(
                    "It seems the model for shuffle %s and trainFraction %s does not exist."
                    % (shuffle, trainFraction)
                )

            # TODO: IMPLEMENT for different batch sizes?
            dlc_cfg["batch_size"] = 1  # due to differently sized images!!!

            joints = dlc_cfg["all_joints_names"]

            # Create folder structure to store results.
            evaluationfolder = os.path.join(
                cfg["project_path"],
                str(
                    auxiliaryfunctions.GetEvaluationFolder(
                        trainFraction, shuffle, cfg, modelprefix=modelprefix
                    )
                ),
            )
            auxiliaryfunctions.attempttomakefolder(evaluationfolder, recursive=True)
            # path_train_config = modelfolder / 'train' / 'pose_cfg.yaml'

            # Check which snapshots are available and sort them by # iterations
            Snapshots = np.array(
                [
                    fn.split(".")[0]
                    for fn in os.listdir(os.path.join(str(modelfolder), "train"))
                    if "index" in fn
                ]
            )
            if len(Snapshots) == 0:
                print(
                    "Snapshots not found! It seems the dataset for shuffle %s and trainFraction %s is not trained.\nPlease train it before evaluating.\nUse the function 'train_network' to do so."
                    % (shuffle, trainFraction)
                )
            else:
                increasing_indices = np.argsort(
                    [int(m.split("-")[1]) for m in Snapshots]
                )
                Snapshots = Snapshots[increasing_indices]

                if cfg["snapshotindex"] == -1:
                    snapindices = [-1]
                elif cfg["snapshotindex"] == "all":
                    snapindices = range(len(Snapshots))
                elif cfg["snapshotindex"] < len(Snapshots):
                    snapindices = [cfg["snapshotindex"]]
                else:
                    print(
                        "Invalid choice, only -1 (last), any integer up to last, or all (as string)!"
                    )

                final_result = []
                ##################################################
                # Compute predictions over images
                ##################################################
                for snapindex in snapindices:
                    dlc_cfg["init_weights"] = os.path.join(
                        str(modelfolder), "train", Snapshots[snapindex]
                    )  # setting weights to corresponding snapshot.
                    trainingsiterations = (
                        dlc_cfg["init_weights"].split(os.sep)[-1]
                    ).split("-")[
                        -1
                    ]  # read how many training siterations that corresponds to.

                    # name for deeplabcut net (based on its parameters)
                    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.GetScorerName(
                        cfg,
                        shuffle,
                        trainFraction,
                        trainingsiterations,
                        modelprefix=modelprefix,
                    )
                    print(
                        "Running ",
                        DLCscorer,
                        " with # of trainingiterations:",
                        trainingsiterations,
                    )
                    (
                        notanalyzed,
                        resultsfilename,
                        DLCscorer,
                    ) = auxiliaryfunctions.CheckifNotEvaluated(
                        str(evaluationfolder),
                        DLCscorer,
                        DLCscorerlegacy,
                        Snapshots[snapindex],
                    )

                    if os.path.isfile(resultsfilename.split(".h5")[0] + "_full.pickle"):
                        print("Model already evaluated.", resultsfilename)
                    else:
                        if plotting:
                            foldername = os.path.join(
                                str(evaluationfolder),
                                "LabeledImages_"
                                + DLCscorer
                                + "_"
                                + Snapshots[snapindex],
                            )
                            auxiliaryfunctions.attempttomakefolder(foldername)

                        # print(dlc_cfg)
                        # Specifying state of model (snapshot / training state)
                        sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)

                        PredicteData = {}
                        dist = np.full((len(all_bpts), len(Data)), np.nan)
                        distnorm = np.full(len(Data), np.nan)
                        print("Analyzing data...")
                        for imageindex, imagename in tqdm(enumerate(Data.index)):
                            image_path = os.path.join(cfg["project_path"], imagename)
                            image = io.imread(image_path)
                            frame = img_as_ubyte(skimage.color.gray2rgb(image))

                            GT = Data.iloc[imageindex]
                            df = GT.unstack("coords").reindex(joints, level='bodyparts')

                            # Evaluate PAF edge lengths to calibrate `distnorm`
                            temp = GT.unstack("bodyparts")[joints]
                            xy = temp.values.reshape((-1, 2, temp.shape[1])).swapaxes(
                                1, 2
                            )
                            edges = xy[:, dlc_cfg["partaffinityfield_graph"]]
                            lengths = np.sum(
                                (edges[:, :, 0] - edges[:, :, 1]) ** 2, axis=2
                            )
                            distnorm[imageindex] = np.nanmax(lengths)

                            # FIXME Is having an empty array vs nan really that necessary?!
                            groundtruthidentity = list(
                                df.index.get_level_values("individuals")
                                .to_numpy()
                                .reshape((-1, 1))
                            )
                            groundtruthcoordinates = list(df.values[:, np.newaxis])
                            for i, coords in enumerate(groundtruthcoordinates):
                                if np.isnan(coords).any():
                                    groundtruthcoordinates[i] = np.empty(
                                        (0, 2), dtype=float
                                    )
                                    groundtruthidentity[i] = np.array([], dtype=str)

                            PredicteData[imagename] = {}
                            PredicteData[imagename]["index"] = imageindex

                            pred = predictma.get_detectionswithcostsandGT(
                                frame,
                                groundtruthcoordinates,
                                dlc_cfg,
                                sess,
                                inputs,
                                outputs,
                                outall=False,
                                nms_radius=dlc_cfg.nmsradius,
                                det_min_score=dlc_cfg.minconfidence,
                                c_engine=c_engine,
                            )
                            PredicteData[imagename]["prediction"] = pred
                            PredicteData[imagename]["groundtruth"] = [
                                groundtruthidentity,
                                groundtruthcoordinates,
                                GT,
                            ]

                            coords_pred = pred["coordinates"][0]
                            probs_pred = pred["confidence"]
                            for bpt, xy_gt in df.groupby(level="bodyparts"):
                                inds_gt = np.flatnonzero(
                                    np.all(~np.isnan(xy_gt), axis=1)
                                )
                                xy = coords_pred[joints.index(bpt)]
                                if inds_gt.size and xy.size:
                                    # Pick the predictions closest to ground truth,
                                    # rather than the ones the model has most confident in
                                    d = cdist(xy_gt.iloc[inds_gt], xy)
                                    rows, cols = linear_sum_assignment(d)
                                    min_dists = d[rows, cols]
                                    inds = np.flatnonzero(all_bpts == bpt)
                                    dist[inds[inds_gt[rows]], imageindex] = min_dists

                            if plotting:
                                fig = visualization.make_multianimal_labeled_image(
                                    frame,
                                    groundtruthcoordinates,
                                    coords_pred,
                                    probs_pred,
                                    colors,
                                    cfg["dotsize"],
                                    cfg["alphavalue"],
                                    cfg["pcutoff"],
                                )

                                visualization.save_labeled_frame(
                                    fig,
                                    image_path,
                                    foldername,
                                    imageindex in trainIndices,
                                )

                        sess.close()  # closes the current tf session

                        # Compute all distance statistics
                        df_dist = pd.DataFrame(dist, index=df.index)
                        write_path = os.path.join(evaluationfolder, "dist.csv")
                        df_dist.to_csv(write_path)

                        stats_per_ind = _compute_stats(df_dist.groupby("individuals"))
                        stats_per_ind.to_csv(
                            write_path.replace("dist.csv", "dist_stats_ind.csv")
                        )
                        stats_per_bpt = _compute_stats(df_dist.groupby("bodyparts"))
                        stats_per_bpt.to_csv(
                            write_path.replace("dist.csv", "dist_stats_bpt.csv")
                        )

                        # For OKS/PCK, compute the standard deviation error across all frames
                        sd = df_dist.groupby("bodyparts").mean().std(axis=1)
                        sd["distnorm"] = np.sqrt(np.nanmax(distnorm))
                        sd.to_csv(write_path.replace("dist.csv", "sd.csv"))

                        if show_errors:
                            print("##########################################&1")
                            print(
                                "Euclidean distance statistics per individual (in pixels)"
                            )
                            print(stats_per_ind.mean(axis=1).unstack().to_string())
                            print("##########################################")
                            print(
                                "Euclidean distance statistics per bodypart (in pixels)"
                            )
                            print(stats_per_bpt.mean(axis=1).unstack().to_string())

                        PredicteData["metadata"] = {
                            "nms radius": dlc_cfg.nmsradius,
                            "minimal confidence": dlc_cfg.minconfidence,
                            "PAFgraph": dlc_cfg.partaffinityfield_graph,
                            "all_joints": [[i] for i in range(len(dlc_cfg.all_joints))],
                            "all_joints_names": [
                                dlc_cfg.all_joints_names[i]
                                for i in range(len(dlc_cfg.all_joints))
                            ],
                            "stride": dlc_cfg.get("stride", 8),
                        }
                        print(
                            "Done and results stored for snapshot: ",
                            Snapshots[snapindex],
                        )

                        dictionary = {
                            "Scorer": DLCscorer,
                            "DLC-model-config file": dlc_cfg,
                            "trainIndices": trainIndices,
                            "testIndices": testIndices,
                            "trainFraction": trainFraction,
                        }
                        metadata = {"data": dictionary}
                        auxfun_multianimal.SaveFullMultiAnimalData(
                            PredicteData, metadata, resultsfilename
                        )

                        tf.reset_default_graph()

    # returning to intial folder
    os.chdir(str(start_path))


def evaluate_multianimal_crossvalidate(
    config,
    Shuffles=[1],
    trainingsetindex=0,
    pbounds=None,
    edgewisecondition=True,
    target="rpck_train",
    inferencecfg=None,
    init_points=20,
    n_iter=50,
    log_file=None,
    dcorr=10.0,
    leastbpts=1,
    printingintermediatevalues=True,
    modelprefix="",
    plotting=False,
):
    """
    Cross-validate inference parameters on evaluation data; optimal parameters will be stored in "inference_cfg.yaml".

    They will then be then used for inference (for analysis of videos). Performs Bayesian Optimization with https://github.com/fmfn/BayesianOptimization

    This is a crucial step. The most important variable (in inferencecfg) to cross-validate is minimalnumberofconnections.
    Pass a reasonable range to optimize (e.g. if you have 5 edges from 1 to 5. If you have 4 bpts and 11 connections from 3 to 9).

    config: string
        Full path of the config.yaml file as a string.

    shuffle: int, optional
        An integer specifying the shuffle index of the training dataset used for training the network. The default is 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    pbounds: dictionary of variables with ranges to crossvalidate.
        By default: pbounds = {
                        'pafthreshold': (0.05, 0.7),
                        'detectionthresholdsquare': (0, 0.9),
                        'minimalnumberofconnections': (1, # connections in your skeleton),
                    }

    inferencecfg: dict, OPTIONAL
        For the variables that are *not* crossvalidated the parameters from inference_cfg.yaml are used, or
        you can overwrite them by passing a dictinary with your preferred parameters.

    edgewisecondition: bool, default True
        Estimates Euclidean distances for each skeleton edge and uses those distance for excluding possible connections.
        If false, uses only one distance for all bodyparts (which is obviously suboptimal).

    target: string, default='rpck_train'
        What metric to optimize. Options are pck/rpck/rmse on train/test set.

    init_points: int, optional (default=10)
        Number of random initial explorations. Probing random regions helps diversify the exploration space.
        Parameter from BayesianOptimization.

    n_iter: int, optional (default=20)
        Number of iterations of Bayesian optimization to perform.
        The larger it is, the higher the likelihood of finding a good extremum.
        Parameter from BayesianOptimization.

    log_file: str, optional (default=None)
        Path to a JSON file containing the progress of a previous Bayesian optimization run.
        Note that previously probed points will not be evaluated again.

    dcorr: float,
        Distance thereshold for percent correct keypoints / relative percent correct keypoints (see paper).

    leastbpts: integer (should be a small number)
        If an animals has less or equal as many body parts in an image it will not be used
        for cross validation. Imagine e.g. if only a single bodypart is present, then
        if animals need a certain minimal number of bodyparts for assembly (minimalnumberofconnections),
        this might not be predictable.

    printingintermediatevalues: bool, default True
        If intermediate metrics RMSE/hits/.. per sample should be printed.


    Examples
    --------

    first run evalute:

    deeplabcut.evaluate_network(path_config_file,Shuffles=[shuffle],plotting=True)

    Then e.g. for finding inference parameters to minimize rmse on test set:

    deeplabcut.evaluate_multianimal_crossvalidate(path_config_file,Shuffles=[shuffle],target='rmse_test')
    """
    from deeplabcut.pose_estimation_tensorflow.lib import crossvalutils
    from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions
    from easydict import EasyDict as edict

    cfg = auxiliaryfunctions.read_config(config)
    trainFraction = cfg["TrainingFraction"][trainingsetindex]
    trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(cfg)
    Data = pd.read_hdf(
        os.path.join(
            cfg["project_path"],
            str(trainingsetfolder),
            "CollectedData_" + cfg["scorer"] + ".h5",
        ),
        "df_with_missing",
    )
    comparisonbodyparts = auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(
        cfg, "all"
    )
    colors = visualization.get_cmap(len(comparisonbodyparts), name=cfg["colormap"])

    # wild guesses for a wide range:
    maxconnections = len(cfg["skeleton"])
    minconnections = 1  # len(cfg['multianimalbodyparts'])-1

    _pbounds = {
        "pafthreshold": (0.05, 0.7),
        "detectionthresholdsquare": (
            0.0,
            0.9,
        ),  # TODO: set to minimum (from pose_cfg.yaml)
        "minimalnumberofconnections": (minconnections, maxconnections),
    }
    if pbounds is not None:
        _pbounds.update(pbounds)

    if "rpck" in target or "pck" in target:
        maximize = True

    if "rmse" in target:
        maximize = False  # i.e. minimize

    for shuffle in Shuffles:
        evaluationfolder = os.path.join(
            cfg["project_path"],
            str(
                auxiliaryfunctions.GetEvaluationFolder(
                    trainFraction, shuffle, cfg, modelprefix=modelprefix
                )
            ),
        )
        auxiliaryfunctions.attempttomakefolder(evaluationfolder, recursive=True)

        datafn, metadatafn = auxiliaryfunctions.GetDataandMetaDataFilenames(
            trainingsetfolder, trainFraction, shuffle, cfg
        )
        _, trainIndices, testIndices, _ = auxiliaryfunctions.LoadMetadata(
            os.path.join(cfg["project_path"], metadatafn)
        )
        modelfolder = os.path.join(
            cfg["project_path"],
            str(
                auxiliaryfunctions.GetModelFolder(
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

        # Check which snapshots are available and sort them by # iterations
        Snapshots = np.array(
            [
                fn.split(".")[0]
                for fn in os.listdir(os.path.join(str(modelfolder), "train"))
                if "index" in fn
            ]
        )
        snapindex = -1
        dlc_cfg["init_weights"] = os.path.join(
            str(modelfolder), "train", Snapshots[snapindex]
        )  # setting weights to corresponding snapshot.
        trainingsiterations = (dlc_cfg["init_weights"].split(os.sep)[-1]).split("-")[
            -1
        ]  # read how many training siterations that corresponds to.

        DLCscorer, _ = auxiliaryfunctions.GetScorerName(
            cfg, shuffle, trainFraction, trainingsiterations, modelprefix=modelprefix
        )

        path_inference_config = Path(modelfolder) / "test" / "inference_cfg.yaml"
        if inferencecfg is None:  # then load or initialize
            inferencecfg = auxfun_multianimal.read_inferencecfg(
                path_inference_config, cfg
            )
        else:
            inferencecfg = edict(inferencecfg)
            auxfun_multianimal.check_inferencecfg_sanity(cfg, inferencecfg)

        # Pick distance threshold for (r)PCK from the statistics computed during evaluation
        stats_file = os.path.join(evaluationfolder, "sd.csv")
        if os.path.isfile(stats_file):
            stats = pd.read_csv(stats_file, header=None, index_col=0)
            inferencecfg.distnormalization = np.round(
                stats.loc["distnorm", 1], 2
            ).item()
            stats = stats.drop("distnorm")
            dcorr = (
                2 * stats.mean().squeeze()
            )  # Taken as 2*SD error between predictions and ground truth
        else:
            dcorr = 10
        inferencecfg.topktoretain = np.inf
        inferencecfg, opt = crossvalutils.bayesian_search(
            config,
            inferencecfg,
            _pbounds,
            edgewisecondition=edgewisecondition,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            target=target,
            maximize=maximize,
            init_points=init_points,
            n_iter=n_iter,
            acq="ei",
            log_file=log_file,
            dcorr=dcorr,
            leastbpts=leastbpts,
            modelprefix=modelprefix,
            printingintermediatevalues=printingintermediatevalues,
        )

        # update number of individuals to retain.
        inferencecfg.topktoretain = len(cfg["individuals"]) + 1 * (
            len(cfg["uniquebodyparts"]) > 0
        )

        # calculating result at best best solution
        DataOptParams, poses_gt, poses = crossvalutils.compute_crossval_metrics(
            config, inferencecfg, shuffle, trainingsetindex, modelprefix
        )

        path_inference_config = str(path_inference_config)
        # print("Quantification:", DataOptParams.head())
        # DataOptParams.to_hdf(
        #     path_inference_config.split(".yaml")[0] + ".h5",
        #     "df_with_missing",
        #     format="table",
        #     mode="w",
        # )
        DataOptParams.to_csv(os.path.join(evaluationfolder, "results.csv"))
        print("Saving optimal inference parameters...")
        print(DataOptParams.to_string())
        auxiliaryfunctions.write_plainconfig(path_inference_config, dict(inferencecfg))

        # Store best predictions
        max_indivs = max(pose.shape[0] for pose in poses)
        bpts = dlc_cfg["all_joints_names"]
        container = np.full((len(poses), max_indivs * len(bpts) * 3), np.nan)
        for n, pose in enumerate(poses):
            temp = pose.flatten()
            container[n, : len(temp)] = temp

        header = pd.MultiIndex.from_product(
            [
                [DLCscorer],
                [f"individual{i}" for i in range(1, max_indivs + 1)],
                bpts,
                ["x", "y", "likelihood"],
            ],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )

        df = pd.DataFrame(container, columns=header)
        df.to_hdf(
            os.path.join(evaluationfolder, f"{DLCscorer}.h5"), key="df_with_missing"
        )

        if plotting:
            foldername = os.path.join(
                str(evaluationfolder),
                "LabeledImages_" + DLCscorer + "_" + Snapshots[snapindex],
            )
            auxiliaryfunctions.attempttomakefolder(foldername)
            for imageindex, imagename in tqdm(enumerate(Data.index)):
                image_path = os.path.join(cfg["project_path"], imagename)
                image = io.imread(image_path)
                frame = img_as_ubyte(skimage.color.gray2rgb(image))
                groundtruthcoordinates = poses_gt[imageindex]
                coords_pred = poses[imageindex][:, :, :2]
                probs_pred = poses[imageindex][:, :, -1:]
                fig = visualization.make_multianimal_labeled_image(
                    frame,
                    groundtruthcoordinates,
                    coords_pred,
                    probs_pred,
                    colors,
                    cfg["dotsize"],
                    cfg["alphavalue"],
                    cfg["pcutoff"],
                )
                visualization.save_labeled_frame(
                    fig, image_path, foldername, imageindex in trainIndices
                )
