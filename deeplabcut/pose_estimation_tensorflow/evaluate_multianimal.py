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

# Dependencies for anaysis
import numpy as np
import pandas as pd
import skimage.color
from skimage import io
from skimage.util import img_as_ubyte
from tqdm import tqdm

from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.utils import visualization


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
    """
    WIP multi animal project.
    """

    import os

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

                (
                    individuals,
                    uniquebodyparts,
                    multianimalbodyparts,
                ) = auxfun_multianimal.extractindividualsandbodyparts(cfg)

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
                        print("Analyzing data...")
                        for imageindex, imagename in tqdm(enumerate(Data.index)):
                            image_path = os.path.join(cfg["project_path"], imagename)
                            image = io.imread(image_path)
                            frame = img_as_ubyte(skimage.color.gray2rgb(image))

                            GT = Data.iloc[imageindex]

                            # Storing GT data as dictionary, so it can be used for calculating connection costs
                            groundtruthcoordinates = []
                            groundtruthidentity = []
                            for bptindex, bpt in enumerate(dlc_cfg["all_joints_names"]):
                                coords = np.zeros([len(individuals), 2]) * np.nan
                                identity = []
                                for prfxindex, prefix in enumerate(individuals):
                                    if bpt in uniquebodyparts and prefix == "single":
                                        coords[prfxindex, :] = np.array(
                                            [
                                                GT[cfg["scorer"]][prefix][bpt]["x"],
                                                GT[cfg["scorer"]][prefix][bpt]["y"],
                                            ]
                                        )
                                        identity.append(prefix)
                                    elif (
                                        bpt in multianimalbodyparts
                                        and prefix != "single"
                                    ):
                                        coords[prfxindex, :] = np.array(
                                            [
                                                GT[cfg["scorer"]][prefix][bpt]["x"],
                                                GT[cfg["scorer"]][prefix][bpt]["y"],
                                            ]
                                        )
                                        identity.append(prefix)
                                    else:
                                        identity.append("nix")

                                groundtruthcoordinates.append(
                                    coords[np.isfinite(coords[:, 0]), :]
                                )
                                groundtruthidentity.append(
                                    np.array(identity)[np.isfinite(coords[:, 0])]
                                )

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

                            if plotting:
                                coords_pred = pred["coordinates"][0]
                                probs_pred = pred["confidence"]
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
    dcorr=10.0,
    leastbpts=1,
    printingintermediatevalues=True,
    modelprefix="",
    plotting=False,
):
    """
    Crossvalidate inference parameters on evaluation data; optimal parametrs will be stored in " inference_cfg.yaml".

    They will then be then used for inference (for analysis of videos). Performs Bayesian Optimization with https://github.com/fmfn/BayesianOptimization

    This is a crucial step. The most important variable (in inferencecfg) to cross-validate is minimalnumberofconnections. Pass
    a reasonable range to optimze (e.g. if you have 5 edges from 1 to 5. If you have 4 bpts and 11 connections from 3 to 9).

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
            0,
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
            dcorr=dcorr,
            leastbpts=leastbpts,
            modelprefix=modelprefix,
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
        DataOptParams.to_hdf(
            path_inference_config.split(".yaml")[0] + ".h5",
            "df_with_missing",
            format="table",
            mode="w",
        )
        DataOptParams.to_csv(path_inference_config.split(".yaml")[0] + ".csv")
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
