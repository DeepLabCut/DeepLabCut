"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""


import argparse
import os
from pathlib import Path

# Dependencies for anaysis
import numpy as np
import pandas as pd
from tqdm import tqdm


def pairwisedistances(DataCombined, scorer1, scorer2, pcutoff=-1, bodyparts=None):
    """ Calculates the pairwise Euclidean distance metric over body parts vs. images"""
    mask = DataCombined[scorer2].xs("likelihood", level=1, axis=1) >= pcutoff
    if bodyparts == None:
        Pointwisesquareddistance = (DataCombined[scorer1] - DataCombined[scorer2]) ** 2
        RMSE = np.sqrt(
            Pointwisesquareddistance.xs("x", level=1, axis=1)
            + Pointwisesquareddistance.xs("y", level=1, axis=1)
        )  # Euclidean distance (proportional to RMSE)
        return RMSE, RMSE[mask]
    else:
        Pointwisesquareddistance = (
            DataCombined[scorer1][bodyparts] - DataCombined[scorer2][bodyparts]
        ) ** 2
        RMSE = np.sqrt(
            Pointwisesquareddistance.xs("x", level=1, axis=1)
            + Pointwisesquareddistance.xs("y", level=1, axis=1)
        )  # Euclidean distance (proportional to RMSE)
        return RMSE, RMSE[mask]


def distance(v, w):
    return np.sqrt(np.sum((v - w) ** 2))


def calculatepafdistancebounds(
    config, shuffle=0, trainingsetindex=0, modelprefix="", numdigits=0, onlytrain=False
):
    """
    Returns distances along paf edges in train/test data

    ----------
    config : string
        Full path of the config.yaml file as a string.

    shuffle: integer
        integers specifying shuffle index of the training dataset. The default is 0.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml). This
        variable can also be set to "all".

    numdigits: number of digits to round for distances.

    """
    import os
    from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal
    from deeplabcut.pose_estimation_tensorflow.config import load_config

    # Read file path for pose_config file. >> pass it on
    cfg = auxiliaryfunctions.read_config(config)

    if cfg["multianimalproject"]:
        (
            individuals,
            uniquebodyparts,
            multianimalbodyparts,
        ) = auxfun_multianimal.extractindividualsandbodyparts(cfg)

        # Loading human annotatated data
        trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(cfg)
        trainFraction = cfg["TrainingFraction"][trainingsetindex]
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

        # Load meta data & annotations
        (
            data,
            trainIndices,
            testIndices,
            trainFraction,
        ) = auxiliaryfunctions.LoadMetadata(
            os.path.join(cfg["project_path"], metadatafn)
        )
        Data = pd.read_hdf(
            os.path.join(
                cfg["project_path"],
                str(trainingsetfolder),
                "CollectedData_" + cfg["scorer"] + ".h5",
            ),
            "df_with_missing",
        )[cfg["scorer"]]

        path_test_config = Path(modelfolder) / "test" / "pose_cfg.yaml"
        dlc_cfg = load_config(str(path_test_config))

        # get the graph!
        partaffinityfield_graph = dlc_cfg.partaffinityfield_graph
        jointnames = [
            dlc_cfg.all_joints_names[i] for i in range(len(dlc_cfg.all_joints))
        ]
        path_inferencebounds_config = (
            Path(modelfolder) / "test" / "inferencebounds.yaml"
        )
        inferenceboundscfg = {}
        for pi, edge in enumerate(partaffinityfield_graph):
            j1, j2 = jointnames[edge[0]], jointnames[edge[1]]
            ds_within = []
            ds_across = []
            for ind in individuals:
                for ind2 in individuals:
                    if ind != "single" and ind2 != "single":
                        if (ind, j1, "x") in Data.keys() and (
                            ind2,
                            j2,
                            "y",
                        ) in Data.keys():
                            distances = np.sqrt(
                                (Data[ind, j1, "x"] - Data[ind2, j2, "x"]) ** 2
                                + (Data[ind, j1, "y"] - Data[ind2, j2, "y"]) ** 2
                            ) / dlc_cfg["stride"]
                        else:
                            distances = None

                        if distances is not None:
                            if onlytrain:
                                distances = distances.iloc[trainIndices]
                            if ind == ind2:
                                ds_within.extend(distances.values.flatten())
                            else:
                                ds_across.extend(distances.values.flatten())

            edgeencoding = str(edge[0]) + "_" + str(edge[1])
            inferenceboundscfg[edgeencoding] = {}
            if len(ds_within) > 0:
                inferenceboundscfg[edgeencoding]["intra_max"] = str(
                    round(np.nanmax(ds_within), numdigits)
                )
                inferenceboundscfg[edgeencoding]["intra_min"] = str(
                    round(np.nanmin(ds_within), numdigits)
                )
            else:
                inferenceboundscfg[edgeencoding]["intra_max"] = str(
                    1e5
                )  # large number (larger than any image diameter)
                inferenceboundscfg[edgeencoding]["intra_min"] = str(0)

            # NOTE: the inter-animal distances are currently not used, but are interesting to compare to intra_*
            if len(ds_across) > 0:
                inferenceboundscfg[edgeencoding]["inter_max"] = str(
                    round(np.nanmax(ds_across), numdigits)
                )
                inferenceboundscfg[edgeencoding]["inter_min"] = str(
                    round(np.nanmin(ds_across), numdigits)
                )
            else:
                inferenceboundscfg[edgeencoding]["inter_max"] = str(
                    1e5
                )  # large number (larger than image diameters in typical experiments)
                inferenceboundscfg[edgeencoding]["inter_min"] = str(0)

        auxiliaryfunctions.write_plainconfig(
            str(path_inferencebounds_config), dict(inferenceboundscfg)
        )
        return inferenceboundscfg
    else:
        print("You might as well bring owls to Athens.")
        return {}


def Plotting(
    cfg, comparisonbodyparts, DLCscorer, trainIndices, DataCombined, foldername
):
    """ Function used for plotting GT and predictions """
    from deeplabcut.utils import visualization

    colors = visualization.get_cmap(len(comparisonbodyparts), name=cfg["colormap"])
    NumFrames = np.size(DataCombined.index)
    for ind in tqdm(np.arange(NumFrames)):
        visualization.plot_and_save_labeled_frame(
            DataCombined,
            ind,
            trainIndices,
            cfg,
            colors,
            comparisonbodyparts,
            DLCscorer,
            foldername,
        )


def return_evaluate_network_data(
    config,
    shuffle=0,
    trainingsetindex=0,
    comparisonbodyparts="all",
    Snapindex=None,
    rescale=False,
    fulldata=False,
    show_errors=True,
    modelprefix="",
    returnjustfns=True,
):
    """
    Returns the results for (previously evaluated) network. deeplabcut.evaluate_network(..)
    Returns list of (per model): [trainingsiterations,trainfraction,shuffle,trainerror,testerror,pcutoff,trainerrorpcutoff,testerrorpcutoff,Snapshots[snapindex],scale,net_type]

    If fulldata=True, also returns (the complete annotation and prediction array)
    Returns list of: (DataMachine, Data, data, trainIndices, testIndices, trainFraction, DLCscorer,comparisonbodyparts, cfg, Snapshots[snapindex])
    ----------
    config : string
        Full path of the config.yaml file as a string.

    shuffle: integer
        integers specifying shuffle index of the training dataset. The default is 0.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml). This
        variable can also be set to "all".

    comparisonbodyparts: list of bodyparts, Default is "all".
        The average error will be computed for those body parts only (Has to be a subset of the body parts).

    rescale: bool, default False
        Evaluate the model at the 'global_scale' variable (as set in the test/pose_config.yaml file for a particular project). I.e. every
        image will be resized according to that scale and prediction will be compared to the resized ground truth. The error will be reported
        in pixels at rescaled to the *original* size. I.e. For a [200,200] pixel image evaluated at global_scale=.5, the predictions are calculated
        on [100,100] pixel images, compared to 1/2*ground truth and this error is then multiplied by 2!. The evaluation images are also shown for the
        original size!

    Examples
    --------
    If you do not want to plot
    >>> deeplabcut._evaluate_network_data('/analysis/project/reaching-task/config.yaml', shuffle=[1])
    --------
    If you want to plot
    >>> deeplabcut.evaluate_network('/analysis/project/reaching-task/config.yaml',shuffle=[1],True)
    """

    import os

    from deeplabcut.pose_estimation_tensorflow.config import load_config
    from deeplabcut.utils import auxiliaryfunctions

    start_path = os.getcwd()
    # Read file path for pose_config file. >> pass it on
    cfg = auxiliaryfunctions.read_config(config)

    # Loading human annotatated data
    trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(cfg)
    # Data=pd.read_hdf(os.path.join(cfg["project_path"],str(trainingsetfolder),'CollectedData_' + cfg["scorer"] + '.h5'),'df_with_missing')

    # Get list of body parts to evaluate network for
    comparisonbodyparts = auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(
        cfg, comparisonbodyparts
    )
    ##################################################
    # Load data...
    ##################################################
    trainFraction = cfg["TrainingFraction"][trainingsetindex]
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
    data, trainIndices, testIndices, trainFraction = auxiliaryfunctions.LoadMetadata(
        os.path.join(cfg["project_path"], metadatafn)
    )

    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError(
            "It seems the model for shuffle %s and trainFraction %s does not exist."
            % (shuffle, trainFraction)
        )

    ########################### RESCALING (to global scale)
    if rescale == True:
        scale = dlc_cfg["global_scale"]
        print("Rescaling Data to ", scale)
        Data = (
            pd.read_hdf(
                os.path.join(
                    cfg["project_path"],
                    str(trainingsetfolder),
                    "CollectedData_" + cfg["scorer"] + ".h5",
                ),
                "df_with_missing",
            )
            * scale
        )
    else:
        scale = 1
        Data = pd.read_hdf(
            os.path.join(
                cfg["project_path"],
                str(trainingsetfolder),
                "CollectedData_" + cfg["scorer"] + ".h5",
            ),
            "df_with_missing",
        )

    evaluationfolder = os.path.join(
        cfg["project_path"],
        str(
            auxiliaryfunctions.GetEvaluationFolder(
                trainFraction, shuffle, cfg, modelprefix=modelprefix
            )
        ),
    )
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
        snapindices = []
    else:
        increasing_indices = np.argsort([int(m.split("-")[1]) for m in Snapshots])
        Snapshots = Snapshots[increasing_indices]
        if Snapindex == None:
            Snapindex = cfg["snapshotindex"]

        if Snapindex == -1:
            snapindices = [-1]
        elif Snapindex == "all":
            snapindices = range(len(Snapshots))
        elif Snapindex < len(Snapshots):
            snapindices = [Snapindex]
        else:
            print(
                "Invalid choice, only -1 (last), any integer up to last, or all (as string)!"
            )

    DATA = []
    results = []
    resultsfns = []
    for snapindex in snapindices:
        dlc_cfg["init_weights"] = os.path.join(
            str(modelfolder), "train", Snapshots[snapindex]
        )  # setting weights to corresponding snapshot.
        trainingsiterations = (dlc_cfg["init_weights"].split(os.sep)[-1]).split("-")[
            -1
        ]  # read how many training siterations that corresponds to.

        # name for deeplabcut net (based on its parameters)
        DLCscorer, DLCscorerlegacy = auxiliaryfunctions.GetScorerName(
            cfg, shuffle, trainFraction, trainingsiterations, modelprefix=modelprefix
        )
        if not returnjustfns:
            print(
                "Retrieving ",
                DLCscorer,
                " with # of trainingiterations:",
                trainingsiterations,
            )

        (
            notanalyzed,
            resultsfilename,
            DLCscorer,
        ) = auxiliaryfunctions.CheckifNotEvaluated(
            str(evaluationfolder), DLCscorer, DLCscorerlegacy, Snapshots[snapindex]
        )
        # resultsfilename=os.path.join(str(evaluationfolder),DLCscorer + '-' + str(Snapshots[snapindex])+  '.h5') # + '-' + str(snapshot)+  ' #'-' + Snapshots[snapindex]+  '.h5')
        print(resultsfilename)
        resultsfns.append(resultsfilename)
        if not returnjustfns:
            if not notanalyzed and os.path.isfile(resultsfilename):  # data exists..
                DataMachine = pd.read_hdf(resultsfilename, "df_with_missing")
                DataCombined = pd.concat([Data.T, DataMachine.T], axis=0).T
                RMSE, RMSEpcutoff = pairwisedistances(
                    DataCombined,
                    cfg["scorer"],
                    DLCscorer,
                    cfg["pcutoff"],
                    comparisonbodyparts,
                )

                testerror = np.nanmean(RMSE.iloc[testIndices].values.flatten())
                trainerror = np.nanmean(RMSE.iloc[trainIndices].values.flatten())
                testerrorpcutoff = np.nanmean(
                    RMSEpcutoff.iloc[testIndices].values.flatten()
                )
                trainerrorpcutoff = np.nanmean(
                    RMSEpcutoff.iloc[trainIndices].values.flatten()
                )
                if show_errors == True:
                    print(
                        "Results for",
                        trainingsiterations,
                        " training iterations:",
                        int(100 * trainFraction),
                        shuffle,
                        "train error:",
                        np.round(trainerror, 2),
                        "pixels. Test error:",
                        np.round(testerror, 2),
                        " pixels.",
                    )
                    print(
                        "With pcutoff of",
                        cfg["pcutoff"],
                        " train error:",
                        np.round(trainerrorpcutoff, 2),
                        "pixels. Test error:",
                        np.round(testerrorpcutoff, 2),
                        "pixels",
                    )
                    print("Snapshot", Snapshots[snapindex])

                r = [
                    trainingsiterations,
                    int(100 * trainFraction),
                    shuffle,
                    np.round(trainerror, 2),
                    np.round(testerror, 2),
                    cfg["pcutoff"],
                    np.round(trainerrorpcutoff, 2),
                    np.round(testerrorpcutoff, 2),
                    Snapshots[snapindex],
                    scale,
                    dlc_cfg["net_type"],
                ]
                results.append(r)
            else:
                print("Model not trained/evaluated!")
            if fulldata == True:
                DATA.append(
                    [
                        DataMachine,
                        Data,
                        data,
                        trainIndices,
                        testIndices,
                        trainFraction,
                        DLCscorer,
                        comparisonbodyparts,
                        cfg,
                        evaluationfolder,
                        Snapshots[snapindex],
                    ]
                )

    os.chdir(start_path)
    if returnjustfns:
        return resultsfns
    else:
        if fulldata == True:
            return DATA, results
        else:
            return results


def evaluate_network(
    config,
    Shuffles=[1],
    trainingsetindex=0,
    plotting=None,
    show_errors=True,
    comparisonbodyparts="all",
    gputouse=None,
    rescale=False,
    modelprefix="",
    c_engine=False,
):
    """

    Evaluates the network based on the saved models at different stages of the training network.\n
    The evaluation results are stored in the .h5 and .csv file under the subdirectory 'evaluation_results'.
    Change the snapshotindex parameter in the config file to 'all' in order to evaluate all the saved models.
    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    Shuffles: list, optional
        List of integers specifying the shuffle indices of the training dataset. The default is [1]

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml). This
        variable can also be set to "all".

    plotting: bool, optional
        Plots the predictions on the train and test images. The default is ``False``; if provided it must be either ``True`` or ``False``

    show_errors: bool, optional
        Display train and test errors. The default is `True``

    comparisonbodyparts: list of bodyparts, Default is "all".
        The average error will be computed for those body parts only (Has to be a subset of the body parts).

    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU put None.
        See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    rescale: bool, default False
        Evaluate the model at the 'global_scale' variable (as set in the test/pose_config.yaml file for a particular project). I.e. every
        image will be resized according to that scale and prediction will be compared to the resized ground truth. The error will be reported
        in pixels at rescaled to the *original* size. I.e. For a [200,200] pixel image evaluated at global_scale=.5, the predictions are calculated
        on [100,100] pixel images, compared to 1/2*ground truth and this error is then multiplied by 2!. The evaluation images are also shown for the
        original size!

    Examples
    --------
    If you do not want to plot
    >>> deeplabcut.evaluate_network('/analysis/project/reaching-task/config.yaml', Shuffles=[1])
    --------
    If you want to plot
    >>> deeplabcut.evaluate_network('/analysis/project/reaching-task/config.yaml',Shuffles=[1],True)

    """
    import os

    start_path = os.getcwd()
    from deeplabcut.utils import auxiliaryfunctions

    cfg = auxiliaryfunctions.read_config(config)

    if cfg.get("multianimalproject", False):
        from deeplabcut.pose_estimation_tensorflow.evaluate_multianimal import (
            evaluate_multianimal_full,
        )

        # TODO: Make this code not so redundant!
        evaluate_multianimal_full(
            config=config,
            Shuffles=Shuffles,
            trainingsetindex=trainingsetindex,
            plotting=plotting,
            comparisonbodyparts=comparisonbodyparts,
            gputouse=gputouse,
            modelprefix=modelprefix,
            c_engine=c_engine,
        )
    else:
        from deeplabcut.utils.auxfun_videos import imread, imresize
        from deeplabcut.pose_estimation_tensorflow.nnet import predict
        from deeplabcut.pose_estimation_tensorflow.config import load_config
        from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import (
            data_to_input,
        )
        from deeplabcut.utils import auxiliaryfunctions
        import tensorflow as tf

        if "TF_CUDNN_USE_AUTOTUNE" in os.environ:
            del os.environ[
                "TF_CUDNN_USE_AUTOTUNE"
            ]  # was potentially set during training

        tf.reset_default_graph()
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  #
        #    tf.logging.set_verbosity(tf.logging.WARN)

        start_path = os.getcwd()
        # Read file path for pose_config file. >> pass it on
        cfg = auxiliaryfunctions.read_config(config)
        if gputouse is not None:  # gpu selectinon
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gputouse)

        if trainingsetindex == "all":
            TrainingFractions = cfg["TrainingFraction"]
        else:
            if (
                trainingsetindex < len(cfg["TrainingFraction"])
                and trainingsetindex >= 0
            ):
                TrainingFractions = [cfg["TrainingFraction"][int(trainingsetindex)]]
            else:
                raise Exception(
                    "Please check the trainingsetindex! ",
                    trainingsetindex,
                    " should be an integer from 0 .. ",
                    int(len(cfg["TrainingFraction"]) - 1),
                )

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

        # Get list of body parts to evaluate network for
        comparisonbodyparts = auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(
            cfg, comparisonbodyparts
        )
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

                # change batch size, if it was edited during analysis!
                dlc_cfg["batch_size"] = 1  # in case this was edited for analysis.

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
                try:  # check if any where found?
                    Snapshots[0]
                except IndexError:
                    raise FileNotFoundError(
                        "Snapshots not found! It seems the dataset for shuffle %s and trainFraction %s is not trained.\nPlease train it before evaluating.\nUse the function 'train_network' to do so."
                        % (shuffle, trainFraction)
                    )

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
                    raise ValueError(
                        "Invalid choice, only -1 (last), any integer up to last, or all (as string)!"
                    )

                final_result = []

                ########################### RESCALING (to global scale)
                if rescale == True:
                    scale = dlc_cfg["global_scale"]
                    Data = (
                        pd.read_hdf(
                            os.path.join(
                                cfg["project_path"],
                                str(trainingsetfolder),
                                "CollectedData_" + cfg["scorer"] + ".h5",
                            ),
                            "df_with_missing",
                        )
                        * scale
                    )
                else:
                    scale = 1

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

                    # Name for deeplabcut net (based on its parameters)
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
                    if notanalyzed:
                        # Specifying state of model (snapshot / training state)
                        sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)
                        Numimages = len(Data.index)
                        PredicteData = np.zeros(
                            (Numimages, 3 * len(dlc_cfg["all_joints_names"]))
                        )
                        print("Analyzing data...")
                        for imageindex, imagename in tqdm(enumerate(Data.index)):
                            image = imread(
                                os.path.join(cfg["project_path"], imagename), mode="RGB"
                            )
                            if scale != 1:
                                image = imresize(image, scale)

                            image_batch = data_to_input(image)
                            # Compute prediction with the CNN
                            outputs_np = sess.run(
                                outputs, feed_dict={inputs: image_batch}
                            )
                            scmap, locref = predict.extract_cnn_output(
                                outputs_np, dlc_cfg
                            )

                            # Extract maximum scoring location from the heatmap, assume 1 person
                            pose = predict.argmax_pose_predict(
                                scmap, locref, dlc_cfg.stride
                            )
                            PredicteData[
                                imageindex, :
                            ] = (
                                pose.flatten()
                            )  # NOTE: thereby     cfg_test['all_joints_names'] should be same order as bodyparts!

                        sess.close()  # closes the current tf session

                        index = pd.MultiIndex.from_product(
                            [
                                [DLCscorer],
                                dlc_cfg["all_joints_names"],
                                ["x", "y", "likelihood"],
                            ],
                            names=["scorer", "bodyparts", "coords"],
                        )

                        # Saving results
                        DataMachine = pd.DataFrame(
                            PredicteData, columns=index, index=Data.index.values
                        )
                        DataMachine.to_hdf(
                            resultsfilename, "df_with_missing", format="table", mode="w"
                        )

                        print(
                            "Done and results stored for snapshot: ",
                            Snapshots[snapindex],
                        )
                        DataCombined = pd.concat(
                            [Data.T, DataMachine.T], axis=0, sort=False
                        ).T

                        RMSE, RMSEpcutoff = pairwisedistances(
                            DataCombined,
                            cfg["scorer"],
                            DLCscorer,
                            cfg["pcutoff"],
                            comparisonbodyparts,
                        )
                        testerror = np.nanmean(RMSE.iloc[testIndices].values.flatten())
                        trainerror = np.nanmean(
                            RMSE.iloc[trainIndices].values.flatten()
                        )
                        testerrorpcutoff = np.nanmean(
                            RMSEpcutoff.iloc[testIndices].values.flatten()
                        )
                        trainerrorpcutoff = np.nanmean(
                            RMSEpcutoff.iloc[trainIndices].values.flatten()
                        )
                        results = [
                            trainingsiterations,
                            int(100 * trainFraction),
                            shuffle,
                            np.round(trainerror, 2),
                            np.round(testerror, 2),
                            cfg["pcutoff"],
                            np.round(trainerrorpcutoff, 2),
                            np.round(testerrorpcutoff, 2),
                        ]
                        final_result.append(results)

                        if show_errors:
                            print(
                                "Results for",
                                trainingsiterations,
                                " training iterations:",
                                int(100 * trainFraction),
                                shuffle,
                                "train error:",
                                np.round(trainerror, 2),
                                "pixels. Test error:",
                                np.round(testerror, 2),
                                " pixels.",
                            )
                            print(
                                "With pcutoff of",
                                cfg["pcutoff"],
                                " train error:",
                                np.round(trainerrorpcutoff, 2),
                                "pixels. Test error:",
                                np.round(testerrorpcutoff, 2),
                                "pixels",
                            )
                            if scale != 1:
                                print(
                                    "The predictions have been calculated for rescaled images (and rescaled ground truth). Scale:",
                                    scale,
                                )
                            print(
                                "Thereby, the errors are given by the average distances between the labels by DLC and the scorer."
                            )

                        if plotting == True:
                            print("Plotting...")
                            foldername = os.path.join(
                                str(evaluationfolder),
                                "LabeledImages_"
                                + DLCscorer
                                + "_"
                                + Snapshots[snapindex],
                            )
                            auxiliaryfunctions.attempttomakefolder(foldername)
                            Plotting(
                                cfg,
                                comparisonbodyparts,
                                DLCscorer,
                                trainIndices,
                                DataCombined * 1.0 / scale,
                                foldername,
                            )  # Rescaling coordinates to have figure in original size!

                        tf.reset_default_graph()
                        # print(final_result)
                    else:
                        DataMachine = pd.read_hdf(resultsfilename, "df_with_missing")
                        if plotting == True:
                            DataCombined = pd.concat(
                                [Data.T, DataMachine.T], axis=0, sort=False
                            ).T
                            print(
                                "Plotting...(attention scale might be inconsistent in comparison to when data was analyzed; i.e. if you used rescale)"
                            )
                            foldername = os.path.join(
                                str(evaluationfolder),
                                "LabeledImages_"
                                + DLCscorer
                                + "_"
                                + Snapshots[snapindex],
                            )
                            auxiliaryfunctions.attempttomakefolder(foldername)
                            Plotting(
                                cfg,
                                comparisonbodyparts,
                                DLCscorer,
                                trainIndices,
                                DataCombined * 1.0 / scale,
                                foldername,
                            )

                if len(final_result) > 0:  # Only append if results were calculated
                    make_results_file(final_result, evaluationfolder, DLCscorer)
                    print(
                        "The network is evaluated and the results are stored in the subdirectory 'evaluation_results'."
                    )
                    print(
                        "If it generalizes well, choose the best model for prediction and update the config file with the appropriate index for the 'snapshotindex'.\nUse the function 'analyze_video' to make predictions on new videos."
                    )
                    print(
                        "Otherwise consider retraining the network (see DeepLabCut workflow Fig 2)"
                    )

    # returning to intial folder
    os.chdir(str(start_path))


def make_results_file(final_result, evaluationfolder, DLCscorer):
    """
    Makes result file in csv format and saves under evaluation_results directory.
    If the file exists (typically, when the network has already been evaluated),
    newer results are appended to it.
    """
    col_names = [
        "Training iterations:",
        "%Training dataset",
        "Shuffle number",
        " Train error(px)",
        " Test error(px)",
        "p-cutoff used",
        "Train error with p-cutoff",
        "Test error with p-cutoff",
    ]
    df = pd.DataFrame(final_result, columns=col_names)
    output_path = os.path.join(str(evaluationfolder), DLCscorer + "-results.csv")
    if os.path.exists(output_path):
        temp = pd.read_csv(output_path, index_col=0)
        df = pd.concat((df, temp)).reset_index(drop=True)

    df.to_csv(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    cli_args = parser.parse_args()
