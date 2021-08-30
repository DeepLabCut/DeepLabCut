"""
DeepLabCut 2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""


import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.color
from scipy.spatial import cKDTree
from skimage import io
from skimage.util import img_as_ubyte
from tqdm import tqdm

from deeplabcut.pose_estimation_tensorflow.core.evaluate import make_results_file
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.lib import crossvalutils
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


def _find_closest_neighbors(xy_true, xy_pred, k=5):
    n_preds = xy_pred.shape[0]
    tree = cKDTree(xy_pred)
    dist, inds = tree.query(xy_true, k=k)
    idx = np.argsort(dist[:, 0])
    neighbors = np.full(len(xy_true), -1, dtype=int)
    picked = set()
    for i, ind in enumerate(inds[idx]):
        for j in ind:
            if j not in picked:
                picked.add(j)
                neighbors[idx[i]] = j
                break
        if len(picked) == n_preds:
            break
    return neighbors


def _calc_prediction_error(data):
    _ = data.pop("metadata", None)
    dists = []
    for n, dict_ in enumerate(tqdm(data.values())):
        gt = np.concatenate(dict_["groundtruth"][1])
        xy = np.concatenate(dict_["prediction"]["coordinates"][0])
        p = np.concatenate(dict_["prediction"]["confidence"])
        neighbors = _find_closest_neighbors(gt, xy)
        found = neighbors != -1
        gt2 = gt[found]
        xy2 = xy[neighbors[found]]
        dists.append(np.c_[np.linalg.norm(gt2 - xy2, axis=1), p[neighbors[found]]])
    return dists


def _calc_train_test_error(data, metadata, pcutoff=0.3):
    train_inds = set(metadata["data"]["trainIndices"])
    dists = _calc_prediction_error(data)
    dists_train, dists_test = [], []
    for n, dist in enumerate(dists):
        if n in train_inds:
            dists_train.append(dist)
        else:
            dists_test.append(dist)
    dists_train = np.concatenate(dists_train)
    dists_test = np.concatenate(dists_test)
    error_train = np.nanmean(dists_train[:, 0])
    error_train_cut = np.nanmean(dists_train[dists_train[:, 1] >= pcutoff, 0])
    error_test = np.nanmean(dists_test[:, 0])
    error_test_cut = np.nanmean(dists_test[dists_test[:, 1] >= pcutoff, 0])
    return error_train, error_test, error_train_cut, error_test_cut


def evaluate_multianimal_full(
    config,
    Shuffles=[1],
    trainingsetindex=0,
    plotting=None,
    show_errors=True,
    comparisonbodyparts="all",
    gputouse=None,
    modelprefix="",
):
    from deeplabcut.pose_estimation_tensorflow.core import (
        predict,
        predict_multianimal as predictma,
    )
    from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal

    import tensorflow as tf

    if "TF_CUDNN_USE_AUTOTUNE" in os.environ:
        del os.environ["TF_CUDNN_USE_AUTOTUNE"]  # was potentially set during training

    tf.compat.v1.reset_default_graph()
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
        )
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

            stride = dlc_cfg["stride"]
            # Ignore best edges possibly defined during a prior evaluation
            _ = dlc_cfg.pop("paf_best", None)
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

                    data_path = resultsfilename.split(".h5")[0] + "_full.pickle"
                    if os.path.isfile(data_path):
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
                            fig, ax = visualization.create_minimal_figure()

                        sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)

                        PredicteData = {}
                        dist = np.full((len(Data), len(all_bpts)), np.nan)
                        conf = np.full_like(dist, np.nan)
                        print("Analyzing data...")
                        for imageindex, imagename in tqdm(enumerate(Data.index)):
                            image_path = os.path.join(cfg["project_path"], imagename)
                            image = io.imread(image_path)
                            if image.ndim == 2 or image.shape[-1] == 1:
                                image = skimage.color.gray2rgb(image)
                            frame = img_as_ubyte(image)

                            GT = Data.iloc[imageindex]
                            if not GT.any():
                                continue
                            df = GT.unstack("coords").reindex(joints, level="bodyparts")

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

                            # Form 2D array of shape (n_rows, 4) where the last dimension
                            # is (sample_index, peak_y, peak_x, bpt_index) to slice the PAFs.
                            temp = df.reset_index(level="bodyparts").dropna()
                            temp["bodyparts"].replace(
                                dict(zip(joints, range(len(joints)))),
                                inplace=True,
                            )
                            temp["sample"] = 0
                            peaks_gt = temp.loc[:, ["sample", "y", "x", "bodyparts"]].to_numpy()
                            peaks_gt[:, 1:3] = (peaks_gt[:, 1:3] - stride // 2) / stride
                            pred = predictma.predict_batched_peaks_and_costs(
                                dlc_cfg,
                                np.expand_dims(frame, axis=0),
                                sess,
                                inputs,
                                outputs,
                                peaks_gt.astype(int),
                            )
                            if not pred:
                                continue
                            else:
                                pred = pred[0]

                            PredicteData[imagename] = {}
                            PredicteData[imagename]["index"] = imageindex
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
                                n_joint = joints.index(bpt)
                                xy = coords_pred[n_joint]
                                if inds_gt.size and xy.size:
                                    # Pick the predictions closest to ground truth,
                                    # rather than the ones the model has most confident in
                                    xy_gt_values = xy_gt.iloc[inds_gt].values
                                    neighbors = _find_closest_neighbors(xy_gt_values, xy, k=3)
                                    found = neighbors != -1
                                    min_dists = np.linalg.norm(
                                        xy_gt_values[found] - xy[neighbors[found]], axis=1,
                                    )
                                    inds = np.flatnonzero(all_bpts == bpt)
                                    sl = imageindex, inds[inds_gt[found]]
                                    dist[sl] = min_dists
                                    conf[sl] = probs_pred[n_joint][neighbors[found]].squeeze()

                            if plotting:
                                temp_xy = GT.unstack("bodyparts")[joints].values
                                gt = temp_xy.reshape(
                                    (-1, 2, temp_xy.shape[1])
                                ).T.swapaxes(1, 2)
                                h, w, _ = np.shape(frame)
                                fig.set_size_inches(w / 100, h / 100)
                                ax.set_xlim(0, w)
                                ax.set_ylim(0, h)
                                ax.invert_yaxis()
                                ax = visualization.make_multianimal_labeled_image(
                                    frame,
                                    gt,
                                    coords_pred,
                                    probs_pred,
                                    colors,
                                    cfg["dotsize"],
                                    cfg["alphavalue"],
                                    cfg["pcutoff"],
                                    ax=ax,
                                )
                                visualization.save_labeled_frame(
                                    fig,
                                    image_path,
                                    foldername,
                                    imageindex in trainIndices,
                                )
                                visualization.erase_artists(ax)

                        sess.close()  # closes the current tf session

                        # Compute all distance statistics
                        df_dist = pd.DataFrame(dist, columns=df.index)
                        df_conf = pd.DataFrame(conf, columns=df.index)
                        df_joint = pd.concat(
                            [df_dist, df_conf],
                            keys=["rmse", "conf"],
                            names=["metrics"],
                            axis=1,
                        )
                        df_joint = df_joint.reorder_levels(
                            list(np.roll(df_joint.columns.names, -1)), axis=1
                        )
                        df_joint.sort_index(
                            axis=1,
                            level=["individuals", "bodyparts"],
                            ascending=[True, True],
                            inplace=True,
                        )
                        write_path = os.path.join(
                            evaluationfolder, f"dist_{trainingsiterations}.csv"
                        )
                        df_joint.to_csv(write_path)

                        # Calculate overall prediction error
                        error = df_joint.xs("rmse", level="metrics", axis=1)
                        mask = (
                            df_joint.xs("conf", level="metrics", axis=1)
                            >= cfg["pcutoff"]
                        )
                        error_masked = error[mask]
                        error_train = np.nanmean(error.iloc[trainIndices])
                        error_train_cut = np.nanmean(error_masked.iloc[trainIndices])
                        error_test = np.nanmean(error.iloc[testIndices])
                        error_test_cut = np.nanmean(error_masked.iloc[testIndices])
                        results = [
                            trainingsiterations,
                            int(100 * trainFraction),
                            shuffle,
                            np.round(error_train, 2),
                            np.round(error_test, 2),
                            cfg["pcutoff"],
                            np.round(error_train_cut, 2),
                            np.round(error_test_cut, 2),
                        ]
                        final_result.append(results)

                        if show_errors:
                            string = (
                                "Results for {} training iterations: {}, shuffle {}:\n"
                                "Train error: {} pixels. Test error: {} pixels.\n"
                                "With pcutoff of {}:\n"
                                "Train error: {} pixels. Test error: {} pixels."
                            )
                            print(string.format(*results))

                            print("##########################################")
                            print(
                                "Average Euclidean distance to GT per individual (in pixels)"
                            )
                            print(
                                error_masked.groupby("individuals", axis=1)
                                .mean()
                                .mean()
                                .to_string()
                            )
                            print(
                                "Average Euclidean distance to GT per bodypart (in pixels)"
                            )
                            print(
                                error_masked.groupby("bodyparts", axis=1)
                                .mean()
                                .mean()
                                .to_string()
                            )

                        PredicteData["metadata"] = {
                            "nms radius": dlc_cfg["nmsradius"],
                            "minimal confidence": dlc_cfg["minconfidence"],
                            "sigma": dlc_cfg.get("sigma", 1),
                            "PAFgraph": dlc_cfg["partaffinityfield_graph"],
                            "PAFinds": np.arange(len(dlc_cfg["partaffinityfield_graph"])),
                            "all_joints": [
                                [i] for i in range(len(dlc_cfg["all_joints"]))
                            ],
                            "all_joints_names": [
                                dlc_cfg["all_joints_names"][i]
                                for i in range(len(dlc_cfg["all_joints"]))
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
                        _ = auxfun_multianimal.SaveFullMultiAnimalData(
                            PredicteData, metadata, resultsfilename
                        )

                        tf.compat.v1.reset_default_graph()

                    # Skip data-driven skeleton selection unless
                    # the model was trained on the full graph.
                    n_multibpts = len(cfg["multianimalbodyparts"])
                    max_n_edges = n_multibpts * (n_multibpts - 1) // 2
                    n_edges = len(dlc_cfg["partaffinityfield_graph"])
                    if n_edges == max_n_edges:
                        print("Selecting best skeleton...")
                        n_graphs = 10
                        paf_inds = None
                    else:
                        n_graphs = 1
                        paf_inds = [list(range(n_edges))]
                    results, paf_scores = crossvalutils.cross_validate_paf_graphs(
                        config,
                        str(path_test_config).replace("pose_", "inference_"),
                        data_path,
                        data_path.replace("_full.", "_meta."),
                        n_graphs=n_graphs,
                        paf_inds=paf_inds,
                    )
                    df = results[1].copy()
                    df.loc(axis=0)[('mAP', 'mean')] = [d['mAP'] for d in results[2]]
                    df.loc(axis=0)[('mAR', 'mean')] = [d['mAR'] for d in results[2]]
                    with open(data_path.replace("_full.", "_map."), "wb") as file:
                        pickle.dump((df, paf_scores), file)

                if len(final_result) > 0:  # Only append if results were calculated
                    make_results_file(final_result, evaluationfolder, DLCscorer)

    os.chdir(str(start_path))
