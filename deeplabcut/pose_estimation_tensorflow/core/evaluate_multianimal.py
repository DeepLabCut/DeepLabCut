"""
DeepLabCut 2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import imgaug.augmenters as iaa
import os
import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

from deeplabcut.pose_estimation_tensorflow.core import (
    predict,
    predict_multianimal as predictma,
)
from deeplabcut import auxiliaryfunctions
from deeplabcut.generate_training_dataset.trainingsetmanipulation import read_image_shape_fast
from deeplabcut.pose_estimation_tensorflow.core.evaluate import make_results_file
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.lib import crossvalutils, inferenceutils
from deeplabcut.utils import visualization, auxfun_videos
from deeplabcut.utils.conversioncode import guarantee_multiindex_rows
from imgaug import Keypoint, KeypointsOnImage


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
    plotting=False,
    show_errors=True,
    comparisonbodyparts="all",
    gputouse=None,
    modelprefix="",
):
    from deeplabcut.pose_estimation_tensorflow.core import (
        predict,
        predict_multianimal as predictma,
    )
    from deeplabcut.utils import (
        auxiliaryfunctions,
        auxfun_multianimal,
        auxfun_videos,
        conversioncode,
    )

    import tensorflow as tf

    if "TF_CUDNN_USE_AUTOTUNE" in os.environ:
        del os.environ["TF_CUDNN_USE_AUTOTUNE"]  # was potentially set during training

    tf.compat.v1.reset_default_graph()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  #
    if gputouse is not None:  # gpu selectinon
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gputouse)

    start_path = os.getcwd()

    if plotting is True:
        plotting = "bodypart"

    ##################################################
    # Load data...
    ##################################################
    cfg = auxiliaryfunctions.read_config(config)
    if trainingsetindex == "all":
        TrainingFractions = cfg["TrainingFraction"]
    else:
        TrainingFractions = [cfg["TrainingFraction"][trainingsetindex]]

    # Loading human annotated data
    trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(cfg)
    path_gt_data = os.path.join(
        cfg["project_path"],
        str(trainingsetfolder),
        "CollectedData_" + cfg["scorer"] + ".h5",
    )
    gt_data = _format_gt_data(path_gt_data)
    images = [
        os.path.join(cfg["project_path"], img)
        for img in gt_data['annotations']
    ]
    image_shapes = [read_image_shape_fast(im)[1:] for im in images]

    gt_df = pd.read_hdf(path_gt_data)
    gt_columns = gt_df.columns

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

            pipeline = iaa.Sequential(random_order=False)
            pre_resize = dlc_cfg.get("pre_resize")
            if pre_resize:
                width, height = pre_resize
                pipeline.add(iaa.Resize({"height": height, "width": width}))

                # Augment GT data
                for i, (k, v) in enumerate(gt_data["annotations"].items()):
                    shape_ = v.shape
                    arr = np.concatenate(v)
                    kpts = KeypointsOnImage(
                        [Keypoint(xy[0], xy[1]) for xy in arr],
                        shape=image_shapes[i],
                    )
                    kpts = pipeline(keypoints=kpts)
                    v = np.concatenate([kpt.xy for kpt in kpts]).reshape(shape_)
                    gt_data["annotations"][k] = v

                for i in range(len(gt_df)):
                    arr = gt_df.iloc[i].to_numpy().reshape((-1, 2)).astype(float)
                    kpts = KeypointsOnImage(
                        [Keypoint(xy[0], xy[1]) for xy in arr],
                        shape=image_shapes[i],
                    )
                    kpts = pipeline(keypoints=kpts)
                    gt_df.iloc[i][:] = np.concatenate([kpt.xy for kpt in kpts]).flatten()

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
                    foldername = os.path.join(
                        str(evaluationfolder),
                        "LabeledImages_" + DLCscorer + "_" + Snapshots[snapindex],
                    )
                    if os.path.isfile(data_path):
                        print("Model already evaluated.", resultsfilename)
                    else:
                        preds = _eval_images(
                            images,
                            dlc_cfg,
                            aug_pipeline=pipeline,
                        )
                        tf.compat.v1.reset_default_graph()

                        PredicteData = {}
                        for imageindex, image in enumerate(images):
                            pred = preds["predictions"][image]
                            if not pred:
                                continue

                            imagename = tuple(
                                os.path.relpath(image, cfg["project_path"]).split(os.sep)
                            )
                            df = (
                                gt_df
                                .iloc[imageindex]
                                .unstack("coords")
                                .reindex(joints, level="bodyparts")
                            )
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
                            PredicteData[imagename]["prediction"] = pred
                            # TODO Get rid of ground truth data stored in the _full.pickle
                            PredicteData[imagename]["groundtruth"] = [
                                groundtruthidentity,
                                groundtruthcoordinates,
                                gt_df.iloc[imageindex],
                            ]

                        PredicteData["metadata"] = {
                            "nms radius": dlc_cfg["nmsradius"],
                            "minimal confidence": dlc_cfg["minconfidence"],
                            "sigma": dlc_cfg.get("sigma", 1),
                            "PAFgraph": dlc_cfg["partaffinityfield_graph"],
                            "PAFinds": np.arange(
                                len(dlc_cfg["partaffinityfield_graph"])
                            ),
                            "all_joints": [
                                [i] for i in range(len(dlc_cfg["all_joints"]))
                            ],
                            "all_joints_names": [
                                dlc_cfg["all_joints_names"][i]
                                for i in range(len(dlc_cfg["all_joints"]))
                            ],
                            "stride": dlc_cfg.get("stride", 8),
                        }
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
                        print(
                            "Done and results stored for snapshot: ",
                            Snapshots[snapindex],
                        )

                        # Compute and format prediction errors
                        errors = calc_prediction_errors(preds, gt_data)
                        cols = (
                            gt_columns
                            .set_levels(["rmse", "conf"], level="coords")
                            .set_names(list(gt_columns.names)[:-1] + ["metrics"])
                        )
                        n_unique = gt_data["metadata"]["n_unique"]
                        if n_unique == 0:
                            temp = errors.reshape((errors.shape[0], -1))
                        else:
                            temp = np.c_[
                                errors[:, :-1, :-n_unique].reshape((errors.shape[0], -1)),
                                errors[:, -1, -n_unique:].reshape((errors.shape[0], -1)),
                            ]
                        df_joint = pd.DataFrame(temp, columns=cols)
                        df_joint.to_csv(
                            os.path.join(
                                evaluationfolder, f"dist_{trainingsiterations}.csv"
                            )
                        )
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
                                "Results for {} training iterations, training fraction of {}, and shuffle {}:\n"
                                "Train error: {} pixels. Test error: {} pixels.\n"
                                "With pcutoff of {}:\n"
                                "Train error: {} pixels. Test error: {} pixels."
                            )
                            print(string.format(*results))

                            print("##########################################")
                            print(
                                "Average Euclidean distance to GT per individual (in pixels; test-only)"
                            )
                            print(
                                error_masked.iloc[testIndices]
                                .groupby("individuals", axis=1)
                                .mean()
                                .mean()
                                .reindex(cfg['individuals'])
                                .to_string()
                            )
                            print(
                                "Average Euclidean distance to GT per bodypart (in pixels; test-only)"
                            )
                            print(
                                error_masked.iloc[testIndices]
                                .groupby("bodyparts", axis=1)
                                .mean()
                                .mean()
                                .reindex(cfg["multianimalbodyparts"] + cfg["uniquebodyparts"])
                                .to_string()
                            )

                        if plotting == "bodypart":
                            preds_train = {
                                "metadata": preds["metadata"],
                                "predictions": {
                                    images[i]: preds["predictions"][images[i]]
                                    for i in trainIndices
                                }
                            }
                            preds_test = {
                                "metadata": preds["metadata"],
                                "predictions": {
                                    images[i]: preds["predictions"][images[i]]
                                    for i in testIndices
                                }
                            }
                            visualization.visualize_predictions(
                                preds_train,
                                gt_data,
                                cfg["pcutoff"],
                                cfg["dotsize"],
                                cfg["alphavalue"],
                                cfg["colormap"],
                                destfolder=os.path.join(foldername, 'train'),
                                aug_pipeline=pipeline,
                            )
                            visualization.visualize_predictions(
                                preds_test,
                                gt_data,
                                cfg["pcutoff"],
                                cfg["dotsize"],
                                cfg["alphavalue"],
                                cfg["colormap"],
                                destfolder=os.path.join(foldername, 'test'),
                                aug_pipeline=pipeline,
                            )

                    n_multibpts = len(cfg["multianimalbodyparts"])
                    if n_multibpts == 1:
                        continue

                    # Skip data-driven skeleton selection unless
                    # the model was trained on the full graph.
                    max_n_edges = n_multibpts * (n_multibpts - 1) // 2
                    n_edges = len(dlc_cfg["partaffinityfield_graph"])
                    if n_edges == max_n_edges:
                        print("Selecting best skeleton...")
                        n_graphs = 10
                        paf_inds = None
                    else:
                        n_graphs = 1
                        paf_inds = [list(range(n_edges))]
                    (
                        results,
                        paf_scores,
                        best_assembler,
                    ) = crossvalutils.cross_validate_paf_graphs(
                        config,
                        str(path_test_config).replace("pose_", "inference_"),
                        data_path,
                        data_path.replace("_full.", "_meta."),
                        n_graphs=n_graphs,
                        paf_inds=paf_inds,
                        oks_sigma=dlc_cfg.get("oks_sigma", 0.1),
                        margin=dlc_cfg.get("bbox_margin", 0),
                        symmetric_kpts=dlc_cfg.get("symmetric_kpts"),
                    )
                    df = results[1].copy()
                    df.loc(axis=0)[("mAP_train", "mean")] = [
                        d[0]["mAP"] for d in results[2]
                    ]
                    df.loc(axis=0)[("mAR_train", "mean")] = [
                        d[0]["mAR"] for d in results[2]
                    ]
                    df.loc(axis=0)[("mAP_test", "mean")] = [
                        d[1]["mAP"] for d in results[2]
                    ]
                    df.loc(axis=0)[("mAR_test", "mean")] = [
                        d[1]["mAR"] for d in results[2]
                    ]
                    with open(data_path.replace("_full.", "_map."), "wb") as file:
                        pickle.dump((df, paf_scores), file)

                    if plotting == "individual":
                        visualization.visualize_predictions(
                            best_assembler,
                            gt_data,
                            cfg["pcutoff"],
                            cfg["dotsize"],
                            cfg["alphavalue"],
                            cfg["colormap"],
                            destfolder=foldername,
                            aug_pipeline=pipeline,
                        )

                if len(final_result) > 0:  # Only append if results were calculated
                    make_results_file(final_result, evaluationfolder, DLCscorer)

    os.chdir(str(start_path))


def _eval_single_image(
    image_path,
    test_cfg,
    pose_setup=None,
    aug_pipeline=None,
    to_coco=False,
):
    if pose_setup is None:
        pose_setup = predict.setup_pose_prediction(test_cfg)
    frame = auxfun_videos.imread(image_path, mode="skimage")
    if aug_pipeline is not None:
        frame = aug_pipeline(images=[frame])[0]
    preds = predictma.predict_batched_peaks_and_costs(
        test_cfg,
        np.expand_dims(frame, axis=0),
        pose_setup.session,
        pose_setup.inputs,
        pose_setup.outputs,
    )
    if not preds:
        return {}
    preds = preds[0]
    if to_coco:
        # TODO
        ...
    return preds


def _eval_images(
    image_paths,
    test_cfg,
    snapshot_path="",
    aug_pipeline=None,
    to_coco=False,
):
    test_cfg["batch_size"] = 1
    if snapshot_path:
        test_cfg["init_weights"] = snapshot_path
    pose_setup = predict.setup_pose_prediction(test_cfg)
    data = defaultdict(dict)
    print("Network evaluation underway...")
    for image_path in tqdm(image_paths):
        preds = _eval_single_image(
            image_path, test_cfg, pose_setup, aug_pipeline, to_coco,
        )
        data["predictions"][image_path] = preds
    data["metadata"] = {"keypoints": test_cfg["all_joints_names"]}
    pose_setup.session.close()
    return data


def _format_gt_data(h5file):
    df = pd.read_hdf(h5file)

    def _get_unique_level_values(header, level):
        return header.get_level_values(level).unique().to_list()

    animals = _get_unique_level_values(df.columns, "individuals")
    kpts = _get_unique_level_values(df.columns, "bodyparts")
    try:
        n_unique = len(_get_unique_level_values(
            df.xs("single", level="individuals", axis=1).columns, "bodyparts")
        )
    except KeyError:
        n_unique = 0
    guarantee_multiindex_rows(df)
    file_paths = [os.path.join(*row) for row in df.index.to_list()]
    temp = (df
            .stack("individuals", dropna=False)
            .reindex(animals, level="individuals")
            .reindex(kpts, level="bodyparts", axis=1))
    data = temp.to_numpy().reshape((len(file_paths), len(animals), -1, 2))
    meta = {"animals": animals, "keypoints": kpts, "n_unique": n_unique}
    return {
        "annotations": dict(zip(file_paths, data)),
        "metadata": meta,
    }


def calc_prediction_errors(preds, gt):
    kpts_gt = gt["metadata"]["keypoints"]
    kpts_pred = preds["metadata"]["keypoints"]
    map_ = {kpts_gt.index(kpt): i for i, kpt in enumerate(kpts_pred)}
    annot = gt["annotations"]

    # Map image paths from predicted data to GT as the first are typically
    # absolute whereas the latter are relative to the project path.
    map_images = auxiliaryfunctions._map(list(preds["predictions"]), list(annot))

    errors = np.full(
        (
            len(preds["predictions"]),
            len(gt["metadata"]["animals"]),
            len(kpts_gt),
            2,  # Hold distance to GT and confidence
        ),
        np.nan,
    )
    for n, (path, preds_) in enumerate(preds["predictions"].items()):
        if not preds_:
            continue
        xy_gt = annot[map_images[path]].swapaxes(0, 1)
        xy_pred = preds_["coordinates"][0]
        conf_pred = preds_["confidence"]
        for i, xy_gt_ in enumerate(xy_gt):
            visible = np.flatnonzero(np.all(~np.isnan(xy_gt_), axis=1))
            xy_pred_ = xy_pred[map_[i]]
            if visible.size and xy_pred_.size:
                # Pick the predictions closest to ground truth,
                # rather than the ones the model has most confident in.
                neighbors = _find_closest_neighbors(
                    xy_gt_[visible], xy_pred_, k=3
                )
                found = neighbors != -1
                if ~np.any(found):
                    continue
                min_dists = np.linalg.norm(
                    xy_gt_[visible][found] - xy_pred_[neighbors[found]],
                    axis=1,
                )
                conf_pred_ = conf_pred[map_[i]]
                errors[n, visible[found], i, 0] = min_dists
                errors[n, visible[found], i, 1] = conf_pred_[neighbors[found], 0]
    return errors


def calc_mAP(
    preds,
    gt,
    paf_graph,
    paf_inds,
    oks_sigma=0.1,
    pcutoff=0.1,
    min_affinity=0.1,
    margin=0,
    symmetric_kpts=None,
    greedy_matching=False,
    add_discarded=True,
    identity_only=False,
):
    kpts = gt["metadata"]["keypoints"]
    n_multi = len(kpts) - gt["metadata"]["n_unique"]
    metadata = {
        "all_joints_names": kpts,
        "PAFgraph": paf_graph,
    }
    data = {"metadata": metadata}
    data.update(preds["predictions"])
    ass = inferenceutils.Assembler(
        data,
        max_n_individuals=len(gt["metadata"]["animals"]),
        n_multibodyparts=n_multi,
        paf_inds=paf_inds,
        pcutoff=pcutoff,
        min_affinity=min_affinity,
        add_discarded=add_discarded,
        identity_only=identity_only,
    )
    ass.assemble()

    gt_images = list(gt["annotations"])
    map_images = auxiliaryfunctions._map(list(preds["predictions"]), gt_images)
    inds_images = [
        gt_images.index(map_images[n]) for n in ass.metadata["imnames"]
    ]
    inds_non_single = [
        i for i, n in enumerate(gt["metadata"]["animals"]) if n != "single"
    ]
    temp = np.stack(list(gt["annotations"].values()))
    ass_true_dict = inferenceutils._parse_ground_truth_data(
        temp[np.ix_(inds_images, inds_non_single, np.arange(n_multi))],
    )
    oks = inferenceutils.evaluate_assembly(
        ass.assemblies,
        ass_true_dict,
        oks_sigma,
        margin=margin,
        symmetric_kpts=symmetric_kpts,
        greedy_matching=greedy_matching,
    )
    return oks["mAP"], ass