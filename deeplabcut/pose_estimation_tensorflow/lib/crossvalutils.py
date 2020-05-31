"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs
from scipy.optimize import linear_sum_assignment

from deeplabcut.pose_estimation_tensorflow import return_evaluate_network_data
from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions


def set_up_evaluation(data):
    params = dict()
    params["joint_names"] = data["metadata"]["all_joints_names"]
    params["num_joints"] = len(params["joint_names"])
    partaffinityfield_graph = data["metadata"]["PAFgraph"]
    params["paf"] = np.arange(len(partaffinityfield_graph))
    params["paf_graph"] = params["paf_links"] = [
        partaffinityfield_graph[l] for l in params["paf"]
    ]
    params["bpts"] = params["ibpts"] = range(params["num_joints"])
    params["imnames"] = [fn for fn in list(data) if fn != "metadata"]
    return params


def compute_crossval_metrics(
    config_path,
    inference_cfg,
    shuffle=1,
    trainingsetindex=0,
    modelprefix="",
    snapshotindex=-1,
    dcorr=5,
    leastbpts=3,
):

    fns = return_evaluate_network_data(
        config_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
    )

    predictionsfn = fns[snapshotindex]
    data, metadata = auxfun_multianimal.LoadFullMultiAnimalData(predictionsfn)
    params = set_up_evaluation(data)

    n_images = len(params["imnames"])
    poses = []
    poses_gt = []
    stats = np.full(
        (n_images, 7), np.nan
    )  # RMSE, hits, misses, false_pos, num_detections, pck
    columns = ["train_iter", "train_frac", "shuffle"]
    columns += [
        "_".join((b, a))
        for a in ("train", "test")
        for b in ("rmse", "hits", "misses", "falsepos", "ndetects", "pck", "rpck")
    ]
    for n, imname in enumerate(params["imnames"]):
        animals = inferenceutils.assemble_individuals(
            inference_cfg,
            data[imname],
            params["num_joints"],
            params["bpts"],
            params["ibpts"],
            params["paf"],
            params["paf_graph"],
            params["paf_links"],
            evaluation=True,
        )
        n_animals = len(animals)
        if n_animals:
            _, _, GT = data[imname]["groundtruth"]
            GT = GT.droplevel("scorer").unstack(level=["bodyparts", "coords"])
            gt = GT.values.reshape((GT.shape[0], -1, 2))
            poses_gt.append(gt)

            if (
                leastbpts > 0
            ):  # ONLY KEEP animals with at least as many bpts (to get rid of crops that cannot be assembled)
                gt = gt[np.nansum(gt, axis=(1, 2)) > leastbpts]

            temp = np.stack(animals).reshape((n_animals, -1, 3))
            poses.append(temp)
            ani = temp[:, : gt.shape[1], :2]
            mat = np.full((gt.shape[0], n_animals), np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for i in range(len(gt)):
                    for j in range(len(animals)):
                        mat[i, j] = np.sqrt(
                            np.nanmean(np.sum((gt[i] - ani[j, :, :2]) ** 2, axis=1))
                        )

            if np.nansum(mat) > 0:  # also assures at least one not nan np.size(mat)>0:
                mat[np.isnan(mat)] = np.nanmax(mat) + 1
                row_indices, col_indices = linear_sum_assignment(mat)
                stats[n, 0] = mat[row_indices, col_indices].mean()  # rmse

                gt_annot = np.any(~np.isnan(gt), axis=2)
                gt_matched = gt_annot[row_indices].flatten()

                dlc_annot = np.any(~np.isnan(ani), axis=2)  # DLC assemblies
                dlc_matched = dlc_annot[col_indices].flatten()

                stats[n, 1] = np.logical_and(gt_matched, dlc_matched).sum()  # hits
                stats[n, 2] = gt_annot.sum() - stats[n, 1]  # misses
                stats[n, 3] = np.logical_and(
                    ~gt_matched, dlc_matched
                ).sum()  # additional detections
                stats[n, 4] = n_animals

                numgtpts = gt_annot.sum()
                # animal & bpt-wise distance!
                if numgtpts > 0:
                    # corrkps=np.sum((gt[row_indices]-ani[col_indices])**2,axis=2)<dcorr**2
                    dists = np.sum((gt[row_indices] - ani[col_indices]) ** 2, axis=2)
                    corrkps = dists[np.isfinite(dists)] < dcorr ** 2
                    pck = (
                        corrkps.sum() * 1.0 / numgtpts
                    )  # weigh by actually annotated ones!
                    rpck = (
                        np.sum(
                            np.exp(-dists[np.isfinite(dists)] * 1.0 / (2 * dcorr ** 2))
                        )
                        * 1.0
                        / numgtpts
                    )

                else:
                    pck = 1.0  # does that make sense? As a convention fully correct...
                    rpck = 1.0

                stats[n, 5] = pck
                stats[n, 6] = rpck

    train_iter = int(predictionsfn.split("-")[-1].split(".")[0])
    train_frac = int(predictionsfn.split("trainset")[1].split("shuffle")[0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        res = np.r_[
            train_iter,
            train_frac,
            shuffle,
            np.nanmean(stats[metadata["data"]["trainIndices"]], axis=0),
            np.nanmean(stats[metadata["data"]["testIndices"]], axis=0),
        ]

    return pd.DataFrame(res.reshape((1, -1)), columns=columns), poses_gt, poses


def compute_crossval_metrics_preloadeddata(
    params,
    columns,
    inference_cfg,
    data,
    trainIndices,
    testIndices,
    train_iter,
    train_frac,
    shuffle,
    lowerbound,
    upperbound,
    dcorr,
    leastbpts,
):
    n_images = len(params["imnames"])
    stats = np.full(
        (n_images, 7), np.nan
    )  # RMSE, hits, misses, false_pos, num_detections, pck, rpck
    for n, imname in enumerate(params["imnames"]):
        animals = inferenceutils.assemble_individuals(
            inference_cfg,
            data[imname],
            params["num_joints"],
            params["bpts"],
            params["ibpts"],
            params["paf"],
            params["paf_graph"],
            params["paf_links"],
            lowerbound,
            upperbound,
            evaluation=True,
        )

        n_animals = len(animals)
        if n_animals:
            _, _, GT = data[imname]["groundtruth"]
            GT = GT.droplevel("scorer").unstack(level=["bodyparts", "coords"])
            gt = GT.values.reshape((GT.shape[0], -1, 2))

            if (
                leastbpts > 0
            ):  # ONLY KEEP animals with at least as many bpts (to get rid of crops that cannot be assembled)
                gt = gt[np.nansum(gt, axis=(1, 2)) > leastbpts]

            ani = np.stack(animals).reshape((n_animals, -1, 3))[:, : gt.shape[1], :2]
            mat = np.full((gt.shape[0], n_animals), np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for i in range(len(gt)):
                    for j in range(len(animals)):
                        mat[i, j] = np.sqrt(
                            np.nanmean(np.sum((gt[i] - ani[j, :, :2]) ** 2, axis=1))
                        )

            if np.nansum(mat) > 0:  # np.size(mat)>0:
                mat[np.isnan(mat)] = np.nanmax(mat) + 1
                row_indices, col_indices = linear_sum_assignment(mat)
                stats[n, 0] = mat[row_indices, col_indices].mean()  # rmse

                gt_annot = np.any(~np.isnan(gt), axis=2)
                gt_matched = gt_annot[row_indices].flatten()

                dlc_annot = np.any(~np.isnan(ani), axis=2)  # DLC assemblies
                dlc_matched = dlc_annot[col_indices].flatten()

                stats[n, 1] = np.logical_and(gt_matched, dlc_matched).sum()  # hits
                stats[n, 2] = gt_annot.sum() - stats[n, 1]  # misses
                stats[n, 3] = np.logical_and(
                    ~gt_matched, dlc_matched
                ).sum()  # additional detections
                stats[n, 4] = n_animals

                numgtpts = gt_annot.sum()
                # animal & bpt-wise distance!
                if numgtpts > 0:
                    # corrkps=np.sum((gt[row_indices]-ani[col_indices])**2,axis=2)<dcorr**2
                    dists = np.sum((gt[row_indices] - ani[col_indices]) ** 2, axis=2)
                    corrkps = dists[np.isfinite(dists)] < dcorr ** 2
                    pck = (
                        corrkps.sum() * 1.0 / numgtpts
                    )  # weigh by actually annotated ones!
                    rpck = (
                        np.sum(
                            np.exp(-dists[np.isfinite(dists)] * 1.0 / (2 * dcorr ** 2))
                        )
                        * 1.0
                        / numgtpts
                    )

                else:
                    pck = 1.0  # does that make sense? As a convention fully correct...
                    rpck = 1.0

                stats[n, 5] = pck
                stats[n, 6] = rpck

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        res = np.r_[
            train_iter,
            train_frac,
            shuffle,
            np.nanmean(stats[trainIndices], axis=0),
            np.nanmean(stats[testIndices], axis=0),
        ]

    return pd.DataFrame(res.reshape((1, -1)), columns=columns)


def bayesian_search(
    config_path,
    inferencecfg,
    pbounds,
    edgewisecondition=True,
    shuffle=1,
    trainingsetindex=0,
    modelprefix="",
    snapshotindex=-1,
    target="rpck_test",
    maximize=True,
    init_points=20,
    n_iter=50,
    acq="ei",
    log_file=None,
    dcorr=5,
    leastbpts=3,
    printingintermediatevalues=True,
):  #

    if "rpck" in target:
        assert maximize == True

    if "rmse" in target:
        assert maximize == False

    cfg = auxiliaryfunctions.read_config(config_path)
    evaluationfolder = os.path.join(
        cfg["project_path"],
        str(
            auxiliaryfunctions.GetEvaluationFolder(
                cfg["TrainingFraction"][int(trainingsetindex)],
                shuffle,
                cfg,
                modelprefix=modelprefix,
            )
        ),
    )

    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.GetScorerName(
        cfg,
        shuffle,
        cfg["TrainingFraction"][int(trainingsetindex)],
        cfg["iteration"],
        modelprefix=modelprefix,
    )

    # load params
    fns = return_evaluate_network_data(
        config_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
    )
    predictionsfn = fns[snapshotindex]
    data, metadata = auxfun_multianimal.LoadFullMultiAnimalData(predictionsfn)
    params = set_up_evaluation(data)
    columns = ["train_iter", "train_frac", "shuffle"]
    columns += [
        "_".join((b, a))
        for a in ("train", "test")
        for b in ("rmse", "hits", "misses", "falsepos", "ndetects", "pck", "rpck")
    ]

    train_iter = trainingsetindex  # int(predictionsfn.split('-')[-1].split('.')[0])
    train_frac = cfg["TrainingFraction"][
        train_iter
    ]  # int(predictionsfn.split('trainset')[1].split('shuffle')[0])
    trainIndices = metadata["data"]["trainIndices"]
    testIndices = metadata["data"]["testIndices"]

    if edgewisecondition:
        mf = str(
            auxiliaryfunctions.GetModelFolder(
                cfg["TrainingFraction"][int(trainingsetindex)],
                shuffle,
                cfg,
                modelprefix=modelprefix,
            )
        )
        modelfolder = os.path.join(cfg["project_path"], mf)
        path_inferencebounds_config = (
            Path(modelfolder) / "test" / "inferencebounds.yaml"
        )
        try:
            inferenceboundscfg = auxiliaryfunctions.read_plainconfig(
                path_inferencebounds_config
            )
        except FileNotFoundError:
            print("Computing distances...")
            from deeplabcut.pose_estimation_tensorflow import calculatepafdistancebounds

            inferenceboundscfg = calculatepafdistancebounds(
                config_path, shuffle, trainingsetindex
            )
            auxiliaryfunctions.write_plainconfig(
                path_inferencebounds_config, inferenceboundscfg
            )

        partaffinityfield_graph = params["paf_graph"]
        upperbound = np.array(
            [
                float(
                    inferenceboundscfg[str(edge[0]) + "_" + str(edge[1])]["intra_max"]
                )
                for edge in partaffinityfield_graph
            ]
        )
        lowerbound = np.array(
            [
                float(
                    inferenceboundscfg[str(edge[0]) + "_" + str(edge[1])]["intra_min"]
                )
                for edge in partaffinityfield_graph
            ]
        )

        upperbound *= inferencecfg["upperbound_factor"]
        lowerbound *= inferencecfg["lowerbound_factor"]

    else:
        lowerbound = None
        upperbound = None

    def dlc_hyperparams(**kwargs):
        inferencecfg.update(kwargs)
        # Ensure type consistency
        for k, (bound, _) in pbounds.items():
            inferencecfg[k] = type(bound)(inferencecfg[k])

        stats = compute_crossval_metrics_preloadeddata(
            params,
            columns,
            inferencecfg,
            data,
            trainIndices,
            testIndices,
            train_iter,
            train_frac,
            shuffle,
            lowerbound,
            upperbound,
            dcorr=dcorr,
            leastbpts=leastbpts,
        )

        # stats = compute_crossval_metrics(config_path, inferencecfg, shuffle,trainingsetindex,
        #                                    dcorr=dcorr,leastbpts=leastbpts,modelprefix=modelprefix)

        if printingintermediatevalues:
            print(
                "rpck",
                stats["rpck_test"].values[0],
                "rpck train:",
                stats["rpck_train"].values[0],
            )
            print(
                "rmse",
                stats["rmse_test"].values[0],
                "miss",
                stats["misses_test"].values[0],
                "hit",
                stats["hits_test"].values[0],
            )

        # val = stats['rmse_test'].values[0]*(1+stats['misses_test'].values[0]*1./stats['hits_test'].values[0])
        val = stats[target].values[0]
        if np.isnan(val):
            if maximize:  # pck case
                val = -1e9  # random small number
            else:  # RMSE, return a large RMSE
                val = 1e9

        if not maximize:
            val = -val

        return val

    opt = BayesianOptimization(f=dlc_hyperparams, pbounds=pbounds, random_state=42)

    # Saving log file
    if log_file:
        load_logs(opt, log_file)

    logger = JSONLogger(
        path=os.path.join(evaluationfolder, "opti_log" + DLCscorer + ".json")
    )

    opt.subscribe(Events.OPTIMIZATION_STEP, logger)
    opt.maximize(init_points=init_points, n_iter=n_iter, acq=acq)

    inferencecfg.update(opt.max["params"])
    for k, (bound, _) in pbounds.items():
        tmp = type(bound)(inferencecfg[k])
        if isinstance(tmp, np.floating):
            tmp = np.round(tmp, 2).item()
        inferencecfg[k] = tmp

    return inferencecfg, opt
