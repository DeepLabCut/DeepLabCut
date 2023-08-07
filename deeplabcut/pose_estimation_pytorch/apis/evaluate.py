"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import albumentations as A
import deeplabcut.pose_estimation_pytorch as dlc
import pandas as pd
import torch
from deeplabcut.pose_estimation_pytorch.apis.inference import inference
from deeplabcut.pose_estimation_pytorch.apis.utils import (
    build_inference_transform,
    build_pose_model,
)
from deeplabcut.pose_estimation_pytorch.models.detectors import DETECTORS
from deeplabcut.pose_estimation_pytorch.models.predictors import PREDICTORS
from deeplabcut.pose_estimation_pytorch.solvers.inference import get_scores
from deeplabcut.pose_estimation_pytorch.solvers.utils import (
    build_predictions_df,
    get_model_folder,
    get_paths,
    get_results_filename,
    get_snapshots,
)
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.visualization import plot_evaluation_results


def ensure_multianimal_df_format(df_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dataframe to 'multianimal' format (with an "individuals" columns index)

    Args:
        df_predictions: the dataframe to convert

    Returns:
        the dataframe in MA format
    """
    df_predictions_ma = df_predictions.copy()
    try:
        df_predictions_ma.columns.get_level_values("individuals").unique().tolist()
    except KeyError:
        new_cols = pd.MultiIndex.from_tuples(
            [(col[0], "single", col[1], col[2]) for col in df_predictions_ma.columns],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )
        df_predictions_ma.columns = new_cols
    return df_predictions_ma


def evaluate_snapshot(
    cfg: Dict,
    shuffle: int = 0,
    trainingsetindex: int = -1,
    snapshotindex: int = -1,
    transform: Union[A.BasicTransform, A.Compose] = None,
    plotting: Union[bool, str] = False,
    show_errors: bool = True,
    modelprefix: str = "",
    batch_size: int = 1,
) -> None:
    """Evaluates a snapshot.
    The evaluation results are stored in the .h5 and TODO .csv file under the subdirectory
    'evaluation_results'.

    Args:
        cfg: the content of the project's config file
        shuffle: shuffle index
        trainingsetindex: the training set fraction to use
        modelprefix: model prefix
        snapshotindex: index (starting at 0) of the snapshot we want to load. To
            evaluate the last one, use -1. To evaluate all snapshots, use "all". For
            example if we have 3 models saved
                - snapshot-0.pt
                - snapshot-50.pt
                - snapshot-100.pt
            and we want to evaluate snapshot-50.pt, snapshotindex should be 1. If None,
            the snapshotindex is loaded from the project configuration.
        transform: transformation pipeline for evaluation
            ** Should normalise the data the same way it was normalised during training **
        plotting: Plots the predictions on the train and test images. If provided it must
            be either ``True``, ``False``, ``"bodypart"``, or ``"individual"``. Setting
            to ``True`` defaults as ``"bodypart"`` for multi-animal projects.
        show_errors: whether to compare predictions and ground truth
        batch_size: the batch size to use for evaluation

    Returns:
        None
    """
    # reading pytorch config
    train_fraction = cfg["TrainingFraction"][trainingsetindex]
    modelfolder = os.path.join(
        cfg["project_path"],
        auxiliaryfunctions.get_model_folder(
            train_fraction,
            shuffle,
            cfg,
            modelprefix=modelprefix,
        ),
    )
    individuals = cfg.get("individuals", ["single"])
    bodyparts = auxiliaryfunctions.get_bodyparts(cfg)
    max_individuals = len(individuals)
    num_joints = len(bodyparts)
    pytorch_config = auxiliaryfunctions.read_plainconfig(
        os.path.join(modelfolder, "train", "pytorch_config.yaml")
    )
    method = pytorch_config.get("method", "bu")
    if method not in ["bu", "td"]:
        raise ValueError(
            f"Method should be set to either 'bu' (Bottom Up) or 'td' (Top Down), "
            f"currently it is {method}"
        )
    device = pytorch_config["device"]

    if transform is None:
        print("No transform passed, using default normalisation from config")
        transform = build_inference_transform(pytorch_config["data"])

    # if images are resized for inference, need to map keypoints back to original space
    images_resized_with_transform = pytorch_config["data"].get("resize", False)

    project = dlc.DLCProject(shuffle=shuffle, proj_root=pytorch_config["project_path"])
    names = get_paths(
        train_fraction=train_fraction,
        model_prefix=modelprefix,
        shuffle=shuffle,
        cfg=project.cfg,
        train_iterations=snapshotindex,
        method=method,
    )
    results_filename = get_results_filename(
        names["evaluation_folder"],
        names["dlc_scorer"],
        names["dlc_scorer_legacy"],
        names["model_path"][:-3],
    )

    pose_cfg = auxiliaryfunctions.read_plainconfig(pytorch_config["pose_cfg_path"])
    model = build_pose_model(pytorch_config["model"], pose_cfg)
    model.load_state_dict(torch.load(names["model_path"])["model_state_dict"])

    predictor = PREDICTORS.build(dict(pytorch_config["predictor"]))
    detector = None
    if method.lower() == "td":
        detector = DETECTORS.build(dict(pytorch_config["detector"]["detector_model"]))
        detector.load_state_dict(
            torch.load(names["detector_path"])["detector_state_dict"]
        )

    df_mode_predictions: List[pd.DataFrame] = []
    for mode in ["train", "test"]:
        dataset = dlc.PoseDataset(project, transform=transform, mode=mode)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        target_df = dataset.dataframe.copy()
        predictions = inference(
            dataloader=dataloader,
            model=model,
            predictor=predictor,
            method=method,
            max_individuals=max_individuals,
            num_keypoints=num_joints,
            device=device,
            align_predictions_to_ground_truth=True,
            images_resized_with_transform=images_resized_with_transform,
            detector=detector,
        )
        df_predictions = build_predictions_df(
            dlc_scorer=names["dlc_scorer"],
            individuals=individuals,
            bodyparts=bodyparts,
            df_index=target_df.index,
            predictions=predictions.reshape(target_df.index.shape[0], -1),
        )
        df_mode_predictions.append(df_predictions)

        df_predictions_ma = ensure_multianimal_df_format(df_predictions)
        if plotting:
            snapshot_name = Path(names["model_path"]).stem
            folder_name = (
                f"{names['evaluation_folder']}/"
                f"LabeledImages_{names['dlc_scorer']}_{snapshot_name}"
            )
            auxiliaryfunctions.attempttomakefolder(folder_name)
            df_combined = df_predictions_ma.merge(
                target_df, left_index=True, right_index=True
            )

            if isinstance(plotting, str):
                plot_mode = plotting
            else:
                plot_mode = "bodypart"

            plot_evaluation_results(
                df_combined=df_combined,
                project_root=cfg["project_path"],
                scorer=cfg["scorer"],
                model_name=names["dlc_scorer"],
                output_folder=folder_name,
                in_train_set=mode == "train",
                mode=plot_mode,
                colormap=cfg["colormap"],
                dot_size=cfg["dotsize"],
                alpha_value=cfg["alphavalue"],
                p_cutoff=cfg["pcutoff"],
            )

        if show_errors:
            scores = get_scores(pose_cfg, df_predictions, target_df)
            print(f"Mode {mode} scores:", scores)

    # Create the output dataframe
    df_all_predictions = pd.concat(df_mode_predictions, axis=0)
    # Re-Index the DataFrame in the same order as the ground truth dataframe
    df_all_predictions = df_all_predictions.reindex(project.dlc_df.index)

    output_filename = Path(results_filename)
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    df_all_predictions.to_hdf(str(output_filename), "df_with_missing")


def evaluate_network(
    config: str,
    shuffles: Iterable[int] = (1,),
    trainingsetindex: Union[int, str] = 0,
    snapshotindex: Optional[Union[int, str]] = None,
    plotting: Union[bool, str] = False,
    show_errors: bool = True,
    transform: Union[A.BasicTransform, A.Compose] = None,
    modelprefix: str = "",
    batch_size: int = 1,
) -> None:
    """Evaluates a snapshot.

    The evaluation results are stored in the .h5 and .csv file under the subdirectory
    'evaluation_results'.

    Args:
        config: path to the project's config file
        shuffles: Iterable of integers specifying the shuffle indices to evaluate.
        trainingsetindex: Integer specifying which training set fraction to use.
            Evaluates all fractions if set to "all"
        snapshotindex: index (starting at 0) of the snapshot we want to load. To
            evaluate the last one, use -1. To evaluate all snapshots, use "all". For
            example if we have 3 models saved
                - snapshot-0.pt
                - snapshot-50.pt
                - snapshot-100.pt
            and we want to evaluate snapshot-50.pt, snapshotindex should be 1. If None,
            the snapshotindex is loaded from the project configuration.
        plotting: Plots the predictions on the train and test images. If provided it must
            be either ``True``, ``False``, ``"bodypart"``, or ``"individual"``. Setting
            to ``True`` defaults as ``"bodypart"`` for multi-animal projects.
        show_errors: display train and test errors.
        transform: transformation pipeline for evaluation
            ** Should normalise the data the same way it was normalised during training **
        modelprefix: directory containing the deeplabcut models to use when evaluating
            the network. By default, they are assumed to exist in the project folder.
        batch_size: the batch size to use for evaluation

    Examples:
        If you want to evaluate on shuffle 1 without plotting predictions.

        >>> deeplabcut.evaluate_network(
                '/analysis/project/reaching-task/config.yaml', shuffles=[1],
            )

        If you want to evaluate shuffles 0 and 1 and plot the predictions.

        >>> deeplabcut.evaluate_network(
                '/analysis/project/reaching-task/config.yaml',
                shuffles=[0, 1],
                plotting=True,
            )

        If you want to plot assemblies for a maDLC project

        >>> deeplabcut.evaluate_network(
                '/analysis/project/reaching-task/config.yaml',
                shuffles=[1],
                plotting="individual",
            )
    """
    cfg = auxiliaryfunctions.read_config(config)

    if isinstance(trainingsetindex, int):
        train_set_indices = [trainingsetindex]
    elif isinstance(trainingsetindex, str) and trainingsetindex.lower() == "all":
        train_set_indices = list(range(len(cfg["TrainingFraction"])))
    else:
        raise ValueError(f"Invalid trainingsetindex: {trainingsetindex}")

    if snapshotindex is None:
        snapshotindex = cfg["snapshotindex"]

    for train_set_index in train_set_indices:
        for shuffle in shuffles:
            if isinstance(snapshotindex, str) and snapshotindex.lower() == "all":
                model_folder = get_model_folder(
                    train_fraction=cfg["TrainingFraction"][train_set_index],
                    shuffle=shuffle,
                    model_prefix=modelprefix,
                    test_cfg=cfg,
                )
                all_snapshots = get_snapshots(Path(model_folder))
                snapshot_indices = list(range(len(all_snapshots)))
            elif isinstance(snapshotindex, int):
                snapshot_indices = [snapshotindex]
            else:
                raise ValueError(f"Invalid snapshotindex: {snapshotindex}")

            for snapshot in snapshot_indices:
                evaluate_snapshot(
                    cfg=cfg,
                    shuffle=shuffle,
                    trainingsetindex=train_set_index,
                    snapshotindex=snapshot,
                    transform=transform,
                    plotting=plotting,
                    show_errors=show_errors,
                    modelprefix=modelprefix,
                    batch_size=batch_size,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--modelprefix", type=str, default="")
    parser.add_argument("--snapshotindex", type=int, default=49)
    parser.add_argument("--plotting", type=bool, default=False)
    parser.add_argument("--show_errors", type=bool, default=True)
    args = parser.parse_args()
    evaluate_network(
        config=args.config,
        modelprefix=args.modelprefix,
        snapshotindex=args.snapshotindex,
        plotting=args.plotting,
        show_errors=args.show_errors,
    )
