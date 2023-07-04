import argparse
import deeplabcut.pose_estimation_pytorch as dlc
import numpy as np
import pandas as pd
import os
import torch
from deeplabcut.utils import auxiliaryfunctions
import deeplabcut.pose_estimation_pytorch.apis.inference_utils as inference_utils
from deeplabcut.pose_estimation_pytorch.apis.utils import (
    build_pose_model,
    build_inference_transform,
)
from deeplabcut.pose_estimation_pytorch.models.detectors import DETECTORS
from deeplabcut.pose_estimation_pytorch.models.predictors import PREDICTORS
from deeplabcut.pose_estimation_pytorch.solvers.inference import get_scores
from deeplabcut.pose_estimation_pytorch.solvers.utils import (
    get_paths,
    get_results_filename,
    save_predictions,
)
from deeplabcut.pose_estimation_tensorflow import Plotting
from deeplabcut.pose_estimation_pytorch.post_processing import (
    rmse_match_prediction_to_gt,
)
import albumentations as A
from typing import Union


def inference_network(
    config_path: str,
    shuffle: int = 0,
    model_prefix: str = "",
    load_epoch: Union[int, str] = -1,
    stride: int = 8,
    transform: Union[A.BasicTransform, A.Compose] = None,
    plot: bool = False,
    evaluate: bool = True,
) -> None:
    """
        Performs inference on the validation dataset and save the results as a dataframe

    Args:
        - config_path : path to the project's config file
        - shuffle : shuffle index
        - model_prefix: model prefix
        - load_epoch:
                    index (starting at 0) of the snapshot we want to load,
                    if -1 loads the last one automatically
                    for example if we have 3 models saved
                        -snapshot-0.pt
                        -snapshot-50.pt
                        -snapshot-100.pt
                    and we want to load the second one, load epoch should be 1
        - stride : unused #TODO We clearly should remove this
        - transform :
                    transformation pipeline for evaluation
                    ** Should normalise the data the same way it was normalised during training **
        - plot: whether to plot the predicted data or not
            #TODO Currently does not work for multinaimal, should be False for multianimal project
             otherwise it breaks
        - evaluate: whether to compare predictions and ground truth

    Returns:
        None
    """
    # reading pytorch config
    cfg = auxiliaryfunctions.read_config(config_path)
    train_fraction = cfg["TrainingFraction"]
    modelfolder = os.path.join(
        cfg["project_path"],
        auxiliaryfunctions.get_model_folder(
            train_fraction[0],
            shuffle,
            cfg,
            modelprefix=model_prefix,
        ),
    )
    individuals = cfg.get("individuals", ["single"])
    max_num_animals = len(individuals)
    num_joints = len(auxiliaryfunctions.get_bodyparts(cfg))
    pytorch_config_path = os.path.join(modelfolder, "train", "pytorch_config.yaml")
    pytorch_config = auxiliaryfunctions.read_plainconfig(pytorch_config_path)
    method = pytorch_config.get("method", "bu")
    if method not in ["bu", "td"]:
        raise ValueError(
            f"Method should be set to either 'bu' (Bottom Up) or 'td' (Top Down), currently it is {method}"
        )
    device = pytorch_config["device"]

    batch_size = pytorch_config["batch_size"]

    if transform is None:
        print("No transform passed, using default normalisation from config")
        transform = build_inference_transform(pytorch_config["data"])
    project = dlc.DLCProject(shuffle=shuffle, proj_root=pytorch_config["project_path"])

    valid_dataset = dlc.PoseDataset(project, transform=transform, mode="test")
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False
    )

    # if images are resized for inference,
    # need to take that into account to go back to original space
    images_resized_with_transform = pytorch_config["data"].get("resize", False)

    names = get_paths(
        train_fraction=train_fraction[0],
        model_prefix=model_prefix,
        shuffle=shuffle,
        cfg=valid_dataset.cfg,
        train_iterations=load_epoch,
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
    model.load_state_dict(torch.load(names["model_path"]))

    predictor = PREDICTORS.build(dict(pytorch_config["predictor"]))

    if method.lower() == "td":
        detector = DETECTORS.build(dict(pytorch_config["detector"]["detector_model"]))
        detector.load_state_dict(torch.load(names["detector_path"]))
        top_down_predictor = PREDICTORS.build(
            {"type": "TopDownPredictor", "format_bbox": "xyxy"}
        )  # I don't think this top down predictor should depend on config since I feel like it's already pretty general
        detector.eval()
        detector.to(device)

    target_df = valid_dataset.dataframe
    predicted_poses = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for item in valid_dataloader:
            item["image"] = item["image"].to(device)
            shape_image = item["image"].shape

            if method.lower() == "td":
                predictions = inference_utils.get_predictions_top_down(
                    detector=detector,
                    top_down_predictor=top_down_predictor,
                    model=model,
                    pose_predictor=predictor,
                    images=item["image"],
                    max_num_animals=max_num_animals,
                    num_keypoints=num_joints,
                    device=device,
                )
            elif method.lower() == "bu":
                predictions = inference_utils.get_predictions_bottom_up(
                    model=model,
                    predictor=predictor,
                    images=item["image"],
                )
            else:
                raise ValueError("This error should not happen !")

            # Matching predictions to ground truth individuals in order to compute rmse and save as dataframe
            if len(individuals) > 1:
                for b in range(predictions.shape[0]):
                    # rmse is more practical than oks
                    # since oks needs at least 2 annotated keypoints per animal (to compute area)
                    match_individuals = rmse_match_prediction_to_gt(
                        predictions[b],
                        item["annotations"]["keypoints"][b].cpu().numpy(),
                        individuals,
                    )
                    predictions[b] = predictions[b][match_individuals]

            # TODO shifting error when padding
            # converts back to original image size if image was resized during the augmentation pipeline
            if images_resized_with_transform:
                for b in range(predictions.shape[0]):
                    resizing_factor = (
                        item["original_size"][0][b] / shape_image[2]
                    ).item(), (item["original_size"][1][b] / shape_image[3]).item()
                    predictions[b, :, :, 0] = (
                        predictions[b, :, :, 0] * resizing_factor[1]
                        + resizing_factor[1] / 2
                    )
                    predictions[b, :, :, 1] = (
                        predictions[b, :, :, 1] * resizing_factor[0]
                        + resizing_factor[0] / 2
                    )
            predicted_poses.append(predictions)

        predicted_poses = np.array(predicted_poses)

    predicted_df = save_predictions(
        names,
        cfg,
        target_df.index,
        predicted_poses.reshape(target_df.index.shape[0], -1),
        results_filename,
    )

    # Convert dataframe to 'multianimal' format in any case, allows for similar post_processing
    try:
        predicted_df.columns.get_level_values("individuals").unique().tolist()
    except KeyError:
        new_cols = pd.MultiIndex.from_tuples(
            [(col[0], "single", col[1], col[2]) for col in predicted_df.columns],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )
        predicted_df.columns = new_cols

    if plot:
        foldername = f'{names["evaluation_folder"]}/LabeledImages_{names["dlc_scorer"]}-{load_epoch}'
        auxiliaryfunctions.attempttomakefolder(foldername)
        combined_df = predicted_df.merge(target_df, left_index=True, right_index=True)
        Plotting(
            valid_dataset.cfg,
            valid_dataset.cfg["bodyparts"],
            names["dlc_scorer"],
            predicted_df.index,
            combined_df,
            foldername,
        )
    if evaluate:
        scores = get_scores(pose_cfg, predicted_df, target_df)
        print(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--modelprefix", type=str, default="")
    parser.add_argument("--load_epoch", type=int, default=49)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--evaluate", type=bool, default=True)
    args = parser.parse_args()
    inference_network(
        config_path=args.config_path,
        shuffle=args.shuffle,
        model_prefix=args.modelprefix,
        load_epoch=args.load_epoch,
        plot=args.plot,
        evaluate=args.evaluate,
    )
