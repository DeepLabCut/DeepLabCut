"""Evaluate LightningPose OOD data"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from deeplabcut.pose_estimation_pytorch import DLCLoader, PoseDatasetParameters
from deeplabcut.pose_estimation_pytorch.apis.scoring import pair_predicted_individuals_with_gt, get_scores
from deeplabcut.pose_estimation_pytorch.apis.utils import get_runners
from deeplabcut.pose_estimation_pytorch.data.utils import map_id_to_annotations
from deeplabcut.pose_estimation_pytorch.runners import Task
from deeplabcut.pose_estimation_pytorch.utils import df_to_generic

from benchmark_lightning_pose import LP_DLC_BENCHMARKS
from utils import Project, Shuffle


def load_ground_truth(gt_data, parameters: PoseDatasetParameters):
    annotations = DLCLoader.filter_annotations(gt_data["annotations"], task=Task.BOTTOM_UP)
    img_to_ann_map = map_id_to_annotations(annotations)

    ground_truth_dict = {}
    for image in gt_data["images"]:
        image_path = image["file_name"]
        individual_keypoints = {
            annotations[i]["individual"]: annotations[i]["keypoints"]
            for i in img_to_ann_map[image["id"]]
        }
        gt_array = np.empty((parameters.max_num_animals, parameters.num_joints, 3))
        gt_array.fill(np.nan)

        # Keep the shape of the ground truth
        for idv_idx, idv in enumerate(parameters.individuals):
            if idv in individual_keypoints:
                keypoints = individual_keypoints[idv].reshape(parameters.num_joints, -1)
                gt_array[idv_idx, :, :] = keypoints[:, :3]

        ground_truth_dict[image_path] = gt_array

    return ground_truth_dict


def evaluate_ood(
    shuffle: Shuffle,
    snapshot_indices: list[int] | None = None,
):
    df_ood_path = shuffle.project.path / "CollectedData_new.csv"
    df_ood = pd.read_csv(
        df_ood_path,
        index_col=0,
        header=[0, 1, 2],
    )
    df_ood = df_ood[~df_ood.index.duplicated(keep="first")]
    images = [shuffle.project.path / Path(img) for img in df_ood.index]

    snapshots = shuffle.snapshots(detector=False)
    if snapshot_indices is not None:
        snapshots = [snapshots[i] for i in snapshot_indices]

    loader = DLCLoader(
        project_root=str(shuffle.project.path),
        model_config_path=str(shuffle.pytorch_cfg_path),
        shuffle=shuffle.index,
    )
    parameters = loader.get_dataset_parameters()

    best_results = {"rmse": 1_000_000}
    for snapshot in snapshots:
        runner, detector_runner = get_runners(
            pytorch_config=shuffle.pytorch_cfg,
            snapshot_path=str(snapshot),
            max_individuals=parameters.max_num_animals,
            num_bodyparts=parameters.num_joints,
            num_unique_bodyparts=parameters.num_unique_bpts,
            with_identity=False,
            transform=None,
            detector_path=None,
            detector_transform=None,
        )
        image_paths = [str(i) for i in images]
        print("Running pose prediction")
        predictions = runner.inference(tqdm(image_paths))
        poses = {
            image_path: image_predictions["bodyparts"][..., :3]
            for image_path, image_predictions in zip(image_paths, predictions)
        }

        gt_data = df_to_generic(str(shuffle.project.path), df_ood, image_id_offset=1)
        annotations_with_bbox = DLCLoader._get_all_bboxes(gt_data["images"], gt_data["annotations"])
        gt_data["annotations"] = annotations_with_bbox
        gt_keypoints = load_ground_truth(gt_data, loader.get_dataset_parameters())

        if parameters.max_num_animals > 1:
            poses = pair_predicted_individuals_with_gt(poses, gt_keypoints)

        results = get_scores(
            poses,
            gt_keypoints,
            pcutoff=0.6,
            unique_bodypart_poses=None,
            unique_bodypart_gt=None,
        )
        print(snapshot, results["rmse"])
        if results["rmse"] < best_results["rmse"]:
            best_results = results

    return best_results


def main(project: Project, train_fraction: float, shuffle_indices: list[int]):
    full_results = {"shuffles": []}
    for idx in shuffle_indices:
        shuffle_results = evaluate_ood(
            shuffle=Shuffle(project=project, train_fraction=train_fraction, index=idx),
            snapshot_indices=None,
        )
        full_results["shuffles"].append(idx)
        for k, v in shuffle_results.items():
            metric_list = full_results.get(k, [])
            metric_list.append(v)
            full_results[k] = metric_list

    print("Results:")
    for k, v in full_results.items():
        print(f"  {k}: {v}")

    print("mean", np.mean(full_results["rmse"]))
    print("std", np.std(full_results["rmse_pcutoff"]))


if __name__ == "__main__":
    main(
        LP_DLC_BENCHMARKS["mirrorFish"],
        train_fraction=0.81,
        shuffle_indices=[36, 37, 38, 39, 40],
    )
