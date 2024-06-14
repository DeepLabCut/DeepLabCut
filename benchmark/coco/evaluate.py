"""Evaluating COCO models"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from deeplabcut.pose_estimation_pytorch import COCOLoader
from deeplabcut.pose_estimation_pytorch.apis.evaluate import evaluate
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.task import Task


def pycocotools_evaluation(
    kpt_oks_sigmas: list[int],
    ground_truth: dict,
    predictions: list[dict],
    annotation_type: str,
) -> None:
    """Evaluation of models using Pycocotools

    Evaluates the predictions using OKS sigma 0.1, margin 0 and prints the results to
    the console.

    Args:
        kpt_oks_sigmas: the OKS sigma for each keypoint
        ground_truth: the ground truth data, in COCO format
        predictions: the predictions, in COCO format
        annotation_type: {"bbox", "keypoints"} the annotation type to evaluate
    """
    print(80 * "-")
    print(f"Attempting `pycocotools` evaluation for {annotation_type}!")
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco = COCO()
        coco.dataset["annotations"] = ground_truth["annotations"]
        coco.dataset["categories"] = ground_truth["categories"]
        coco.dataset["images"] = ground_truth["images"]
        coco.createIndex()

        coco_det = coco.loadRes(predictions)
        coco_eval = COCOeval(coco, coco_det, iouType=annotation_type)
        coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    except Exception as err:
        print(f"Could not evaluate with `pycocotools`: {err}")
    finally:
        print(80 * "-")


def main(
    project_root: str,
    train_file: str,
    test_file: str,
    pytorch_config_path: str,
    device: str | None,
    snapshot_path: str,
    detector_path: str | None,
    pcutoff: float,
    oks_sigma: float,
):
    loader = COCOLoader(
        project_root=project_root,
        model_config_path=pytorch_config_path,
        train_json_filename=train_file,
        test_json_filename=test_file,
    )
    parameters = loader.get_dataset_parameters()
    if device is not None:
        loader.model_cfg["device"] = device

    pose_runner, detector_runner = get_inference_runners(
        model_config=loader.model_cfg,
        snapshot_path=snapshot_path,
        max_individuals=parameters.max_num_animals,
        num_bodyparts=parameters.num_joints,
        num_unique_bodyparts=parameters.num_unique_bpts,
        with_identity=False,
        transform=None,
        detector_path=detector_path,
        detector_transform=None,
    )

    output_path = Path(pytorch_config_path).parent.parent / "results"
    output_path.mkdir(exist_ok=True)
    for mode in ["train", "test"]:
        scores, predictions = evaluate(
            pose_task=Task(loader.model_cfg["method"]),
            pose_runner=pose_runner,
            loader=loader,
            mode=mode,
            detector_runner=detector_runner,
            pcutoff=pcutoff,
        )
        coco_predictions = loader.predictions_to_coco(predictions, mode=mode)
        model_name = Path(snapshot_path).stem
        if detector_path is not None:
            model_name += Path(detector_path).stem
        predictions_file = output_path / f"{model_name}-{mode}-predictions.json"
        with open(predictions_file, "w") as f:
            json.dump(coco_predictions, f)

        annotation_types = ["keypoints"]
        if detector_runner is not None:
            annotation_types.append("bbox")

        ground_truth = loader.load_data(mode=mode)
        for annotation_type in annotation_types:
            kpt_oks_sigmas = oks_sigma * np.ones(parameters.num_joints)
            pycocotools_evaluation(
                ground_truth=ground_truth,
                predictions=coco_predictions,
                kpt_oks_sigmas=kpt_oks_sigmas,
                annotation_type=annotation_type,
            )

        print(80 * "-")
        print(f"{mode} results")
        for k, v in scores.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root")
    parser.add_argument("pytorch_config_path")
    parser.add_argument("snapshot_path")
    parser.add_argument("--train_file", default="train.json")
    parser.add_argument("--test_file", default="test.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--detector_path", default=None)
    parser.add_argument("--pcutoff", type=float, default=0.6)
    parser.add_argument("--oks_sigma", type=float, default=0.1)
    args = parser.parse_args()
    main(
        args.project_root,
        args.train_file,
        args.test_file,
        args.pytorch_config_path,
        args.device,
        args.snapshot_path,
        args.detector_path,
        args.pcutoff,
        args.oks_sigma,
    )
