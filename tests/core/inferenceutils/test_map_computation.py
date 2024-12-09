"""Tests mAP computation from inferenceutils"""

from __future__ import annotations

import numpy as np
import pytest

from deeplabcut.core import inferenceutils
from deeplabcut.pose_estimation_pytorch.data.utils import bbox_from_keypoints


@pytest.mark.parametrize(
    "ground_truth",
    [
        {
            "img0": [
                [
                    [100.0, 10.0, 2],
                    [150.0, 15.0, 2],
                    [202.0, 20.0, 2],
                ],
            ],
        },
        {
            "img0": [
                [
                    [90.0, 12.0, 2],
                    [140.0, 17.0, 2],
                    [192.0, 22.0, 2],
                ],
            ],
        },
    ],
)
@pytest.mark.parametrize(
    "predictions",
    [
        {
            "img0": [
                [
                    [100.0, 10.0, 0.9],
                    [150.0, 15.0, 0.7],
                    [202.0, 20.0, 0.8],
                ],
            ],
        },
        {
            "img0": [
                [
                    [90.0, 12.0, 0.9],
                    [140.0, 17.0, 0.7],
                    [192.0, 22.0, 0.8],
                ],
                [
                    [97.0, 11.0, 0.5],
                    [148.0, 14.0, 0.2],
                    [202.0, 21.0, 0.3],
                ],
            ],
        },
        {
            "img0": [
                [
                    [90.0, 12.0, 0.9],
                    [np.nan, np.nan, 0.0],
                    [192.0, 22.0, 0.8],
                ],
                [
                    [97.0, 11.0, 0.5],
                    [148.0, 14.0, 0.2],
                    [202.0, 21.0, 0.3],
                ],
            ],
        },
    ],
)
def test_map_single_image_simple(ground_truth: dict, predictions: dict):
    gt = {k: np.array(v) for k, v in ground_truth.items()}
    pred = {k: np.array(v) for k, v in predictions.items()}
    _evaluate(gt, pred)


@pytest.mark.parametrize(
    "ground_truth",
    [
        {
            "img0": [
                [
                    [100.0, 10.0, 2],
                    [150.0, 15.0, 2],
                    [202.0, 20.0, 2],
                ],
            ],
        },
        {
            "img0": [
                [
                    [90.0, 12.0, 2],
                    [140.0, 17.0, 2],
                    [192.0, 22.0, 2],
                ],
                [
                    [726.0, 325.0, 2],
                    [326.0, 236.0, 2],
                    [457.0, 832.0, 2],
                ],
            ],
        },
        {
            "img0": [
                [
                    [90.0, 12.0, 2],
                    [140.0, 17.0, 2],
                    [192.0, 22.0, 2],
                ],
                [
                    [726.0, 325.0, 2],
                    [0.0, 0.0, 0],
                    [457.0, 832.0, 2],
                ],
            ],
        },
        {
            "img0": [
                [
                    [90.0, 12.0, 2],
                    [140.0, 17.0, 2],
                    [192.0, 22.0, 2],
                ],
                [
                    [726.0, 325.0, 2],
                    [0, 0, 0],
                    [457.0, 832.0, 2],
                ],
                [
                    [452.0, 321.0, 2],
                    [213.0, 387.0, 2],
                    [213.0, 832.0, 2],
                ],
                [
                    [253.0, 238.0, 2],
                    [213.0, 238.0, 2],
                    [457.0, 832.0, 2],
                ],
            ],
        },
    ],
)
def test_map_single_image_random_errors(ground_truth: dict):
    rng = np.random.default_rng(seed=0)

    gt = {k: np.array(v) for k, v in ground_truth.items()}
    pred = {}
    for k, gt_kpts in gt.items():
        num_idv, num_bpt = gt_kpts.shape[:2]

        error = rng.integers(low=-30, high=30, size=(num_idv, num_bpt, 2))
        scores = rng.random(size=(num_idv, num_bpt))

        pred[k] = np.zeros(shape=(num_idv, num_bpt, 3))
        pred[k][..., :2] = np.clip(gt_kpts[..., :2] + error, 0, 1024)
        pred[k][..., 2] = scores

    _evaluate(gt, pred)


@pytest.mark.parametrize("num_images", [1, 2, 5, 10])
@pytest.mark.parametrize("num_joints", [2, 5, 8, 20])
@pytest.mark.parametrize("max_error", [1, 2, 5, 20, 40])
def test_random_map_computation(num_images, num_joints, max_error):
    rng = np.random.default_rng(seed=0)

    num_individuals = rng.integers(low=0, high=20, size=(num_images, 2))
    max_idv = num_individuals.max(initial=0)

    gt = {}
    pred = {}
    for i, (gt_idv, pred_idv) in enumerate(num_individuals):
        # padding needed as we then stack
        gt_kpts = np.zeros((max_idv, num_joints, 3))
        pred_kpts = -np.ones((max_idv, num_joints, 3))

        gt_kpts[:gt_idv] = 2 * np.ones((gt_idv, num_joints, 3))
        gt_kpts[:gt_idv, :, :2] = rng.integers(
            low=0, high=1024, size=(gt_idv, num_joints, 2)
        )
        gt[f"img_{i}"] = gt_kpts

        # set scores
        pred_kpts[:pred_idv, :, 2] = rng.random(size=(pred_idv, num_joints))

        # predictions that are ground truth + error
        matched = min(gt_idv, pred_idv)
        if matched > 0:
            error = rng.integers(
                low=-max_error, high=max_error, size=(matched, num_joints, 2)
            )
            matched_pred = gt_kpts[:matched, :, :2] + error
            pred_kpts[:matched, :, :2] = np.clip(matched_pred, 0, 1024)

        # random predictions
        unmatched = pred_idv - matched
        if unmatched > 0:
            pred_kpts[matched:pred_idv, :, :2] = rng.integers(
                low=0, high=1024, size=(unmatched, num_joints, 2)
            )

        pred[f"img_{i}"] = pred_kpts

    _evaluate(gt, pred)


@pytest.mark.parametrize("num_images", [1, 2, 5, 10])
@pytest.mark.parametrize("num_joints", [2, 5, 8, 20])
@pytest.mark.parametrize("max_error", [1, 2, 5, 20, 40])
def test_random_map_computation_with_missing_kpts(num_images, num_joints, max_error):
    rng = np.random.default_rng(seed=0)

    num_individuals = rng.integers(low=0, high=20, size=(num_images, 2))
    max_idv = num_individuals.max(initial=0)

    gt = {}
    pred = {}
    for i, (gt_idv, pred_idv) in enumerate(num_individuals):
        # padding needed as we then stack
        gt_kpts = np.zeros((max_idv, num_joints, 3))
        pred_kpts = -np.ones((max_idv, num_joints, 3))

        gt_kpts[:gt_idv] = 2 * np.ones((gt_idv, num_joints, 3))
        gt_kpts[:gt_idv, :, :2] = rng.integers(
            low=0, high=1024, size=(gt_idv, num_joints, 2)
        )
        gt[f"img_{i}"] = gt_kpts

        # drop some ground truth keypoints
        gt_vis_mask = rng.random(size=(max_idv, num_joints)) < 0.2
        gt_kpts[gt_vis_mask, 2] = 0

        # set scores
        pred_kpts[:pred_idv, :, 2] = rng.random(size=(pred_idv, num_joints))

        # predictions that are ground truth + error
        matched = min(gt_idv, pred_idv)
        if matched > 0:
            error = rng.integers(
                low=-max_error, high=max_error, size=(matched, num_joints, 2)
            )
            matched_pred = gt_kpts[:matched, :, :2] + error
            pred_kpts[:matched, :, :2] = np.clip(matched_pred, 0, 1024)

        # random predictions
        unmatched = pred_idv - matched
        if unmatched > 0:
            pred_kpts[matched:pred_idv, :, :2] = rng.integers(
                low=0, high=1024, size=(unmatched, num_joints, 2)
            )

        pred[f"img_{i}"] = pred_kpts

    _evaluate(gt, pred)


def _evaluate(gt: dict[str, np.ndarray], pred: dict[str, np.ndarray]):
    for k, v in gt.items():
        print(20 * "-")
        print(k)
        print("GT")
        print(v)
        print("PR")
        print(pred[k])

    gt_assemblies = _to_assemblies(gt, ground_truth=True)
    pred_assemblies = _to_assemblies(pred, ground_truth=False)
    oks = inferenceutils.evaluate_assembly_greedy(
        assemblies_gt=gt_assemblies,
        assemblies_pred=pred_assemblies,
        oks_sigma=0.1,
        oks_thresholds=np.linspace(0.5, 0.95, 10),
        margin=0.0,
        symmetric_kpts=None,
    )

    num_joints = gt[list(gt.keys())[0]].shape[1]
    coco_gt = _to_coco_ground_truth(gt, num_joints, bbox_margin=0)
    coco_pred = _to_coco_predictions(coco_gt, pred, bbox_margin=0)
    coco_oks = eval_coco(coco_gt, coco_pred, num_joints)
    print(20 * "-")
    print(f"dlc mAP:")
    for k, v in oks.items():
        print(k)
        print(v)
        print()
    print(20 * "-")
    print(f"pycocotools mAP: {coco_oks}")
    print()
    assert oks["mAP"] == coco_oks


def _to_assemblies(
    data: dict[str, np.ndarray], ground_truth: bool,
) -> dict[str, list[inferenceutils.Assembly]]:
    images = list(data.keys())
    raw_data = np.stack([data[i] for i in images], axis=0)

    # mask not visible entries
    mask = raw_data[..., 2] <= 0
    raw_data[mask] = np.nan

    # set the "score" to 1 for ground truth
    if ground_truth:
        raw_data[~mask, 2] = 1

    return {
        images[i]: assembly
        for i, assembly in inferenceutils._parse_ground_truth_data(raw_data).items()
    }


def _to_coco_ground_truth(
    data: dict[str, np.ndarray],
    num_joints: int,
    bbox_margin: int = 0,
    image_size: tuple[int, int] = (1024, 1024),
) -> dict[str, list[dict]]:
    w, h = image_size
    anns, images = [], []
    for path, image_keypoints in data.items():
        id_ = len(images) + 1
        images.append(dict(id=id_, file_name=path, width=w, height=h))

        assert image_keypoints.shape[1] == num_joints
        for idv_id, kpts in enumerate(image_keypoints):
            visible = kpts[:, 2] > 0
            num_keypoints = visible.sum()

            if num_keypoints > 1:
                bbox = bbox_from_keypoints(
                    keypoints=kpts,
                    image_h=h,
                    image_w=w,
                    margin=bbox_margin,
                )
                area = bbox[2].item() * bbox[3].item()
                anns.append(
                    {
                        "id": len(anns) + 1,
                        "image_id": id_,
                        "category_id": 1,
                        "area": area,
                        "bbox": bbox.tolist(),
                        "keypoints": kpts.reshape(-1).tolist(),
                        "iscrowd": 0,
                        "num_keypoints": num_keypoints,
                    }
                )

    keypoints = [f"bpt{i}" for i in range(num_joints)]
    category = dict(id=1, name="animal", supercategory="animal", keypoints=keypoints)
    return {"annotations": anns, "categories": [category], "images": images}


def _to_coco_predictions(
    ground_truth: dict,
    predictions: dict[str, np.ndarray],
    bbox_margin: int = 0,
    image_size: tuple[int, int] = (1024, 1024),
) -> list[dict]:
    w, h = image_size
    num_joints = len(ground_truth["categories"][0]["keypoints"])
    path_to_id = {img["file_name"]: img["id"] for img in ground_truth["images"]}

    coco_predictions = []
    for path, image_keypoints in predictions.items():
        assert image_keypoints.shape[1] == num_joints

        img_id = path_to_id[path]
        valid_predictions = [
            kpt for kpt in image_keypoints  if np.any(np.all(~np.isnan(kpt), axis=-1))
        ]
        for kpts in valid_predictions:
            score = float(np.nanmean(kpts[:, 2]).item())
            kpts = kpts.copy()
            kpts[:, 2] = 2

            # NaN predictions to infinity
            kpts[np.isnan(kpts)] = np.inf

            bbox = bbox_from_keypoints(
                keypoints=kpts,
                image_h=h,
                image_w=w,
                margin=bbox_margin,
            )
            area = bbox[2].item() * bbox[3].item()
            coco_predictions.append(
                {
                    "image_id": img_id,
                    "category_id": 1,
                    "keypoints": kpts.reshape(-1).tolist(),
                    "bbox": bbox.tolist(),
                    "area": area,
                    "score": score,
                }
            )

    return coco_predictions


def eval_coco(
    ground_truth: dict,
    predictions: list[dict],
    num_joints: int,
) -> float | None:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco = COCO()
        coco.dataset["annotations"] = ground_truth["annotations"]
        coco.dataset["categories"] = ground_truth["categories"]
        coco.dataset["images"] = ground_truth["images"]
        coco.createIndex()

        coco_det = coco.loadRes(predictions)
        coco_eval = COCOeval(coco, coco_det, iouType="keypoints")
        coco_eval.params.kpt_oks_sigmas = np.array(num_joints * [0.1])
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return float(coco_eval.stats[0])

    except ModuleNotFoundError as err:
        print(f"pycocotools is not installed")
