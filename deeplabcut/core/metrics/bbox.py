#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Bounding box metrics

Metrics are currently computed using pycocotools, which can be installed with `pypi`
(see https://github.com/ppwwyyxx/cocoapi/tree/master).
"""
from __future__ import annotations

import numpy as np

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with_pycocotools = True
except ModuleNotFoundError as err:
    with_pycocotools = False


def compute_bbox_metrics(
    ground_truth: dict[str, dict],
    detections: dict[str, dict],
) -> dict[str, float]:
    """Computes bbox mAP and mAR metrics for bounding boxes.

    Args:
        ground_truth: A dictionary mapping image UIDs (such as image paths or filenames)
            to a ground truth labels dict. The labels dict should contain the keys
            "width" (image width), "height" (image height) and "bboxes" (a numpy array
            of shape (num_gt_bboxes, 4) containing the ground truth bounding boxes in
            format xywh).
        detections: A dictionary mapping image UIDs (such as image paths or filenames)
            to a predicted bounding box dict. The detections dict should contain the
            keys "bboxes" (a numpy array of shape (num_detected_bboxes, 4) containing
            the predicted bounding boxes in format xywh) and "scores" (a numpy array of
            length num_detected_bboxes containing the confidence score for each
            predicted bounding box).

    Returns:
        The bounding box mAP/mAR metrics in a dictionary.

    Raises:
        ModuleNotFoundError: if ``pycocotools`` is not installed
        ValueError: if there are mismatches in the keys of ground_truth and detections
    """
    if not with_pycocotools:
        raise ModuleNotFoundError("pycocotools not installed! can't compute bbox mAP")

    if len(detections) != len(ground_truth):
        raise ValueError()

    coco = COCO()
    coco.dataset["annotations"] = []
    coco.dataset["categories"] = [{"id": 1, "name": "animals", "supercategory": "obj"}]
    coco.dataset["images"] = []
    predictions = []
    for idx, (img, gt) in enumerate(ground_truth.items()):
        img_id = idx + 1
        coco.dataset["images"].append(
            {
                "id": img_id,
                "file_name": img,
                "width": gt["width"],
                "height": gt["height"],
            }
        )
        for bbox in gt["bboxes"][:, :4]:
            ann_id = len(coco.dataset["annotations"]) + 1
            coco.dataset["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "area": max(1, (bbox[2] * bbox[3]).item()),
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )

        for bbox, score in zip(detections[img]["bboxes"], detections[img]["scores"]):
            predictions.append(np.array([img_id, *bbox, score, 1]))

    if len(predictions) == 0:
        return {
            "mAP@50:95": 0.0,
            "mAP@50": 0.0,
            "mAP@75": 0.0,
            "mAR@50:95": 0.0,
            "mAR@50": 0.0,
            "mAR@75": 0.0,
        }

    predictions = np.stack(predictions, axis=0)
    coco.createIndex()
    coco_det = coco.loadRes(predictions)
    coco_eval = COCOeval(coco, coco_det, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    return {
        name: val
        for name, val in [
            _get_metric(coco_eval, recall=False),
            _get_metric(coco_eval, recall=False, iou_threshold=0.5),
            _get_metric(coco_eval, recall=False, iou_threshold=0.75),
            _get_metric(coco_eval, recall=True),
            _get_metric(coco_eval, recall=True, iou_threshold=0.5),
            _get_metric(coco_eval, recall=True, iou_threshold=0.75),
        ]
    }


def _get_metric(
    coco_eval: COCOeval,
    recall: bool = False,
    iou_threshold: float | None = None,
    area_rng: str = "all",
    max_dets: int = 100,
) -> tuple[str, float]:
    metric_name = "mAR" if recall else "mAP"
    if iou_threshold is not None:
        thresh = f"{int(100 * iou_threshold)}"
    else:
        low, high = coco_eval.params.iouThrs[0], coco_eval.params.iouThrs[-1]
        thresh = f"{int(100 * low)}:{int(100 * high)}"

    aind = [i for i, aRng in enumerate(coco_eval.params.areaRngLbl) if aRng == area_rng]
    mind = [i for i, mDet in enumerate(coco_eval.params.maxDets) if mDet == max_dets]
    if recall:
        s = coco_eval.eval["recall"]
        if iou_threshold is not None:
            t = np.where(iou_threshold == coco_eval.params.iouThrs)[0]
            s = s[t]
        s = s[:, :, aind, mind]
    else:
        s = coco_eval.eval["precision"]
        if iou_threshold is not None:
            t = np.where(iou_threshold == coco_eval.params.iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]

    if len(s[s > -1]) == 0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s > -1])

    return f"{metric_name}@{thresh}", mean_s
