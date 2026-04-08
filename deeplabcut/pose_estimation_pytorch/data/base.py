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
from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

import albumentations as A
import numpy as np
from scipy.optimize import linear_sum_assignment

import deeplabcut.core.config as config_utils
import deeplabcut.pose_estimation_pytorch.config as config
from deeplabcut.pose_estimation_pytorch.data.bboxes import BBoxComputationMethod
from deeplabcut.pose_estimation_pytorch.data.dataset import (
    PoseDataset,
    PoseDatasetParameters,
)
from deeplabcut.pose_estimation_pytorch.data.generative_sampling import (
    GenSamplingConfig,
)
from deeplabcut.pose_estimation_pytorch.data.snapshots import Snapshot, list_snapshots
from deeplabcut.pose_estimation_pytorch.data.utils import (
    _compute_crop_bounds,
    bbox_from_keypoints,
    map_id_to_annotations,
)
from deeplabcut.pose_estimation_pytorch.task import Task

logger = logging.getLogger(__name__)


class DetectorRunnerLike(Protocol):
    """Minimal protocol for any detector inference runner used by the data layer."""

    def inference(
        self,
        images,
        shelf_writer=None,
    ) -> list[dict[str, np.ndarray]]:
        """
        Expected final postprocessed DLC output per image, e.g.
            {"bboxes": np.ndarray[N, 4], "bbox_scores": np.ndarray[N]}
        """
        ...


class Loader(ABC):
    """Abstract class that represents a blueprint for loading and processing dataset
    information.

    Methods:
        load_data(mode: str = 'train') -> dict:
            Abstract method to convert the project configuration to a standard COCO format.
        create_dataset(images: dict = None, annotations: dict = None, transform: object = None,
            mode: str = "train", task: Task = Task.BOTTOM_UP) -> PoseDataset:
            Creates and returns a PoseDataset given a set of images, annotations, and other parameters.
        _compute_bboxes(images, annotations, method:  BBoxComputationMethod | str = BBoxComputationMethod.GT) -> dict:
            Retrieves all bounding boxes based on the specified method.
        get_dataset_parameters(*args, **kwargs) -> dict:
            Returns a dictionary containing dataset parameters derived from the configuration.
    """

    def __init__(
        self,
        project_root: str | Path,
        image_root: str | Path,
        model_config_path: str | Path,
    ) -> None:
        self.project_root = Path(project_root)
        self.image_root = Path(image_root)
        self.model_config_path = Path(model_config_path)
        self.model_cfg = config_utils.read_config_as_dict(str(model_config_path))
        self.pose_task = Task(self.model_cfg["method"])
        self._loaded_data: dict[str, dict[str, list[dict]]] = {}

    @property
    def model_folder(self) -> Path:
        """Returns: The path of the folder containing the model data"""
        return self.model_config_path.parent

    def snapshots(
        self,
        detector: bool = False,
        best_in_last: bool = True,
    ) -> list[Snapshot]:
        """Lists snapshots saved for the model.

        Args:
            detector: If the Loader is for a Top-Down model, passing detector=True
                will return the snapshots for the detector. Otherwise, the snapshots
                for the pose model are returned.
            best_in_last: Whether to place the snapshot with the best performance in the
                last position in the list, even if it wasn't the last epoch.

        Returns:
            The snapshots stored in a folder, sorted by the number of epochs they were
            trained for. If best_in_last=True and a best snapshot exists, it will be the
            last one in the list.
        """
        prefix = self.pose_task.snapshot_prefix
        if detector:
            prefix = Task.DETECT.snapshot_prefix
        return list_snapshots(self.model_folder, prefix, best_in_last=best_in_last)

    def update_model_cfg(self, updates: dict) -> None:
        """Updates the model configuration.

        Args:
            updates: the items to update in the model configuration
        """
        self.model_cfg = config.update_config_by_dotpath(self.model_cfg, updates)
        config_utils.write_config(self.model_config_path, self.model_cfg)

    @abstractmethod
    def load_data(self, mode: str = "train") -> dict[str, list[dict]]:
        """Abstract method to convert the project configuration to a standard coco
        format.

        Raises:
            NotImplementedError: This method must be implemented in the derived classes.
        """
        raise NotImplementedError

    def image_filenames(self, mode: str = "train") -> list[str]:
        """
        Args:
            mode: {"train", "test"} whether to load train or test data

        Returns:
            the image paths for this mode
        """
        if mode not in self._loaded_data:
            self._loaded_data[mode] = self.load_data(mode)

        data = self._loaded_data[mode]
        return [image["file_name"] for image in data["images"]]

    def default_bbox_method(self, task: Task) -> BBoxComputationMethod | None:
        """
        Returns the default bbox source for this loader/task.
        Subclasses may override this to preserve legacy behavior.
        """
        if task in (Task.TOP_DOWN, Task.DETECT):
            return BBoxComputationMethod.GT
        return None

    def ground_truth_keypoints(self, mode: str = "train", unique_bodypart: bool = False) -> dict[str, np.ndarray]:
        """Creates a dictionary containing the ground truth data.

        TODO: make more efficient

        Args:
            mode: {"train", "test"} whether to load train or test data
            unique_bodypart: returns the ground truth for unique bodyparts

        Raises:
            ValueError if unique_bodypart=True but there are no unique bodyparts

        Returns:
            A dict mapping image paths to the ground truth annotations for the mode in
            the format:
                {'image': keypoints with shape (num_individuals, num_keypoints, 2)}
        """
        parameters = self.get_dataset_parameters()
        if unique_bodypart:
            if not parameters.num_unique_bpts > 0:
                raise ValueError("There are no unique bodyparts in this dataset!")
            individuals = ["single"]
            num_bodyparts = parameters.num_unique_bpts
        else:
            individuals = parameters.individuals
            num_bodyparts = parameters.num_joints

        if "weight_init" in self.model_cfg["train_settings"]:
            weight_init_cfg = self.model_cfg["train_settings"]["weight_init"]
            if weight_init_cfg["memory_replay"]:
                conversion_array = weight_init_cfg["conversion_array"]
                num_bodyparts = len(conversion_array)

        if mode not in self._loaded_data:
            self._loaded_data[mode] = self.load_data(mode)
        data = self._loaded_data[mode]

        annotations = self.filter_annotations(data["annotations"], task=Task.BOTTOM_UP)
        img_to_ann_map = map_id_to_annotations(annotations)

        ground_truth_dict = {}
        for image in data["images"]:
            image_path = image["file_name"]
            individual_keypoints = {
                annotations[i]["individual"]: annotations[i]["keypoints"] for i in img_to_ann_map[image["id"]]
            }
            gt_array = np.zeros((len(individuals), num_bodyparts, 3))
            # Keep the shape of the ground truth
            for idv_idx, idv in enumerate(individuals):
                if idv in individual_keypoints:
                    keypoints = individual_keypoints[idv].reshape(num_bodyparts, -1)
                    gt_array[idv_idx, :, :] = keypoints[:, :3]

            ground_truth_dict[image_path] = gt_array

        return ground_truth_dict

    def ground_truth_bboxes(self, mode: str = "train") -> dict[str, dict]:
        """Creates a dictionary containing the ground truth bounding boxes.

        Args:
            mode: {"train", "test"} whether to load train or test data

        Returns:
            A dict mapping image paths to the ground truth annotations for the mode in
            the format:
                {
                    'path/to/image000.png': {
                        "width": (int) the width of the image, in pixels
                        "height": (int) the height of the image, in pixels
                        "bboxes": (np.ndarray) bboxes with shape (num_individuals, xywh)
                    },
                    'path/to/image000.png': {...},
                }
        """
        if mode not in self._loaded_data:
            self._loaded_data[mode] = self.load_data(mode)
        data = self._loaded_data[mode]

        annotations = self.filter_annotations(data["annotations"], task=Task.DETECT)
        img_to_ann_map = map_id_to_annotations(annotations)

        ground_truth_dict = {}
        for image in data["images"]:
            image_path = image["file_name"]
            img_shape = image["height"], image["width"], 3
            bboxes = [annotations[i]["bbox"] for i in img_to_ann_map[image["id"]]]
            if len(bboxes) == 0:
                bboxes = np.zeros((0, 4))
            else:
                bboxes = _compute_crop_bounds(np.stack(bboxes, axis=0), img_shape)

            ground_truth_dict[image_path] = dict(
                width=image["width"],
                height=image["height"],
                bboxes=bboxes,
            )

        return ground_truth_dict

    def create_dataset(
        self,
        transform: A.BaseCompose | None = None,
        mode: str = "train",
        task: Task = Task.BOTTOM_UP,
        detector_runner: DetectorRunnerLike | None = None,
    ) -> PoseDataset:
        """Creates a PoseDataset based on provided arguments."""

        parameters = self.get_dataset_parameters()
        data = self.load_data(mode)

        # load_data() is cached -> never mutate cached annotations
        images = data["images"]
        annotations = copy.deepcopy(data["annotations"])

        if task in (Task.TOP_DOWN, Task.DETECT):
            bbox_method = self._resolve_bbox_method(task=task, detector_runner=detector_runner)
            annotations = self._compute_bboxes(
                images=images,
                annotations=annotations,
                method=bbox_method,
                bbox_margin=self.model_cfg["data"].get("bbox_margin", 20),
                detector_runner=detector_runner,
                bbox_iou_threshold=self.model_cfg["data"].get("bbox_match_iou_threshold", 0.1),
                fallback_to_gt=self.model_cfg["data"].get("bbox_fallback_to_gt", True),
            )

        annotations = self.filter_annotations(annotations, task)

        ctd_config = None
        if self.pose_task == Task.COND_TOP_DOWN:
            ctd_config = GenSamplingConfig(
                bbox_margin=self.model_cfg["data"].get("bbox_margin", 20),
                **self.model_cfg["data"].get("gen_sampling", {}),
            )

        dataset = PoseDataset(
            images=images,
            annotations=annotations,
            transform=transform,
            mode=mode,
            task=task,
            parameters=parameters,
            ctd_config=ctd_config,
        )
        return dataset

    @abstractmethod
    def get_dataset_parameters(self) -> PoseDatasetParameters:
        """Retrieves dataset parameters based on the instance's configuration.

        Returns:
            An instance of the PoseDatasetParameters with the parameters set.
        """
        raise NotImplementedError

    @staticmethod
    def filter_annotations(annotations: list[dict], task: Task) -> list[dict]:
        """Filters annotations based on the task, removing empty annotations.

        For pose estimation tasks, annotations with empty keypoints are removed. For
        detection task, annotations with no bounding boxes are removed

        Args:
            annotations: the annotations to filter
            task: the task for which to filter

        Returns:
            list: the filtered annotations
        """
        filtered_annotations = []
        for annotation in annotations:
            keypoints = annotation["keypoints"].reshape(-1, 3)
            if task in (Task.DETECT, Task.TOP_DOWN) and (annotation["bbox"][2] <= 0 or annotation["bbox"][3] <= 0):
                continue
            elif task != Task.DETECT and np.all(keypoints[:, :2] <= 0):
                continue

            filtered_annotations.append(annotation)

        return filtered_annotations

    def _resolve_bbox_method(
        self,
        task: Task,
        detector_runner: DetectorRunnerLike | None,
    ) -> BBoxComputationMethod | None:
        """
        Priority:
        1. detector_runner provided -> detector boxes
        2. explicit config bbox_source
        3. loader/task default
        """
        if detector_runner is not None:
            return BBoxComputationMethod.DETECTION_BBOX

        configured = self.model_cfg["data"].get("bbox_source")
        if configured is not None:
            return self._coerce_bbox_method(configured)

        default = self.default_bbox_method(task)
        if default is not None:
            return self._coerce_bbox_method(default)

        return None

    @staticmethod
    def _coerce_bbox_method(
        method: BBoxComputationMethod | str | None,
    ) -> BBoxComputationMethod | None:
        if method is None:
            return None
        if isinstance(method, BBoxComputationMethod):
            return method

        normalized = method.strip().lower().replace(" ", "_")
        aliases = {
            "gt": BBoxComputationMethod.GT,
            "keypoints": BBoxComputationMethod.KEYPOINTS,
            "detection_bbox": BBoxComputationMethod.DETECTION_BBOX,
            "detector": BBoxComputationMethod.DETECTION_BBOX,
            "segmentation_mask": BBoxComputationMethod.SEGMENTATION_MASK,
        }
        try:
            return aliases[normalized]
        except KeyError as e:
            raise ValueError(f"Invalid bbox computation method: {method}") from e

    @staticmethod
    def _compute_bboxes(
        images: list[dict],
        annotations: list[dict],
        method: BBoxComputationMethod | str = BBoxComputationMethod.GT,
        bbox_margin: int = 20,
        detector_runner: DetectorRunnerLike | None = None,
        bbox_iou_threshold: float = 0.1,
        fallback_to_gt: bool = True,
    ):
        """TODO: Nastya method of bbox computation (detection bbox, seg. mask, ...)
        Retrieves all bounding boxes based on the given method.

        Args:
            images: A list of images.
            annotations: A list of annotations corresponding to images.
            method (str, optional): Method to use for retrieving bounding boxes. Defaults to 'gt'.
                - 'gt': Ground truth bounding boxes.
                - 'detection bbox': Bounding boxes from detection.
                - 'keypoints': Bounding boxes from keypoints.
                - 'segmentation mask': Bounding boxes from segmentation masks.
            bbox_margin: Margin to add around keypoints when generating bounding boxes.

        Returns:
            list: Updated annotations based on the given method.

        Raises:
            ValueError: If 'bbox' is not found in annotation when method is 'gt'.
            ValueError: If method is not one of 'gt', 'detection bbox', 'keypoints', or 'segmentation mask'.
        """

        if not method:
            return annotations
        if isinstance(method, str):
            try:
                method = BBoxComputationMethod[method.upper()]
            except KeyError as e:
                raise ValueError(f"Invalid bbox computation method: {method}") from e

        if method == BBoxComputationMethod.GT:
            for annotation in annotations:
                if "bbox" not in annotation:
                    raise ValueError(
                        f"Bounding box not found in annotation {annotation}, "
                        "please choose another bbox computation method"
                    )
            return annotations

        elif method == BBoxComputationMethod.KEYPOINTS:
            min_area = 1
            img_id_to_annotations = map_id_to_annotations(annotations)
            for img in images:
                anns = [annotations[idx] for idx in img_id_to_annotations[img["id"]]]
                for a in anns:
                    a["bbox"] = bbox_from_keypoints(
                        keypoints=a["keypoints"],
                        image_h=img["height"],
                        image_w=img["width"],
                        margin=bbox_margin,
                    )
                    a["area"] = max(min_area, (a["bbox"][2] * a["bbox"][3]).item())
            return annotations

        elif method == BBoxComputationMethod.DETECTION_BBOX:
            if detector_runner is None:
                raise ValueError("detector_runner must be provided when method='detection bbox'")

            img_id_to_annotations = map_id_to_annotations(annotations)
            image_inputs = [img["file_name"] for img in images]
            predictions = detector_runner.inference(image_inputs)

            if len(predictions) != len(images):
                raise ValueError(f"Detector returned {len(predictions)} predictions for {len(images)} images")

            num_unmatched = 0
            num_total = 0

            for img, pred in zip(images, predictions, strict=False):
                ann_indices = img_id_to_annotations[img["id"]]

                # Only match real individuals, not unique-bodypart-only annotations
                candidate_ann_indices = [idx for idx in ann_indices if annotations[idx].get("category_id", 1) == 1]

                if len(candidate_ann_indices) == 0:
                    continue

                pred_bboxes = np.asarray(pred.get("bboxes", np.zeros((0, 4))), dtype=np.float32).reshape(-1, 4)
                pred_scores = np.asarray(
                    pred.get("bbox_scores", np.ones((len(pred_bboxes),), dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(-1)

                gt_bboxes = np.stack(
                    [np.asarray(annotations[idx]["bbox"], dtype=np.float32) for idx in candidate_ann_indices],
                    axis=0,
                )

                matches = Loader._match_bboxes_iou(
                    gt_bboxes=gt_bboxes,
                    pred_bboxes=pred_bboxes,
                    pred_scores=pred_scores,
                    iou_threshold=bbox_iou_threshold,
                )

                num_total += len(candidate_ann_indices)

                for local_gt_idx, ann_idx in enumerate(candidate_ann_indices):
                    pred_idx = matches.get(local_gt_idx, None)

                    if pred_idx is None:
                        num_unmatched += 1
                        if not fallback_to_gt:
                            annotations[ann_idx]["bbox"] = np.zeros((4,), dtype=np.float32)
                            annotations[ann_idx]["area"] = 0.0
                        continue

                    matched_bbox = pred_bboxes[pred_idx].astype(np.float32, copy=True)
                    annotations[ann_idx]["bbox"] = matched_bbox
                    annotations[ann_idx]["area"] = max(1.0, float(matched_bbox[2] * matched_bbox[3]))

            if num_total > 0 and num_unmatched > 0:
                logging.info(
                    f"Detector bbox matching: {num_total - num_unmatched}/{num_total} annotations matched "
                    f"(fallback_to_gt={fallback_to_gt})"
                )

            return annotations

        if method == "segmentation mask":
            raise NotImplementedError

        raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        """Convert boxes from xywh -> xyxy."""
        boxes = np.asarray(boxes, dtype=np.float32).copy()
        if boxes.size == 0:
            return boxes.reshape(0, 4)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return boxes

    @staticmethod
    def _bbox_iou_xywh(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """
        Compute pairwise IoU between two sets of boxes in xywh format.
        Returns matrix of shape [len(boxes_a), len(boxes_b)].
        """
        boxes_a = Loader._xywh_to_xyxy(boxes_a)
        boxes_b = Loader._xywh_to_xyxy(boxes_b)

        if len(boxes_a) == 0 or len(boxes_b) == 0:
            return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

        ious = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
        for i, a in enumerate(boxes_a):
            ax1, ay1, ax2, ay2 = a
            a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)

            for j, b in enumerate(boxes_b):
                bx1, by1, bx2, by2 = b
                b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

                ix1 = max(ax1, bx1)
                iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2)
                iy2 = min(ay2, by2)

                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih

                union = a_area + b_area - inter
                if union > 0:
                    ious[i, j] = inter / union

        return ious

    @staticmethod
    def _match_bboxes_iou(
        gt_bboxes: np.ndarray,
        pred_bboxes: np.ndarray,
        pred_scores: np.ndarray | None = None,
        iou_threshold: float = 0.1,
    ) -> dict[int, int]:
        """
        Match predicted boxes to GT boxes using Hungarian assignment on IoU cost.

        Returns:
            dict mapping local_gt_index -> pred_index
        """
        if len(gt_bboxes) == 0 or len(pred_bboxes) == 0:
            return {}

        iou = Loader._bbox_iou_xywh(gt_bboxes, pred_bboxes)

        # Prefer higher score very slightly when IoUs are tied
        cost = 1.0 - iou
        if pred_scores is not None and len(pred_scores) == pred_bboxes.shape[0]:
            score_penalty = (1.0 - pred_scores.reshape(1, -1)) * 1e-6
            cost = cost + score_penalty

        gt_idx, pred_idx = linear_sum_assignment(cost)

        matches: dict[int, int] = {}
        for g, p in zip(gt_idx, pred_idx, strict=False):
            if iou[g, p] >= iou_threshold:
                matches[int(g)] = int(p)

        return matches
