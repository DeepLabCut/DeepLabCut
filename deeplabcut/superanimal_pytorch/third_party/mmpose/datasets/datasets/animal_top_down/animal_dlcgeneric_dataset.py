# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from collections import OrderedDict, defaultdict

import json_tricks as json
import numpy as np
from mmcv import Config
from xtcocotools.cocoeval import COCOeval

from ....core.post_processing import oks_nms, soft_oks_nms
from ...builder import DATASETS
from ..base import Kpt2dSviewRgbImgTopDownDataset
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance


@DATASETS.register_module()
class TopDownDLCGenericDataset(Kpt2dSviewRgbImgTopDownDataset):
    """Animal-Pose dataset for animal pose estimation.

    "Cross-domain Adaptation For Animal Pose Estimation" ICCV'2019
    More details can be found in the `paper
    <https://arxiv.org/abs/1908.05806>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    Animal-Pose keypoint indexes::

        0: 'L_Eye',
        1: 'R_Eye',
        2: 'L_EarBase',
        3: 'R_EarBase',
        4: 'Nose',
        5: 'Throat',
        6: 'TailBase',
        7: 'Withers',
        8: 'L_F_Elbow',
        9: 'R_F_Elbow',
        10: 'L_B_Elbow',
        11: 'R_B_Elbow',
        12: 'L_F_Knee',
        13: 'R_F_Knee',
        14: 'L_B_Knee',
        15: 'R_B_Knee',
        16: 'L_F_Paw',
        17: 'R_F_Paw',
        18: 'L_B_Paw',
        19: 'R_B_Paw'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(
        self,
        ann_file,
        img_prefix,
        data_cfg,
        pipeline,
        mask_keypoints=[],
        keep_keypoints=[],
        kpts_sparsity=0.0,
        dataset_info=None,
        test_mode=False,
    ):

        if dataset_info is None:
            warnings.warn(
                "dataset_info is missing. "
                "Check https://github.com/open-mmlab/mmpose/pull/663 "
                "for details.",
                DeprecationWarning,
            )
            # cfg = Config.fromfile('configs/_base_/datasets/animalpose.py')
            # dataset_info = cfg._cfg_dict['dataset_info']

        dataset_info = json.load(dataset_info)["dataset_info"]
        self.keypoint_info = dataset_info["keypoint_info"]

        self.kpt_name_2_id = {}
        for k_id, k in enumerate(self.keypoint_info.values()):
            self.kpt_name_2_id[(k["name"])] = k_id

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode,
        )

        self.kpts_sparsity = kpts_sparsity
        self.use_gt_bbox = data_cfg["use_gt_bbox"]
        self.bbox_file = data_cfg["bbox_file"]
        self.det_bbox_thr = data_cfg.get("det_bbox_thr", 0.0)
        self.use_nms = data_cfg.get("use_nms", True)
        self.soft_nms = data_cfg["soft_nms"]
        self.nms_thr = data_cfg["nms_thr"]
        self.oks_thr = data_cfg["oks_thr"]
        self.vis_thr = data_cfg["vis_thr"]

        if self.kpts_sparsity > 0:
            np.random.seed(0)
            n_kpts = len(self.kpt_name_2_id)
            full_kpts = np.arange(n_kpts)
            mask_keypoints = np.random.choice(
                n_kpts, int(self.kpts_sparsity * n_kpts), replace=False
            )
            self.mask_keypoints = np.array(mask_keypoints)

        else:
            self.mask_keypoints = np.array(mask_keypoints)

        self.keep_keypoints = np.array(keep_keypoints)

        print("mask_keypoints", self.mask_keypoints)

        self.ann_info["use_different_joint_weights"] = False
        self.db = self._get_db()

        print(f"=> num_images: {self.num_images}")
        print(f"=> load {len(self.db)} samples")

    def _get_db(self):

        if (not self.test_mode) or self.use_gt_bbox:
            print("using gt box")
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()

        else:
            # use bbox from detection
            print("using detector box")
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_person_detection_results(self):
        """Load coco person detection results."""
        num_joints = self.ann_info["num_joints"]
        all_boxes = None
        with open(self.bbox_file, "r") as f:
            all_boxes = json.load(f)

        if not all_boxes:
            raise ValueError("=> Load %s fail!" % self.bbox_file)

        print(f"=> Total boxes: {len(all_boxes)}")

        kpt_db = []
        bbox_id = 0
        for det_res in all_boxes:

            # in coco you have to pick category_id =1, but not the case here
            # if det_res['category_id'] != 1:
            #    continue
            image_file = os.path.join(
                self.img_prefix, self.id2name[det_res["image_id"]]
            )
            box = det_res["bbox"]
            score = det_res["score"]

            if score < self.det_bbox_thr:
                print("thr continue")
                continue

            center, scale = self._xywh2cs(*box[:4])
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.ones((num_joints, 3), dtype=np.float32)
            kpt_db.append(
                {
                    "image_file": image_file,
                    "center": center,
                    "scale": scale,
                    "rotation": 0,
                    "bbox": box[:4],
                    "bbox_score": score,
                    "dataset": self.dataset_name,
                    "joints_3d": joints_3d,
                    "joints_3d_visible": joints_3d_visible,
                    "bbox_id": bbox_id,
                }
            )
            bbox_id = bbox_id + 1
        print(
            f"=> Total boxes after filter " f"low score@{self.det_bbox_thr}: {bbox_id}"
        )
        return kpt_db

    def _load_coco_keypoint_annotations(self):
        """Ground truth bbox and keypoints."""
        gt_db = []
        for img_id in self.img_ids:
            gt_db.extend(self._load_coco_keypoint_annotation_kernel(img_id))

        return gt_db

    def _load_coco_keypoint_annotation_kernel(self, img_id):
        """load annotation from COCOAPI.

        Note:
            bbox:[x1, y1, w, h]

        Args:
            img_id: coco image id

        Returns:
            dict: db entry
        """

        img_ann = self.coco.loadImgs(img_id)[0]

        width = img_ann["width"]
        height = img_ann["height"]
        num_joints = self.ann_info["num_joints"]

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)

        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            if "bbox" not in obj:
                print("no bbox continue")
                continue
            x, y, w, h = obj["bbox"]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if ("area" not in obj or obj["area"] > 0) and x2 > x1 and y2 > y1:
                obj["clean_bbox"] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)

        objs = valid_objs

        bbox_id = 0
        rec = []
        for obj in objs:
            if "keypoints" not in obj:
                print("no keypoint continue")
                continue
            if max(obj["keypoints"]) == 0:
                print("max 0 continue")
                continue
            if "num_keypoints" in obj and obj["num_keypoints"] == 0:
                print("num keypoint continue")
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            keypoints = np.array(obj["keypoints"]).reshape(-1, 3)

            if len(self.mask_keypoints) > 0:
                temp_mask = np.array([False] * keypoints.shape[0])
                temp_mask[self.mask_keypoints] = True
                keypoints[temp_mask, :] = 0

            if len(self.keep_keypoints) > 0:
                temp_mask = np.array([True] * keypoints.shape[0])
                for kpt in self.keep_keypoints:
                    temp_mask[self.kpt_name_2_id[kpt]] = False
                keypoints[temp_mask, :] = 0

            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

            center, scale = self._xywh2cs(*obj["clean_bbox"][:4])

            image_file = os.path.join(self.img_prefix, self.id2name[img_id])
            rec.append(
                {
                    "image_file": image_file,
                    "center": center,
                    "scale": scale,
                    "bbox": obj["clean_bbox"][:4],
                    "rotation": 0,
                    "joints_3d": joints_3d,
                    "joints_3d_visible": joints_3d_visible,
                    "dataset": self.dataset_name,
                    "bbox_score": 1,
                    "bbox_id": bbox_id,
                }
            )
            bbox_id = bbox_id + 1

        return rec

    def match_gts_kpts(self, gts, preds):
        # after match, the indices should match
        # this is only for multiple animal
        # assuming gts and preds now point to the same image
        gts_list = [gt["keypoints"] for gt in gts]
        preds_list = [kpt["keypoints"] for kpt in preds]

        arranged_preds_list = []
        num_gts = len(gts_list)
        num_preds = len(preds_list)
        cost_matrix = np.zeros((num_gts, num_preds))

        for i in range(num_gts):
            for j in range(num_preds):
                cost_matrix[i, j] = distance.euclidean(
                    gts_list[i][..., :2].flatten(), preds_list[j][..., :2].flatten()
                )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for i in range(len(gts_list)):
            arranged_preds_list.append(preds[col_ind[i]])

        return gts, arranged_preds_list

    def make_mixed_keypoints(self, image_id, gts, preds, replace=True):
        # assuming gts and kpts are already matched

        for i in range(len(gts[image_id])):

            _gt_keypoints = gts[image_id][i]["keypoints"]

            if i > len(preds[image_id]) - 1:
                continue
            _pred_keypoints = preds[image_id][i]["keypoints"]

            mixed_keypoints = []

            for j in range(_gt_keypoints.shape[0]):
                if _gt_keypoints[j][0] > 0:
                    if replace:
                        mixed_keypoints.append(_gt_keypoints[j])
                    else:
                        mixed_keypoints.append(_pred_keypoints[j])
                else:
                    mixed_keypoints.append(_pred_keypoints[j])

            mixed_keypoints = np.array(mixed_keypoints)

            preds[image_id][i]["keypoints"] = mixed_keypoints
            preds[image_id][i]["bbox"] = gts[image_id][i]["bbox"]
            x1, y1, w, h = gts[image_id][i]["bbox"]
            preds[image_id][i]["num_keypoints"] = gts[image_id][i]["keypoints"].shape[0]
            preds[image_id][i]["area"] = w * h

    def evaluate(
        self,
        outputs,
        res_folder,
        metric="mAP",
        skip_evaluation=False,
        domain_adaptation=False,
        memory_replay=False,
        **kwargs,
    ):
        """Evaluate coco keypoint results. The pose prediction results will be
        saved in ``${res_folder}/result_keypoints.json``.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            outputs (list[dict]): Outputs containing the following items.

                - preds (np.ndarray[N,K,3]): The first two dimensions are \
                    coordinates, score is the third dimension of the array.
                - boxes (np.ndarray[N,6]): [center[0], center[1], scale[0], \
                    scale[1],area, score]
                - image_paths (list[str]): For example, ['data/coco/val2017\
                    /000000393226.jpg']
                - heatmap (np.ndarray[N, K, H, W]): model output heatmap
                - bbox_id (list(int)).
            res_folder (str): Path of directory to save the results.
            metric (str | list[str]): Metric to be performed. Defaults: 'mAP'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ["mAP"]
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f"metric {metric} is not supported")

        res_file = os.path.join(res_folder, "result_keypoints.json")

        kpts = defaultdict(list)
        gts = defaultdict(list)
        _gts = self._get_db()

        for _gt in _gts:
            image_path = _gt["image_file"]
            if image_path not in self.name2id:
                continue
            image_id = self.name2id[image_path]
            gt_keypoints = _gt["joints_3d"]
            # make the confidence 1
            gt_keypoints[..., 2] = 1
            gts[image_id].append(
                {
                    "keypoints": gt_keypoints,
                    "center": _gt["center"],
                    "image_id": image_id,
                    "bbox": _gt["bbox"],
                }
            )

        for output in outputs:
            preds = output["preds"]
            boxes = output["boxes"]
            image_paths = output["image_paths"]
            bbox_ids = output["bbox_ids"]
            batch_size = len(image_paths)
            for i in range(batch_size):
                # this should work for both relative path and abs path
                if image_paths[i] in self.name2id:
                    image_id = self.name2id[image_paths[i]]
                else:
                    image_id = self.name2id[image_paths[i][len(self.img_prefix) :]]

                _pred_keypoints = preds[i]

                bbox = []

                if domain_adaptation or memory_replay:
                    bbox = gts[image_id][i]

                kpts[image_id].append(
                    {
                        "image_path": image_paths[i],
                        "keypoints": _pred_keypoints,
                        "center": boxes[i][0:2],
                        "scale": boxes[i][2:4],
                        "area": boxes[i][4],
                        "score": boxes[i][5],
                        "image_id": image_id,
                        "bbox_id": bbox_ids[i],
                        "bbox": bbox,
                    }
                )

        kpts = self._sort_and_unique_bboxes(kpts)
        num_predictions = 0

        if domain_adaptation or memory_replay:
            for image_id in kpts:
                _gts = gts[image_id]
                _kpts = kpts[image_id]

                _, arranged_pred_kpts = self.match_gts_kpts(_gts, _kpts)
                num_predictions += len(arranged_pred_kpts)
                kpts[image_id] = arranged_pred_kpts
                self.make_mixed_keypoints(image_id, gts, kpts, replace=memory_replay)

        # rescoring and oks nms
        num_joints = self.ann_info["num_joints"]
        vis_thr = self.vis_thr
        oks_thr = self.oks_thr
        valid_kpts = []

        if memory_replay or domain_adaptation:
            self.use_nms = False

        for image_id in kpts.keys():
            img_kpts = kpts[image_id]
            for n_p in img_kpts:
                box_score = n_p["score"]
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p["keypoints"][n_jt][2]
                    if t_s > vis_thr:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p["score"] = kpt_score * box_score

            if self.use_nms:
                nms = soft_oks_nms if self.soft_nms else oks_nms
                keep = nms(list(img_kpts), oks_thr, sigmas=self.sigmas)
                valid_kpts.append([img_kpts[_keep] for _keep in keep])
            else:
                valid_kpts.append(img_kpts)

        num_predictions = 0
        for e in valid_kpts:
            num_predictions += len(e)
        print("num prediction for valid", num_predictions)

        self._write_coco_keypoint_results(valid_kpts, res_file)
        info_str = self._do_python_keypoint_eval(res_file)
        name_value = OrderedDict(info_str)
        if memory_replay or domain_adaptation:
            return valid_kpts
        else:
            return name_value

    def _write_coco_keypoint_results(self, keypoints, res_file):
        """Write results into a json file."""
        data_pack = [
            {
                "cat_id": self._class_to_coco_ind[cls],
                "cls_ind": cls_ind,
                "cls": cls,
                "ann_type": "keypoints",
                "keypoints": keypoints,
            }
            for cls_ind, cls in enumerate(self.classes)
            if not cls == "__background__"
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])

        with open(res_file, "w") as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        """Get coco keypoint results."""
        cat_id = data_pack["cat_id"]
        keypoints = data_pack["keypoints"]
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                print("img kpts 0 length, continue")
                continue

            _key_points = np.array([img_kpt["keypoints"] for img_kpt in img_kpts])
            key_points = _key_points.reshape(-1, self.ann_info["num_joints"] * 3)

            result = [
                {
                    "image_id": img_kpt["image_id"],
                    "category_id": cat_id,
                    "keypoints": key_point.tolist(),
                    "score": float(img_kpt["score"]),
                    "center": img_kpt["center"].tolist(),
                    "scale": img_kpt["scale"].tolist(),
                }
                for img_kpt, key_point in zip(img_kpts, key_points)
            ]

            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_det, "keypoints", self.sigmas)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            "AP",
            "AP .5",
            "AP .75",
            "AP (M)",
            "AP (L)",
            "AR",
            "AR .5",
            "AR .75",
            "AR (M)",
            "AR (L)",
        ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str

    def _sort_and_unique_bboxes(self, kpts, key="bbox_id"):
        """sort kpts and remove the repeated ones."""
        for img_id, persons in kpts.items():
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts
