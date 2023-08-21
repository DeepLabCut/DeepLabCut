# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser
import numpy as np
import cv2


from mmpose.apis import (
    inference_top_down_pose_model,
    init_pose_model,
    process_mmdet_results,
    vis_pose_result,
)
from mmpose.datasets import DatasetInfo
from collections import deque
import traceback

try:
    from mmdet.apis import inference_detector, init_detector

    print(inference_detector, init_detector)
    has_mmdet = True
except (ImportError, ModuleNotFoundError) as e:

    has_mmdet = False
    traceback.print_exc()

import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


import numpy as np
from scipy.ndimage import median_filter


class MedianFilter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def update(self, bbox):
        self.buffer.append(bbox)
        return self.compute_median()

    def compute_median(self):
        # Transpose to get separate arrays of x1, y1, x2, y2
        transposed = np.transpose(self.buffer)
        # Compute median for each coordinate
        return [np.median(coordinate) for coordinate in transposed]


class KeypointsMedianFilter:
    def __init__(self, num_kpts, window_size):
        self.window_size = window_size
        self.num_kpts = num_kpts
        # A buffer for each keypoint
        self.buffers = [
            [deque(maxlen=window_size) for _ in range(2)] for _ in range(num_kpts)
        ]

    def update(self, keypoints):
        # Add new keypoints to buffers and compute new median keypoints
        median_kpts = []
        for i, (x, y, _) in enumerate(keypoints):
            self.buffers[i][0].append(x)
            self.buffers[i][1].append(y)
            median_x = np.median(self.buffers[i][0])
            median_y = np.median(self.buffers[i][1])
            median_kpts.append((median_x, median_y))
        return np.array(median_kpts)


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument("det_config", help="Config file for detection")
    parser.add_argument("det_checkpoint", help="Checkpoint file for detection")
    parser.add_argument("pose_config", help="Config file for pose")
    parser.add_argument("pose_checkpoint", help="Checkpoint file for pose")
    parser.add_argument("--video-path", type=str, help="Video path")
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="whether to show visualizations.",
    )
    parser.add_argument("--kpt-median-filter", action="store_true", default=False)
    parser.add_argument(
        "--out-video-root",
        default="",
        help="Root of the output video file. "
        "Default not saving the visualization video.",
    )
    parser.add_argument("--dataset-info", default="")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--det-cat-id",
        type=int,
        default=1,
        help="Category id for bounding box detection model",
    )
    parser.add_argument(
        "--bbox-thr", type=float, default=0.0, help="Bounding box score threshold"
    )
    parser.add_argument(
        "--kpt-thr", type=float, default=0.0, help="Keypoint score threshold"
    )
    parser.add_argument(
        "--radius", type=int, default=8, help="Keypoint radius for visualization"
    )
    parser.add_argument(
        "--thickness", type=int, default=3, help="Link thickness for visualization"
    )

    assert has_mmdet, "Please install mmdet to run the demo."

    args = parser.parse_args()

    assert args.show or (args.out_video_root != "")
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower()
    )
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower()
    )

    dataset = pose_model.cfg.data["test"]["type"]
    if args.dataset_info:
        dataset_info = args.dataset_info
    else:
        dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
    if dataset_info is None:
        warnings.warn(
            "Please set `dataset_info` in the config."
            "Check https://github.com/open-mmlab/mmpose/pull/663 for details.",
            DeprecationWarning,
        )
    else:

        with open(dataset_info, "r") as f:
            dataset_info = json.load(f)["dataset_info"]
        dataset_info = DatasetInfo(dataset_info)

    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), f"Faild to load video file {args.video_path}"

    if args.out_video_root == "":
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        videoWriter = cv2.VideoWriter(
            os.path.join(
                args.out_video_root, f"vis_{os.path.basename(args.video_path)}"
            ),
            fourcc,
            fps,
            size,
        )

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    frame_id = 0
    ret = {}

    time_accumulated = 0
    frame_count = 0
    window_size = 5
    median_filter_instance = MedianFilter(window_size)
    kpt_median_filter_instance = KeypointsMedianFilter(39, 5)
    import time

    while cap.isOpened():
        flag, img = cap.read()
        frame_count += 1
        if not flag:
            print("frame_id", frame_id)
            break
        # test a single image, the resulting box is (x1, y1, x2, y2)
        start = time.time()
        mmdet_results = inference_detector(det_model, img)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

        # test a single image, with a list of bboxes.

        bbox = person_results[0]["bbox"][:4]
        bbox = median_filter_instance.update(bbox)
        person_results[0]["bbox"][:4] = bbox

        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=args.bbox_thr,
            format="xyxy",
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names,
        )
        end = time.time()

        if args.kpt_median_filter:
            kpts = pose_results[0]["keypoints"]
            kpts = kpt_median_filter_instance.update(kpts)
            pose_results[0]["keypoints"][:, :2] = kpts

        time_accumulated += end - start

        ret[frame_id] = pose_results
        frame_id += 1

        # show the results

        vis_img = vis_pose_result(
            pose_model,
            img,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=False,
        )

        if args.show:
            cv2.imshow("Image", vis_img)

        if save_out_video:
            videoWriter.write(vis_img)

        if args.show and cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("frame_count", frame_count)
    print("time_accumulated", time_accumulated)
    print("frame_count / time_accumulated", frame_count / time_accumulated)

    cap.release()
    if save_out_video:
        videoWriter.release()
    if args.show:
        cv2.destroyAllWindows()

    videoname = args.video_path.split("/")[-1].replace(".avi", "")
    with open(os.path.join(args.out_video_root, videoname + ".json"), "w") as f:
        json.dump(ret, f, cls=NumpyEncoder)


if __name__ == "__main__":
    main()
