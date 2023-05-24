#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import os
import subprocess

import numpy as np
from tqdm import tqdm
import cv2

try:
    from openvino.runtime import Core, AsyncInferQueue

    is_openvino_available = True
except ImportError:
    is_openvino_available = False


class OpenVINOSession:
    def __init__(self, cfg, device):
        self.core = Core()
        self.xml_path = cfg["init_weights"] + ".xml"
        self.device = device

        # Convert a frozen graph to OpenVINO IR format
        if not os.path.exists(self.xml_path):
            subprocess.run(
                [
                    "mo",
                    "--output_dir",
                    os.path.dirname(cfg["init_weights"]),
                    "--input_model",
                    cfg["init_weights"] + ".pb",
                    "--input_shape",
                    "[1, 747, 832, 3]",
                    "--extensions",
                    os.path.join(os.path.dirname(__file__), "mo_extensions"),
                    "--data_type",
                    "FP16",
                ],
                check=True,
            )

        # Read network into memory
        self.net = self.core.read_model(self.xml_path)
        self.input_name = self.net.inputs[0].get_any_name()
        self.output_name = self.net.outputs[0].get_any_name()
        self.infer_queue = None

    def _init_model(self, inp_h, inp_w):
        # For better efficiency, model is initialized for batch_size 1 and every sample processed independently
        inp_shape = [1, inp_h, inp_w, 3]
        self.net.reshape({self.input_name: inp_shape})

        # Load network to device
        if "CPU" in self.device:
            self.core.set_property(
                "CPU",
                {
                    "CPU_THROUGHPUT_STREAMS": "CPU_THROUGHPUT_AUTO",
                    "CPU_BIND_THREAD": "YES",
                },
            )
        if "GPU" in self.device:
            self.core.set_property(
                "GPU", {"GPU_THROUGHPUT_STREAMS": "GPU_THROUGHPUT_AUTO"}
            )

        compiled_model = self.core.compile_model(self.net, self.device)
        num_requests = compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        print(f"OpenVINO uses {num_requests} inference requests")
        self.infer_queue = AsyncInferQueue(compiled_model, num_requests)

    def run(self, out_name, feed_dict):
        inp_name, inp = next(iter(feed_dict.items()))

        if self.infer_queue is None:
            self._init_model(inp.shape[1], inp.shape[2])

        batch_size = inp.shape[0]
        batch_output = np.zeros(
            [batch_size] + self.net.outputs[out_name].shape, dtype=np.float32
        )

        def completion_callback(request, inp_id):
            output = next(iter(request.results.values()))
            batch_output[out_id] = output

        self.infer_queue.set_callback(completion_callback)

        for inp_id in range(batch_size):
            self.infer_queue.start_async({inp_name: inp[inp_id : inp_id + 1]}, inp_id)

        self.infer_queue.wait_all()

        return batch_output.reshape(-1, 3)


def GetPoseF_OV(cfg, dlc_cfg, sess, inputs, outputs, cap, nframes, batchsize):
    """Prediction of pose"""
    PredictedData = np.zeros((nframes, 3 * len(dlc_cfg["all_joints_names"])))
    ny, nx = int(cap.get(4)), int(cap.get(3))
    if cfg["cropping"]:
        ny, nx = checkcropping(cfg, cap)

    sess._init_model(ny, nx)

    pbar = tqdm(total=nframes)
    counter = 0
    step = max(10, int(nframes / 100))

    def completion_callback(request, inp_id):
        pose = next(iter(request.results.values()))

        pose[:, [0, 1, 2]] = pose[:, [1, 0, 2]]  # change order to have x,y,confidence
        pose = np.reshape(pose, (1, -1))  # bring into batchsize times x,y,conf etc.
        PredictedData[inp_id] = pose

    sess.infer_queue.set_callback(completion_callback)

    while cap.isOpened():
        if counter % step == 0:
            pbar.update(step)
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare input data
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if cfg["cropping"]:
            frame = frame[cfg["y1"] : cfg["y2"], cfg["x1"] : cfg["x2"]]

        sess.infer_queue.start_async(
            {sess.input_name: np.expand_dims(frame, axis=0)}, counter
        )

        counter += 1

    sess.infer_queue.wait_all()

    pbar.close()
    return PredictedData, nframes
