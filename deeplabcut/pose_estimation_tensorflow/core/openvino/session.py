import os
import sys
import subprocess

import numpy as np
from tqdm import tqdm
import cv2
from openvino.inference_engine import IECore, StatusCode

class OpenVINOSession:
    def __init__(self, cfg, device):
        self.ie = IECore()
        self.xml_path = cfg["init_weights"] + ".xml"
        self.device = device

        # Convert a frozen graph to OpenVINO IR format
        if not os.path.exists(self.xml_path):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
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
        self.net = self.ie.read_network(self.xml_path)
        self.input_name = next(iter(self.net.input_info.keys()))
        self.output_name = next(iter(self.net.outputs.keys()))
        self.net.input_info[self.input_name].precision = "U8"
        self.exec_net = None

    def _get_idle_infer_request(self):
        infer_request_id = self.exec_net.get_idle_request_id()
        if infer_request_id < 0:
            status = self.exec_net.wait(num_requests=1)
            if status != StatusCode.OK:
                raise Exception("Wait for idle request failed!")
            infer_request_id = self.exec_net.get_idle_request_id()
            if infer_request_id < 0:
                raise Exception("Invalid request id!")
        return infer_request_id


    def _init_model(self, inp_h, inp_w):
        # For better efficiency, model is initialized for batch_size 1 and every sample processed independently
        inp_shape = [1, 3, inp_h, inp_w]
        self.net.reshape({self.input_name: inp_shape})

        # Load network to device
        config = {}
        if "CPU" in self.device:
            config["CPU_THROUGHPUT_STREAMS"] = "CPU_THROUGHPUT_AUTO"
        if "GPU" in self.device:
            config["GPU_THROUGHPUT_STREAMS"] = "GPU_THROUGHPUT_AUTO"
        self.exec_net = self.ie.load_network(self.net, self.device, config, num_requests=0)


    def run(self, out_name, feed_dict):
        inp_name, inp = next(iter(feed_dict.items()))

        if self.exec_net is None:
            self._init_model(inp.shape[1], inp.shape[2])

        batch_size = inp.shape[0]
        batch_output = np.zeros([batch_size] + self.net.outputs[out_name].shape, dtype=np.float32)

        # List that maps infer requests to index of processed sample from batch.
        # -1 means that request has not been started yet.
        infer_request_input_id = [-1] * len(self.exec_net.requests)

        for inp_id in range(batch_size):
            infer_request_id = self._get_idle_infer_request()
            out_id = infer_request_input_id[infer_request_id]
            request = self.exec_net.requests[infer_request_id]

            # Copy output prediction
            if out_id != -1:
                batch_output[out_id] = request.output_blobs[out_name].buffer

            # Start this request on new data
            infer_request_input_id[infer_request_id] = inp_id
            request.async_infer({inp_name: inp[inp_id].transpose(2, 0, 1)})

        # Wait for the rest of requests
        status = self.exec_net.wait()
        if status != StatusCode.OK:
            raise Exception("Wait for idle request failed!")

        for infer_request_id, out_id in enumerate(infer_request_input_id):
            if out_id == -1:
                continue
            request = self.exec_net.requests[infer_request_id]
            batch_output[out_id] = request.output_blobs[out_name].buffer

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
    inds = [-1] * len(sess.exec_net.requests)  # Keep indices of frames which are currently in processing
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

        # Get idle iference request. If there is processed data - copy to output list
        infer_request_id = sess._get_idle_infer_request()
        out_id = inds[infer_request_id]
        request = sess.exec_net.requests[infer_request_id]

        # Copy output prediction
        if out_id != -1:
            pose = request.output_blobs[sess.output_name].buffer
            pose[:, [0, 1, 2]] = pose[
                :, [1, 0, 2]
            ]  # change order to have x,y,confidence
            pose = np.reshape(
                pose, (1, -1)
            )  # bring into batchsize times x,y,conf etc.
            PredictedData[out_id] = pose

        # Start this request on new data
        inds[infer_request_id] = counter
        request.async_infer({sess.input_name: frame.transpose(2, 0, 1)})

        counter += 1

    # Wait for the rest of requests
    status = sess.exec_net.wait()
    if status != StatusCode.OK:
        raise Exception("Wait for idle request failed!")

    for infer_request_id, out_id in enumerate(inds):
        if out_id == -1:
            continue
        request = sess.exec_net.requests[infer_request_id]
        pose = request.output_blobs[sess.output_name].buffer
        PredictedData[out_id] = pose[:, [1, 0, 2]].reshape(1, -1)

    pbar.close()
    return PredictedData, nframes
