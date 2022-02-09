import os
import sys
import subprocess

import numpy as np
from openvino.inference_engine import IECore, StatusCode

class OpenVINOSession:
    def __init__(self, cfg):
        self.ie = IECore()
        self.xml_path = cfg["init_weights"] + ".xml"

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
                ],
                check=True,
            )

        # Read network into memory
        self.net = self.ie.read_network(self.xml_path)
        self.input_name = next(iter(self.net.input_info.keys()))
        self.output_name = next(iter(self.net.outputs.keys()))
        self.net.input_info[self.input_name].precision = "U8"

        # Load network to device
        config = {'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'}
        self.exec_net = self.ie.load_network(self.net, "CPU", config, num_requests=0)


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


    def run(self, output, feed_dict):
        inp = next(iter(feed_dict.values()))

        # For better efficiency, model is initialized for batch_size 1 and every sample processed independently
        batch_size = inp.shape[0]
        batch_output = np.zeros([batch_size] + self.net.outputs[self.output_name].shape, dtype=np.float32)

        # List that maps infer requests to index of processed sample from batch.
        # -1 means that request has not been started yet.
        infer_request_input_id = [-1] * len(self.exec_net.requests)

        for inp_id in range(batch_size):
            infer_request_id = self._get_idle_infer_request()
            out_id = infer_request_input_id[infer_request_id]
            request = self.exec_net.requests[infer_request_id]

            # Copy output prediction
            if out_id != -1:
                batch_output[out_id] = request.output_blobs[self.output_name].buffer

            # Start this request on new data
            infer_request_input_id[infer_request_id] = inp_id
            request.async_infer({self.input_name: inp[inp_id].transpose(2, 0, 1)})

        # Wait for the rest of requests
        status = self.exec_net.wait()
        if status != StatusCode.OK:
            raise Exception("Wait for idle request failed!")

        for infer_request_id, out_id in enumerate(infer_request_input_id):
            if out_id == -1:
                continue
            request = self.exec_net.requests[infer_request_id]
            batch_output[out_id] = request.output_blobs[self.output_name].buffer

        return batch_output.reshape(-1, 3)
