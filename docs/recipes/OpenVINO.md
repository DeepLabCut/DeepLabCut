# Intel OpenVINO backend

::::{warning}
This feature is currently implemented for TensorFlow-based models only.
::::

DeepLabCut provides an option to run deep learning model with [OpenVINO](https://github.com/openvinotoolkit/openvino) backend.
To enable OpenVINO in your pipeline, use `use_openvino` flag of `analyze_videos` method with one of string values
indicating device:
* ```"CPU"``` - Use CPU. This is a default value.
* ```"GPU"``` - Use GPU (requires OpenCL to be installed). First launch might take some time for kernels initialization.
* ```"MULTI:CPU,GPU"``` - Use CPU and GPU simultaneously. In most cases this option provides the best efficiency.

```python
def analyze_videos(
    ...
    use_openvino="MULTI:CPU,GPU",
)
```

OpenVINO is an optional dependency. You can install it with DeepLabCut by the following command:

```bash
pip install deeplabcut[openvino]
```
