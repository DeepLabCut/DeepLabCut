---
deeplabcut:
  last_content_updated: '2026-02-10'
  last_metadata_updated: '2026-03-06'
  ignore: false
  visibility: online
  status: outdated
  recommendation: update
  notes: Useful but needs to be updated and clarified.
---

(file:hardware-requirements)=

# Technical & hardware considerations

## Quick summary

On our {ref}`install page <sec:hardware-considerations-during-install>`
we highlight that for GPU computing through standard installation you need a NVIDIA GPU, with at least 8 GB of memory. If you have an Intel or AMD GPU, and are on windows, there is an alternative method of installation available which is shown on the [installation tips page](installation-tips) under "How to install Deeplabcut for Intel and AMD GPUs".
Note, some info is repeated here, and will be updated as systems and hardware changes.

### Computer

For reference, we use e.g. Dell workstations (79xx series) with **Ubuntu 16.04 LTS, 18.04 LTS, or 20.04 LTS** and run a Docker container that has TensorFlow, etc. installed (https://github.com/DeepLabCut/Docker4DeepLabCut2.0).

### Computer hardware

Ideally, you will use a strong GPU with *at least* 8GB memory such as the [NVIDIA GeForce 1080 Ti, 2080 Ti, or 3090](https://marketplace.nvidia.com/en-us/consumer/graphics-cards/). A GPU is not strictly necessary, but on a CPU the (training and evaluation) code is considerably slower (10x) for ResNets, but MobileNets and EfficientNets are slightly faster. Still, a GPU will give you a massive speed boost. You might also consider using cloud computing services like [Google cloud/amazon web services](https://github.com/DeepLabCut/DeepLabCut/issues/47) or Google Colaboratory.

```{note}
If you encounter errors during inference related to
`torch.inference_mode` and DirectML, set the environment variable
`DLC_DIRECTML_NO_GRAD=true` before starting Python. This switches the inference
context to `torch.no_grad`, which is compatible with the DirectML execution path.
```

#### Apple Silicon (MPS)

On Apple Silicon Macs, the PyTorch engine can use the GPU through Metal (`mps`):

- **Pose models**: with `device: auto`, ResNet backbones run on MPS. Other
  backbones (e.g. HRNets) stay on the CPU unless `device` is set explicitly in
  `pytorch_config.yaml`.
- **Object detectors** (top-down / multi-animal models): a detector is trained
  on MPS only when MPS is available, the installed torch is a release
  `>= 2.12`, and the detector variant has been validated on MPS. Currently
  validated: `ssdlite` (the default detector), on Apple Silicon with torch
  2.12.1 and 2.13.0. Other variants are trained on the CPU, and a warning says
  why when MPS was requested: on older torch versions, detectors hang on MPS
  (see [#3155](https://github.com/DeepLabCut/DeepLabCut/issues/3155)), and
  training the Faster R-CNN variants on MPS hung the GPU hard enough to trigger a macOS
  watchdog kernel panic and reboot. The cause is a bug in the MPS backward
  kernel of torchvision's `roi_align`
  ([pytorch/vision#9510](https://github.com/pytorch/vision/pull/9510)), fixed
  in torchvision 0.29.0 (which requires torch 2.14). Faster R-CNN training on
  MPS has not been validated against that release yet, so it stays on the CPU
  for now.
- **Which device the detector trains on**: the `device` argument of
  `train_network` applies to both models. Otherwise the detector inherits the
  top-level `device` of `pytorch_config.yaml` unless `detector.device` is set,
  and picks its own device by the rule above when both are `auto` (so a
  top-down HRNet model that stays on the CPU still trains its `ssdlite`
  detector on MPS). To keep the detector on the CPU, set `detector.device: cpu`
  or pass `device="cpu"`; both silence the warning.

### Camera Hardware

The software is very robust to track data from any camera (cell phone cameras, grayscale, color; captured under infrared light, different manufacturers, etc.). See demos on our [website](https://www.mousemotorlab.org/deeplabcut/).

### Software

**Operating System:** Linux (Ubuntu), MacOS\* (Mojave), or Windows 10. However, the authors strongly recommend Ubuntu! \*MacOS does not support NVIDIA GPUs (easily); on Apple Silicon the PyTorch engine can use the GPU through MPS for some models (see *Apple Silicon (MPS)* above). Otherwise we suggest this option for CPU use or a case where the user wants to label data, refine data, etc and then push the project to a cloud resource for GPU computing steps, or use MobileNets.

**Anaconda/Python3:** Anaconda: a free and open source distribution of the Python programming language (download from https://www.anaconda.com/). DeepLabCut is written in Python 3 (https://www.python.org/) and not compatible with Python 2.

**For the TensorFlow Engine:** You will need [TensorFlow](https://www.tensorflow.org/).
We used version 1.0 in the paper, later versions also work with the provided code (we
tested **TensorFlow versions 1.0 to 1.15, and 2.0 to 2.18**); we
recommend TF2.12 for Python 3.10 with GPU support. Note that native GPU support for Windows was dropped after TF version 2.10. We recommend Windows users to install [the Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) if they want to keep GPU support with TensorFlow.

To note, is it possible to run DeepLabCut on your CPU, but it will be VERY slow (see:
[Mathis & Warren](https://www.biorxiv.org/content/early/2018/10/30/457242)). However, this is the preferred path if you want to test
DeepLabCut on your own computer/data before purchasing a GPU, with the added benefit of
a straightforward installation! Otherwise, use our COLAB notebooks for GPU access for
testing.

Docker: We highly recommend advanced users use the supplied [Docker container](docker-containers).

NOTE: [Currently GPU support in Docker Desktop is only available on Windows with the
WSL2 backend.](https://docs.docker.com/desktop/features/gpu/)
