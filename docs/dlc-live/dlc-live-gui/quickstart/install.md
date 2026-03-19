# Installation

This page explains how to install **DeepLabCut-live-GUI** for interactive, real‑time pose estimation.

We support various installation methods, including `uv` and `mamba`/`conda`.

```{tip}
If you feel confident you meet the requirements and you just want to get started quickly, see the {ref}`sec:dlclivegui-install-quickstart` section below.
```

---

## System requirements

### Key takeaways

- **On Windows**: If you want TensorFlow, use Python 3.10
- **On macOS**: TensorFlow is only supported on CPU
- **On Linux**: Full support for both PyTorch and TensorFlow

### OS support

| OS | PyTorch | TensorFlow | Notes & recommendations |
| -- | ------- | ---------- | ----- |
| Windows | ✅ | ❌ | Limited TensorFlow support due to lack of official Windows builds for Python 3.11+ onwards |
| Linux | ✅ | ✅ | Full support for both backends |
| macOS | ✅ | ⚠️ | PyTorch MPS support is improving but still has limitations; TensorFlow only supports CPU on macOS |

### Hardware requirements

- Any **compatible camera** (see *{ref}`file:dlclivegui-camera-support`*):
  - **USB webcam, OBS virtual camera** → OpenCV-recognized cameras are accessible by default
  - **Basler**
  - *GenTL [^exp]*
  - *Aravis [^exp]*
- Optional but recommended:
  - **CUDA-capable GPU** (for real‑time inference)
  - NVIDIA drivers compatible with your PyTorch/TensorFlow version

```{note}
If you use an OpenCV-compatible camera (e.g. USB webcam, OBS virtual camera), you can simply install the package as it comes with OpenCV support by default.
```

### Software requirements

- **Python 3.10, 3.11 or 3.12**
- A machine learning framework for inference (instructions below for both):
  - **PyTorch** (recommended for best performance and compatibility)
  - **TensorFlow** (for backwards compatibility with existing models)
- A working camera backend (see *{ref}`file:dlclivegui-camera-support`*)

---
(sec:dlclivegui-install-quickstart)=
## Quickstart (recommended defaults)

```bash
mkdir -p dlclivegui
cd dlclivegui
uv venv -p 3.12 # or desired Python version
source .venv/bin/activate   # Windows: see tabs below
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu<your-cuda-version> # e.g. cu128 for CUDA 12.8, or skip for CPU-only
uv pip install --pre "deeplabcut-live-gui[pytorch]"
dlclivegui
```

## Choose your installation method

Below instructions cover installation with `uv` and `mamba`/`conda`, but you may also install with other package managers like `pdm` if preferred.

```{note}
The main DeepLabCut package and its GUI are not required to use this software, as it is designed to be a lightweight interface for real‑time pose estimation.
```

### Install DeepLabCut-live-GUI

```{important}
The current release is distributed on PyPI as a **pre-release**.
Use `--pre` when installing with `pip`/`uv pip` so the release candidate can be resolved.
```

#### Install with `uv`

We recommend installing with [`uv`](https://github.com/astral-sh/uv),
but also support installation with `pip` or `conda` (see next section).

##### Create and activate a new virtual environment

`````{tab-set}
````{tab-item} Linux / macOS
```bash
uv venv -p 3.12 # or desired Python version
source .venv/bin/activate
```
````

````{tab-item} Windows (Command Prompt)
```cmd
uv venv -p 3.12 # or desired Python version
.\.venv\Scripts\activate.bat
```
````

````{tab-item} Windows (PowerShell)
```powershell
uv venv -p 3.12 # or desired Python version
.\.venv\Scripts\Activate.ps1
```
````
`````

##### Choose inference backend

We offer two distinct inference backends: **PyTorch** and **TensorFlow**.
You may install either or both, but you must **choose at least one** to run the pose estimation models.

`````{tab-set}

````{tab-item} PyTorch
```{important}
To **enable GPU support** and obtain detailed installation instructions,
please refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) and install PyTorch **before** installing the GUI package.
```
```bash
uv pip install --pre "deeplabcut-live-gui[pytorch]"
```
````

````{tab-item} TensorFlow
```{caution}
Please note TensorFlow is **no longer available** on **Windows** for **Python > 3.10**.
```

```bash
uv pip install --pre "deeplabcut-live-gui[tf]"
```
```{note}
For detailed installation instructions, please refer to the [official TensorFlow installation guide](https://www.tensorflow.org/install/pip).
```
````

````{tab-item} Both backends
```{caution}
Installing both backends may increase environment size and dependency resolution time.
```

```bash
uv pip install --pre "deeplabcut-live-gui[pytorch,tf]"
```
````
`````

#### Install with `mamba` or `conda`

##### Create and activate a new conda environment

If you prefer using `mamba` or `conda`, you can create a new environment and install the package with:

```bash
conda create -n dlclivegui python=3.12 # pick your desired Python version
conda activate dlclivegui
```

##### Choose inference backend

We offer two distinct inference backends: **PyTorch** and **TensorFlow**.
You may install either or both, but you must **choose at least one** to run the pose estimation models.

`````{tab-set}

````{tab-item} PyTorch
```{important}
To **enable GPU support** and obtain detailed installation instructions,
please refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) and install PyTorch **before** installing the GUI package.
```
```bash
pip install --pre "deeplabcut-live-gui[pytorch]"
```
````

````{tab-item} TensorFlow
```{caution}
Please note TensorFlow is **no longer available** on **Windows** for **Python > 3.10**.
```

```bash
pip install --pre "deeplabcut-live-gui[tf]"
```
```{note}
For detailed installation instructions, please refer to the [official TensorFlow installation guide](https://www.tensorflow.org/install/pip).
```
````

````{tab-item} Both backends
```{caution}
Installing both backends may increase environment size and dependency resolution time.
```

```bash
pip install --pre "deeplabcut-live-gui[pytorch,tf]"
```
````
`````

## Verify installation

After installation, you can verify that the package is installed correctly with:

```bash
dlclivegui --help
```

## Download and export a model from the model zoo

See the {ref}`file:dlclivegui-pretrained-models` page for instructions on how to programmatically download and export pre-trained models from the DeepLabCut Model Zoo for use in the GUI.

```{important}
We may in the future add more direct, built-in support for browsing and downloading compatible models.
For now, you can use the `dlclive.modelzoo` API to fetch and export models as described in the linked documentation.
```

## Run the application

After installation, you can start the DeepLabCut-live-GUI application with:

```bash
dlclivegui # OR uv run dlclivegui
```

```{important}
Make sure your venv or conda environment is activated before running the application, so it can access the installed dependencies.
```

[^exp]: Support for this backend is currently experimental and may not work out of the box. Please refer to the backend-specific documentation for details and troubleshooting tips, and report any issues you encounter on GitHub to help us improve support for these backends.
