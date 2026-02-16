# Installation

This page explains how to install **DeepLabCut-live-GUI** for interactive, real‑time pose estimation.

We support various installation methods to fit different user preferences and workflows, including `uv` and `mamba`/`conda`.

---

## System requirements

## Supported OSes

``````{toggle}
`````{tab-set}
```{tab-item} Windows
- PyTorch supported
- TensorFlow not available for Python > 3.10
```

```{tab-item} Linux
- Full support for both PyTorch and TensorFlow
```

```{tab-item} macOS
- PyTorch MPS support is limited
- TensorFlow CPU-only
```
`````
``````

### Hardware requirements

- Any compatible camera:
  - USB webcam, OBS virtual camera -< OpenCV-compatible camera backends are supported by default
  - Basler
  - Gentl
  - Aravis
- Optional but recommended:
  - **GPU with CUDA** (for real‑time inference)
  - NVIDIA drivers compatible with your PyTorch/TensorFlow version

```{note}
If you use an OpenCV-compatible camera (e.g. USB webcam, OBS virtual camera), you can simply install the package as it comes with OpenCV support by default.
```

### Software requirements

- **Python ≥ 3.10**
- A machine learning framework for inference (instructions below for both):
  - **PyTorch** (recommended for best performance and compatibility)
  - **TensorFlow** (for backwards compatibility with existing models)
- A working camera backend (see *{ref}`file:dlclivegui-camera-support`*)

---

## Choose your installation method

We support several installation methods to fit different user preferences and workflows.

Below we document installation with `uv` and `mamba`/`conda`, but you can also install with other package managers like `pdm` if you prefer.

```{note}
The entire DeepLabCut package and its GUI are not required to use this software, as it is designed to be a lightweight interface for real‑time pose estimation.
```

### Install with `uv`

We recommend installing with [`uv`](https://github.com/astral-sh/uv),
but also support installation with `pip` or `conda` (see next section).

#### Create and activate a new virtual environment

`````{tab-set}
````{tab-item} Linux / macOS
```bash
uv venv dlclivegui
source dlclivegui/bin/activate
```
````

````{tab-item} Windows (Command Prompt)
```cmd
uv venv dlclivegui
.\dlclivegui\Scripts\activate.bat
```
````

````{tab-item} Windows (PowerShell)
```powershell
uv venv dlclivegui
.\dlclivegui\Scripts\Activate.ps1
```
````
`````

#### Install DeepLabCut-live-GUI

```{danger}
This pre-release version of the package is not currently on PyPI.
For this reason, the install process requires cloning the repository and installing from source.
```

#### Clone the repository

```bash
git clone https://github.com/DeepLabCut/DeepLabCut-live-GUI.git
cd DeepLabCut-live-GUI
```

#### Choose inference backend

We offer two distinct inference backends: **PyTorch** and **TensorFlow**.
You can install either or both, but you must choose at least one to run the pose estimation models.

`````{tab-set}

````{tab-item} PyTorch
```bash
uv pip install -e .[pytorch]
```
```{note}
For detailed installation instructions, please refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).
```
````

````{tab-item} TensorFlow
```{caution}
Please note TensorFlow is **no longer available** on **Windows** for **Python > 3.10**.
```

```bash
uv pip install -e .[tf]
```
```{note}
For detailed installation instructions, please refer to the [official TensorFlow installation guide](https://www.tensorflow.org/install/pip).
```
````
`````

### Install with `mamba` or `conda`

#### Create and activate a new conda environment

If you prefer using `mamba` or `conda`, you can create a new environment and install the package with:

```bash
conda create -n dlclivegui python=3.12 # pick your desired Python version
conda activate dlclivegui
```

#### Clone the repository

```bash
git clone https://github.com/DeepLabCut/DeepLabCut-live-GUI.git
cd DeepLabCut-live-GUI
```

#### Install DeepLabCut-live-GUI

Then, install the package with the desired backend:

`````{tab-set}
````{tab-item} PyTorch
```bash
pip install -e .[pytorch]
```
```{note}
For detailed installation instructions, please refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).
```
````

````{tab-item} TensorFlow
```{caution}
Please note TensorFlow is **no longer available** on **Windows** for **Python > 3.10**.
```
```bash
pip install -e .[tf]
```
```{note}
For detailed installation instructions, please refer to the [official TensorFlow installation guide](https://www.tensorflow.org/install/pip).
```
````
`````

## Run the application

After installation, you can start the DeepLabCut-live-GUI application with:

```bash
dlclivegui # OR uv run dlclivegui
```

```{important}
Make sure your venv or conda environment is activated before running the application, so it can access the installed dependencies.
```
