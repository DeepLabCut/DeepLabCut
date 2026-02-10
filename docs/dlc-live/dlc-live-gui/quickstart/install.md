# Installation

This page explains how to install **DeepLabCut Live GUI** for interactive, real‑time pose estimation.

We **recommend `uv`** for most users because it is fast, reliable, and handles optional dependencies cleanly.
**Conda is also supported**, especially if you already use it for DeepLabCut or GPU workflows.

---

## System requirements

- **Python ≥ 3.10**
- A working camera backend (see *{doc}`User Guide → Cameras<docs/dlc-live/dlc-live-gui/user_guide/cameras>`*)
- Optional but recommended:
  - **GPU with CUDA** (for real‑time inference)
  - NVIDIA drivers compatible with your PyTorch/TensorFlow version

```{note}
If you use an OpenCV-compatible camera (e.g. USB webcam, OBS virtual camera), you can simply install the package as it comes with OpenCV support by default.
```

---

## Recommended: Install with `uv`

We recommend installing with [`uv`](https://github.com/astral-sh/uv),
but also support installation with `pip` or `conda` (see next section).


### Create and activate a new environment

::::{tab-set}
:::{tab-item} Linux / macOS
```bash
uv venv create dlclivegui
source uv venv activate dlclivegui
```
:::

:::{tab-item} Windows (Command Prompt)
```cmd
uv venv create dlclivegui
.\dlclivegui\Scripts\activate.bat
```
:::

:::{tab-item} Windows (PowerShell)
```powershell
uv venv create dlclivegui
.\dlclivegui\Scripts\Activate.ps1
```
:::
::::

### Install DeepLabCut-Live-GUI

```{danger}
This pre-release version of the package is not currently on PyPI.
Additionally, the package requires `deeplabcut-live` >= 2.0.0 which is similarly not yet on PyPI.
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

:::::{tab-set}

::::{tab-item} PyTorch
```bash
uv pip install -e .[live-latest-pytorch]
```
:::{note}
For detailed installation instructions, please refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).
:::
::::

::::{tab-item} TensorFlow
:::{caution}
Please note TensorFlow is no longer available on Windows for Python > 3.10.
:::
```bash
uv pip install -e .[live-latest-tensorflow]
```
:::{note}
For detailed installation instructions, please refer to the [official TensorFlow installation guide](https://www.tensorflow.org/install/pip).
:::
::::
:::::

### Run the application

After installation, you can start the DeepLabCut Live GUI application with:

```bash
uv run dlclivegui
```

```{important}
Make sure your venv or conda environment is activated before running the application, so it can access the installed dependencies.
```
