# Installation

This page explains how to install **DeepLabCut Live GUI** for interactive, real‑time experiments.

We **recommend `uv`** for most users because it is fast, reliable, and handles optional dependencies cleanly.
**Conda is also supported**, especially if you already use it for DeepLabCut or GPU workflows.

---

## System requirements

- **Python ≥ 3.10**
- A working camera backend (see *User Guide → Cameras*)
- Optional but recommended:
  - **GPU with CUDA** (for real‑time inference)
  - NVIDIA drivers compatible with your PyTorch/TensorFlow version

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

As the package is not currently on PyPI, install directly from GitHub:


```bash
git clone https://github.com/DeepLabCut/DeepLabCut-live-GUI.git
cd DeepLabCut-live-GUI
```

We offer two distinct inference backends:

:::::{tab-set}

::::{tab-item} PyTorch
```bash
uv pip install -e .[pytorch]
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
uv pip install -e .[tensorflow]
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
