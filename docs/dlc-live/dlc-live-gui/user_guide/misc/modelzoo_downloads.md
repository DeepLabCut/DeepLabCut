(file:dlclivegui-pretrained-models)=
# Pre-trained models

This page explains how to programmatically download and export **pre-trained, GUI-compatible** models from the DeepLabCut Model Zoo using the `dlclive.modelzoo` API, and convert them for use in DLC-ive and by extension, the GUI.

For a reference of available models and their capabilities, see {ref}`file:model-zoo` page.

```{important}
The `superhuman` model is currently not available in the model zoo due to a missing detector export. We are working on adding it back as soon as possible.
```

The core idea is:

- Fetch a *SuperAnimal* model snapshot (weights) from the model zoo.
- Package it together with the corresponding config into a single export artifact (e.g. `exported_superanimal_quadruped_resnet_50.pt`).
- Point the GUI (or your config) to that exported model checkpoint.

```{caution}
The example below targets the PyTorch engine.
If you are using **TensorFlow models**, you will typically point the GUI to a DLC model directory instead.
```

---

## Quick start (PyTorch)

```{note}
This example assumes you have already installed the GUI and its dependencies, including PyTorch.
```

### Example constants

```python
from pathlib import Path

MODELS_DIR = Path("./models")

TORCH_MODEL = "resnet_50"
TORCH_CONFIG = {
    "checkpoint": MODELS_DIR / f"exported_quadruped_{TORCH_MODEL}.pt",
    "super_animal": "superanimal_quadruped",
}
```

### Download + export

```python
from dlclive.modelzoo.pytorch_model_zoo_export import export_modelzoo_model

export_modelzoo_model(
    export_path=TORCH_CONFIG["checkpoint"],
    super_animal=TORCH_CONFIG["super_animal"],
    model_name=TORCH_MODEL,
)

assert TORCH_CONFIG["checkpoint"].exists(), "Export failed"
```

What this does:

1. Creates the destination directory if needed.
2. Downloads the correct model snapshot (weights) for the specified `super_animal` + `model_name`.
3. Writes a **single `.pt` export file** containing the model config and weights.

---

## API reference

### `export_modelzoo_model(export_path, super_animal, model_name, detector_name=None)`

- `export_path` (str | Path): Output path for the exported `.pt` file.
- `super_animal` (str): The model zoo dataset key (e.g. `superanimal_quadruped`).
- `model_name` (str): Backbone / architecture key (e.g. `resnet_50`).
- `detector_name` (str | None): Optional detector weights to bundle alongside the pose model.

Behavior:

- If `export_path` already exists, the function **skips** exporting (and emits a warning).
- If `detector_name` is provided, it downloads and exports the detector weights as well.

---

## What gets saved in the exported `.pt`

The `.pt` file created by `export_modelzoo_model` is a `torch.save(...)` dictionary with (at least) these keys:

- `config`: model configuration loaded via `load_super_animal_config(...)`
- `pose`: a PyTorch `state_dict` (OrderedDict) for the pose model
- `detector`: a PyTorch `state_dict` for the detector (or `None` if not used)


## Example full script

```python
import warnings
from pathlib import Path
from dlclive.modelzoo.pytorch_model_zoo_export import export_modelzoo_model

MODELS_DIR = Path("./models")
model_name = "resnet_50"
super_animal = "superanimal_quadruped"

export_path = MODELS_DIR / "exported_models" / f"exported_{super_animal}_{model_name}.pt"

export_modelzoo_model(
    export_path=export_path,
    super_animal=super_animal,
    model_name=model_name,
)

print(f"Exported model zoo checkpoint to: {export_path}")
```

---

## In the future

- We may in the future integrate the model zoo functionality more tightly into the GUI, allowing you to browse and download models directly from the interface.
