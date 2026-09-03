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
"""Tests for the stub-driven lazy loading of ``deeplabcut``'s top-level API.

``deeplabcut/__init__.pyi`` is the single declarative source of truth for the
public API. ``lazy_loader.attach_stub`` reads it at runtime to install
``__getattr__``, ``__dir__``, and ``__all__``, importing each implementation
module only when its attribute is first accessed. These tests assert that the
stub drives both static discovery and lazy runtime resolution.
"""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import deeplabcut

_STUB_PATH = Path(deeplabcut.__file__).with_name("__init__.pyi")


def _stub_public_names() -> set[str]:
    """Return every public name declared in ``deeplabcut/__init__.pyi``."""
    tree = ast.parse(_STUB_PATH.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _stub_gui_names() -> set[str]:
    """Return the stub's exports that come from ``deeplabcut.gui``.

    Derived from the stub rather than hardcoded, so adding a GUI export cannot
    silently leave it out of the skip below.
    """
    tree = ast.parse(_STUB_PATH.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "gui":
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def _module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def test_expected_top_level_api() -> None:
    assert callable(deeplabcut.analyze_images)
    assert callable(deeplabcut.analyze_videos)
    assert callable(deeplabcut.train_network)


def test_flat_import_remains_supported() -> None:
    from deeplabcut import analyze_images

    assert callable(analyze_images)


def test_all_exports_appear_in_dir() -> None:
    assert set(deeplabcut.__all__) <= set(dir(deeplabcut))


def test_lazy_export_returns_stable_object() -> None:
    from deeplabcut.api.pose_estimation import analyze_images as canonical

    first = deeplabcut.analyze_images
    second = deeplabcut.analyze_images

    assert first is second
    assert first is canonical


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="No deeplabcut attribute"):
        _ = deeplabcut.this_name_does_not_exist


def test_stub_declares_every_runtime_export() -> None:
    missing = set(deeplabcut.__all__) - _stub_public_names()
    assert not missing, f"Runtime exports missing from stub: {sorted(missing)}"


def test_stub_declares_no_unexpected_exports() -> None:
    # ``DEBUG`` is declared directly (``DEBUG: bool``) but is eagerly defined in
    # ``__init__.py``, so ``lazy_loader`` does not add it to ``__all__``.
    extra = _stub_public_names() - set(deeplabcut.__all__)
    assert extra <= {"DEBUG"}, f"Stub declares unexpected names: {sorted(extra)}"


def test_stub_and_py_typed_ship_with_package() -> None:
    pkg_dir = Path(deeplabcut.__file__).parent
    assert (pkg_dir / "__init__.pyi").is_file()
    assert (pkg_dir / "py.typed").is_file()


def test_pose_estimation_is_loaded_lazily() -> None:
    code = (
        "import sys\n"
        "import deeplabcut\n"
        "assert 'deeplabcut.api.pose_estimation' not in sys.modules\n"
        "_ = deeplabcut.analyze_images\n"
        "assert 'deeplabcut.api.pose_estimation' in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_gui_module_is_not_loaded_eagerly() -> None:
    code = "import sys\nimport deeplabcut\nassert 'deeplabcut.gui' not in sys.modules\n"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_torch_tracking_module_is_not_loaded_eagerly() -> None:
    code = "import sys\nimport deeplabcut\nassert 'deeplabcut.pose_tracking_pytorch' not in sys.modules\n"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_import_deeplabcut_is_lightweight() -> None:
    code = "\n".join(
        [
            "import sys",
            "import deeplabcut",
            "heavy = [",
            "    'deeplabcut.api',",
            "    'deeplabcut.create_project',",
            "    'deeplabcut.generate_training_dataset',",
            "    'deeplabcut.utils',",
            "    'deeplabcut.pose_estimation_3d',",
            "    'deeplabcut.pose_estimation_pytorch',",
            "    'deeplabcut.gui',",
            "    'deeplabcut.pose_tracking_pytorch',",
            "    'torch',",
            "    'tensorflow',",
            "]",
            "for mod in heavy:",
            "    assert mod not in sys.modules, f'{mod} imported eagerly'",
        ]
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_every_export_resolves() -> None:
    """Resolve the whole public surface, so lazy loading cannot hide a breakage.

    GUI exports are skipped when Qt is absent
    """
    gui_available = _module_available("PySide6") and _module_available("napari")
    gui_names = _stub_gui_names()
    failures: dict[str, str] = {}
    for name in deeplabcut.__all__:
        if not gui_available and name in gui_names:
            continue
        try:
            getattr(deeplabcut, name)
        except Exception as exc:  # noqa: BLE001 - report every failure
            failures[name] = f"{type(exc).__name__}: {exc}"
    assert not failures, "exports that fail to resolve:\n" + "\n".join(
        f"  {name}: {error}" for name, error in sorted(failures.items())
    )


@pytest.mark.skipif(
    not (_module_available("torch") and _module_available("PySide6")),
    reason="Full dependency set (torch + GUI) required for eager-import validation",
)
def test_eager_import_mode_resolves_all_exports() -> None:
    env = {**os.environ, "EAGER_IMPORT": "1"}
    subprocess.run([sys.executable, "-c", "import deeplabcut"], check=True, env=env)
