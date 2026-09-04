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
import importlib.metadata
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement

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


def _missing_requirements(extra: str = "") -> set[str]:
    """Return the distributions DeepLabCut needs here that are not installed."""
    try:
        requirements = importlib.metadata.requires("deeplabcut") or []
    except importlib.metadata.PackageNotFoundError:
        return {"deeplabcut"}  # not installed as a distribution; metadata unavailable

    missing: set[str] = set()
    for spec in requirements:
        requirement = Requirement(spec)
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": extra}):
            continue
        try:
            importlib.metadata.distribution(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            missing.add(requirement.name)
    return missing


# Distributions from the [gui] extra that are absent here. The base requirements
# are subtracted so this names only the GUI-specific gap: a broken base install
# should fail the tests below, not quietly narrow them to the non-GUI surface.
_MISSING_GUI_REQUIREMENTS = sorted(_missing_requirements("gui") - _missing_requirements())


def test_expected_top_level_api() -> None:
    assert callable(deeplabcut.analyze_images)
    assert callable(deeplabcut.analyze_videos)
    assert callable(deeplabcut.train_network)


def test_flat_import_remains_supported() -> None:
    from deeplabcut import analyze_images

    assert callable(analyze_images)


def test_all_exports_appear_in_dir() -> None:
    assert set(deeplabcut.__all__) == set(dir(deeplabcut))


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
    extra = _stub_public_names() - set(deeplabcut.__all__)
    assert not extra, f"Stub declares unexpected names: {sorted(extra)}"


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


def _unresolvable_names() -> set[str]:
    """Exports that cannot resolve here, so tests cover the rest rather than skipping."""
    return _stub_gui_names() if _MISSING_GUI_REQUIREMENTS else set()


def test_every_export_resolves() -> None:
    """Resolve the whole public surface, so lazy loading cannot hide a breakage.

    GUI exports are excluded when the GUI extra is not installed.
    """
    skip = _unresolvable_names()
    failures: dict[str, str] = {}
    for name in deeplabcut.__all__:
        if name in skip:
            continue
        try:
            getattr(deeplabcut, name)
        except Exception as exc:  # noqa: BLE001 - report every failure
            failures[name] = f"{type(exc).__name__}: {exc}"
    assert not failures, "exports that fail to resolve:\n" + "\n".join(
        f"  {name}: {error}" for name, error in sorted(failures.items())
    )


def test_all_exports_resolve_in_a_fresh_process() -> None:
    """Resolve the public API in a clean interpreter.

    Complements ``test_every_export_resolves``, which runs inside the pytest
    process. GUI exports are excluded when the GUI extra is missing.
    """
    code = "\n".join(
        [
            "import deeplabcut",
            f"skip = {sorted(_unresolvable_names())!r}",
            "failures = []",
            "for name in deeplabcut.__all__:",
            "    if name in skip:",
            "        continue",
            "    try:",
            "        getattr(deeplabcut, name)",
            "    except Exception as exc:",
            "        failures.append(f'  {name}: {type(exc).__name__}: {exc}')",
            "if failures:",
            "    raise SystemExit('exports that fail to resolve:\\n' + '\\n'.join(failures))",
        ]
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# -----------------------------------------------------------------------------
# Test for deeplabcut.utils namespace
# -----------------------------------------------------------------------------
# These tests are a smoke test for the deeplabcut.utils namespace which used to
# be a flat namespace that was polluted by star imports. They currently lock
# the previously existing behavior. We should consider removing (some of) these
# symbols as public API.
#
# See https://github.com/DeepLabCut/DeepLabCut/pull/3459
# -----------------------------------------------------------------------------


def test_star_imported_names_remain_importable() -> None:
    """A few names from each formerly star-imported module, as a smoke test."""
    from deeplabcut.utils import (  # noqa: F401
        CropVideo,
        KmeansbasedFrameselection,
        VideoProcessor,
        VideoReader,
        convert2_maDLC,
        convertcsv2h5,
        create_labeled_video,
        plot_trajectories,
        read_config,
    )


def test_star_import_pollution_is_not_restored() -> None:
    """Star imports also leaked third-party modules; those stay gone."""
    leaked = [name for name in ("np", "os", "pd", "cv2", "plt", "logger") if name in deeplabcut.utils.__all__]
    assert not leaked, f"star-import leakage should not be re-exported: {leaked}"


def test_namespace_stays_lazy() -> None:
    """Importing the package must not pull in the submodules behind it."""
    code = "\n".join(
        [
            "import sys",
            "import deeplabcut.utils",
            "assert 'deeplabcut.utils.make_labeled_video' not in sys.modules",
            "_ = deeplabcut.utils.VideoReader",
            "assert 'deeplabcut.utils.auxfun_videos' in sys.modules",
        ]
    )
    subprocess.run([sys.executable, "-c", code], check=True)
