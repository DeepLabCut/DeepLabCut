import sys

import pytest

pytest.importorskip("PySide6")

from deeplabcut.gui.utils import _build_update_commands, _package_specs_for_update


def test_package_specs_for_update_adds_gui_extra_to_deeplabcut():
    assert _package_specs_for_update(["deeplabcut"]) == ["deeplabcut[gui]"]


def test_package_specs_for_update_preserves_other_packages():
    assert _package_specs_for_update(["napari-deeplabcut"]) == ["napari-deeplabcut"]


def test_package_specs_for_update_handles_mixed_packages():
    assert _package_specs_for_update(["deeplabcut", "napari-deeplabcut"]) == [
        "deeplabcut[gui]",
        "napari-deeplabcut",
    ]


def test_package_specs_for_update_strips_whitespace():
    assert _package_specs_for_update([" deeplabcut ", " napari-deeplabcut "]) == [
        "deeplabcut[gui]",
        "napari-deeplabcut",
    ]


@pytest.mark.parametrize(
    ("available_installers", "expected_backends"),
    [
        ({}, ["pip"]),
        ({"uv": "/mock/bin/uv"}, ["uv", "pip"]),
    ],
)
def test_build_update_commands_backend_order(monkeypatch, available_installers, expected_backends):
    def fake_which(name):
        return available_installers.get(name)

    monkeypatch.setattr("deeplabcut.gui.utils.shutil.which", fake_which)

    commands = _build_update_commands(["deeplabcut", "napari-deeplabcut"])

    assert [backend for backend, _program, _args in commands] == expected_backends


def test_build_update_commands_uses_uv_when_available(monkeypatch):
    monkeypatch.setattr(
        "deeplabcut.gui.utils.shutil.which",
        lambda name: "/mock/bin/uv" if name == "uv" else None,
    )

    commands = _build_update_commands(["deeplabcut", "napari-deeplabcut"])

    assert commands == [
        (
            "uv",
            "/mock/bin/uv",
            [
                "pip",
                "install",
                "--python",
                sys.executable,
                "-U",
                "deeplabcut[gui]",
                "napari-deeplabcut",
            ],
        ),
        (
            "pip",
            sys.executable,
            [
                "-m",
                "pip",
                "install",
                "-U",
                "deeplabcut[gui]",
                "napari-deeplabcut",
            ],
        ),
    ]


def test_build_update_commands_uses_uv_then_pip(monkeypatch):
    installers = {"uv": "/mock/bin/uv"}

    monkeypatch.setattr(
        "deeplabcut.gui.utils.shutil.which",
        lambda name: installers.get(name),
    )

    commands = _build_update_commands(["deeplabcut"])

    assert commands == [
        (
            "uv",
            "/mock/bin/uv",
            [
                "pip",
                "install",
                "--python",
                sys.executable,
                "-U",
                "deeplabcut[gui]",
            ],
        ),
        (
            "pip",
            sys.executable,
            [
                "-m",
                "pip",
                "install",
                "-U",
                "deeplabcut[gui]",
            ],
        ),
    ]


def test_build_update_commands_always_has_pip_fallback(monkeypatch):
    monkeypatch.setattr("deeplabcut.gui.utils.shutil.which", lambda _name: None)

    commands = _build_update_commands(["deeplabcut"])

    assert commands == [
        (
            "pip",
            sys.executable,
            [
                "-m",
                "pip",
                "install",
                "-U",
                "deeplabcut[gui]",
            ],
        )
    ]
