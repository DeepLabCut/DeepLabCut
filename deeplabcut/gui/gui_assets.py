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
from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from PySide6.QtGui import QIcon, QPixmap

ASSETS_DIR = files("deeplabcut.gui").joinpath("assets")


def resource_bytes(*parts: str) -> bytes:
    """Read a resource bundled inside deeplabcut.gui."""
    return ASSETS_DIR.joinpath(*parts).read_bytes()


def resource_text(*parts: str, encoding: str = "utf-8") -> str:
    """Read a text resource bundled inside deeplabcut.gui."""
    return ASSETS_DIR.joinpath(*parts).read_text(encoding=encoding)


def get_assets_dir() -> Path:
    """Get the path to the assets directory."""
    return Path(ASSETS_DIR)


def get_style_qss() -> str:
    """Get the contents of the style.qss file."""
    return resource_text("style.qss")


def pixmap_from_resource(*parts: str) -> QPixmap:
    pixmap = QPixmap()
    data = resource_bytes(*parts)

    if not pixmap.loadFromData(data):
        joined = "/".join(parts)
        raise FileNotFoundError(f"Could not load GUI resource as QPixmap: {joined}")

    return pixmap


def icon_from_resource(*parts: str) -> QIcon:
    return QIcon(pixmap_from_resource(*parts))
