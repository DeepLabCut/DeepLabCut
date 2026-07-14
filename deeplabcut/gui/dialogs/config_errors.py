#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""User-facing formatting for project configuration errors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError


@dataclass(frozen=True)
class ConfigErrorReport:
    """Description of a configuration error for presentation in the GUI."""

    title: str
    summary: str
    details: str
    technical_details: str


def _format_location(location: tuple[Any, ...]) -> str:
    """Format a Pydantic error location as a readable field path."""
    if not location:
        return "Configuration"

    parts: list[str] = []

    for item in location:
        if isinstance(item, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{item}]"
            else:
                parts.append(f"[{item}]")
        else:
            parts.append(str(item))

    return ".".join(parts)


def _format_input(
    value: Any,
    max_length: int = 120,
) -> str:
    """Format an invalid value without flooding the dialog."""
    text = repr(value)

    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."


def format_config_error(
    config_path: str | Path,
    error: Exception,
) -> ConfigErrorReport:
    """Build a concise, user-facing report for a configuration error."""
    path = Path(config_path)

    if isinstance(error, ValidationError):
        entries: list[str] = []

        for detail in error.errors(include_url=False):
            location = _format_location(tuple(detail.get("loc", ())))
            error_type = detail.get("type")
            message = detail.get(
                "msg",
                "Invalid value",
            )

            if error_type == "extra_forbidden":
                message = "This setting is not supported by the installed DeepLabCut version."
            elif error_type == "missing":
                message = "This required setting is missing."

            entry = f"• {location}: {message}"

            if "input" in detail:
                entry += f"\n  Received: {_format_input(detail['input'])}"

            entries.append(entry)

        count = error.error_count()
        noun = "problem" if count == 1 else "problems"

        return ConfigErrorReport(
            title="Invalid project configuration",
            summary=(f"DeepLabCut found {count} {noun} in the project configuration."),
            details=(f"Configuration file:\n{path}\n\n" + "\n\n".join(entries)),
            technical_details=str(error),
        )

    if isinstance(error, FileNotFoundError):
        return ConfigErrorReport(
            title="Project configuration not found",
            summary=("The selected project configuration does not exist."),
            details=f"Configuration file:\n{path}",
            technical_details=repr(error),
        )

    if isinstance(error, PermissionError):
        return ConfigErrorReport(
            title="Cannot read project configuration",
            summary=("DeepLabCut does not have permission to read the selected configuration."),
            details=f"Configuration file:\n{path}",
            technical_details=repr(error),
        )

    return ConfigErrorReport(
        title="Cannot load project configuration",
        summary=("DeepLabCut could not read the selected project configuration."),
        details=(f"Configuration file:\n{path}\n\n{error}"),
        technical_details=repr(error),
    )
