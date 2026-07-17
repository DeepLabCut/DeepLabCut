#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests for user-facing formatting of configuration errors."""

import pytest

pytest.importorskip("PySide6")  # deeplabcut.gui imports qtpy at package level

from pathlib import Path

from pydantic import ValidationError

from deeplabcut.core.config import ProjectConfig
from deeplabcut.gui.dialogs.config_errors import (
    CONFIG_LOAD_ERRORS,
    format_config_error,
)

CONFIG_PATH = "C:/projects/demo/config.yaml"
# The report renders the path in native form (backslashes on Windows).
CONFIG_PATH_DISPLAY = str(Path(CONFIG_PATH))


def _make_validation_error(cfg: dict) -> ValidationError:
    with pytest.raises(ValidationError) as exc_info:
        ProjectConfig.from_dict(cfg)
    return exc_info.value


def test_validation_error_report_names_the_field():
    error = _make_validation_error({"multianimalproject": "banana"})

    report = format_config_error(CONFIG_PATH, error)

    assert report.title == "Invalid project configuration"
    assert "1 problem" in report.summary
    assert CONFIG_PATH_DISPLAY in report.details
    assert "multianimalproject" in report.details
    assert "'banana'" in report.details  # the received value is shown
    assert report.technical_details  # raw pydantic message preserved


def test_extra_forbidden_gets_friendly_message():
    error = _make_validation_error({"not_a_dlc_setting": 1})

    report = format_config_error(CONFIG_PATH, error)

    assert "not_a_dlc_setting" in report.details
    assert "not supported by the installed DeepLabCut version" in report.details


def test_multiple_errors_are_all_listed():
    error = _make_validation_error({"not_a_dlc_setting": 1, "multianimalproject": "banana"})

    report = format_config_error(CONFIG_PATH, error)

    assert "2 problems" in report.summary
    assert "not_a_dlc_setting" in report.details
    assert "multianimalproject" in report.details


def test_file_not_found_report():
    report = format_config_error(CONFIG_PATH, FileNotFoundError(CONFIG_PATH))

    assert report.title == "Project configuration not found"
    assert CONFIG_PATH_DISPLAY in report.details


def test_permission_error_report():
    report = format_config_error(CONFIG_PATH, PermissionError("denied"))

    assert report.title == "Cannot read project configuration"
    assert "permission" in report.summary.lower()


def test_generic_error_report_includes_message():
    report = format_config_error(CONFIG_PATH, ValueError("config is empty or null"))

    assert report.title == "Cannot load project configuration"
    assert "config is empty or null" in report.details


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("x"),
        PermissionError("x"),
        OSError("x"),
    ],
)
def test_config_load_errors_cover_common_failures(error):
    assert isinstance(error, CONFIG_LOAD_ERRORS)


@pytest.mark.parametrize(
    "error",
    [
        TypeError("x"),
        ValueError("x"),
    ],
)
def test_config_load_errors_excludes_overly_broad_types(error):
    """TypeError / ValueError are too broad to be treated as config-load errors."""
    assert not isinstance(error, CONFIG_LOAD_ERRORS)
