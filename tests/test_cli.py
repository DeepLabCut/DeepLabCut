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
"""Unit tests for the ``deeplabcut.cli`` delegation decorator and helpers.

These tests exercise ``delegate_to_api`` against a fake API function so the
Click parameter-source logic (which decides whether a CLI option was supplied
or left at its ``DEFAULT``) is verified without running real DeepLabCut code.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Annotated

import pytest
import typer
from typer.testing import CliRunner

import deeplabcut as dlc
from deeplabcut import cli as cli_module
from deeplabcut.cli import _optional_api_parameters, _parse_overrides, delegate_to_api

runner = CliRunner()


# ---------------------------------------------------------------------------
# _parse_overrides
# ---------------------------------------------------------------------------


def test_parse_overrides_empty_and_none():
    assert _parse_overrides(None) == {}
    assert _parse_overrides([]) == {}


def test_parse_overrides_yaml_values():
    assert _parse_overrides(["shuffle=2", "batch_size=8"]) == {
        "shuffle": 2,
        "batch_size": 8,
    }
    assert _parse_overrides(["flag=true", "ratio=0.5"]) == {
        "flag": True,
        "ratio": 0.5,
    }
    assert _parse_overrides(["items=[1, 2, 3]"]) == {"items": [1, 2, 3]}


def test_parse_overrides_null():
    assert _parse_overrides(["batch_size=null"]) == {"batch_size": None}


def test_parse_overrides_missing_separator():
    with pytest.raises(typer.BadParameter, match="Expected KEY=VALUE"):
        _parse_overrides(["noequals"])


def test_parse_overrides_duplicate_key():
    with pytest.raises(typer.BadParameter, match="supplied more than once"):
        _parse_overrides(["a=1", "a=2"])


# ---------------------------------------------------------------------------
# delegate_to_api
# ---------------------------------------------------------------------------


def _recording_api(calls: list[dict]) -> Callable[..., str]:
    def api_fn(
        config: str,
        shuffle: int = 1,
        batch_size: int | None = None,
        verbose: bool = False,
    ) -> str:
        calls.append(
            {
                "config": config,
                "shuffle": shuffle,
                "batch_size": batch_size,
                "verbose": verbose,
            }
        )
        return "ok"

    return api_fn


def _make_app(api_fn: Callable[..., str]) -> typer.Typer:
    app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

    @delegate_to_api(api_fn)
    def cmd(
        config: Annotated[str, typer.Argument()],
        shuffle: Annotated[int | None, typer.Option()] = None,
    ) -> None:
        pass

    app.command("cmd")(cmd)
    return app


def test_required_arg_forwarded_and_defaults_omitted():
    calls: list[dict] = []
    result = runner.invoke(_make_app(_recording_api(calls)), ["config.yaml"])

    assert result.exit_code == 0
    assert calls == [{"config": "config.yaml", "shuffle": 1, "batch_size": None, "verbose": False}]


def test_explicit_option_forwarded():
    calls: list[dict] = []
    result = runner.invoke(_make_app(_recording_api(calls)), ["config.yaml", "--shuffle", "2"])

    assert result.exit_code == 0
    assert calls == [{"config": "config.yaml", "shuffle": 2, "batch_size": None, "verbose": False}]


def test_set_merges_optional_params():
    calls: list[dict] = []
    result = runner.invoke(
        _make_app(_recording_api(calls)),
        ["config.yaml", "--set", "batch_size=8", "--set", "verbose=true"],
    )

    assert result.exit_code == 0
    assert calls == [{"config": "config.yaml", "shuffle": 1, "batch_size": 8, "verbose": True}]


def test_set_null_passes_none():
    calls: list[dict] = []
    result = runner.invoke(_make_app(_recording_api(calls)), ["config.yaml", "--set", "batch_size=null"])

    assert result.exit_code == 0
    assert calls[0]["batch_size"] is None


def test_set_conflicts_with_exposed_option():
    result = runner.invoke(
        _make_app(_recording_api([])),
        ["config.yaml", "--shuffle", "2", "--set", "shuffle=3"],
        standalone_mode=False,
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, typer.BadParameter)
    assert "must not be supplied through --set" in str(result.exception)


def test_set_unknown_param_rejected():
    result = runner.invoke(
        _make_app(_recording_api([])),
        ["config.yaml", "--set", "nonexistent=1"],
        standalone_mode=False,
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, typer.BadParameter)
    assert "Unknown parameter(s)" in str(result.exception)


def test_command_must_not_define_override_kwargs():
    def api_fn(config: str) -> None:
        pass

    with pytest.raises(TypeError, match="must not define '_override_kwargs'"):

        @delegate_to_api(api_fn)
        def cmd(config: str, _override_kwargs=None) -> None:
            pass


# ---------------------------------------------------------------------------
# **kwargs passthrough
# ---------------------------------------------------------------------------


def test_set_unknown_param_allowed_when_delegate_accepts_kwargs():
    calls: list[dict] = []

    def api_fn(config: str, **kwargs) -> None:
        calls.append({"config": config, **kwargs})

    app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

    @delegate_to_api(api_fn)
    def cmd(config: Annotated[str, typer.Argument()]) -> None:
        pass

    app.command("cmd")(cmd)

    result = runner.invoke(
        app,
        ["config.yaml", "--set", "snapshot_index=-1", "--set", "detector_snapshot_index=0"],
    )

    assert result.exit_code == 0
    assert calls == [{"config": "config.yaml", "snapshot_index": -1, "detector_snapshot_index": 0}]


def test_set_conflict_with_exposed_option_rejected_even_with_kwargs():
    def api_fn(config: str, **kwargs) -> None:
        pass

    app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

    @delegate_to_api(api_fn)
    def cmd(config: Annotated[str, typer.Argument()]) -> None:
        pass

    app.command("cmd")(cmd)

    result = runner.invoke(
        app,
        ["config.yaml", "--set", "config=other"],
        standalone_mode=False,
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, typer.BadParameter)
    assert "must not be supplied through --set" in str(result.exception)


# ---------------------------------------------------------------------------
# Conformance of the real CLI against the public API
# ---------------------------------------------------------------------------

# Command name -> public API function it delegates to. Keeping this explicit
# means a newly added command fails the mapping test and must be reviewed.
COMMAND_DELEGATES = {
    "create-new-project": dlc.create_new_project,
    "add-new-videos": dlc.add_new_videos,
    "extract-frames": dlc.extract_frames,
    "label-frames": dlc.label_frames,
    "check-labels": dlc.check_labels,
    "refine-labels": dlc.refine_labels,
    "create-training-dataset": dlc.create_training_dataset,
    "train-network": dlc.train_network,
    "evaluate-network": dlc.evaluate_network,
    "analyze-videos": dlc.analyze_videos,
    "extract-outlier-frames": dlc.extract_outlier_frames,
    "create-labeled-video": dlc.create_labeled_video,
    "plot-trajectories": dlc.plot_trajectories,
}


def _command_callback(name: str) -> Callable[..., None]:
    for info in cli_module.app.registered_commands:
        if info.name == name:
            return info.callback
    raise KeyError(name)


def test_all_registered_commands_are_mapped():
    registered = {info.name for info in cli_module.app.registered_commands}
    assert registered == set(COMMAND_DELEGATES)


def test_cli_params_are_known_api_params():
    for name, delegate in COMMAND_DELEGATES.items():
        api_params = set(inspect.signature(delegate).parameters)
        callback = _command_callback(name)
        cli_params = inspect.signature(callback.__wrapped__).parameters
        for param in cli_params:
            assert param in api_params, f"{name}: CLI param {param!r} not in API signature"


def test_required_api_params_are_exposed():
    for name, delegate in COMMAND_DELEGATES.items():
        api_sig = inspect.signature(delegate)
        required = {
            p.name
            for p in api_sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        }
        callback = _command_callback(name)
        cli_params = set(inspect.signature(callback.__wrapped__).parameters)
        missing = required - cli_params
        assert not missing, f"{name}: required API param(s) {sorted(missing)} not exposed"


def test_removed_options_left_to_set():
    analyze_cli = set(inspect.signature(_command_callback("analyze-videos").__wrapped__).parameters)
    assert "robust_nframes" not in analyze_cli
    assert "robust_nframes" in _optional_api_parameters(dlc.analyze_videos)

    outlier_cli = set(inspect.signature(_command_callback("extract-outlier-frames").__wrapped__).parameters)
    for param in ("ARdegree", "MAdegree"):
        assert param not in outlier_cli
        assert param in _optional_api_parameters(dlc.extract_outlier_frames)
