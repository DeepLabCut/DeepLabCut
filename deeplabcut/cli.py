# DeepLabCut Toolbox (deeplabcut.org)
# Licensed under GNU Lesser General Public License v3.0
"""Command-line interface for DeepLabCut.

CLI API rules
-------------
* Every required parameter of a public DeepLabCut API function MUST be
  redefined as a regular argument in the corresponding CLI command.
* Optional API parameters MAY be redefined as named CLI options when they are
  commonly used, deserve dedicated help, or have types that are inconvenient
  or ambiguous to express as YAML.
* Optional parameters not explicitly exposed remain available through the
  repeatable ``--set KEY=VALUE`` option added by ``@delegate_to_api``.
* Parameters exposed as regular CLI arguments or named options MUST NOT also be
  supplied through ``--set``.
* Named options omitted by the user are detected through Click's parameter
  source and are not forwarded, so the public Python API remains the single
  source of truth for defaults.
* Use ``--set KEY=null`` to pass Python ``None`` explicitly for an API parameter
  that is not exposed as a regular CLI argument or named option.
* CLI parameter names SHOULD match the corresponding Python API parameter names.
  User-facing option spellings may still be customized with ``typer.Option``.

The decorated CLI functions intentionally contain no implementation body. They
only declare the stable command-line interface; ``@delegate_to_api`` validates
and forwards the invocation to the public DeepLabCut API.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypeVar

import click
import typer
import yaml
from typing_extensions import ParamSpec

import deeplabcut as dlc

P = ParamSpec("P")
R = TypeVar("R")

app = typer.Typer(
    name="dlc",
    no_args_is_help=True,
    add_completion=True,
    help="DeepLabCut command-line interface.",
)

ConfigArg = Annotated[
    Path,
    typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the project's config.yaml.",
        metavar="CONFIG",
    ),
]

VideosArg = Annotated[
    list[Path],
    typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="One or more video files or directories containing videos.",
        metavar="VIDEO_OR_DIR [VIDEO_OR_DIR ...]",
    ),
]


def _parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """Parse repeatable KEY=VALUE API overrides."""
    parsed: dict[str, Any] = {}

    for pair in pairs or ():
        key, separator, raw_value = pair.partition("=")
        key = key.strip()

        if not separator or not key:
            raise typer.BadParameter(
                f"Expected KEY=VALUE, received {pair!r}.",
                param_hint="--set",
            )
        if key in parsed:
            raise typer.BadParameter(
                f"{key!r} was supplied more than once.",
                param_hint="--set",
            )

        try:
            parsed[key] = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise typer.BadParameter(
                f"Could not parse the YAML value for {key!r}: {raw_value!r}.",
                param_hint="--set",
            ) from exc

    return parsed


def _optional_api_parameters(delegate: Callable[..., Any]) -> tuple[str, ...]:
    """Return named API parameters that may be supplied through --set."""
    return tuple(
        name
        for name, parameter in inspect.signature(delegate).parameters.items()
        if parameter.default is not inspect.Parameter.empty
        and parameter.kind
        not in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
    )


def _overrides_option(delegate: Callable[..., Any]) -> Any:
    """Build command-specific --set help from the API signature."""
    parameters = ", ".join(_optional_api_parameters(delegate))
    accepted = (
        f" Accepted API parameters: {parameters}." if parameters else " This API call has no optional parameters."
    )

    return typer.Option(
        "--set",
        metavar="KEY=VALUE",
        show_default=False,
        help=(
            "Set an optional parameter of the underlying Python API call. "
            "VALUE is parsed as YAML; repeat --set for multiple parameters. "
            "Use --set KEY=null to pass Python None explicitly." + accepted
        ),
    )


def delegate_to_api(
    delegate: Callable[..., R],
) -> Callable[[Callable[P, Any]], Callable[P, R]]:
    """Create a thin CLI command backed by a DeepLabCut API function.

    The decorated function defines the required arguments, any selected named
    options, and command-specific help. This decorator injects ``--set``,
    validates supplied parameter names, uses Click's parameter source to omit
    option defaults, and delegates the call to ``delegate``.
    """
    api_signature = inspect.signature(delegate)
    api_parameters = set(api_signature.parameters)
    accepts_extra_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in api_signature.parameters.values()
    )

    def decorator(command: Callable[P, Any]) -> Callable[P, R]:
        command_signature = inspect.signature(command)
        if "_override_kwargs" in command_signature.parameters:
            raise TypeError(
                f"{command.__name__} must not define '_override_kwargs'; @delegate_to_api adds it automatically."
            )

        overrides_parameter = inspect.Parameter(
            "_override_kwargs",
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=Annotated[
                list[str] | None,
                _overrides_option(delegate),
            ],
        )

        exposed_parameters = set(command_signature.parameters)

        @functools.wraps(command)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            context = click.get_current_context()
            overrides = _parse_overrides(kwargs.pop("_override_kwargs", None))

            forbidden = exposed_parameters & overrides.keys()
            if forbidden:
                names = ", ".join(sorted(forbidden))
                raise typer.BadParameter(
                    f"Parameters exposed by this command must not be supplied through --set: {names}.",
                    param_hint="--set",
                )

            explicit: dict[str, Any] = {}
            for name, value in kwargs.items():
                source = context.get_parameter_source(name)
                if source.name != "DEFAULT":
                    explicit[name] = value

            unknown = overrides.keys() - api_parameters
            if unknown and not accepts_extra_kwargs:
                names = ", ".join(sorted(unknown))
                raise typer.BadParameter(
                    f"Unknown parameter(s) for {delegate.__name__}: {names}.",
                    param_hint="--set",
                )

            return delegate(*args, **explicit, **overrides)

        wrapper.__signature__ = command_signature.replace(  # type: ignore[attr-defined]
            parameters=[
                *command_signature.parameters.values(),
                overrides_parameter,
            ]
        )
        return wrapper

    return decorator


@app.callback()
def root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the DeepLabCut version and exit.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Run a DeepLabCut workflow command."""
    if version:
        typer.echo(dlc.__version__)
        raise typer.Exit()


# Project setup


@app.command("create-new-project")
@delegate_to_api(dlc.create_new_project)
def create_new_project(
    project: Annotated[str, typer.Argument(help="Project name.")],
    experimenter: Annotated[str, typer.Argument(help="Experimenter name.")],
    videos: VideosArg,
    working_directory: Annotated[
        Path | None,
        typer.Option("--working-directory", "-d", file_okay=False, show_default=False),
    ] = None,
    copy_videos: Annotated[
        bool | None,
        typer.Option("--copy-videos/--symlink-videos", show_default=False),
    ] = None,
    video_extensions: Annotated[
        list[str] | None,
        typer.Option("--video-extension", show_default=False, help="Repeatable."),
    ] = None,
    multianimal: Annotated[
        bool | None,
        typer.Option("--multianimal/--single-animal", show_default=False),
    ] = None,
    individuals: Annotated[
        list[str] | None,
        typer.Option("--individual", show_default=False, help="Repeatable."),
    ] = None,
) -> None:
    """Create a project and its config.yaml."""


@app.command("add-new-videos")
@delegate_to_api(dlc.add_new_videos)
def add_new_videos(
    config: ConfigArg,
    videos: VideosArg,
    copy_videos: Annotated[bool | None, typer.Option("--copy-videos/--symlink-videos", show_default=False)] = None,
    extract_frames: Annotated[
        bool | None, typer.Option("--extract-frames/--no-extract-frames", show_default=False)
    ] = None,
) -> None:
    """Add videos to an existing project."""


# Data labeling


@app.command("extract-frames")
@delegate_to_api(dlc.extract_frames)
def extract_frames(
    config: ConfigArg,
    mode: Annotated[str | None, typer.Option(show_default=False)] = None,
    algo: Annotated[str | None, typer.Option(show_default=False)] = None,
    crop: Annotated[bool | None, typer.Option("--crop/--no-crop", show_default=False)] = None,
    userfeedback: Annotated[bool | None, typer.Option("--userfeedback/--no-userfeedback", show_default=False)] = None,
) -> None:
    """Extract frames from project videos for labeling."""


@app.command("label-frames")
@delegate_to_api(dlc.label_frames)
def label_frames(config_path: ConfigArg) -> None:
    """Open the interface for labeling extracted frames."""


@app.command("check-labels")
@delegate_to_api(dlc.check_labels)
def check_labels(
    config: ConfigArg,
    scale: Annotated[float | None, typer.Option(show_default=False)] = None,
    dpi: Annotated[int | None, typer.Option(show_default=False)] = None,
    draw_skeleton: Annotated[
        bool | None, typer.Option("--draw-skeleton/--no-draw-skeleton", show_default=False)
    ] = None,
    visualizeindividuals: Annotated[
        bool | None,
        typer.Option("--visualize-individuals/--no-visualize-individuals", show_default=False),
    ] = None,
) -> None:
    """Visualize labeled frames for inspection."""


@app.command("refine-labels")
@delegate_to_api(dlc.refine_labels)
def refine_labels(config_path: ConfigArg) -> None:
    """Open the interface for refining labels."""


# Training data and models


@app.command("create-training-dataset")
@delegate_to_api(dlc.create_training_dataset)
def create_training_dataset(
    config: ConfigArg,
    num_shuffles: Annotated[int | None, typer.Option("--num-shuffles", "-n", show_default=False)] = None,
    net_type: Annotated[str | None, typer.Option(show_default=False)] = None,
    detector_type: Annotated[str | None, typer.Option(show_default=False)] = None,
    augmenter_type: Annotated[str | None, typer.Option(show_default=False)] = None,
    engine: Annotated[
        dlc.Engine | None,
        typer.Option(show_default=False, help="Training engine."),
    ] = None,
    userfeedback: Annotated[bool | None, typer.Option("--userfeedback/--no-userfeedback", show_default=False)] = None,
) -> None:
    """Create training and test datasets from labeled data."""


@app.command("train-network")
@delegate_to_api(dlc.train_network)
def train_network(
    config: ConfigArg,
    shuffle: Annotated[int | None, typer.Option("--shuffle", "-s", show_default=False)] = None,
    trainingsetindex: Annotated[int | None, typer.Option(show_default=False)] = None,
    modelprefix: Annotated[str | None, typer.Option(show_default=False)] = None,
    device: Annotated[str | None, typer.Option(show_default=False)] = None,
    snapshot_path: Annotated[Path | None, typer.Option(show_default=False)] = None,
    detector_path: Annotated[Path | None, typer.Option(show_default=False)] = None,
    load_head_weights: Annotated[
        bool | None, typer.Option("--load-head-weights/--no-load-head-weights", show_default=False)
    ] = None,
    batch_size: Annotated[int | None, typer.Option(show_default=False)] = None,
    epochs: Annotated[int | None, typer.Option(show_default=False)] = None,
    detector_epochs: Annotated[int | None, typer.Option(show_default=False)] = None,
) -> None:
    """Train a pose-estimation network."""


@app.command("evaluate-network")
@delegate_to_api(dlc.evaluate_network)
def evaluate_network(
    config: ConfigArg,
    shuffles: Annotated[
        list[int] | None, typer.Option("--shuffle", "-s", show_default=False, help="Repeatable.")
    ] = None,
    device: Annotated[str | None, typer.Option(show_default=False)] = None,
    show_errors: Annotated[bool | None, typer.Option("--show-errors/--no-show-errors", show_default=False)] = None,
    comparison_bodyparts: Annotated[
        list[str] | None, typer.Option("--comparison-bodypart", show_default=False, help="Repeatable.")
    ] = None,
    per_keypoint_evaluation: Annotated[
        bool | None,
        typer.Option("--per-keypoint-evaluation/--no-per-keypoint-evaluation", show_default=False),
    ] = None,
) -> None:
    """Evaluate a trained network and store its metrics."""


@app.command("analyze-videos")
@delegate_to_api(dlc.analyze_videos)
def analyze_videos(
    config: ConfigArg,
    videos: VideosArg,
    video_extensions: Annotated[
        list[str] | None, typer.Option("--video-extension", show_default=False, help="Repeatable.")
    ] = None,
    shuffle: Annotated[int | None, typer.Option("--shuffle", "-s", show_default=False)] = None,
    trainingsetindex: Annotated[int | None, typer.Option(show_default=False)] = None,
    save_as_csv: Annotated[bool | None, typer.Option("--save-as-csv/--no-save-as-csv", show_default=False)] = None,
    device: Annotated[str | None, typer.Option(show_default=False)] = None,
    destfolder: Annotated[Path | None, typer.Option(show_default=False)] = None,
    batch_size: Annotated[int | None, typer.Option(show_default=False)] = None,
    detector_batch_size: Annotated[int | None, typer.Option(show_default=False)] = None,
    auto_track: Annotated[bool | None, typer.Option("--auto-track/--no-auto-track", show_default=False)] = None,
    n_tracks: Annotated[int | None, typer.Option(show_default=False)] = None,
    overwrite: Annotated[bool | None, typer.Option("--overwrite/--no-overwrite", show_default=False)] = None,
) -> None:
    """Analyze one or more videos with a trained network."""


@app.command("extract-outlier-frames")
@delegate_to_api(dlc.extract_outlier_frames)
def extract_outlier_frames(
    config: ConfigArg,
    videos: VideosArg,
    shuffle: Annotated[int | None, typer.Option("--shuffle", "-s", show_default=False)] = None,
    outlieralgorithm: Annotated[str | None, typer.Option(show_default=False)] = None,
    epsilon: Annotated[float | None, typer.Option(show_default=False)] = None,
    p_bound: Annotated[float | None, typer.Option("--p-bound", show_default=False)] = None,
    automatic: Annotated[bool | None, typer.Option("--automatic/--interactive", show_default=False)] = None,
) -> None:
    """Extract candidate outlier frames for relabeling."""


# Visualization


@app.command("create-labeled-video")
@delegate_to_api(dlc.create_labeled_video)
def create_labeled_video(
    config: ConfigArg,
    videos: VideosArg,
    shuffle: Annotated[int | None, typer.Option("--shuffle", "-s", show_default=False)] = None,
    filtered: Annotated[bool | None, typer.Option("--filtered/--unfiltered", show_default=False)] = None,
    save_frames: Annotated[bool | None, typer.Option("--save-frames/--no-save-frames", show_default=False)] = None,
    keypoints_only: Annotated[bool | None, typer.Option("--keypoints-only/--full-frame", show_default=False)] = None,
    displayedbodyparts: Annotated[str | None, typer.Option("--bodyparts", show_default=False)] = None,
    displayedindividuals: Annotated[str | None, typer.Option("--individuals", show_default=False)] = None,
    codec: Annotated[str | None, typer.Option(show_default=False)] = None,
    destfolder: Annotated[Path | None, typer.Option(show_default=False)] = None,
) -> None:
    """Render analyzed videos with predicted keypoints overlaid."""


@app.command("plot-trajectories")
@delegate_to_api(dlc.plot_trajectories)
def plot_trajectories(
    config: ConfigArg,
    videos: VideosArg,
    shuffle: Annotated[int | None, typer.Option("--shuffle", "-s", show_default=False)] = None,
    filtered: Annotated[bool | None, typer.Option("--filtered/--unfiltered", show_default=False)] = None,
    displayedbodyparts: Annotated[str | None, typer.Option("--bodyparts", show_default=False)] = None,
    displayedindividuals: Annotated[str | None, typer.Option("--individuals", show_default=False)] = None,
    showfigures: Annotated[bool | None, typer.Option("--show/--no-show", show_default=False)] = None,
    destfolder: Annotated[Path | None, typer.Option(show_default=False)] = None,
    imagetype: Annotated[str | None, typer.Option(show_default=False)] = None,
) -> None:
    """Plot body-part trajectories for analyzed videos."""


def main() -> None:
    app()


if __name__ == "__main__":
    main()
