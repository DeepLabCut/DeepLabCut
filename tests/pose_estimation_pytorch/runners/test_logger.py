#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests loggers."""

from pathlib import Path
from typing import Any

import pytest
import torch

import deeplabcut.pose_estimation_pytorch.runners.logger as logging


class MockImageLogger(logging.ImageLoggerMixin):
    """Mock image logger."""

    def log_images(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, torch.Tensor],
        targets: dict[str, dict[str, torch.Tensor]],
        step: int,
    ) -> None:
        pass


@pytest.mark.parametrize(
    "keypoints",
    [
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        [
            [[float("nan"), float("nan")], [float("nan"), float("nan")]],
        ],
        [
            [[0.0, 0.0], [1, 1], [2, 2]],
        ],
        [[[float("nan"), 0.0], [1, 1], [2, 2]]],
        [[[-1.0, -1.0], [1, 1], [2, 2]]],
        [
            [[-1.0, -1.0], [-1.0, -1.0]],
        ],
        [
            [[-1.0, -1.0], [-1.0, -1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ],
    ],
)
@pytest.mark.parametrize("denormalize", [True, False])
def test_prepare_image(keypoints: list[list[float]], denormalize: bool) -> None:
    image = torch.ones((3, 256, 256))
    keypoints = torch.tensor(keypoints)

    print()
    print(f"IMAGE: {image.shape}")
    print(f"KEYPOINTS: {keypoints.shape}")
    for k in keypoints:
        print(k)
    print()
    print()

    logger = MockImageLogger()
    logger._prepare_image(
        image=image,
        denormalize=denormalize,
        keypoints=keypoints,
        bboxes=None,
    )


def test_csv_logger_resume(tmp_path: Path) -> None:
    """Test CSVLogger preserves data when resuming from snapshot."""
    log_file = tmp_path / "learning_stats.csv"

    # Initial training: log some metrics
    logger1 = logging.CSVLogger(str(tmp_path), "learning_stats.csv")
    logger1.log({"loss": 0.5, "accuracy": 0.8}, step=1)
    logger1.log({"loss": 0.4, "accuracy": 0.9}, step=2)

    assert log_file.exists()
    assert len(logger1._steps) == 2

    # Resume training: should load existing data
    logger2 = logging.CSVLogger(str(tmp_path), "learning_stats.csv")
    assert len(logger2._steps) == 2
    assert logger2._steps == [1, 2]
    assert logger2._metric_store[0]["loss"] == 0.5
    assert logger2._metric_store[1]["accuracy"] == 0.9

    # Log new data: should append, not overwrite
    logger2.log({"loss": 0.3, "accuracy": 0.95}, step=3)
    assert len(logger2._steps) == 3
    assert logger2._steps == [1, 2, 3]


class _FakeRun:
    """Minimal stand-in for a wandb Run."""

    entity = "test-entity"
    project = "test-project"
    id = "test-id"

    def watch(self, model: Any) -> None:
        pass


class _FakeWandb:
    """Fake ``wandb`` module recording ``init`` calls."""

    def __init__(self) -> None:
        self.run = None
        self.init_calls: list[dict[str, Any]] = []
        self.init_error: Exception | None = None

    def init(self, **kwargs: Any) -> _FakeRun:
        self.init_calls.append(kwargs)
        if self.init_error is not None:
            raise self.init_error
        return _FakeRun()

    def finish(self) -> None:
        self.run = None


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> _FakeWandb:
    fake = _FakeWandb()
    monkeypatch.setattr(logging, "wandb", fake, raising=False)
    monkeypatch.setattr(logging, "has_wandb", True)
    return fake


def test_wandb_logger_reports_unknown_init_option(fake_wandb: _FakeWandb, tmp_path: Path) -> None:
    """An option wandb.init rejects must be reported against the project config."""
    cause = TypeError("init() got an unexpected keyword argument 'run_nmae'")
    fake_wandb.init_error = cause

    with pytest.raises(ValueError) as excinfo:
        logging.WandbLogger(model=object(), train_folder=str(tmp_path), run_nmae="typo")

    message = str(excinfo.value)
    assert "run_nmae" in message
    assert "wandb_kwargs" in message
    assert excinfo.value.__cause__ is cause


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"train_folder": "folder"}, "Specify the model to track!"),
        ({"model": object()}, "Specify the train folder!"),
    ],
    ids=["no-model", "no-train-folder"],
)
def test_wandb_logger_validates_before_opening_a_run(
    fake_wandb: _FakeWandb, kwargs: dict[str, Any], match: str
) -> None:
    """Unusable arguments must be rejected before a run is created, to avoid orphans."""
    with pytest.raises(ValueError, match=match):
        logging.WandbLogger(**kwargs)

    assert fake_wandb.init_calls == []


def test_wandb_logger_forwards_options_to_init(fake_wandb: _FakeWandb, tmp_path: Path) -> None:
    logging.WandbLogger(
        project_name="mice",
        run_name="exp1",
        model=object(),
        train_folder=str(tmp_path),
        tags=["a"],
    )

    assert fake_wandb.init_calls == [{"project": "mice", "name": "exp1", "tags": ["a"]}]
    assert (tmp_path / "wandb_info.yaml").exists()
