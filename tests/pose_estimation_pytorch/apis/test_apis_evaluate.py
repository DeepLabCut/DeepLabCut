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
from dataclasses import dataclass
from unittest.mock import Mock, patch

import numpy as np
import pytest

import deeplabcut.pose_estimation_pytorch.apis as apis
import deeplabcut.pose_estimation_pytorch.data as data


PREDICT = Mock()


@patch("deeplabcut.pose_estimation_pytorch.apis.evaluation.predict", PREDICT)
@pytest.mark.parametrize("num_individuals", [1, 2, 5])
@pytest.mark.parametrize(
    "bodyparts, error",
    [
        (["nose", "left_ear"], [5, 10]),
        (["nose", "left_ear", "right_ear"], [2, 3, 4]),
    ]
)
def test_evaluate_basic(
    num_individuals: int,
    bodyparts: list[str],
    error: list[float],
) -> None:
    print()
    gt, pred = generate_data(1, num_individuals, len(bodyparts), error)

    pose_runner = Mock()

    PREDICT.return_value = {img: {"bodyparts": pose} for img, pose in pred.items()}
    loader = build_mock_loader(gt, num_individuals, bodyparts)
    results, preds = apis.evaluate(pose_runner, loader, mode="test")
    print("results", results)
    np.testing.assert_almost_equal(results["rmse"], np.mean(error))


@patch("deeplabcut.pose_estimation_pytorch.apis.evaluation.predict", PREDICT)
@pytest.mark.parametrize("num_individuals", [1, 2, 5])
@pytest.mark.parametrize(
    "bodyparts, error",
    [
        (["nose", "left_ear"], [5, 10]),
        (["nose", "left_ear", "right_ear"], [2, 3, 4]),
    ]
)
@pytest.mark.parametrize(
    "unique_bodyparts, unique_error",
    [
        (["top_left"], [2]),
        (["top_left", "bottom_right"], [2, 3]),
    ]
)
def test_evaluate_with_unique_bodyparts(
    num_individuals: int,
    bodyparts: list[str],
    error: list[float],
    unique_bodyparts: list[str],
    unique_error: list[float],
) -> None:
    print()
    num_images = 5
    gt, pred = generate_data(num_images, num_individuals, len(bodyparts), error)
    gt_unique, pred_unique = generate_data(
        num_images, 1, len(unique_bodyparts), unique_error
    )

    pose_runner = Mock()
    PREDICT.return_value = {
        img: {"bodyparts": pose, "unique_bodyparts": pred_unique[img]}
        for img, pose in pred.items()
    }
    loader = build_mock_loader(
        gt, num_individuals, bodyparts, gt_unique=gt_unique, unique=unique_bodyparts
    )
    results, preds = apis.evaluate(pose_runner, loader, mode="test")
    idv_errors = np.tile(error, (num_individuals, 1)).reshape(-1)
    expected_rmse = np.mean(np.concatenate([idv_errors, unique_error]))
    print(num_individuals)
    print(error)
    print(idv_errors)
    print(unique_error)
    print(np.concatenate([idv_errors, unique_error]))
    print(expected_rmse)
    print("results", results)
    np.testing.assert_almost_equal(results["rmse"], expected_rmse)


@dataclass
class CompTestConfig:
    num_individuals: int = 1
    bodyparts: tuple[str, ...] = ("nose", "left_ear")
    error: tuple[float, ...] = (5, 10)
    unique_bodyparts: tuple[str, ...] = ("top_left", )
    unique_error: tuple[float, ...] = (2, )
    comparison_bodyparts: str | list[str] | None = None
    expected_error: float = (2 + 5 + 10) / 3

    def num_bpt(self) -> int:
        return len(self.bodyparts)

    def num_unique(self) -> int:
        return len(self.unique_bodyparts)


@patch("deeplabcut.pose_estimation_pytorch.apis.evaluation.predict", PREDICT)
@pytest.mark.parametrize(
    "cfg",
    [
        CompTestConfig(comparison_bodyparts=None),
        CompTestConfig(comparison_bodyparts="all"),
        CompTestConfig(comparison_bodyparts=["nose", "left_ear", "top_left"]),
        CompTestConfig(num_individuals=2, expected_error=(2 + 5 + 5 + 10 + 10) / 5),
        CompTestConfig(comparison_bodyparts="nose", expected_error=5),
        CompTestConfig(comparison_bodyparts=["nose"], expected_error=5),
        CompTestConfig(comparison_bodyparts=["left_ear"], expected_error=10),
        CompTestConfig(comparison_bodyparts=["nose", "left_ear"], expected_error=7.5),
        CompTestConfig(comparison_bodyparts="top_left", expected_error=2),
        CompTestConfig(comparison_bodyparts=["top_left"], expected_error=2),
        CompTestConfig(
            unique_bodyparts=("a", "b", "c"),
            unique_error=(3.0, 4.0, 5.0),
            comparison_bodyparts=["a", "b", "c"],
            expected_error=4,
        ),
        CompTestConfig(
            num_individuals=1,
            unique_bodyparts=("a", "b", "c"),
            unique_error=(3.0, 4.0, 5.0),
            comparison_bodyparts=["nose", "a", "b", "c"],
            expected_error=(5.0 + 3.0 + 4.0 + 5.0) / 4,
        ),
        CompTestConfig(
            num_individuals=7,
            unique_bodyparts=("a", "b", "c"),
            unique_error=(3.0, 4.0, 5.0),
            comparison_bodyparts=["nose", "left_ear", "a", "b"],
            expected_error=((7 * 5) + (7 * 10) + 3.0 + 4.0) / (7 + 7 + 2),
        ),
    ]
)
def test_evaluate_with_comparison_bodyparts(cfg: CompTestConfig) -> None:
    print()
    num_images = 5
    gt, pred = generate_data(num_images, cfg.num_individuals, cfg.num_bpt(), cfg.error)
    gt_unique, pred_unique = generate_data(num_images, 1, cfg.num_unique(), cfg.unique_error)

    pose_runner = Mock()
    PREDICT.return_value = {
        img: {"bodyparts": pose, "unique_bodyparts": pred_unique[img]}
        for img, pose in pred.items()
    }
    loader = build_mock_loader(
        gt,
        cfg.num_individuals,
        cfg.bodyparts,
        gt_unique=gt_unique,
        unique=cfg.unique_bodyparts,
    )
    results, preds = apis.evaluate(
        pose_runner, loader, mode="test", comparison_bodyparts=cfg.comparison_bodyparts,
    )
    print(cfg)
    print("results", results)
    np.testing.assert_almost_equal(results["rmse"], cfg.expected_error)


@dataclass
class KeypointData:
    img: int
    idv: int
    bodypart: str
    gt: tuple[float, float]
    pred: tuple[float, float]
    score: float

    def image(self) -> str:
        return f"image_{self.img:04d}.png"

    def error(self) -> float:
        return np.linalg.norm(
            np.asarray(self.gt, dtype=float) - np.asarray(self.pred, dtype=float)
        ).item()


@patch("deeplabcut.pose_estimation_pytorch.apis.evaluation.predict", PREDICT)
@pytest.mark.parametrize(
    "pcutoff", [0.4, 0.6, 0.8, [0.3, 0.5, 0.7]],
)
@pytest.mark.parametrize(
    "keypoints", [
        [
            KeypointData(img=0, idv=0, bodypart="a", gt=(10, 10), pred=(11, 10), score=0.7),
            KeypointData(img=0, idv=0, bodypart="b", gt=(20, 20), pred=(21, 20), score=0.7),
            KeypointData(img=0, idv=0, bodypart="c", gt=(20, 20), pred=(20, 22), score=0.5),
        ],
        [
            KeypointData(img=0, idv=0, bodypart="a", gt=(10, 10), pred=(11, 10), score=0.7),
            KeypointData(img=0, idv=0, bodypart="b", gt=(20, 20), pred=(21, 20), score=0.5),
            KeypointData(img=0, idv=0, bodypart="c", gt=(30, 30), pred=(30, 32), score=0.2),
            KeypointData(img=0, idv=1, bodypart="a", gt=(40, 10), pred=(41, 10), score=0.7),
            KeypointData(img=0, idv=1, bodypart="b", gt=(50, 20), pred=(49, 20), score=0.5),
            KeypointData(img=0, idv=1, bodypart="c", gt=(60, 20), pred=(58, 20), score=0.2),
        ],
    ]
)
def test_evaluate_with_pcutoff(
    pcutoff: float | list[float],
    keypoints: list[KeypointData],
) -> None:
    print()

    images = {d.image() for d in keypoints}
    individuals = list({d.idv for d in keypoints if d.idv != -1})
    bodyparts = list({d.bodypart for d in keypoints if d.idv != -1})
    unique_bodyparts = list({d.bodypart for d in keypoints if d.idv == -1})

    num_idv = len(individuals)
    num_bodyparts = len(bodyparts)
    num_unique = len(unique_bodyparts)

    gt, pred = {}, {}
    for img in images:
        gt[img] = np.zeros((num_idv, num_bodyparts, 3))
        pred[img] = np.zeros((num_idv, num_bodyparts, 3))

    errors = []
    errors_cutoff = []
    for kpt in keypoints:
        img = kpt.image()
        bpt = bodyparts.index(kpt.bodypart)

        gt[img][kpt.idv, bpt, :2] = kpt.gt
        gt[img][kpt.idv, bpt, 2] = 2
        pred[img][kpt.idv, bpt, :2] = kpt.pred
        pred[img][kpt.idv, bpt, 2] = kpt.score

        if isinstance(pcutoff, list):
            bpt_cutoff = pcutoff[bpt]
        else:
            bpt_cutoff = pcutoff

        errors.append(kpt.error())
        if kpt.score >= bpt_cutoff:
            errors_cutoff.append(kpt.error())

    print(errors)
    print(errors_cutoff)

    pose_runner = Mock()
    PREDICT.return_value = {img: {"bodyparts": pose} for img, pose in pred.items()}
    loader = build_mock_loader(gt, num_idv, bodyparts)
    results, preds = apis.evaluate(pose_runner, loader, mode="test", pcutoff=pcutoff)
    print("results", results)
    np.testing.assert_almost_equal(results["rmse"], np.mean(errors))
    np.testing.assert_almost_equal(results["rmse_pcutoff"], np.mean(errors_cutoff))
    if "rmse_detections" in results:
        np.testing.assert_almost_equal(
            results["rmse_detections"], np.mean(errors)
        )
        np.testing.assert_almost_equal(
            results["rmse_detections_pcutoff"], np.mean(errors_cutoff)
        )


@patch("deeplabcut.pose_estimation_pytorch.apis.evaluation.predict", PREDICT)
@pytest.mark.parametrize(
    "pcutoff", [
        0.4,
        0.6,
        0.8,
        [0.3, 0.5, 0.7, 0.4, 0.6],
        [0.25, 0.43, 0.61, 0.46, 0.92],
        [0.12, 0.15, 0.92, 0.97, 0.85],
        [0.92, 0.97, 0.85, 0.12, 0.15],
    ],
)
@pytest.mark.parametrize(
    "keypoints", [
        [
            KeypointData(img=0, idv=0, bodypart="a", gt=(10, 10), pred=(11, 10), score=0.7),
            KeypointData(img=0, idv=0, bodypart="b", gt=(20, 20), pred=(21, 20), score=0.7),
            KeypointData(img=0, idv=0, bodypart="c", gt=(20, 20), pred=(20, 22), score=0.5),
            KeypointData(img=0, idv=-1, bodypart="u1", gt=(20, 20), pred=(20, 22), score=0.5),
            KeypointData(img=0, idv=-1, bodypart="u2", gt=(20, 20), pred=(20, 22), score=0.3),
        ],
        [
            KeypointData(img=0, idv=0, bodypart="a", gt=(10, 10), pred=(11, 10), score=0.7),
            KeypointData(img=0, idv=0, bodypart="b", gt=(20, 20), pred=(21, 20), score=0.5),
            KeypointData(img=0, idv=0, bodypart="c", gt=(30, 30), pred=(30, 32), score=0.2),
            KeypointData(img=0, idv=1, bodypart="a", gt=(40, 10), pred=(41, 10), score=0.7),
            KeypointData(img=0, idv=1, bodypart="b", gt=(50, 20), pred=(49, 20), score=0.5),
            KeypointData(img=0, idv=1, bodypart="c", gt=(60, 20), pred=(58, 20), score=0.2),
            KeypointData(img=0, idv=-1, bodypart="u1", gt=(2, 3), pred=(3, 3), score=0.7),
            KeypointData(img=0, idv=-1, bodypart="u2", gt=(20, 20), pred=(20, 22), score=0.9),
        ],
        [
            KeypointData(img=0, idv=0, bodypart="a", gt=(8, 13), pred=(11, 10), score=0.7),
            KeypointData(img=0, idv=0, bodypart="b", gt=(20, 27), pred=(21, 20), score=0.5),
            KeypointData(img=0, idv=0, bodypart="c", gt=(30, 36), pred=(30, 32), score=0.2),
            KeypointData(img=0, idv=-1, bodypart="u1", gt=(2, 3), pred=(3, 3), score=0.7),
            KeypointData(img=0, idv=-1, bodypart="u2", gt=(20, 20), pred=(20, 22), score=0.9),
            KeypointData(img=1, idv=0, bodypart="a", gt=(15, 20), pred=(41, 10), score=0.7),
            KeypointData(img=1, idv=0, bodypart="b", gt=(20, 12), pred=(49, 20), score=0.5),
            KeypointData(img=1, idv=0, bodypart="c", gt=(17, 32), pred=(58, 20), score=0.2),
            KeypointData(img=1, idv=-1, bodypart="u1", gt=(37, 4), pred=(3, 3), score=0.7),
            KeypointData(img=1, idv=-1, bodypart="u2", gt=(12, 6), pred=(20, 22), score=0.9),
        ],
        [
            KeypointData(img=0, idv=0, bodypart="a", gt=(8, 13), pred=(11, 10), score=0.7),
            KeypointData(img=0, idv=0, bodypart="b", gt=(20, 27), pred=(21, 20), score=0.5),
            KeypointData(img=0, idv=-1, bodypart="u1", gt=(30, 36), pred=(30, 32), score=0.2),
            KeypointData(img=0, idv=-1, bodypart="u2", gt=(2, 3), pred=(3, 3), score=0.7),
            KeypointData(img=0, idv=-1, bodypart="u3", gt=(20, 20), pred=(20, 22), score=0.9),
            KeypointData(img=1, idv=0, bodypart="a", gt=(15, 20), pred=(41, 10), score=0.7),
            KeypointData(img=1, idv=0, bodypart="b", gt=(20, 12), pred=(49, 20), score=0.5),
            KeypointData(img=1, idv=-1, bodypart="u1", gt=(17, 32), pred=(58, 20), score=0.2),
            KeypointData(img=1, idv=-1, bodypart="u2", gt=(37, 4), pred=(3, 3), score=0.7),
            KeypointData(img=1, idv=-1, bodypart="u3", gt=(12, 6), pred=(20, 22), score=0.9),
        ]
    ]
)
def test_evaluate_with_pcutoff_and_unique_bodyparts(
    pcutoff: float | list[float],
    keypoints: list[KeypointData],
) -> None:
    print()

    images = {d.image() for d in keypoints}
    individuals = list({d.idv for d in keypoints if d.idv != -1})
    bodyparts = list({d.bodypart for d in keypoints if d.idv != -1})
    unique_bodyparts = list({d.bodypart for d in keypoints if d.idv == -1})

    num_idv = len(individuals)
    num_bodyparts = len(bodyparts)
    num_unique = len(unique_bodyparts)

    gt, pred, gt_unique, pred_unique = {}, {}, {}, {}
    for img in images:
        gt[img] = np.zeros((num_idv, num_bodyparts, 3))
        pred[img] = np.zeros((num_idv, num_bodyparts, 3))
        gt_unique[img] = np.zeros((1, num_unique, 3))
        pred_unique[img] = np.zeros((1, num_unique, 3))

    errors, errors_cutoff = [], []
    for kpt in keypoints:
        img = kpt.image()
        if kpt.idv == -1:
            idv, bpt = 0, unique_bodyparts.index(kpt.bodypart)
            pcutoff_idx = bpt + len(bodyparts)  # offset by number of bodyparts
            gt_data, pred_data = gt_unique[img], pred_unique[img]
        else:
            idv, bpt = kpt.idv, bodyparts.index(kpt.bodypart)
            pcutoff_idx = bpt
            gt_data, pred_data = gt[img], pred[img]

        gt_data[idv, bpt, :2] = kpt.gt
        gt_data[idv, bpt, 2] = 2
        pred_data[idv, bpt, :2] = kpt.pred
        pred_data[idv, bpt, 2] = kpt.score

        if isinstance(pcutoff, list):
            bpt_cutoff = pcutoff[pcutoff_idx]
        else:
            bpt_cutoff = pcutoff

        errors.append(kpt.error())
        if kpt.score >= bpt_cutoff:
            errors_cutoff.append(kpt.error())

    print(errors)
    print(errors_cutoff)

    pose_runner = Mock()
    PREDICT.return_value = {
        img: {"bodyparts": pose, "unique_bodyparts": pred_unique[img]}
        for img, pose in pred.items()
    }
    loader = build_mock_loader(gt, num_idv, bodyparts, gt_unique, unique_bodyparts)
    results, preds = apis.evaluate(pose_runner, loader, mode="test", pcutoff=pcutoff)

    print("results", results)
    np.testing.assert_almost_equal(results["rmse"], np.mean(errors))
    np.testing.assert_almost_equal(results["rmse_pcutoff"], np.mean(errors_cutoff))
    if "rmse_detections" in results:
        np.testing.assert_almost_equal(
            results["rmse_detections"], np.mean(errors)
        )
        np.testing.assert_almost_equal(
            results["rmse_detections_pcutoff"], np.mean(errors_cutoff)
        )


def generate_data(
    num_images: int,
    num_individuals: int,
    num_bodyparts: int,
    error: list[float] | tuple[float, ...] | np.ndarray,
    cutoffs: list[float] | tuple[float, ...] | np.ndarray | None = None,
    error_cutoff: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    num_elems = num_individuals * num_bodyparts
    shape = num_individuals, num_bodyparts, 3
    error = np.asarray(error)
    coord_error = (np.sqrt(2) / 2) * error

    gt, pred = {}, {}
    for img in range(num_images):
        gt_pose = 100 * np.arange(3 * num_elems, dtype=float).reshape(shape)
        gt_pose[..., 2] = 2
        gt[f"img_{img:04d}.png"] = gt_pose

        pred_pose = np.ones(shape, dtype=float)
        pred_pose[..., :2] = gt_pose[..., :2]
        pred_pose[:, :, 0] += coord_error
        pred_pose[:, :, 1] += coord_error
        pred[f"img_{img:04d}.png"] = pred_pose

    if error_cutoff is not None and cutoffs is not None:
        for img in range(num_images):
            gt_pose = 100 * np.arange(3 * num_elems, dtype=float).reshape(shape)
            gt_pose[..., 2] = 2
            gt[f"img_{num_images + img:04d}.png"] = gt_pose

            pred_pose = np.ones(shape, dtype=float)
            pred_pose[..., :2] = gt_pose[..., :2]
            pred_pose[..., 2] = cutoffs
            pred_pose[:, :, 0] += coord_error
            pred_pose[:, :, 1] += coord_error
            pred[f"img_{num_images + img:04d}.png"] = pred_pose

    return gt, pred


def build_mock_loader(
    gt: dict[str, np.ndarray],
    num_individuals: int,
    bodyparts: list[str] | tuple[str, ...],
    gt_unique: dict[str, np.ndarray] | None = None,
    unique: list[str] | tuple[str, ...] | None = None,
) -> Mock:
    if unique is None:
        unique = []

    def _gt(mode: str, unique_bodypart: bool = False) -> dict[str, np.ndarray]:
        if unique_bodypart:
            print("LOADING UNIQUE GT")
            return gt_unique
        print("LOADING GT")
        return gt

    individuals = [f"animal_{i:03d}" for i in range(num_individuals)]
    loader = Mock()
    loader.get_dataset_parameters.return_value = data.PoseDatasetParameters(
        bodyparts=bodyparts,
        unique_bpts=unique,
        individuals=individuals,
    )
    loader.ground_truth_keypoints = _gt
    loader.model_cfg = {
        "metadata": {
            "bodyparts": bodyparts,
            "unique_bodyparts": unique,
            "individuals": individuals,
            "with_identity": False,
        },
        "train_settings": {},
    }
    return loader
