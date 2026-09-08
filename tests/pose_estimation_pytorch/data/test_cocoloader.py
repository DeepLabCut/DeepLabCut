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
"""Tests for COCOLoader dataset parameter handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from deeplabcut.core.config import ProjectConfig
from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
from deeplabcut.pose_estimation_pytorch.data.cocoloader import COCOLoader

BODYPARTS = ["nose", "tail"]


def _ann(image_id: int, ann_id: int, keypoints: list[float] | None = None) -> dict:
    if keypoints is None:
        # two bodyparts visible
        keypoints = [10.0, 10.0, 2.0, 20.0, 20.0, 2.0]
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "keypoints": keypoints,
        "bbox": [5.0, 5.0, 30.0, 30.0],
        "num_keypoints": 2,
        "iscrowd": 0,
        "area": 900.0,
    }


def _coco_dict(
    image_name: str,
    image_id: int,
    n_individuals: int,
    start_ann_id: int = 1,
    bodyparts: list[str] | None = None,
) -> dict:
    return {
        "images": [
            {
                "id": image_id,
                "file_name": image_name,
                "width": 64,
                "height": 64,
            }
        ],
        "annotations": [_ann(image_id, start_ann_id + i) for i in range(n_individuals)],
        "categories": [
            {
                "id": 1,
                "name": "animal",
                "keypoints": bodyparts or BODYPARTS,
                "skeleton": [],
            }
        ],
    }


def _write_project(
    tmp_path: Path,
    *,
    train_n: int,
    test_n: int | None,
    max_individuals: int,
    bodyparts: list[str] | None = None,
    json_bodyparts: list[str] | None = None,
) -> tuple[Path, PoseConfig]:
    """Writes a minimal COCO project (train + optional test json) and a matching PoseConfig.

    Args:
        train_n: number of individuals annotated on the (single) train image.
        test_n: number of individuals annotated on the (single) test image, or None to
            skip writing a test.json altogether.
        max_individuals: the number of individuals to configure in the PoseConfig.
        bodyparts: the bodyparts to put in the PoseConfig (defaults to BODYPARTS).
        json_bodyparts: the bodyparts to put in the COCO json category (defaults to
            `bodyparts`, i.e. matching). Set to something else to simulate a mismatch.
    """
    bodyparts = bodyparts or BODYPARTS
    json_bodyparts = json_bodyparts if json_bodyparts is not None else bodyparts

    project_root = tmp_path / "coco_project"
    ann_dir = project_root / "annotations"
    img_dir = project_root / "images"
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_dir / "train.png")
    Image.new("RGB", (64, 64), color=(64, 64, 64)).save(img_dir / "test.png")

    train = _coco_dict("train.png", image_id=1, n_individuals=train_n, start_ann_id=1, bodyparts=json_bodyparts)
    (ann_dir / "train.json").write_text(json.dumps(train))

    if test_n is not None:
        test = _coco_dict("test.png", image_id=2, n_individuals=test_n, start_ann_id=100, bodyparts=json_bodyparts)
        (ann_dir / "test.json").write_text(json.dumps(test))

    if max_individuals > 1:
        project_config = ProjectConfig(
            project_path=project_root,
            bodyparts="MULTI!",
            multianimalbodyparts=bodyparts,
            individuals=[f"individual{i}" for i in range(max_individuals)],
            multianimalproject=True,
        )
    else:
        project_config = ProjectConfig(
            project_path=project_root,
            bodyparts=bodyparts,
            individuals=["individual0"],
            multianimalproject=False,
        )
    pose_config_path = project_root / "pytorch_config.yaml"
    pose_config = PoseConfig.build(
        project_config,
        pose_config_path,
        top_down=False,
        net_type="resnet_50",
        multi_animal=max_individuals > 1,
    )
    return project_root, pose_config


def _make_loader(project_root: Path, pose_config: PoseConfig, has_test: bool) -> COCOLoader:
    return COCOLoader(
        project_root=project_root,
        model_config=pose_config,
        test_json_filename="test.json" if has_test else "",
    )


def test_max_individuals_in_json():
    coco = _coco_dict("a.png", image_id=1, n_individuals=3)
    coco["annotations"].append(_ann(image_id=1, ann_id=99))  # 4 on same image
    assert COCOLoader._max_individuals_in_json(coco) == 4
    assert COCOLoader._max_individuals_in_json({"annotations": []}) == 0


def test_loader_accepts_capacity_covering_both_splits(tmp_path: Path):
    # train max=3, test max=4, config capacity=4 -> should load without error, using
    # the config's capacity (not the train json's).
    project_root, pose_config = _write_project(tmp_path, train_n=3, test_n=4, max_individuals=4)
    loader = _make_loader(project_root, pose_config, has_test=True)

    params = loader.get_dataset_parameters()
    assert params.max_num_animals == 4
    assert list(params.individuals) == list(pose_config.metadata.individuals)
    assert list(params.bodyparts) == BODYPARTS


def test_loader_raises_when_individuals_exceed_capacity(tmp_path: Path):
    # train max=3, test max=4, config capacity=3 -> test.json needs more individuals
    # than the model supports; this must fail loudly, at construction time.
    project_root, pose_config = _write_project(tmp_path, train_n=3, test_n=4, max_individuals=3)

    with pytest.raises(ValueError, match=r"test\.json has an image with 4 individuals"):
        _make_loader(project_root, pose_config, has_test=True)


def test_loader_raises_when_train_alone_exceeds_capacity(tmp_path: Path):
    # No test.json: the check must still trigger for train.json.
    project_root, pose_config = _write_project(tmp_path, train_n=4, test_n=None, max_individuals=3)

    with pytest.raises(ValueError, match=r"train\.json has an image with 4 individuals"):
        _make_loader(project_root, pose_config, has_test=False)


def test_loader_raises_on_bodypart_mismatch(tmp_path: Path):
    project_root, pose_config = _write_project(
        tmp_path,
        train_n=1,
        test_n=1,
        max_individuals=1,
        bodyparts=["snout", "tailbase"],
        json_bodyparts=BODYPARTS,  # differs from the PoseConfig's bodyparts
    )

    with pytest.raises(ValueError, match="don't match model_cfg.metadata.bodyparts"):
        _make_loader(project_root, pose_config, has_test=True)


def test_loader_ok_without_test_json(tmp_path: Path):
    project_root, pose_config = _write_project(tmp_path, train_n=2, test_n=None, max_individuals=2)
    loader = _make_loader(project_root, pose_config, has_test=False)

    params = loader.get_dataset_parameters()
    assert params.max_num_animals == 2


def test_get_project_parameters_train_only():
    train = _coco_dict("train.png", image_id=1, n_individuals=3)
    num_individuals, bodyparts = COCOLoader.get_project_parameters(train)
    assert num_individuals == 3
    assert list(bodyparts) == BODYPARTS


def test_get_project_parameters_considers_test_json():
    # see https://github.com/DeepLabCut/DeepLabCut/issues/3432
    train = _coco_dict("train.png", image_id=1, n_individuals=3)
    test = _coco_dict("test.png", image_id=2, n_individuals=4)

    num_individuals, bodyparts = COCOLoader.get_project_parameters(train)
    assert num_individuals == 3

    num_individuals, bodyparts = COCOLoader.get_project_parameters(train, test)
    assert num_individuals == 4
    assert list(bodyparts) == BODYPARTS


def test_get_project_parameters_raises_on_empty_train_json():
    empty = _coco_dict("a.png", image_id=1, n_individuals=0)
    with pytest.raises(ValueError, match="No images found"):
        COCOLoader.get_project_parameters(empty)


def test_get_project_parameters_warns_on_multiple_categories():
    train = _coco_dict("train.png", image_id=1, n_individuals=2)
    train["categories"].append({"id": 2, "name": "other", "keypoints": ["eye"], "skeleton": []})

    with pytest.warns(UserWarning, match="more than 1 category"):
        num_individuals, bodyparts = COCOLoader.get_project_parameters(train)

    assert num_individuals == 2
    assert list(bodyparts) == BODYPARTS


def test_get_project_parameters_warns_on_multiple_categories_in_test_json():
    # The same category validation/normalization applied to train.json must also be
    # applied to test.json, not skipped just because we don't read its bodyparts.
    train = _coco_dict("train.png", image_id=1, n_individuals=2)
    test = _coco_dict("test.png", image_id=2, n_individuals=2)
    test["categories"].append({"id": 2, "name": "other", "keypoints": ["eye"], "skeleton": []})

    with pytest.warns(UserWarning, match="more than 1 category"):
        num_individuals, bodyparts = COCOLoader.get_project_parameters(train, test)

    assert num_individuals == 2
    assert list(bodyparts) == BODYPARTS
