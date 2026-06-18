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
"""Tests bbox schema + precomputed detector runner integration with DLC code."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deeplabcut.pose_estimation_pytorch.data.bboxes import (
    BBoxEntry,
    BBoxes,
)
from deeplabcut.pose_estimation_pytorch.data.postprocessor import build_detector_postprocessor
from deeplabcut.pose_estimation_pytorch.data.preprocessor import build_bottom_up_preprocessor
from deeplabcut.pose_estimation_pytorch.data.transforms import build_transforms
from deeplabcut.pose_estimation_pytorch.models.detectors.external import EXTERNAL_DETECTORS, PrecomputedDetectorRunner
from deeplabcut.pose_estimation_pytorch.models.detectors.external.base import (
    build_precomputed_detector_runner_from_config,
    precompute_detector_bboxes,
    validate_precomputed_bboxes_for_loader,
)

# Important: ensure the mock detector module is imported so registry population happens
from deeplabcut.pose_estimation_pytorch.runners.inference import build_inference_runner
from deeplabcut.pose_estimation_pytorch.task import Task


def test_bbox_entry_from_detector_context_roundtrip_xywh():
    """
    Schema should faithfully round-trip DLC detector context in xywh format.
    """
    context = {
        "bboxes": np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
        "bbox_scores": np.array([0.9], dtype=np.float32),
    }

    entry = BBoxEntry.from_detector_context(
        context,
        image_path=Path("img0.png"),
        bbox_format="xywh",
    )

    assert entry.image_path == Path("img0.png")
    assert entry.bbox_format == "xywh"
    assert np.allclose(entry.bbox_scores, [0.9])

    restored = entry.to_detector_context(target_format="xywh")
    np.testing.assert_allclose(restored["bboxes"], context["bboxes"])
    np.testing.assert_allclose(restored["bbox_scores"], context["bbox_scores"])


def test_precomputed_detector_runner_inference_matches_dlc_contract():
    """
    PrecomputedDetectorRunner should behave like a DLC detector runner:
    inference(images) -> list[{"bboxes": ..., "bbox_scores": ...}]
    """
    bboxes = BBoxes(
        train=[
            BBoxEntry(
                bboxes=[(1.0, 2.0, 3.0, 4.0)],
                bbox_scores=[0.8],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ]
    )

    runner = PrecomputedDetectorRunner.from_bboxes(
        bboxes,
        mode="train",
        target_format="xywh",
        validate_image_paths=True,
    )

    outputs = runner.inference([Path("img0.png")])

    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert "bboxes" in outputs[0]
    assert "bbox_scores" in outputs[0]

    np.testing.assert_allclose(
        outputs[0]["bboxes"],
        np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        outputs[0]["bbox_scores"],
        np.array([0.8], dtype=np.float32),
    )


def test_create_dataset_accepts_precomputed_detector_runner(fake_dlc_loader):
    """
    DLC loader.create_dataset(...) should be able to consume PrecomputedDetectorRunner
    and rewrite annotation bboxes accordingly.
    """
    loader = fake_dlc_loader

    precomputed = BBoxes(
        train=[
            BBoxEntry(
                bboxes=[(11.0, 12.0, 13.0, 14.0)],
                bbox_scores=[0.95],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ]
    )

    detector_runner = PrecomputedDetectorRunner.from_bboxes(
        precomputed,
        mode="train",
        target_format="xywh",
        validate_image_paths=True,
    )

    dataset = loader.create_dataset(
        transform=None,
        mode="train",
        task=Task.TOP_DOWN,
        detector_runner=detector_runner,
    )

    ann = dataset.annotations[0]
    actual_bbox = np.asarray(ann["bbox"], dtype=np.float32)

    np.testing.assert_allclose(
        actual_bbox,
        np.array([11.0, 12.0, 13.0, 14.0], dtype=np.float32),
    )


def test_create_dataset_with_precomputed_detector_runner_does_not_mutate_cached_load_data(fake_dlc_loader):
    """
    Regression test: create_dataset() must deep-copy cached annotations before rewriting
    bboxes, otherwise load_data() becomes stateful and unsafe.
    """
    loader = fake_dlc_loader

    precomputed = BBoxes(
        train=[
            BBoxEntry(
                bboxes=[(21.0, 22.0, 23.0, 24.0)],
                bbox_scores=[0.88],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ]
    )

    detector_runner = PrecomputedDetectorRunner.from_bboxes(
        precomputed,
        mode="train",
        target_format="xywh",
        validate_image_paths=True,
    )

    raw_before = np.asarray(loader.load_data("train")["annotations"][0]["bbox"], dtype=np.float32).copy()
    np.testing.assert_allclose(raw_before, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    dataset = loader.create_dataset(
        transform=None,
        mode="train",
        task=Task.TOP_DOWN,
        detector_runner=detector_runner,
    )

    np.testing.assert_allclose(
        np.asarray(dataset.annotations[0]["bbox"], dtype=np.float32),
        np.array([21.0, 22.0, 23.0, 24.0], dtype=np.float32),
    )

    # Cached raw annotations must remain untouched
    raw_after = np.asarray(loader.load_data("train")["annotations"][0]["bbox"], dtype=np.float32)
    np.testing.assert_allclose(raw_after, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))


def test_live_mock_detector_can_roundtrip_through_schema_and_precomputed_runner(fake_dlc_loader):
    """
    Strong integration test:

      live mock external detector
        -> DLC DetectorInferenceRunner
        -> detector context
        -> BBoxEntry.from_detector_context(...)
        -> PrecomputedDetectorRunner
        -> loader.create_dataset(..., detector_runner=...)

    This proves the schema/adapter layer can bridge live detector outputs
    back into DLC’s training/data path.
    """
    # 1. Build live mock external detector
    detector = EXTERNAL_DETECTORS.build(
        {
            "type": "MockExternalDetector",
            "score": 0.9,
            "label": 1,
        }
    )
    detector.eval()

    # 2. Build DLC detector inference runner around it
    transform = build_transforms({"scale_to_unit_range": True})

    runner = build_inference_runner(
        task=Task.DETECT,
        model=detector,
        device="cpu",
        snapshot_path=None,
        batch_size=1,
        preprocessor=build_bottom_up_preprocessor(
            color_mode="RGB",
            transform=transform,
        ),
        postprocessor=build_detector_postprocessor(
            max_individuals=1,
            min_bbox_score=0.0,
        ),
    )

    # 3. Run detector on a mock image
    image = np.zeros((128, 256, 3), dtype=np.uint8)
    live_outputs = runner.inference([image])

    assert len(live_outputs) == 1
    live_context = live_outputs[0]

    assert "bboxes" in live_context
    assert "bbox_scores" in live_context

    # 4. Convert live DLC detector output -> schema
    entry = BBoxEntry.from_detector_context(
        live_context,
        image_path=Path("img0.png"),
        bbox_format="xywh",  # DLC postprocessed outputs are expected here
    )

    # 5. Build precomputed runner from that schema
    precomputed = BBoxes(train=[entry])

    precomputed_runner = PrecomputedDetectorRunner.from_bboxes(
        precomputed,
        mode="train",
        target_format="xywh",
        validate_image_paths=True,
    )

    # 6. Use in DLC create_dataset(...)
    loader = fake_dlc_loader
    dataset = loader.create_dataset(
        transform=None,
        mode="train",
        task=Task.TOP_DOWN,
        detector_runner=precomputed_runner,
    )

    actual_bbox = np.asarray(dataset.annotations[0]["bbox"], dtype=np.float32)
    expected_bbox = np.asarray(live_context["bboxes"][0], dtype=np.float32)

    np.testing.assert_allclose(actual_bbox, expected_bbox)


def test_precomputed_detector_runner_supports_path_based_subset_lookup():
    bboxes = BBoxes(
        train=[
            BBoxEntry(
                bboxes=[(1.0, 2.0, 3.0, 4.0)],
                bbox_scores=[0.1],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            ),
            BBoxEntry(
                bboxes=[(5.0, 6.0, 7.0, 8.0)],
                bbox_scores=[0.9],
                bbox_format="xywh",
                image_path=Path("img1.png"),
            ),
        ]
    )

    runner = PrecomputedDetectorRunner.from_bboxes(bboxes, mode="train")

    outputs = runner.inference([Path("img1.png")])

    assert len(outputs) == 1
    np.testing.assert_allclose(
        outputs[0]["bboxes"],
        np.array([[5.0, 6.0, 7.0, 8.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        outputs[0]["bbox_scores"],
        np.array([0.9], dtype=np.float32),
    )


def test_precomputed_detector_runner_preserves_requested_path_order():
    bboxes = BBoxes(
        train=[
            BBoxEntry(
                bboxes=[(1.0, 2.0, 3.0, 4.0)],
                bbox_scores=[0.1],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            ),
            BBoxEntry(
                bboxes=[(5.0, 6.0, 7.0, 8.0)],
                bbox_scores=[0.9],
                bbox_format="xywh",
                image_path=Path("img1.png"),
            ),
        ]
    )

    runner = PrecomputedDetectorRunner.from_bboxes(bboxes, mode="train")

    outputs = runner.inference([Path("img1.png"), Path("img0.png")])

    np.testing.assert_allclose(outputs[0]["bboxes"], np.array([[5.0, 6.0, 7.0, 8.0]], dtype=np.float32))
    np.testing.assert_allclose(outputs[1]["bboxes"], np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))


def test_precomputed_detector_runner_raises_for_unknown_requested_path():
    bboxes = BBoxes(
        train=[
            BBoxEntry(
                bboxes=[(1.0, 2.0, 3.0, 4.0)],
                bbox_scores=[0.1],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ]
    )

    runner = PrecomputedDetectorRunner.from_bboxes(bboxes, mode="train")

    with pytest.raises(ValueError, match="No precomputed bbox entry found"):
        runner.inference([Path("missing.png")])


def test_validate_precomputed_bboxes_for_loader_requires_train_and_test(tmp_path, fake_dlc_loader):
    loader = fake_dlc_loader

    bboxes = BBoxes(
        train=[
            BBoxEntry(
                bboxes=[(1.0, 2.0, 3.0, 4.0)],
                bbox_scores=[0.9],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ],
        test=[],
    )

    bbox_file = tmp_path / "precomputed_bboxes.json"
    bboxes.dump_json(bbox_file)

    with pytest.raises(ValueError, match="\\[test\\] Expected"):
        validate_precomputed_bboxes_for_loader(
            loader,
            bbox_file,
            required_modes=("train", "test"),
            require_image_paths=True,
        )


def test_validate_precomputed_bboxes_allows_empty_bboxes(tmp_path, fake_dlc_loader):
    loader = fake_dlc_loader

    bboxes = BBoxes(
        train=[
            BBoxEntry(
                bboxes=[],
                bbox_scores=[],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ],
        test=[
            BBoxEntry(
                bboxes=[],
                bbox_scores=[],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ],
    )

    bbox_file = tmp_path / "precomputed_bboxes.json"
    bboxes.dump_json(bbox_file)

    summary = validate_precomputed_bboxes_for_loader(
        loader,
        bbox_file,
        required_modes=("train", "test"),
        require_image_paths=True,
        allow_empty_bboxes=True,
    )

    assert summary["train"]["entries_without_bboxes"] == 1


class RecordingDetectorRunner:
    """
    Tiny detector runner used to test precompute resume/recompute policies.
    """

    def __init__(
        self,
        *,
        bbox: tuple[float, float, float, float] = (91.0, 92.0, 93.0, 94.0),
        score: float = 0.99,
    ):
        self.bbox = bbox
        self.score = score
        self.calls: list[Path] = []

    def inference(self, images, shelf_writer=None, **kwargs):
        images = list(images)
        self.calls.extend([Path(p) for p in images])

        return [
            {
                "bboxes": np.asarray([self.bbox], dtype=np.float32),
                "bbox_scores": np.asarray([self.score], dtype=np.float32),
            }
            for _ in images
        ]


def test_precompute_resume_reuses_valid_entries_and_computes_missing_mode(tmp_path, fake_dlc_loader):
    """
    resume should reuse valid existing entries and compute only missing entries.

    FakeDLCLoader has one train image and one test image, both named img0.png.
    Here train exists and test is missing, so only test should be computed.
    """
    loader = fake_dlc_loader
    bbox_file = tmp_path / "precomputed_bboxes.json"

    existing = BBoxes(
        train=[
            BBoxEntry(
                bboxes=[(1.0, 2.0, 3.0, 4.0)],
                bbox_scores=[0.8],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ],
        test=[],
    )
    existing.dump_json(bbox_file)

    detector = RecordingDetectorRunner(
        bbox=(11.0, 12.0, 13.0, 14.0),
        score=0.95,
    )

    artifact = precompute_detector_bboxes(
        loader=loader,
        detector_runner=detector,
        output_file=bbox_file,
        modes=("train", "test"),
        bbox_format="xywh",
        recompute="resume",
        validate_image_paths=True,
    )

    assert len(detector.calls) == 1

    np.testing.assert_allclose(
        artifact.train[0].to_xywh(),
        np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        artifact.test[0].to_xywh(),
        np.asarray([[11.0, 12.0, 13.0, 14.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(artifact.test[0].bbox_scores, [0.95])


def test_precompute_resume_recomputes_invalid_existing_entry(tmp_path, fake_dlc_loader):
    """
    resume should recompute an existing entry if it is structurally invalid.
    """
    loader = fake_dlc_loader
    bbox_file = tmp_path / "precomputed_bboxes.json"

    # Invalid: one box, zero scores.
    existing = BBoxes(
        train=[
            BBoxEntry(
                bboxes=[(1.0, 2.0, 3.0, 4.0)],
                bbox_scores=[],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ]
    )
    existing.dump_json(bbox_file)

    detector = RecordingDetectorRunner(
        bbox=(21.0, 22.0, 23.0, 24.0),
        score=0.91,
    )

    artifact = precompute_detector_bboxes(
        loader=loader,
        detector_runner=detector,
        output_file=bbox_file,
        modes=("train",),
        bbox_format="xywh",
        recompute="resume",
        validate_image_paths=True,
    )

    assert len(detector.calls) == 1
    np.testing.assert_allclose(
        artifact.train[0].to_xywh(),
        np.asarray([[21.0, 22.0, 23.0, 24.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(artifact.train[0].bbox_scores, [0.91])


def test_precompute_recompute_none_raises_for_missing_entry(tmp_path, fake_dlc_loader):
    """
    recompute='none' should be validation-only. Missing entries should raise.
    """
    loader = fake_dlc_loader
    bbox_file = tmp_path / "precomputed_bboxes.json"

    BBoxes(train=[]).dump_json(bbox_file)

    detector = RecordingDetectorRunner()

    with pytest.raises(ValueError, match="Missing bbox entry"):
        precompute_detector_bboxes(
            loader=loader,
            detector_runner=detector,
            output_file=bbox_file,
            modes=("train",),
            bbox_format="xywh",
            recompute="none",
            validate_image_paths=True,
        )

    assert len(detector.calls) == 0


def test_precompute_recompute_all_recomputes_valid_existing_entry(tmp_path, fake_dlc_loader):
    """
    recompute='all' should ignore valid existing entries and recompute everything.
    """
    loader = fake_dlc_loader
    bbox_file = tmp_path / "precomputed_bboxes.json"

    existing = BBoxes(
        train=[
            BBoxEntry(
                bboxes=[(1.0, 2.0, 3.0, 4.0)],
                bbox_scores=[0.8],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ]
    )
    existing.dump_json(bbox_file)

    detector = RecordingDetectorRunner(
        bbox=(31.0, 32.0, 33.0, 34.0),
        score=0.93,
    )

    artifact = precompute_detector_bboxes(
        loader=loader,
        detector_runner=detector,
        output_file=bbox_file,
        modes=("train",),
        bbox_format="xywh",
        recompute="all",
        validate_image_paths=True,
    )

    assert len(detector.calls) == 1
    np.testing.assert_allclose(
        artifact.train[0].to_xywh(),
        np.asarray([[31.0, 32.0, 33.0, 34.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(artifact.train[0].bbox_scores, [0.93])


def test_precompute_resume_keeps_empty_entry_when_empty_policy_valid(tmp_path, fake_dlc_loader):
    """
    Empty bbox entries should be reusable when empty_policy='valid'.
    This matters for true empty/no-animal frames.
    """
    loader = fake_dlc_loader
    bbox_file = tmp_path / "precomputed_bboxes.json"

    existing = BBoxes(
        train=[
            BBoxEntry(
                bboxes=[],
                bbox_scores=[],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ]
    )
    existing.dump_json(bbox_file)

    detector = RecordingDetectorRunner()

    artifact = precompute_detector_bboxes(
        loader=loader,
        detector_runner=detector,
        output_file=bbox_file,
        modes=("train",),
        bbox_format="xywh",
        recompute="resume",
        empty_policy="valid",
        validate_image_paths=True,
    )

    assert len(detector.calls) == 0
    assert len(artifact.train[0].bboxes) == 0
    assert len(artifact.train[0].bbox_scores) == 0


def test_precompute_resume_recomputes_empty_entry_when_empty_policy_recompute(tmp_path, fake_dlc_loader):
    """
    empty_policy='recompute' should select empty entries for recomputation under resume.
    """
    loader = fake_dlc_loader
    bbox_file = tmp_path / "precomputed_bboxes.json"

    existing = BBoxes(
        train=[
            BBoxEntry(
                bboxes=[],
                bbox_scores=[],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ]
    )
    existing.dump_json(bbox_file)

    detector = RecordingDetectorRunner(
        bbox=(41.0, 42.0, 43.0, 44.0),
        score=0.97,
    )

    artifact = precompute_detector_bboxes(
        loader=loader,
        detector_runner=detector,
        output_file=bbox_file,
        modes=("train",),
        bbox_format="xywh",
        recompute="resume",
        empty_policy="recompute",
        validate_image_paths=True,
    )

    assert len(detector.calls) == 1
    np.testing.assert_allclose(
        artifact.train[0].to_xywh(),
        np.asarray([[41.0, 42.0, 43.0, 44.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(artifact.train[0].bbox_scores, [0.97])


def test_precompute_resume_recomputes_existing_entry_below_min_score(tmp_path, fake_dlc_loader):
    """
    min_existing_score should let users selectively refresh low-confidence entries.
    """
    loader = fake_dlc_loader
    bbox_file = tmp_path / "precomputed_bboxes.json"

    existing = BBoxes(
        train=[
            BBoxEntry(
                bboxes=[(1.0, 2.0, 3.0, 4.0)],
                bbox_scores=[0.1],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ]
    )
    existing.dump_json(bbox_file)

    detector = RecordingDetectorRunner(
        bbox=(51.0, 52.0, 53.0, 54.0),
        score=0.89,
    )

    artifact = precompute_detector_bboxes(
        loader=loader,
        detector_runner=detector,
        output_file=bbox_file,
        modes=("train",),
        bbox_format="xywh",
        recompute="resume",
        empty_policy="valid",
        min_existing_score=0.5,
        validate_image_paths=True,
    )

    assert len(detector.calls) == 1
    np.testing.assert_allclose(
        artifact.train[0].to_xywh(),
        np.asarray([[51.0, 52.0, 53.0, 54.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(artifact.train[0].bbox_scores, [0.89])


def test_precomputed_detector_runner_filters_to_highest_score():
    bboxes = BBoxes(
        train=[
            BBoxEntry(
                bboxes=[
                    (0.0, 0.0, 100.0, 100.0),
                    (10.0, 10.0, 20.0, 20.0),
                ],
                bbox_scores=[0.2, 0.9],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ]
    )

    runner = PrecomputedDetectorRunner.from_bboxes(
        bboxes,
        mode="train",
        max_detections=1,
        selection_strategy="score",
    )

    out = runner.inference([Path("img0.png")])[0]

    np.testing.assert_allclose(
        out["bboxes"],
        np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        out["bbox_scores"],
        np.array([0.9], dtype=np.float32),
    )


def test_precomputed_detector_runner_filters_to_largest_box():
    bboxes = BBoxes(
        train=[
            BBoxEntry(
                bboxes=[
                    (0.0, 0.0, 100.0, 100.0),
                    (10.0, 10.0, 20.0, 20.0),
                ],
                bbox_scores=[0.2, 0.9],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ]
    )

    runner = PrecomputedDetectorRunner.from_bboxes(
        bboxes,
        mode="train",
        max_detections=1,
        selection_strategy="largest",
    )

    out = runner.inference([Path("img0.png")])[0]

    np.testing.assert_allclose(
        out["bboxes"],
        np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        out["bbox_scores"],
        np.array([0.2], dtype=np.float32),
    )


def test_build_precomputed_detector_runner_from_config_uses_bbox_filter_config(tmp_path):
    bbox_file = tmp_path / "bboxes.json"

    BBoxes(
        test=[
            BBoxEntry(
                bboxes=[
                    (0.0, 0.0, 100.0, 100.0),
                    (10.0, 10.0, 20.0, 20.0),
                ],
                bbox_scores=[0.2, 0.9],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ]
    ).dump_json(bbox_file)

    model_cfg = {
        "data": {
            "precomputed_bboxes": str(bbox_file),
            "bbox_max_detections": 1,
            "bbox_selection_strategy": "score",
        }
    }

    runner = build_precomputed_detector_runner_from_config(
        model_cfg,
        mode="test",
    )

    out = runner.inference([Path("img0.png")])[0]

    np.testing.assert_allclose(
        out["bboxes"],
        np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32),
    )
