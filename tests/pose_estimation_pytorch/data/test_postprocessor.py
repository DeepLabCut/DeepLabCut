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
"""Tests the pre-processors"""
import numpy as np
import pytest

from deeplabcut.pose_estimation_pytorch.data.postprocessor import (
    PredictKeypointIdentities,
    PrepareBackboneFeatures,
    RescaleAndOffset,
    TrimOutputs,
)


@pytest.mark.parametrize(
    "data",
    [
        {
            "predictions": [[[0, 0, 0.95], [20, 30, 0.5]]],
            "offsets": [(0, 0)],
            "scales": [(1, 1)],
            "rescaled": [[[0, 0, 0.95], [20, 30, 0.5]]],
        },
        {
            "predictions": [
                [[0, 0, 0.12], [1000, 0, 0.5]],  # individual 1
                [[18, 2, 0.24], [0, 1000, 0.6]],  # individual 2
            ],
            "offsets": [(0, 0), (0, 0)],
            "scales": [(1, 1), (0.5, 1.0)],
            "rescaled": [
                [[0, 0, 0.12], [1000, 0, 0.5]],  # individual 1
                [[9, 2, 0.24], [0, 1000, 0.6]],  # individual 2
            ],
        },
        {
            "predictions": [
                [[0, 0, 0.95], [20, 30, 0.5]],  # individual 1
                [[110, 5, 0.95], [60, 1200, 0.5]],  # individual 2
            ],
            "offsets": [(12, 5), (27, 10)],
            "scales": [(0.5, 0.5), (0.2, 0.2)],
            "rescaled": [
                [[12, 5, 0.95], [22, 20, 0.5]],  # individual 1
                [[49, 11, 0.95], [39, 250, 0.5]],  # individual 2
            ],
        },
    ],
)
def test_rescale_topdown(data):
    """expects x_processed = x * scale + offset"""
    postprocessor = RescaleAndOffset(
        keys_to_rescale=["bodyparts"],
        mode=RescaleAndOffset.Mode.KEYPOINT_TD,
    )
    context = {"scales": data["scales"], "offsets": data["offsets"]}
    predictions = {"bodyparts": np.array(data["predictions"])}
    predictions, context = postprocessor(predictions, context=context)
    print(predictions["bodyparts"].tolist())
    print(data["rescaled"])
    np.testing.assert_array_equal(predictions["bodyparts"], np.array(data["rescaled"]))


@pytest.mark.parametrize(
    "data",
    [
        {
            "bboxes": [[0, 0, 0, 0], [1, 1, 1, 1]],
            "bbox_scores": [0, 0],
            "max_individuals": {"bboxes": 1, "bbox_scores": 1},
        },
        {
            "bboxes": [[0, 0, 0, 0], [1, 1, 1, 1]],
            "bbox_scores": [0, 0],
            "max_individuals": {"bboxes": 2, "bbox_scores": 2},
        },
    ],
)
def test_trim_outputs(data):
    """expects x_processed = x * scale + offset"""
    postprocessor = TrimOutputs(max_individuals=data["max_individuals"])
    context = {}
    predictions = {"bboxes": np.array(data["bboxes"]), "bbox_scores": np.array(data["bbox_scores"])}
    predictions, context = postprocessor(predictions, context=context)
    print(predictions["bboxes"].tolist())
    print(predictions["bbox_scores"].tolist())
    assert len(predictions["bboxes"]) == data["max_individuals"]["bboxes"]
    assert len(predictions["bbox_scores"]) == data["max_individuals"]["bbox_scores"]


@pytest.mark.parametrize(
    "data",
    [
        {
            "predictions": [[[0, 0, 0.95], [20, 30, 0.5]]],
            "offsets": (0, 0),
            "scales": (1, 1),
            "rescaled": [[[0, 0, 0.95], [20, 30, 0.5]]],
        },
        {
            "predictions": [
                [[0, 0, 0.12], [10, 0, 0.5]],  # individual 1
                [[1000, 500, 0.24], [50, 250, 0.6]],  # individual 2
            ],
            "offsets": (5, 7),
            "scales": (0.2, 0.5),
            "rescaled": [
                [[5, 7, 0.12], [7, 7, 0.5]],  # individual 1
                [[205, 257, 0.24], [15, 132, 0.6]],  # individual 2
            ],
        },
    ],
)
def test_rescale_bottom_up(data):
    """expects x_processed = x * scale + offset"""
    postprocessor = RescaleAndOffset(
        keys_to_rescale=["bodyparts"],
        mode=RescaleAndOffset.Mode.KEYPOINT,
    )
    context = {"scales": data["scales"], "offsets": data["offsets"]}
    predictions = {"bodyparts": np.array(data["predictions"])}
    predictions, context = postprocessor(predictions, context=context)
    print(predictions["bodyparts"].tolist())
    print(data["rescaled"])
    np.testing.assert_array_equal(predictions["bodyparts"], np.array(data["rescaled"]))


@pytest.mark.parametrize(
    "data",
    [
        {
            "bboxes": [[222.0, 562.0, 721.0, 637.0]],
            "offsets": (0, 0),
            "scales": (1, 1),
            "rescaled": [[222.0, 562.0, 721.0, 637.0]],
        },
        {
            "bboxes": [[386.71875, 219.53125, 281.640625, 248.828125]],
            "offsets": (-768, 0),
            "scales": (2.56, 2.56),
            "rescaled": [[222.0, 562.0, 721.0, 637.0]],
        },
        {
            "bboxes": [
                [0, 0, 100, 100],
                [5, 10, 100, 100],
                [5, 10, 10, 20],
            ],
            "offsets": (3, 7),
            "scales": (2, 0.5),
            "rescaled": [
                [3, 7, 200, 50],
                [13, 12, 200, 50],
                [13, 12, 20, 10],
            ],
        },
    ],
)
def test_rescale_detector(data):
    """expects x_processed = x * scale + offset"""
    postprocessor = RescaleAndOffset(
        keys_to_rescale=["bboxes"],
        mode=RescaleAndOffset.Mode.BBOX_XYWH,
    )
    context = {"scales": data["scales"], "offsets": data["offsets"]}
    predictions = {"bboxes": np.array(data["bboxes"])}
    predictions, context = postprocessor(predictions, context=context)
    print(predictions["bboxes"].tolist())
    print(data["rescaled"])
    np.testing.assert_array_equal(predictions["bboxes"], np.array(data["rescaled"]))


@pytest.mark.parametrize(
    "data",
    [
        {
            "bodyparts": [
                [[3.1, 1, 0.8], [1, 0, 0.9]],  # assembly 1  (x, y, score)
                [[2.2, 1.6, 0.5], [3, 3, 0.4]],  # assembly 2  (x, y, score)
            ],
            "id_heatmap": [  # id1, id2 score for each pixel
                [[0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.4, 0.1]],
                [[0.1, 0.2], [0.2, 0.2], [0.3, 0.2], [0.4, 0.2]],
                [[0.1, 0.3], [0.2, 0.3], [0.3, 0.3], [0.4, 0.3]],
                [[0.1, 0.4], [0.2, 0.4], [0.3, 0.4], [0.4, 0.4]],
            ],
            "id_scores": [  # id1, id2 score for each bodypart
                [[0.4, 0.2], [0.2, 0.1]],  # assembly 1 (id_1 proba, id_2 proba)
                [[0.3, 0.3], [0.4, 0.4]],  # assembly 2 (id_1 proba, id_2 proba)
            ],
        },
    ],
)
def test_assign_id_scores(data):
    p = PredictKeypointIdentities(
        identity_key="keypoint_identity",
        identity_map_key="identity_map",
        pose_key="bodyparts",
        keep_id_maps=True,
    )
    bodyparts = np.array(data["bodyparts"])
    id_heatmap = np.array(data["id_heatmap"])
    expected_ids = np.array(data["id_scores"])
    print()
    print(bodyparts.shape)
    print(id_heatmap.shape)
    print(expected_ids.shape)
    predictions_in = {"bodyparts": bodyparts, "identity_map": id_heatmap}
    predictions, _ = p(predictions_in, {})
    np.testing.assert_array_equal(
        predictions["keypoint_identity"],
        expected_ids,
    )


def test_prepare_backbone_features():
    p = PrepareBackboneFeatures(top_down=False)

    img_w, img_h = 256, 128
    features = np.zeros((1, img_h, img_w))

    features[0, 15, 10] = 1
    features[0, 25, 20] = 2
    features[0, 35, 30] = 3

    pose = np.array([
        [
            [10.1, 15.1, 0.95],
            [20.1, 25.1, 0.95],
            [29.9, 34.9, 0.95],
        ],
    ])

    predictions = [dict(backbone=dict(features=features), bodypart=dict(poses=pose))]
    context = dict(image_size=(img_w, img_h))
    predictions_out, context_out = p(predictions, context)

    assert len(predictions_out) == 1
    assert len(context_out) == 1
    preds = predictions_out[0]

    assert "backbone" in preds
    assert "bodypart_features" in preds["backbone"]
    bodypart_features = preds["backbone"]["bodypart_features"]
    print(f"Bodypart features: {bodypart_features.shape}")
    print(bodypart_features)
    assert bodypart_features.shape == (1, 3, 1)
    assert bodypart_features.reshape(-1).tolist() == [1, 2, 3]


def test_prepare_top_down_backbone_features():
    p = PrepareBackboneFeatures(top_down=True)

    img_w, img_h = 256, 256

    features = np.zeros((2, 1, img_h, img_w))
    features[0, 0, 15, 10] = 1
    features[0, 0, 25, 20] = 2
    features[0, 0, 35, 30] = 3
    features[1, 0, 95, 10] = 11
    features[1, 0, 85, 20] = 12
    features[1, 0, 75, 30] = 13

    pose_idv0 = np.array([
        [
            [10.1, 15.1, 0.95],
            [20.1, 25.1, 0.95],
            [29.9, 34.9, 0.95],
        ],
    ])
    pose_idv1 = np.array(
        [
            [
                [10.1, 95.1, 0.95],
                [20.1, 85.1, 0.95],
                [29.9, 74.9, 0.95],
            ],
        ]
    )

    predictions = [
        dict(backbone=dict(features=features[0]), bodypart=dict(poses=pose_idv0)),
        dict(backbone=dict(features=features[1]), bodypart=dict(poses=pose_idv1)),
    ]
    context = dict(top_down_crop_size=(img_w, img_h))
    predictions_out, context_out = p(predictions, context)

    assert len(predictions_out) == 2
    assert len(context_out) == 1
    for preds, expected in zip(predictions_out, [[1, 2, 3], [11, 12, 13]]):
        assert "backbone" in preds
        assert "bodypart_features" in preds["backbone"]
        bodypart_features = preds["backbone"]["bodypart_features"]
        print(f"Bodypart features: {bodypart_features.shape}")
        print(bodypart_features)
        assert bodypart_features.shape == (1, 3, 1)
        assert bodypart_features.reshape(-1).tolist() == expected
