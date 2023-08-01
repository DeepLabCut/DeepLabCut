# Copyright (c) OpenMMLab. All rights reserved.
"""Tests the Assigner objects.

CommandLine:
    pytest tests/test_utils/test_assigner.py
    xdoctest tests/test_utils/test_assigner.py zero
"""
import pytest
import torch

from mmdet.core.bbox.assigners import (ApproxMaxIoUAssigner,
                                       CenterRegionAssigner, HungarianAssigner,
                                       MaxIoUAssigner, PointAssigner,
                                       TaskAlignedAssigner, UniformAssigner)


def test_max_iou_assigner():
    self = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
    )
    bboxes = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15],
        [32, 32, 38, 42],
    ])
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    gt_labels = torch.LongTensor([2, 3])
    assign_result = self.assign(bboxes, gt_bboxes, gt_labels=gt_labels)
    assert len(assign_result.gt_inds) == 4
    assert len(assign_result.labels) == 4

    expected_gt_inds = torch.LongTensor([1, 0, 2, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_max_iou_assigner_with_ignore():
    self = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        ignore_iof_thr=0.5,
        ignore_wrt_candidates=False,
    )
    bboxes = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15],
        [30, 32, 40, 42],
    ])
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    gt_bboxes_ignore = torch.Tensor([
        [30, 30, 40, 40],
    ])
    assign_result = self.assign(
        bboxes, gt_bboxes, gt_bboxes_ignore=gt_bboxes_ignore)

    expected_gt_inds = torch.LongTensor([1, 0, 2, -1])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_max_iou_assigner_with_empty_gt():
    """Test corner case where an image might have no true detections."""
    self = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
    )
    bboxes = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15],
        [32, 32, 38, 42],
    ])
    gt_bboxes = torch.empty(0, 4)
    assign_result = self.assign(bboxes, gt_bboxes)

    expected_gt_inds = torch.LongTensor([0, 0, 0, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_max_iou_assigner_with_empty_boxes():
    """Test corner case where a network might predict no boxes."""
    self = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
    )
    bboxes = torch.empty((0, 4))
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    gt_labels = torch.LongTensor([2, 3])

    # Test with gt_labels
    assign_result = self.assign(bboxes, gt_bboxes, gt_labels=gt_labels)
    assert len(assign_result.gt_inds) == 0
    assert tuple(assign_result.labels.shape) == (0, )

    # Test without gt_labels
    assign_result = self.assign(bboxes, gt_bboxes, gt_labels=None)
    assert len(assign_result.gt_inds) == 0
    assert assign_result.labels is None


def test_max_iou_assigner_with_empty_boxes_and_ignore():
    """Test corner case where a network might predict no boxes and
    ignore_iof_thr is on."""
    self = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        ignore_iof_thr=0.5,
    )
    bboxes = torch.empty((0, 4))
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    gt_bboxes_ignore = torch.Tensor([
        [30, 30, 40, 40],
    ])
    gt_labels = torch.LongTensor([2, 3])

    # Test with gt_labels
    assign_result = self.assign(
        bboxes,
        gt_bboxes,
        gt_labels=gt_labels,
        gt_bboxes_ignore=gt_bboxes_ignore)
    assert len(assign_result.gt_inds) == 0
    assert tuple(assign_result.labels.shape) == (0, )

    # Test without gt_labels
    assign_result = self.assign(
        bboxes, gt_bboxes, gt_labels=None, gt_bboxes_ignore=gt_bboxes_ignore)
    assert len(assign_result.gt_inds) == 0
    assert assign_result.labels is None


def test_max_iou_assigner_with_empty_boxes_and_gt():
    """Test corner case where a network might predict no boxes and no gt."""
    self = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
    )
    bboxes = torch.empty((0, 4))
    gt_bboxes = torch.empty((0, 4))
    assign_result = self.assign(bboxes, gt_bboxes)
    assert len(assign_result.gt_inds) == 0


def test_point_assigner():
    self = PointAssigner()
    points = torch.FloatTensor([  # [x, y, stride]
        [0, 0, 1],
        [10, 10, 1],
        [5, 5, 1],
        [32, 32, 1],
    ])
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    assign_result = self.assign(points, gt_bboxes)
    expected_gt_inds = torch.LongTensor([1, 2, 1, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_point_assigner_with_empty_gt():
    """Test corner case where an image might have no true detections."""
    self = PointAssigner()
    points = torch.FloatTensor([  # [x, y, stride]
        [0, 0, 1],
        [10, 10, 1],
        [5, 5, 1],
        [32, 32, 1],
    ])
    gt_bboxes = torch.FloatTensor([])
    assign_result = self.assign(points, gt_bboxes)

    expected_gt_inds = torch.LongTensor([0, 0, 0, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_point_assigner_with_empty_boxes_and_gt():
    """Test corner case where an image might predict no points and no gt."""
    self = PointAssigner()
    points = torch.FloatTensor([])
    gt_bboxes = torch.FloatTensor([])
    assign_result = self.assign(points, gt_bboxes)
    assert len(assign_result.gt_inds) == 0


def test_approx_iou_assigner():
    self = ApproxMaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
    )
    bboxes = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15],
        [32, 32, 38, 42],
    ])
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    approxs_per_octave = 1
    approxs = bboxes
    squares = bboxes
    assign_result = self.assign(approxs, squares, approxs_per_octave,
                                gt_bboxes)

    expected_gt_inds = torch.LongTensor([1, 0, 2, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_approx_iou_assigner_with_empty_gt():
    """Test corner case where an image might have no true detections."""
    self = ApproxMaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
    )
    bboxes = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15],
        [32, 32, 38, 42],
    ])
    gt_bboxes = torch.FloatTensor([])
    approxs_per_octave = 1
    approxs = bboxes
    squares = bboxes
    assign_result = self.assign(approxs, squares, approxs_per_octave,
                                gt_bboxes)

    expected_gt_inds = torch.LongTensor([0, 0, 0, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_approx_iou_assigner_with_empty_boxes():
    """Test corner case where an network might predict no boxes."""
    self = ApproxMaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
    )
    bboxes = torch.empty((0, 4))
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    approxs_per_octave = 1
    approxs = bboxes
    squares = bboxes
    assign_result = self.assign(approxs, squares, approxs_per_octave,
                                gt_bboxes)
    assert len(assign_result.gt_inds) == 0


def test_approx_iou_assigner_with_empty_boxes_and_gt():
    """Test corner case where an network might predict no boxes and no gt."""
    self = ApproxMaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
    )
    bboxes = torch.empty((0, 4))
    gt_bboxes = torch.empty((0, 4))
    approxs_per_octave = 1
    approxs = bboxes
    squares = bboxes
    assign_result = self.assign(approxs, squares, approxs_per_octave,
                                gt_bboxes)
    assert len(assign_result.gt_inds) == 0


def test_random_assign_result():
    """Test random instantiation of assign result to catch corner cases."""
    from mmdet.core.bbox.assigners.assign_result import AssignResult
    AssignResult.random()

    AssignResult.random(num_gts=0, num_preds=0)
    AssignResult.random(num_gts=0, num_preds=3)
    AssignResult.random(num_gts=3, num_preds=3)
    AssignResult.random(num_gts=0, num_preds=3)
    AssignResult.random(num_gts=7, num_preds=7)
    AssignResult.random(num_gts=7, num_preds=64)
    AssignResult.random(num_gts=24, num_preds=3)


def test_center_region_assigner():
    self = CenterRegionAssigner(pos_scale=0.3, neg_scale=1)
    bboxes = torch.FloatTensor([[0, 0, 10, 10], [10, 10, 20, 20], [8, 8, 9,
                                                                   9]])
    gt_bboxes = torch.FloatTensor([
        [0, 0, 11, 11],  # match bboxes[0]
        [10, 10, 20, 20],  # match bboxes[1]
        [4.5, 4.5, 5.5, 5.5],  # match bboxes[0] but area is too small
        [0, 0, 10, 10],  # match bboxes[1] and has a smaller area than gt[0]
    ])
    gt_labels = torch.LongTensor([2, 3, 4, 5])
    assign_result = self.assign(bboxes, gt_bboxes, gt_labels=gt_labels)
    assert len(assign_result.gt_inds) == 3
    assert len(assign_result.labels) == 3
    expected_gt_inds = torch.LongTensor([4, 2, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)
    shadowed_labels = assign_result.get_extra_property('shadowed_labels')
    # [8, 8, 9, 9] in the shadowed region of [0, 0, 11, 11] (label: 2)
    assert torch.any(shadowed_labels == torch.LongTensor([[2, 2]]))
    # [8, 8, 9, 9] in the shadowed region of [0, 0, 10, 10] (label: 5)
    assert torch.any(shadowed_labels == torch.LongTensor([[2, 5]]))
    # [0, 0, 10, 10] is already assigned to [4.5, 4.5, 5.5, 5.5].
    #   Therefore, [0, 0, 11, 11] (label: 2) is shadowed
    assert torch.any(shadowed_labels == torch.LongTensor([[0, 2]]))


def test_center_region_assigner_with_ignore():
    self = CenterRegionAssigner(
        pos_scale=0.5,
        neg_scale=1,
    )
    bboxes = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
    ])
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 10],  # match bboxes[0]
        [10, 10, 20, 20],  # match bboxes[1]
    ])
    gt_bboxes_ignore = torch.FloatTensor([
        [0, 0, 10, 10],  # match bboxes[0]
    ])
    gt_labels = torch.LongTensor([1, 2])
    assign_result = self.assign(
        bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_bboxes_ignore,
        gt_labels=gt_labels)
    assert len(assign_result.gt_inds) == 2
    assert len(assign_result.labels) == 2

    expected_gt_inds = torch.LongTensor([-1, 2])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_center_region_assigner_with_empty_bboxes():
    self = CenterRegionAssigner(
        pos_scale=0.5,
        neg_scale=1,
    )
    bboxes = torch.empty((0, 4)).float()
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 10],  # match bboxes[0]
        [10, 10, 20, 20],  # match bboxes[1]
    ])
    gt_labels = torch.LongTensor([1, 2])
    assign_result = self.assign(bboxes, gt_bboxes, gt_labels=gt_labels)
    assert assign_result.gt_inds is None or assign_result.gt_inds.numel() == 0
    assert assign_result.labels is None or assign_result.labels.numel() == 0


def test_center_region_assigner_with_empty_gts():
    self = CenterRegionAssigner(
        pos_scale=0.5,
        neg_scale=1,
    )
    bboxes = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
    ])
    gt_bboxes = torch.empty((0, 4)).float()
    gt_labels = torch.empty((0, )).long()
    assign_result = self.assign(bboxes, gt_bboxes, gt_labels=gt_labels)
    assert len(assign_result.gt_inds) == 2
    expected_gt_inds = torch.LongTensor([0, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_hungarian_match_assigner():
    self = HungarianAssigner()
    assert self.iou_cost.iou_mode == 'giou'

    # test no gt bboxes
    bbox_pred = torch.rand((10, 4))
    cls_pred = torch.rand((10, 81))
    gt_bboxes = torch.empty((0, 4)).float()
    gt_labels = torch.empty((0, )).long()
    img_meta = dict(img_shape=(10, 8, 3))
    assign_result = self.assign(bbox_pred, cls_pred, gt_bboxes, gt_labels,
                                img_meta)
    assert torch.all(assign_result.gt_inds == 0)
    assert torch.all(assign_result.labels == -1)

    # test with gt bboxes
    gt_bboxes = torch.FloatTensor([[0, 0, 5, 7], [3, 5, 7, 8]])
    gt_labels = torch.LongTensor([1, 20])
    assign_result = self.assign(bbox_pred, cls_pred, gt_bboxes, gt_labels,
                                img_meta)
    assert torch.all(assign_result.gt_inds > -1)
    assert (assign_result.gt_inds > 0).sum() == gt_bboxes.size(0)
    assert (assign_result.labels > -1).sum() == gt_bboxes.size(0)

    # test iou mode
    self = HungarianAssigner(
        iou_cost=dict(type='IoUCost', iou_mode='iou', weight=1.0))
    assert self.iou_cost.iou_mode == 'iou'
    assign_result = self.assign(bbox_pred, cls_pred, gt_bboxes, gt_labels,
                                img_meta)
    assert torch.all(assign_result.gt_inds > -1)
    assert (assign_result.gt_inds > 0).sum() == gt_bboxes.size(0)
    assert (assign_result.labels > -1).sum() == gt_bboxes.size(0)

    # test focal loss mode
    self = HungarianAssigner(
        iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0),
        cls_cost=dict(type='FocalLossCost', weight=1.))
    assert self.iou_cost.iou_mode == 'giou'
    assign_result = self.assign(bbox_pred, cls_pred, gt_bboxes, gt_labels,
                                img_meta)
    assert torch.all(assign_result.gt_inds > -1)
    assert (assign_result.gt_inds > 0).sum() == gt_bboxes.size(0)
    assert (assign_result.labels > -1).sum() == gt_bboxes.size(0)


def test_uniform_assigner():
    self = UniformAssigner(0.15, 0.7, 1)
    pred_bbox = torch.FloatTensor([
        [1, 1, 12, 8],
        [4, 4, 20, 20],
        [1, 5, 15, 15],
        [30, 5, 32, 42],
    ])
    anchor = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15],
        [32, 32, 38, 42],
    ])
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    gt_labels = torch.LongTensor([2, 3])
    assign_result = self.assign(
        pred_bbox, anchor, gt_bboxes, gt_labels=gt_labels)
    assert len(assign_result.gt_inds) == 4
    assert len(assign_result.labels) == 4

    expected_gt_inds = torch.LongTensor([-1, 0, 2, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_uniform_assigner_with_empty_gt():
    """Test corner case where an image might have no true detections."""
    self = UniformAssigner(0.15, 0.7, 1)
    pred_bbox = torch.FloatTensor([
        [1, 1, 12, 8],
        [4, 4, 20, 20],
        [1, 5, 15, 15],
        [30, 5, 32, 42],
    ])
    anchor = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15],
        [32, 32, 38, 42],
    ])
    gt_bboxes = torch.empty(0, 4)
    assign_result = self.assign(pred_bbox, anchor, gt_bboxes)

    expected_gt_inds = torch.LongTensor([0, 0, 0, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_uniform_assigner_with_empty_boxes():
    """Test corner case where a network might predict no boxes."""
    self = UniformAssigner(0.15, 0.7, 1)
    pred_bbox = torch.empty((0, 4))
    anchor = torch.empty((0, 4))
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    gt_labels = torch.LongTensor([2, 3])

    # Test with gt_labels
    assign_result = self.assign(
        pred_bbox, anchor, gt_bboxes, gt_labels=gt_labels)
    assert len(assign_result.gt_inds) == 0
    assert tuple(assign_result.labels.shape) == (0, )

    # Test without gt_labels
    assign_result = self.assign(pred_bbox, anchor, gt_bboxes, gt_labels=None)
    assert len(assign_result.gt_inds) == 0


def test_task_aligned_assigner():
    with pytest.raises(AssertionError):
        TaskAlignedAssigner(topk=0)

    self = TaskAlignedAssigner(topk=13)
    pred_score = torch.FloatTensor([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4],
                                    [0.4, 0.5]])
    pred_bbox = torch.FloatTensor([
        [1, 1, 12, 8],
        [4, 4, 20, 20],
        [1, 5, 15, 15],
        [30, 5, 32, 42],
    ])
    anchor = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15],
        [32, 32, 38, 42],
    ])
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    gt_labels = torch.LongTensor([0, 1])
    assign_result = self.assign(
        pred_score,
        pred_bbox,
        anchor,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels)
    assert len(assign_result.gt_inds) == 4
    assert len(assign_result.labels) == 4

    # test empty gt
    gt_bboxes = torch.empty(0, 4)
    gt_labels = torch.empty(0, 2)
    assign_result = self.assign(
        pred_score, pred_bbox, anchor, gt_bboxes=gt_bboxes)
    expected_gt_inds = torch.LongTensor([0, 0, 0, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)
