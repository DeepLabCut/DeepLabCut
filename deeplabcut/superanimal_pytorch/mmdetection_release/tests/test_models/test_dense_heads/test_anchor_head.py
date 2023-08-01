# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch

from mmdet.models.dense_heads import AnchorHead


def test_anchor_head_loss():
    """Tests anchor head loss when truth is empty and non-empty."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]

    cfg = mmcv.Config(
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False))
    self = AnchorHead(num_classes=4, in_channels=1, train_cfg=cfg)

    # Anchor head expects a multiple levels of features per image
    feat = [
        torch.rand(1, 1, s // (2**(i + 2)), s // (2**(i + 2)))
        for i in range(len(self.anchor_generator.strides))
    ]
    cls_scores, bbox_preds = self.forward(feat)

    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]

    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                                img_metas, gt_bboxes_ignore)
    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    empty_cls_loss = sum(empty_gt_losses['loss_cls'])
    empty_box_loss = sum(empty_gt_losses['loss_bbox'])
    assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert empty_box_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                              img_metas, gt_bboxes_ignore)
    onegt_cls_loss = sum(one_gt_losses['loss_cls'])
    onegt_box_loss = sum(one_gt_losses['loss_bbox'])
    assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_box_loss.item() > 0, 'box loss should be non-zero'
