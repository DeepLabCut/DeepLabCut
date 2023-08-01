_base_ = [
    '../common/mstrain-poly_3x_coco_instance.py',
    '../_base_/models/mask_rcnn_r50_fpn.py'
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='RegNet',
        arch='regnetx_800mf',
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://regnetx_800mf')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 288, 672],
        out_channels=256,
        num_outs=5))

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.00005)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
