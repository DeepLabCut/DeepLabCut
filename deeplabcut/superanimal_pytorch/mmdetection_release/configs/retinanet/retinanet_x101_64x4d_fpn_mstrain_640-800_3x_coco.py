_base_ = [
    '../_base_/models/retinanet_r50_fpn.py', '../common/mstrain_3x_coco.py'
]
# optimizer
model = dict(
    pretrained='open-mmlab://resnext101_64x4d',
    backbone=dict(type='ResNeXt', depth=101, groups=64, base_width=4))
optimizer = dict(type='SGD', lr=0.01)
