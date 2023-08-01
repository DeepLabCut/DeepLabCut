_base_ = [
    '../_base_/models/retinanet_r50_fpn.py', '../common/mstrain_3x_coco.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
