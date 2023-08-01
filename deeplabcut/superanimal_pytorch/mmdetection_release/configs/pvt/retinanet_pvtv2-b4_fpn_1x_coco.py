_base_ = 'retinanet_pvtv2-b0_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        embed_dims=64,
        num_layers=[3, 8, 27, 3],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
                      'releases/download/v2/pvt_v2_b4.pth')),
    neck=dict(in_channels=[64, 128, 320, 512]))
# optimizer
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0001 / 1.4, weight_decay=0.0001)
# dataset settings
data = dict(samples_per_gpu=1, workers_per_gpu=1)
