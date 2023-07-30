#_base_ = ['../../../../_base_/datasets/animalpose.py']
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=1, metric='mAP', save_best='AP')
kpts_num = 17

optimizer = dict(
    type='Adam',
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 400
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        #dict(type='WandbLoggerHook',
        #     init_kwargs=dict(
        #         project='vit_pose_estimation',
        #         name='base')
        #     )
    ])

channel_cfg = dict(
    num_output_channels=kpts_num,
    dataset_joints=kpts_num,
    dataset_channel=range(kpts_num),
    inference_channel=range(kpts_num)
)

# model settings
model = dict(
    type='TopDown',
    pretrained='https://download.openmmlab.com/mmpose/'
               'pretrain_models/hrnet_w32-36af842e.pth',
    backbone=dict(
        type='TokenPose_S',
        # HIDDEN_HEATMAP_DIM=384,
        DIM=256,
        IMAGE_SIZE=[256, 256],
        POS_EMBEDDING_TYPE='sine-full',
        PATCH_SIZE=[2, 2],
        HEATMAP_SIZE=[64, 64],
        #         MULTI_TRANSFORMER_DEPTH=[12, 12],
        #         MULTI_TRANSFORMER_HEADS=[16, 16],
        #         MULTI_DIM=[48, 48],
        TRANSFORMER_MLP_RATIO=3,
        TRANSFORMER_DEPTH=12,
        TRANSFORMER_HEADS=8,
        NUM_JOINTS=kpts_num,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=17,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        vit_dim=256,
        vit_neck=True,
        extra=dict(final_conv_kernel=1),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))


data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='data/allquadruped_val_bbox_AP69.json',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

#data_root = 'data/ma_superquadruped_coco'
data_root = 'data/ap-10k'
dataset_info = 'data/ap-10k/dataset.json'

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='TopDownDLCGenericDataset',
        ann_file=f'{data_root}/annotations/train.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info=dataset_info,
    ),
    val=dict(
        type='TopDownDLCGenericDataset',
        ann_file=f'{data_root}/annotations/test.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info=dataset_info
    ),
    test=dict(
        type='TopDownDLCGenericDataset',
        ann_file=f'{data_root}/annotations/test.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info=dataset_info
    ),
)
