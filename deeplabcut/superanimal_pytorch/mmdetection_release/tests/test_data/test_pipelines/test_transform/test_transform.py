# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import mmcv
import numpy as np
import pytest
import torch
from mmcv.utils import build_from_cfg

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets.builder import PIPELINES
from .utils import create_random_bboxes


def test_resize():
    # test assertion if img_scale is a list
    with pytest.raises(AssertionError):
        transform = dict(type='Resize', img_scale=[1333, 800], keep_ratio=True)
        build_from_cfg(transform, PIPELINES)

    # test assertion if len(img_scale) while ratio_range is not None
    with pytest.raises(AssertionError):
        transform = dict(
            type='Resize',
            img_scale=[(1333, 800), (1333, 600)],
            ratio_range=(0.9, 1.1),
            keep_ratio=True)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid multiscale_mode
    with pytest.raises(AssertionError):
        transform = dict(
            type='Resize',
            img_scale=[(1333, 800), (1333, 600)],
            keep_ratio=True,
            multiscale_mode='2333')
        build_from_cfg(transform, PIPELINES)

    # test assertion if both scale and scale_factor are set
    with pytest.raises(AssertionError):
        results = dict(
            img_prefix=osp.join(osp.dirname(__file__), '../../../data'),
            img_info=dict(filename='color.jpg'))
        load = dict(type='LoadImageFromFile')
        load = build_from_cfg(load, PIPELINES)
        transform = dict(type='Resize', img_scale=(1333, 800), keep_ratio=True)
        transform = build_from_cfg(transform, PIPELINES)
        results = load(results)
        results['scale'] = (1333, 800)
        results['scale_factor'] = 1.0
        results = transform(results)

    transform = dict(type='Resize', img_scale=(1333, 800), keep_ratio=True)
    resize_module = build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['img_fields'] = ['img', 'img2']

    results = resize_module(results)
    assert np.equal(results['img'], results['img2']).all()

    results.pop('scale')
    results.pop('scale_factor')
    transform = dict(
        type='Resize',
        img_scale=(1280, 800),
        multiscale_mode='value',
        keep_ratio=False)
    resize_module = build_from_cfg(transform, PIPELINES)
    results = resize_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (800, 1280, 3)
    assert results['img'].dtype == results['img'].dtype == np.uint8

    results_seg = {
        'img': img,
        'img_shape': img.shape,
        'ori_shape': img.shape,
        'gt_semantic_seg': copy.deepcopy(img),
        'gt_seg': copy.deepcopy(img),
        'seg_fields': ['gt_semantic_seg', 'gt_seg']
    }
    transform = dict(
        type='Resize',
        img_scale=(640, 400),
        multiscale_mode='value',
        keep_ratio=False)
    resize_module = build_from_cfg(transform, PIPELINES)
    results_seg = resize_module(results_seg)
    assert results_seg['gt_semantic_seg'].shape == results_seg['gt_seg'].shape
    assert results_seg['img_shape'] == (400, 640, 3)
    assert results_seg['img_shape'] != results_seg['ori_shape']
    assert results_seg['gt_semantic_seg'].shape == results_seg['img_shape']
    assert np.equal(results_seg['gt_semantic_seg'],
                    results_seg['gt_seg']).all()


def test_flip():
    # test assertion for invalid flip_ratio
    with pytest.raises(AssertionError):
        transform = dict(type='RandomFlip', flip_ratio=1.5)
        build_from_cfg(transform, PIPELINES)
    # test assertion for 0 <= sum(flip_ratio) <= 1
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomFlip',
            flip_ratio=[0.7, 0.8],
            direction=['horizontal', 'vertical'])
        build_from_cfg(transform, PIPELINES)

    # test assertion for mismatch between number of flip_ratio and direction
    with pytest.raises(AssertionError):
        transform = dict(type='RandomFlip', flip_ratio=[0.4, 0.5])
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid direction
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomFlip', flip_ratio=1., direction='horizonta')
        build_from_cfg(transform, PIPELINES)

    transform = dict(type='RandomFlip', flip_ratio=1.)
    flip_module = build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['img_fields'] = ['img', 'img2']

    results = flip_module(results)
    assert np.equal(results['img'], results['img2']).all()

    flip_module = build_from_cfg(transform, PIPELINES)
    results = flip_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert np.equal(original_img, results['img']).all()

    # test flip_ratio is float, direction is list
    transform = dict(
        type='RandomFlip',
        flip_ratio=0.9,
        direction=['horizontal', 'vertical', 'diagonal'])
    flip_module = build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['img_fields'] = ['img']
    results = flip_module(results)
    if results['flip']:
        assert np.array_equal(
            mmcv.imflip(original_img, results['flip_direction']),
            results['img'])
    else:
        assert np.array_equal(original_img, results['img'])

    # test flip_ratio is list, direction is list
    transform = dict(
        type='RandomFlip',
        flip_ratio=[0.3, 0.3, 0.2],
        direction=['horizontal', 'vertical', 'diagonal'])
    flip_module = build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['img_fields'] = ['img']
    results = flip_module(results)
    if results['flip']:
        assert np.array_equal(
            mmcv.imflip(original_img, results['flip_direction']),
            results['img'])
    else:
        assert np.array_equal(original_img, results['img'])


def test_random_crop():
    # test assertion for invalid random crop
    with pytest.raises(AssertionError):
        transform = dict(type='RandomCrop', crop_size=(-1, 0))
        build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img

    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # TODO: add img_fields test
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    transform = dict(type='RandomCrop', crop_size=(h - 20, w - 20))
    crop_module = build_from_cfg(transform, PIPELINES)
    results = crop_module(results)
    assert results['img'].shape[:2] == (h - 20, w - 20)
    # All bboxes should be reserved after crop
    assert results['img_shape'][:2] == (h - 20, w - 20)
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes'].shape[0] == 8
    assert results['gt_bboxes_ignore'].shape[0] == 2

    def area(bboxes):
        return np.prod(bboxes[:, 2:4] - bboxes[:, 0:2], axis=1)

    assert (area(results['gt_bboxes']) <= area(gt_bboxes)).all()
    assert (area(results['gt_bboxes_ignore']) <= area(gt_bboxes_ignore)).all()
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32

    # test assertion for invalid crop_type
    with pytest.raises(ValueError):
        transform = dict(
            type='RandomCrop', crop_size=(1, 1), crop_type='unknown')
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid crop_size
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCrop', crop_type='relative', crop_size=(0, 0))
        build_from_cfg(transform, PIPELINES)

    def _construct_toy_data():
        img = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8)
        img = np.stack([img, img, img], axis=-1)
        results = dict()
        # image
        results['img'] = img
        results['img_shape'] = img.shape
        results['img_fields'] = ['img']
        # bboxes
        results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']
        results['gt_bboxes'] = np.array([[0., 0., 2., 1.]], dtype=np.float32)
        results['gt_bboxes_ignore'] = np.array([[2., 0., 3., 1.]],
                                               dtype=np.float32)
        # labels
        results['gt_labels'] = np.array([1], dtype=np.int64)
        return results

    # test crop_type "relative_range"
    results = _construct_toy_data()
    transform = dict(
        type='RandomCrop',
        crop_type='relative_range',
        crop_size=(0.3, 0.7),
        allow_negative_crop=True)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    h, w = results_transformed['img_shape'][:2]
    assert int(2 * 0.3 + 0.5) <= h <= int(2 * 1 + 0.5)
    assert int(4 * 0.7 + 0.5) <= w <= int(4 * 1 + 0.5)
    assert results_transformed['gt_bboxes'].dtype == np.float32
    assert results_transformed['gt_bboxes_ignore'].dtype == np.float32

    # test crop_type "relative"
    transform = dict(
        type='RandomCrop',
        crop_type='relative',
        crop_size=(0.3, 0.7),
        allow_negative_crop=True)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    h, w = results_transformed['img_shape'][:2]
    assert h == int(2 * 0.3 + 0.5) and w == int(4 * 0.7 + 0.5)
    assert results_transformed['gt_bboxes'].dtype == np.float32
    assert results_transformed['gt_bboxes_ignore'].dtype == np.float32

    # test crop_type "absolute"
    transform = dict(
        type='RandomCrop',
        crop_type='absolute',
        crop_size=(1, 2),
        allow_negative_crop=True)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    h, w = results_transformed['img_shape'][:2]
    assert h == 1 and w == 2
    assert results_transformed['gt_bboxes'].dtype == np.float32
    assert results_transformed['gt_bboxes_ignore'].dtype == np.float32

    # test crop_type "absolute_range"
    transform = dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(1, 20),
        allow_negative_crop=True)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    h, w = results_transformed['img_shape'][:2]
    assert 1 <= h <= 2 and 1 <= w <= 4
    assert results_transformed['gt_bboxes'].dtype == np.float32
    assert results_transformed['gt_bboxes_ignore'].dtype == np.float32


def test_min_iou_random_crop():
    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img

    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(1, w, h)
    gt_bboxes_ignore = create_random_bboxes(1, w, h)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    transform = dict(type='MinIoURandomCrop')
    crop_module = build_from_cfg(transform, PIPELINES)

    # Test for img_fields
    results_test = copy.deepcopy(results)
    results_test['img1'] = results_test['img']
    results_test['img_fields'] = ['img', 'img1']
    with pytest.raises(AssertionError):
        crop_module(results_test)
    results = crop_module(results)
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32

    patch = np.array([0, 0, results['img_shape'][1], results['img_shape'][0]])
    ious = bbox_overlaps(patch.reshape(-1, 4),
                         results['gt_bboxes']).reshape(-1)
    ious_ignore = bbox_overlaps(
        patch.reshape(-1, 4), results['gt_bboxes_ignore']).reshape(-1)
    mode = crop_module.mode
    if mode == 1:
        assert np.equal(results['gt_bboxes'], gt_bboxes).all()
        assert np.equal(results['gt_bboxes_ignore'], gt_bboxes_ignore).all()
    else:
        assert (ious >= mode).all()
        assert (ious_ignore >= mode).all()


def test_pad():
    # test assertion if both size_divisor and size is None
    with pytest.raises(AssertionError):
        transform = dict(type='Pad')
        build_from_cfg(transform, PIPELINES)

    transform = dict(type='Pad', size_divisor=32)
    transform = build_from_cfg(transform, PIPELINES)
    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['img_fields'] = ['img', 'img2']

    results = transform(results)
    assert np.equal(results['img'], results['img2']).all()
    # original img already divisible by 32
    assert np.equal(results['img'], original_img).all()
    img_shape = results['img'].shape
    assert img_shape[0] % 32 == 0
    assert img_shape[1] % 32 == 0

    resize_transform = dict(
        type='Resize', img_scale=(1333, 800), keep_ratio=True)
    resize_module = build_from_cfg(resize_transform, PIPELINES)
    results = resize_module(results)
    results = transform(results)
    img_shape = results['img'].shape
    assert np.equal(results['img'], results['img2']).all()
    assert img_shape[0] % 32 == 0
    assert img_shape[1] % 32 == 0

    # test the size and size_divisor must be None when pad2square is True
    with pytest.raises(AssertionError):
        transform = dict(type='Pad', size_divisor=32, pad_to_square=True)
        build_from_cfg(transform, PIPELINES)

    transform = dict(type='Pad', pad_to_square=True)
    transform = build_from_cfg(transform, PIPELINES)
    results['img'] = img
    results = transform(results)
    assert results['img'].shape[0] == results['img'].shape[1]

    # test the pad_val is converted to a dict
    transform = dict(type='Pad', size_divisor=32, pad_val=0)
    with pytest.deprecated_call():
        transform = build_from_cfg(transform, PIPELINES)

    assert isinstance(transform.pad_val, dict)
    results = transform(results)
    img_shape = results['img'].shape
    assert img_shape[0] % 32 == 0
    assert img_shape[1] % 32 == 0


def test_normalize():
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
    transform = dict(type='Normalize', **img_norm_cfg)
    transform = build_from_cfg(transform, PIPELINES)
    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['img_fields'] = ['img', 'img2']

    results = transform(results)
    assert np.equal(results['img'], results['img2']).all()

    mean = np.array(img_norm_cfg['mean'])
    std = np.array(img_norm_cfg['std'])
    converted_img = (original_img[..., ::-1] - mean) / std
    assert np.allclose(results['img'], converted_img)


def test_albu_transform():
    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../../../data'),
        img_info=dict(filename='color.jpg'))

    # Define simple pipeline
    load = dict(type='LoadImageFromFile')
    load = build_from_cfg(load, PIPELINES)

    albu_transform = dict(
        type='Albu', transforms=[dict(type='ChannelShuffle', p=1)])
    albu_transform = build_from_cfg(albu_transform, PIPELINES)

    normalize = dict(type='Normalize', mean=[0] * 3, std=[0] * 3, to_rgb=True)
    normalize = build_from_cfg(normalize, PIPELINES)

    # Execute transforms
    results = load(results)
    results = albu_transform(results)
    results = normalize(results)

    assert results['img'].dtype == np.float32


def test_random_center_crop_pad():
    # test assertion for invalid crop_size while test_mode=False
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=(-1, 0),
            test_mode=False,
            test_pad_mode=None)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid ratios while test_mode=False
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=(511, 511),
            ratios=(1.0),
            test_mode=False,
            test_pad_mode=None)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid mean, std and to_rgb
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=(511, 511),
            mean=None,
            std=None,
            to_rgb=None,
            test_mode=False,
            test_pad_mode=None)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid crop_size while test_mode=True
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=(511, 511),
            ratios=None,
            border=None,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
            test_mode=True,
            test_pad_mode=('logical_or', 127))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid ratios while test_mode=True
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=None,
            ratios=(0.9, 1.0, 1.1),
            border=None,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
            test_mode=True,
            test_pad_mode=('logical_or', 127))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid border while test_mode=True
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=None,
            ratios=None,
            border=128,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
            test_mode=True,
            test_pad_mode=('logical_or', 127))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid test_pad_mode while test_mode=True
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=None,
            ratios=None,
            border=None,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
            test_mode=True,
            test_pad_mode=('do_nothing', 100))
        build_from_cfg(transform, PIPELINES)

    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../../../data'),
        img_info=dict(filename='color.jpg'))

    load = dict(type='LoadImageFromFile', to_float32=True)
    load = build_from_cfg(load, PIPELINES)
    results = load(results)
    test_results = copy.deepcopy(results)

    h, w, _ = results['img_shape']
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    train_transform = dict(
        type='RandomCenterCropPad',
        crop_size=(h - 20, w - 20),
        ratios=(1.0, ),
        border=128,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
        test_mode=False,
        test_pad_mode=None)
    crop_module = build_from_cfg(train_transform, PIPELINES)
    train_results = crop_module(results)
    assert train_results['img'].shape[:2] == (h - 20, w - 20)
    # All bboxes should be reserved after crop
    assert train_results['pad_shape'][:2] == (h - 20, w - 20)
    assert train_results['gt_bboxes'].shape[0] == 8
    assert train_results['gt_bboxes_ignore'].shape[0] == 2
    assert train_results['gt_bboxes'].dtype == np.float32
    assert train_results['gt_bboxes_ignore'].dtype == np.float32

    test_transform = dict(
        type='RandomCenterCropPad',
        crop_size=None,
        ratios=None,
        border=None,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
        test_mode=True,
        test_pad_mode=('logical_or', 127))
    crop_module = build_from_cfg(test_transform, PIPELINES)

    test_results = crop_module(test_results)
    assert test_results['img'].shape[:2] == (h | 127, w | 127)
    assert test_results['pad_shape'][:2] == (h | 127, w | 127)
    assert 'border' in test_results


def test_multi_scale_flip_aug():
    # test assertion if give both scale_factor and img_scale
    with pytest.raises(AssertionError):
        transform = dict(
            type='MultiScaleFlipAug',
            scale_factor=1.0,
            img_scale=[(1333, 800)],
            transforms=[dict(type='Resize')])
        build_from_cfg(transform, PIPELINES)

    # test assertion if both scale_factor and img_scale are None
    with pytest.raises(AssertionError):
        transform = dict(
            type='MultiScaleFlipAug',
            scale_factor=None,
            img_scale=None,
            transforms=[dict(type='Resize')])
        build_from_cfg(transform, PIPELINES)

    # test assertion if img_scale is not tuple or list of tuple
    with pytest.raises(AssertionError):
        transform = dict(
            type='MultiScaleFlipAug',
            img_scale=[1333, 800],
            transforms=[dict(type='Resize')])
        build_from_cfg(transform, PIPELINES)

    # test assertion if flip_direction is not str or list of str
    with pytest.raises(AssertionError):
        transform = dict(
            type='MultiScaleFlipAug',
            img_scale=[(1333, 800)],
            flip_direction=1,
            transforms=[dict(type='Resize')])
        build_from_cfg(transform, PIPELINES)

    scale_transform = dict(
        type='MultiScaleFlipAug',
        img_scale=[(1333, 800), (1333, 640)],
        transforms=[dict(type='Resize', keep_ratio=True)])
    transform = build_from_cfg(scale_transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['img_fields'] = ['img']

    scale_results = transform(copy.deepcopy(results))
    assert len(scale_results['img']) == 2
    assert scale_results['img'][0].shape == (750, 1333, 3)
    assert scale_results['img_shape'][0] == (750, 1333, 3)
    assert scale_results['img'][1].shape == (640, 1138, 3)
    assert scale_results['img_shape'][1] == (640, 1138, 3)

    scale_factor_transform = dict(
        type='MultiScaleFlipAug',
        scale_factor=[0.8, 1.0, 1.2],
        transforms=[dict(type='Resize', keep_ratio=False)])
    transform = build_from_cfg(scale_factor_transform, PIPELINES)
    scale_factor_results = transform(copy.deepcopy(results))
    assert len(scale_factor_results['img']) == 3
    assert scale_factor_results['img'][0].shape == (230, 409, 3)
    assert scale_factor_results['img_shape'][0] == (230, 409, 3)
    assert scale_factor_results['img'][1].shape == (288, 512, 3)
    assert scale_factor_results['img_shape'][1] == (288, 512, 3)
    assert scale_factor_results['img'][2].shape == (345, 614, 3)
    assert scale_factor_results['img_shape'][2] == (345, 614, 3)

    # test pipeline of coco_detection
    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../../../data'),
        img_info=dict(filename='color.jpg'))
    load_cfg, multi_scale_cfg = mmcv.Config.fromfile(
        'configs/_base_/datasets/coco_detection.py').test_pipeline
    load = build_from_cfg(load_cfg, PIPELINES)
    transform = build_from_cfg(multi_scale_cfg, PIPELINES)
    results = transform(load(results))
    assert len(results['img']) == 1
    assert len(results['img_metas']) == 1
    assert isinstance(results['img'][0], torch.Tensor)
    assert isinstance(results['img_metas'][0], mmcv.parallel.DataContainer)
    assert results['img_metas'][0].data['ori_shape'] == (288, 512, 3)
    assert results['img_metas'][0].data['img_shape'] == (750, 1333, 3)
    assert results['img_metas'][0].data['pad_shape'] == (768, 1344, 3)
    assert results['img_metas'][0].data['scale_factor'].tolist() == [
        2.603515625, 2.6041667461395264, 2.603515625, 2.6041667461395264
    ]


def test_cutout():
    # test n_holes
    with pytest.raises(AssertionError):
        transform = dict(type='CutOut', n_holes=(5, 3), cutout_shape=(8, 8))
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='CutOut', n_holes=(3, 4, 5), cutout_shape=(8, 8))
        build_from_cfg(transform, PIPELINES)
    # test cutout_shape and cutout_ratio
    with pytest.raises(AssertionError):
        transform = dict(type='CutOut', n_holes=1, cutout_shape=8)
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='CutOut', n_holes=1, cutout_ratio=0.2)
        build_from_cfg(transform, PIPELINES)
    # either of cutout_shape and cutout_ratio should be given
    with pytest.raises(AssertionError):
        transform = dict(type='CutOut', n_holes=1)
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(
            type='CutOut',
            n_holes=1,
            cutout_shape=(2, 2),
            cutout_ratio=(0.4, 0.4))
        build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')

    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['pad_shape'] = img.shape
    results['img_fields'] = ['img']

    transform = dict(type='CutOut', n_holes=1, cutout_shape=(10, 10))
    cutout_module = build_from_cfg(transform, PIPELINES)
    cutout_result = cutout_module(copy.deepcopy(results))
    assert cutout_result['img'].sum() < img.sum()

    transform = dict(type='CutOut', n_holes=1, cutout_ratio=(0.8, 0.8))
    cutout_module = build_from_cfg(transform, PIPELINES)
    cutout_result = cutout_module(copy.deepcopy(results))
    assert cutout_result['img'].sum() < img.sum()

    transform = dict(
        type='CutOut',
        n_holes=(2, 4),
        cutout_shape=[(10, 10), (15, 15)],
        fill_in=(255, 255, 255))
    cutout_module = build_from_cfg(transform, PIPELINES)
    cutout_result = cutout_module(copy.deepcopy(results))
    assert cutout_result['img'].sum() > img.sum()

    transform = dict(
        type='CutOut',
        n_holes=1,
        cutout_ratio=(0.8, 0.8),
        fill_in=(255, 255, 255))
    cutout_module = build_from_cfg(transform, PIPELINES)
    cutout_result = cutout_module(copy.deepcopy(results))
    assert cutout_result['img'].sum() > img.sum()


def test_random_shift():
    # test assertion for invalid shift_ratio
    with pytest.raises(AssertionError):
        transform = dict(type='RandomShift', shift_ratio=1.5)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid max_shift_px
    with pytest.raises(AssertionError):
        transform = dict(type='RandomShift', max_shift_px=-1)
        build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img
    # TODO: add img_fields test
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']

    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    transform = dict(type='RandomShift', shift_ratio=1.0)
    random_shift_module = build_from_cfg(transform, PIPELINES)
    results = random_shift_module(results)

    assert results['img'].shape[:2] == (h, w)
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32


def test_random_affine():
    # test assertion for invalid translate_ratio
    with pytest.raises(AssertionError):
        transform = dict(type='RandomAffine', max_translate_ratio=1.5)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid scaling_ratio_range
    with pytest.raises(AssertionError):
        transform = dict(type='RandomAffine', scaling_ratio_range=(1.5, 0.5))
        build_from_cfg(transform, PIPELINES)

    with pytest.raises(AssertionError):
        transform = dict(type='RandomAffine', scaling_ratio_range=(0, 0.5))
        build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']

    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    transform = dict(type='RandomAffine')
    random_affine_module = build_from_cfg(transform, PIPELINES)
    results = random_affine_module(results)

    assert results['img'].shape[:2] == (h, w)
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32

    # test filter bbox
    gt_bboxes = np.array([[0, 0, 1, 1], [0, 0, 3, 100]], dtype=np.float32)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    transform = dict(
        type='RandomAffine',
        max_rotate_degree=0.,
        max_translate_ratio=0.,
        scaling_ratio_range=(1., 1.),
        max_shear_degree=0.,
        border=(0, 0),
        min_bbox_size=2,
        max_aspect_ratio=20,
        skip_filter=False)
    random_affine_module = build_from_cfg(transform, PIPELINES)

    results = random_affine_module(results)

    assert results['gt_bboxes'].shape[0] == 0
    assert results['gt_labels'].shape[0] == 0
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32


def test_mosaic():
    # test assertion for invalid img_scale
    with pytest.raises(AssertionError):
        transform = dict(type='Mosaic', img_scale=640)
        build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img
    # TODO: add img_fields test
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']

    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    transform = dict(type='Mosaic', img_scale=(10, 12))
    mosaic_module = build_from_cfg(transform, PIPELINES)

    # test assertion for invalid mix_results
    with pytest.raises(AssertionError):
        mosaic_module(results)

    results['mix_results'] = [copy.deepcopy(results)] * 3
    results = mosaic_module(results)
    assert results['img'].shape[:2] == (20, 24)
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32


def test_mixup():
    # test assertion for invalid img_scale
    with pytest.raises(AssertionError):
        transform = dict(type='MixUp', img_scale=640)
        build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img
    # TODO: add img_fields test
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']

    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    transform = dict(type='MixUp', img_scale=(10, 12))
    mixup_module = build_from_cfg(transform, PIPELINES)

    # test assertion for invalid mix_results
    with pytest.raises(AssertionError):
        mixup_module(results)

    with pytest.raises(AssertionError):
        results['mix_results'] = [copy.deepcopy(results)] * 2
        mixup_module(results)

    results['mix_results'] = [copy.deepcopy(results)]
    results = mixup_module(results)
    assert results['img'].shape[:2] == (288, 512)
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32


def test_photo_metric_distortion():
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    transform = dict(type='PhotoMetricDistortion')
    distortion_module = build_from_cfg(transform, PIPELINES)

    # test assertion for invalid img_fields
    with pytest.raises(AssertionError):
        results = dict()
        results['img'] = img
        results['img2'] = img
        results['img_fields'] = ['img', 'img2']
        distortion_module(results)

    # test uint8 input
    results = dict()
    results['img'] = img
    results = distortion_module(results)
    assert results['img'].dtype == np.float32

    # test float32 input
    results = dict()
    results['img'] = img.astype(np.float32)
    results = distortion_module(results)
    assert results['img'].dtype == np.float32
