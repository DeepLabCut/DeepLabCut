#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import os

import numpy as np
import pytest

from deeplabcut.core.config import read_config_as_dict, write_config
from deeplabcut.pose_estimation_tensorflow.datasets import (
    Batch,
    PoseDatasetFactory,
    pose_multianimal_imgaug,
)

tf = pytest.importorskip(
    "tensorflow",
    reason="TensorFlow not installed (use a project extra such as .[tf])",
)


def mock_imread(path, mode):
    return (np.random.rand(400, 400, 3) * 255).astype(np.uint8)


pose_multianimal_imgaug.imread = mock_imread


@pytest.fixture()
def ma_dataset(test_data_dir):
    ## TODO @deruyter92 2026-06-15: this test config is currently invalid and needs to be
    # updated. For now it is updated in place. (see https://github.com/DeepLabCut/UnitTestData/issues/4)
    for cfg_name in ("config.yaml", "pose_cfg.yaml"):
        cfg_path = os.path.join(test_data_dir, cfg_name)
        cfg = read_config_as_dict(cfg_path)
        if len(cfg.get("bodyparts", [])) > 0 and len(cfg.get("multianimalbodyparts", [])) > 0:
            cfg["bodyparts"] = "MULTI!"
        write_config(cfg_path, cfg, overwrite=True)

    cfg = read_config_as_dict(os.path.join(test_data_dir, "pose_cfg.yaml"))
    cfg["project_path"] = test_data_dir
    cfg["dataset"] = "trimouse_train_data.pickle"
    return PoseDatasetFactory.create(cfg)


@pytest.mark.parametrize(
    "scale, stride",
    [
        (0.6, 2),
        (0.6, 4),
        (0.6, 8),
        (0.8, 4),
        (1.0, 8),
        (1.2, 8),
        (0.6, 4),
        (0.8, 8),
    ],
)
def test_calc_target_and_scoremap_sizes(
    ma_dataset,
    scale,
    stride,
):
    ma_dataset.cfg["global_scale"] = scale
    ma_dataset.cfg["stride"] = stride
    # Disable stochastic scale jitter
    ma_dataset.cfg["scale_jitter_lo"] = 1
    ma_dataset.cfg["scale_jitter_up"] = 1
    target_size, sm_size = ma_dataset.calc_target_and_scoremap_sizes()
    np.testing.assert_equal(np.asarray([400, 400]) * scale, target_size)
    np.testing.assert_equal(target_size / stride, sm_size)


def test_get_batch(ma_dataset):
    for batch_size in 1, 4, 8, 16:
        ma_dataset.batch_size = batch_size
        batch_images, joint_ids, batch_joints, data_items = ma_dataset.get_batch()
        assert len(batch_images) == len(joint_ids) == len(batch_joints) == len(data_items) == batch_size
        for data_item, joint_id, batch_joint in zip(data_items, joint_ids, batch_joints, strict=False):
            assert len(data_item.joints) == len(joint_id)
            assert len(batch_joint) == len(np.concatenate(joint_id))
            start = 0
            mask = ~np.isnan(batch_joint).any(axis=1)
            for joints, id_ in zip(data_item.joints.values(), joint_id, strict=False):
                inds = id_ + start
                mask_ = mask[inds]
                np.testing.assert_equal(joints[:, 0], id_[mask_])
                np.testing.assert_equal(joints[:, 1:], batch_joint[inds][mask_])
                start += id_.size


def test_build_augmentation_pipeline(ma_dataset):
    for prob in (0.3, 0.5):
        _ = ma_dataset.build_augmentation_pipeline(prob)


@pytest.mark.parametrize("num_idchannel", range(4))
def test_get_targetmaps(ma_dataset, num_idchannel):
    ma_dataset.cfg["num_idchannel"] = num_idchannel
    batch = ma_dataset.get_batch()[1:]
    target_size, sm_size = ma_dataset.calc_target_and_scoremap_sizes()
    scale = np.mean(target_size / ma_dataset.default_size)
    maps = ma_dataset.get_targetmaps_update(*batch, sm_size, scale)
    assert all(len(map_) == ma_dataset.batch_size for map_ in maps.values())
    assert maps[Batch.part_score_targets][0].shape == maps[Batch.part_score_weights][0].shape
    assert maps[Batch.part_score_targets][0].shape[2] == ma_dataset.cfg["num_joints"] + num_idchannel
    assert maps[Batch.locref_targets][0].shape == maps[Batch.locref_mask][0].shape
    assert maps[Batch.locref_targets][0].shape[2] == 2 * ma_dataset.cfg["num_joints"]
    assert maps[Batch.pairwise_targets][0].shape == maps[Batch.pairwise_targets][0].shape
    assert maps[Batch.pairwise_targets][0].shape[2] == 2 * ma_dataset.cfg["num_limbs"]


def test_batching(ma_dataset):
    for _ in range(10):
        ma_dataset.next_batch()
