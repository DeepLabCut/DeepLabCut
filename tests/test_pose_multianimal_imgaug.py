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
import numpy as np
import os
import pytest
from conftest import TEST_DATA_DIR
from deeplabcut.pose_estimation_tensorflow.datasets import (
    Batch,
    pose_multianimal_imgaug,
    PoseDatasetFactory,
)
from deeplabcut.utils import read_plainconfig


def mock_imread(path, mode):
    return (np.random.rand(400, 400, 3) * 255).astype(np.uint8)


pose_multianimal_imgaug.imread = mock_imread


@pytest.fixture()
def ma_dataset():
    cfg = read_plainconfig(os.path.join(TEST_DATA_DIR, "pose_cfg.yaml"))
    cfg["project_path"] = TEST_DATA_DIR
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
        batch_images, joint_ids, batch_joints, _, data_items = ma_dataset.get_batch()
        assert (
            len(batch_images)
            == len(joint_ids)
            == len(batch_joints)
            == len(data_items)
            == batch_size
        )
        for data_item, joint_id in zip(data_items, joint_ids):
            assert len(data_item.joints) == len(joint_id)
            for joints, id_ in zip(data_item.joints.values(), joint_id):
                np.testing.assert_equal(joints[:, 0], id_)


def test_build_augmentation_pipeline(ma_dataset):
    for prob in (0.3, 0.5):
        _ = ma_dataset.build_augmentation_pipeline(prob)


@pytest.mark.parametrize("num_idchannel", range(4))
def test_get_targetmaps(ma_dataset, num_idchannel):
    ma_dataset.cfg["num_idchannel"] = num_idchannel
    batch = list(ma_dataset.get_batch()[1:])
    batch.pop(2)
    target_size, sm_size = ma_dataset.calc_target_and_scoremap_sizes()
    scale = np.mean(target_size / ma_dataset.default_size)
    maps = ma_dataset.get_targetmaps_update(*batch, sm_size, scale)
    assert all(len(map_) == ma_dataset.batch_size for map_ in maps.values())
    assert (
        maps[Batch.part_score_targets][0].shape
        == maps[Batch.part_score_weights][0].shape
    )
    assert (
        maps[Batch.part_score_targets][0].shape[2]
        == ma_dataset.cfg["num_joints"] + num_idchannel
    )
    assert maps[Batch.locref_targets][0].shape == maps[Batch.locref_mask][0].shape
    assert maps[Batch.locref_targets][0].shape[2] == 2 * ma_dataset.cfg["num_joints"]
    assert (
        maps[Batch.pairwise_targets][0].shape == maps[Batch.pairwise_targets][0].shape
    )
    assert maps[Batch.pairwise_targets][0].shape[2] == 2 * ma_dataset.cfg["num_limbs"]


def test_batching(ma_dataset):
    for _ in range(10):
        batch = ma_dataset.next_batch()
