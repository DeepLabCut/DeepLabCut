import numpy as np
import os
import pytest
from deeplabcut.pose_estimation_tensorflow.dataset import (
    Batch,
    pose_multianimal_imgaug,
    create,
)
from deeplabcut.utils import read_plainconfig

def mock_imread(path, mode):
    return (np.random.rand(400, 400, 3) * 255).astype(np.uint8)

pose_multianimal_imgaug.imread = mock_imread


@pytest.fixture()
def ma_dataset():
    TEST_DATA_DIR = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data"
    )
    cfg = read_plainconfig(os.path.join(TEST_DATA_DIR, "pose_cfg.yaml"))
    cfg["project_path"] = TEST_DATA_DIR
    cfg["dataset"] = "trimouse_train_data.pickle"
    return create(cfg)


@pytest.mark.parametrize(
    "batch_size, scale, stride",
    [
        (1, 0.6, 2),
        (1, 0.6, 4),
        (1, 0.6, 8),
        (8, 0.8, 4),
        (8, 1.0, 8),
        (8, 1.2, 8),
        (16, 0.6, 4),
        (16, 0.8, 8),
    ]
)
def test_get_batch(
    ma_dataset,
    batch_size,
    scale,
    stride,
):
    ma_dataset.batch_size = batch_size
    ma_dataset.cfg["global_scale"] = scale
    ma_dataset.cfg["stride"] = stride
    (
        batch_images,
        joint_ids,
        batch_joints,
        data_items,
        sm_size,
        target_size,
    ) = ma_dataset.get_batch()
    assert len(batch_images) == len(joint_ids) == len(batch_joints) \
           == len(data_items) == batch_size
    for data_item, joint_id in zip(data_items, joint_ids):
        assert len(data_item.joints) == len(joint_id)
        for joints, id_ in zip(data_item.joints.values(), joint_id):
            np.testing.assert_equal(joints[:, 0], id_)
    np.testing.assert_equal(np.asarray([400, 400]) * scale, target_size)
    np.testing.assert_equal(target_size / stride, sm_size)


@pytest.mark.parametrize(
    "height, width, prob",
    [
        (None, None, 0.5),
        (200, 250, 0.3),
    ]
)
def test_build_augmentation_pipeline(ma_dataset, height, width, prob):
    _ = ma_dataset.build_augmentation_pipeline(height, width, prob)


@pytest.mark.parametrize("num_idchannel", range(4))
def test_get_targetmaps(ma_dataset, num_idchannel):
    ma_dataset.cfg["num_idchannel"] = num_idchannel
    batch = ma_dataset.get_batch()[1:]
    maps = ma_dataset.get_targetmaps_update(*batch)
    assert all(len(map_) == ma_dataset.batch_size for map_ in maps.values())
    assert maps[Batch.part_score_targets][0].shape \
           == maps[Batch.part_score_weights][0].shape
    assert maps[Batch.part_score_targets][0].shape[2] \
           == ma_dataset.cfg["num_joints"] + num_idchannel
    assert maps[Batch.locref_targets][0].shape \
           == maps[Batch.locref_mask][0].shape
    assert maps[Batch.locref_targets][0].shape[2] \
           == 2 * ma_dataset.cfg["num_joints"]
    assert maps[Batch.pairwise_targets][0].shape \
           == maps[Batch.pairwise_targets][0].shape
    assert maps[Batch.pairwise_targets][0].shape[2] \
           == 2 * ma_dataset.cfg["num_limbs"]


def test_batching(ma_dataset):
    for _ in range(10):
        batch = ma_dataset.next_batch()
