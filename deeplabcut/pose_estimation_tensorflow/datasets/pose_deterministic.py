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


import logging
import numpy as np
import os
import scipy.io as sio
from deeplabcut.utils.auxfun_videos import imread, imresize
from deeplabcut.utils.conversioncode import robust_split_path
from .factory import PoseDatasetFactory
from .pose_base import BasePoseDataset
from .utils import (
    DataItem,
    mirror_joints_map,
    crop_image,
    Batch,
    data_to_input,
)


@PoseDatasetFactory.register("deterministic")
class DeterministicPoseDataset(BasePoseDataset):
    def __init__(self, cfg):
        super(DeterministicPoseDataset, self).__init__(cfg)
        self.data = self.load_dataset()
        self.num_images = len(self.data)
        if self.cfg["mirror"]:
            self.symmetric_joints = mirror_joints_map(
                cfg["all_joints"], cfg["num_joints"]
            )
        self.curr_img = 0
        self.scale = cfg["global_scale"]
        self.locref_scale = 1.0 / cfg["locref_stdev"]
        self.stride = cfg["stride"]
        self.half_stride = self.stride / 2
        self.set_shuffle(cfg["shuffle"])

    def load_dataset(self):
        cfg = self.cfg
        file_name = os.path.join(self.cfg["project_path"], cfg["dataset"])
        mlab = sio.loadmat(file_name)
        self.raw_data = mlab
        mlab = mlab["dataset"]

        num_images = mlab.shape[1]
        data = []
        has_gt = True

        for i in range(num_images):
            sample = mlab[0, i]

            item = DataItem()
            item.image_id = i
            im_path = sample[0][0]
            if isinstance(im_path, str):
                im_path = robust_split_path(im_path)
            else:
                im_path = [s.strip() for s in im_path]
            item.im_path = os.path.join(*im_path)
            item.im_size = sample[1][0]
            if len(sample) >= 3:
                joints = sample[2][0][0]
                joint_id = joints[:, 0]
                # make sure joint ids are 0-indexed
                if joint_id.size != 0:
                    assert np.any(joint_id < cfg["num_joints"])
                joints[:, 0] = joint_id
                item.joints = [joints]
            else:
                has_gt = False
            # if cfg.crop:
            #    crop = sample[3][0] - 1
            #    item.crop = extend_crop(crop, cfg.crop_pad, item.im_size)
            data.append(item)

        self.has_gt = has_gt
        return data

    def set_test_mode(self, test_mode):
        self.has_gt = not test_mode

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle
        if not shuffle:
            assert not self.cfg["mirror"]
            self.image_indices = np.arange(self.num_images)

    def mirror_joint_coords(self, joints, image_width):
        # horizontally flip the x-coordinate, keep y unchanged
        joints[:, 1] = image_width - joints[:, 1] - 1
        return joints

    def mirror_joints(self, joints, symmetric_joints, image_width):
        # joint ids are 0 indexed
        res = np.copy(joints)
        res = self.mirror_joint_coords(res, image_width)
        # swap the joint_id for a symmetric one
        joint_id = joints[:, 0].astype(int)
        res[:, 0] = symmetric_joints[joint_id]
        return res

    def shuffle_images(self):
        if self.cfg["deterministic"]:
            np.random.seed(42)
        num_images = self.num_images
        if self.cfg["mirror"]:
            image_indices = np.random.permutation(num_images * 2)
            self.mirrored = image_indices >= num_images
            image_indices[self.mirrored] = image_indices[self.mirrored] - num_images
            self.image_indices = image_indices
        else:
            self.image_indices = np.random.permutation(num_images)

    def num_training_samples(self):
        num = self.num_images
        if self.cfg["mirror"]:
            num *= 2
        return num

    def next_training_sample(self):
        if self.curr_img == 0 and self.shuffle:
            self.shuffle_images()

        curr_img = self.curr_img
        self.curr_img = (self.curr_img + 1) % self.num_training_samples()

        imidx = self.image_indices[curr_img]
        mirror = self.cfg["mirror"] and self.mirrored[curr_img]

        return imidx, mirror

    def get_training_sample(self, imidx):
        return self.data[imidx]

    def next_batch(self):
        while True:
            imidx, mirror = self.next_training_sample()
            data_item = self.get_training_sample(imidx)
            scale = self.sample_scale()

            if not self.is_valid_size(data_item.im_size, scale):
                continue

            return self.make_batch(data_item, scale, mirror)

    def is_valid_size(self, image_size, scale):
        if "min_input_size" in self.cfg and "max_input_size" in self.cfg:
            input_width = image_size[2] * scale
            input_height = image_size[1] * scale
            if (
                input_height < self.cfg["min_input_size"]
                or input_width < self.cfg["min_input_size"]
            ):
                return False
            if input_height * input_width > self.cfg["max_input_size"] ** 2:
                return False

        return True

    def make_batch(self, data_item, scale, mirror):
        im_file = data_item.im_path
        logging.debug("image %s", im_file)
        logging.debug("mirror %r", mirror)
        image = imread(os.path.join(self.cfg["project_path"], im_file), mode="skimage")

        if self.has_gt:
            joints = np.copy(data_item.joints)

        if self.cfg["crop"]:  # adapted cropping for DLC
            if np.random.rand() < self.cfg["cropratio"]:
                j = np.random.randint(np.shape(joints)[1])
                joints, image = crop_image(
                    joints, image, joints[0, j, 1], joints[0, j, 2], self.cfg
                )

        img = imresize(image, scale) if scale != 1 else image
        scaled_img_size = np.array(img.shape[0:2])

        if mirror:
            img = np.fliplr(img)

        batch = {Batch.inputs: img}

        if self.has_gt:
            stride = self.cfg["stride"]
            if mirror:
                joints = [
                    self.mirror_joints(
                        person_joints, self.symmetric_joints, image.shape[1]
                    )
                    for person_joints in joints
                ]
            sm_size = np.ceil(scaled_img_size / (stride * 2)).astype(int) * 2
            scaled_joints = [person_joints[:, 1:3] * scale for person_joints in joints]
            joint_id = [person_joints[:, 0].astype(int) for person_joints in joints]
            (
                part_score_targets,
                part_score_weights,
                locref_targets,
                locref_mask,
            ) = self.compute_target_part_scoremap(
                joint_id, scaled_joints, data_item, sm_size, scale
            )

            batch.update(
                {
                    Batch.part_score_targets: part_score_targets,
                    Batch.part_score_weights: part_score_weights,
                    Batch.locref_targets: locref_targets,
                    Batch.locref_mask: locref_mask,
                }
            )

        batch = {key: data_to_input(data) for (key, data) in batch.items()}

        batch[Batch.data_item] = data_item

        return batch

    def compute_target_part_scoremap(self, joint_id, coords, data_item, size, scale):
        dist_thresh = self.cfg["pos_dist_thresh"] * scale
        dist_thresh_sq = dist_thresh**2
        num_joints = self.cfg["num_joints"]
        scmap = np.zeros(np.concatenate([size, np.array([num_joints])]))
        locref_size = np.concatenate([size, np.array([num_joints * 2])])
        locref_mask = np.zeros(locref_size)
        locref_map = np.zeros(locref_size)
        width = size[1]
        height = size[0]

        for person_id in range(len(coords)):
            for k, j_id in enumerate(joint_id[person_id]):
                joint_pt = coords[person_id][k, :]
                j_x = np.asarray(joint_pt[0]).item()
                j_y = np.asarray(joint_pt[1]).item()

                # don't loop over entire heatmap, but just relevant locations
                j_x_sm = round((j_x - self.half_stride) / self.stride)
                j_y_sm = round((j_y - self.half_stride) / self.stride)
                min_x = round(max(j_x_sm - dist_thresh - 1, 0))
                max_x = round(min(j_x_sm + dist_thresh + 1, width - 1))
                min_y = round(max(j_y_sm - dist_thresh - 1, 0))
                max_y = round(min(j_y_sm + dist_thresh + 1, height - 1))

                for j in range(min_y, max_y + 1):  # range(height):
                    pt_y = j * self.stride + self.half_stride
                    for i in range(min_x, max_x + 1):  # range(width):
                        # pt = arr([i*stride+half_stride, j*stride+half_stride])
                        # diff = joint_pt - pt
                        # The code above is too slow in python
                        pt_x = i * self.stride + self.half_stride
                        dx = j_x - pt_x
                        dy = j_y - pt_y
                        dist = dx**2 + dy**2
                        # print(la.norm(diff))
                        if dist <= dist_thresh_sq:
                            scmap[j, i, j_id] = 1
                            locref_mask[j, i, j_id * 2 + 0] = 1
                            locref_mask[j, i, j_id * 2 + 1] = 1
                            locref_map[j, i, j_id * 2 + 0] = dx * self.locref_scale
                            locref_map[j, i, j_id * 2 + 1] = dy * self.locref_scale

        weights = self.compute_scmap_weights(scmap.shape, joint_id, data_item)

        return scmap, weights, locref_map, locref_mask

    def compute_scmap_weights(self, scmap_shape, joint_id, data_item):
        if self.cfg["weigh_only_present_joints"]:
            weights = np.zeros(scmap_shape)
            for person_joint_id in joint_id:
                for j_id in person_joint_id:
                    weights[:, :, j_id] = 1.0
        else:
            weights = np.ones(scmap_shape)
        return weights
