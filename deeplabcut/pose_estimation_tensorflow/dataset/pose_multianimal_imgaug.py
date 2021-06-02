"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import logging
import os
import pickle
import random as rand

import imageio
import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables import Keypoint, KeypointsOnImage
from numpy import array as arr
from numpy import concatenate as cat

from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import (
    Batch,
    DataItem,
    data_to_input_batch,
)
from deeplabcut.utils.auxfun_videos import imread


class MAPoseDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data = self.load_dataset()
        self.num_images = len(self.data)
        self.batch_size = cfg["batch_size"]
        print("Batch Size is %d" % self.batch_size)

    def load_dataset(self):
        cfg = self.cfg
        file_name = os.path.join(self.cfg["project_path"], cfg["dataset"])
        with open(os.path.join(self.cfg["project_path"], file_name), "rb") as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickledata = pickle.load(f)

        self.raw_data = pickledata
        num_images = len(pickledata)
        data = []
        has_gt = True

        for i in range(num_images):
            sample = pickledata[i]  # mlab[0, i]
            item = DataItem()
            item.image_id = i
            item.im_path = sample["image"]  # [0][0]
            item.im_size = sample["size"]  # sample[1][0]
            if "joints" in sample.keys():
                Joints = sample["joints"]
                if (
                    np.size(
                        np.concatenate(
                            [Joints[person_id][:, 1:3] for person_id in Joints.keys()]
                        )
                    )
                    > 0
                ):
                    item.joints = Joints
                else:
                    has_gt = False  # no animal has joints!
                # item.numanimals=len(item.joints)-1 #as there are also the parts that are not per animal
            else:
                has_gt = False
            data.append(item)

        self.has_gt = has_gt
        return data

    def build_augmentation_pipeline(self, height=None, width=None, apply_prob=0.5):
        sometimes = lambda aug: iaa.Sometimes(apply_prob, aug)
        pipeline = iaa.Sequential(random_order=False)
        cfg = self.cfg
        if cfg.get("fliplr", False):
            opt = cfg.get("fliplr", False)
            if type(opt) == int:
                pipeline.add(sometimes(iaa.Fliplr(opt)))
            else:
                pipeline.add(sometimes(iaa.Fliplr(0.5)))
        if cfg.get("rotation", False):
            opt = cfg.get("rotation", False)
            if type(opt) == int:
                pipeline.add(sometimes(iaa.Affine(rotate=(-opt, opt))))
            else:
                pipeline.add(sometimes(iaa.Affine(rotate=(-10, 10))))
        if cfg.get("hist_eq", False):
            pipeline.add(sometimes(iaa.AllChannelsHistogramEqualization()))
        if cfg.get("motion_blur", False):
            opts = cfg.get("motion_blur", False)
            if type(opts) == list:
                opts = dict(opts)
                pipeline.add(sometimes(iaa.MotionBlur(**opts)))
            else:
                pipeline.add(sometimes(iaa.MotionBlur(k=7, angle=(-90, 90))))
        if cfg.get("covering", False):
            pipeline.add(
                sometimes(iaa.CoarseDropout((0, 0.02), size_percent=(0.01, 0.05)))
            )  # , per_channel=0.5)))
        if cfg.get("elastic_transform", False):
            pipeline.add(sometimes(iaa.ElasticTransformation(sigma=5)))
        if cfg.get("gaussian_noise", False):
            opt = cfg.get("gaussian_noise", False)
            if type(opt) == int or type(opt) == float:
                pipeline.add(
                    sometimes(
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, opt), per_channel=0.5
                        )
                    )
                )
            else:
                pipeline.add(
                    sometimes(
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        )
                    )
                )
        if height is not None and width is not None:
            pipeline.add(
                iaa.Sometimes(
                    cfg.get("cropratio", 0.4),
                    iaa.CropAndPad(percent=(-0.3, 0.1), keep_size=False),
                )
            )
            pipeline.add(iaa.Resize({"height": height, "width": width}))
        return pipeline

    def get_batch(self):
        img_idx = np.random.choice(self.num_images, size=self.batch_size, replace=True)
        batch_images = []
        batch_joints = []
        joint_ids = []
        data_items = []

        # Scale is sampled only once (per batch) to transform all of the images into same size.
        scale = self.get_scale()
        while True:
            idx = np.random.choice(self.num_images)
            scale = self.get_scale()
            size = self.data[idx].im_size
            target_size = np.ceil(size[1:3] * scale).astype(int)
            if self.is_valid_size(target_size):
                break

        stride = self.cfg["stride"]
        for i in range(self.batch_size):
            data_item = self.data[img_idx[i]]

            data_items.append(data_item)
            im_file = data_item.im_path

            logging.debug("image %s", im_file)
            image = imread(os.path.join(self.cfg["project_path"], im_file), mode="RGB")
            if self.has_gt:
                Joints = data_item.joints
                joint_id = [
                    Joints[person_id][:, 0].astype(int) for person_id in Joints.keys()
                ]
                joint_points = np.concatenate(
                    [Joints[person_id][:, 1:3] for person_id in Joints.keys()]
                )
                joint_ids.append(joint_id)
                batch_joints.append(arr(joint_points))

            batch_images.append(image)

        sm_size = np.ceil(target_size / (stride * self.cfg.get("smfactor", 2))).astype(
            int
        ) * self.cfg.get("smfactor", 2)

        if stride == 2:
            sm_size = np.ceil(target_size / 16).astype(int)
            sm_size *= 8

        # assert len(batch_images) == self.batch_size
        return batch_images, joint_ids, batch_joints, data_items, sm_size, target_size

    def get_targetmaps_update(
        self, joint_ids, joints, data_items, sm_size, target_size
    ):
        part_score_targets, part_score_weights, locref_targets, locref_masks = (
            [],
            [],
            [],
            [],
        )
        partaffinityfield_targets, partaffinityfield_masks = [], []
        for i in range(len(data_items)):
            # Approximating the scale
            scale = min(
                target_size[0] / data_items[i].im_size[1],
                target_size[1] / data_items[i].im_size[2],
            )
            if self.cfg.get("scmap_type", None) == "gaussian":
                assert 0 == 1  # not implemented for pafs!
                (
                    part_score_target,
                    part_score_weight,
                    locref_target,
                    locref_mask,
                ) = self.gaussian_scmap(
                    joint_ids[i], [joints[i]], data_items[i], sm_size, scale
                )
            else:
                (
                    part_score_target,
                    part_score_weight,
                    locref_target,
                    locref_mask,
                    partaffinityfield_target,
                    partaffinityfield_mask,
                ) = self.compute_target_part_scoremap_numpy(
                    joint_ids[i], [joints[i]], data_items[i], sm_size, scale
                )

            part_score_targets.append(part_score_target)
            part_score_weights.append(part_score_weight)
            locref_targets.append(locref_target)
            locref_masks.append(locref_mask)
            partaffinityfield_targets.append(partaffinityfield_target)
            partaffinityfield_masks.append(partaffinityfield_mask)

        return {
            Batch.part_score_targets: part_score_targets,
            Batch.part_score_weights: part_score_weights,
            Batch.locref_targets: locref_targets,
            Batch.locref_mask: locref_masks,
            Batch.pairwise_targets: partaffinityfield_targets,
            Batch.pairwise_mask: partaffinityfield_masks,
        }

    def next_batch(self, plotting=False):
        while True:
            (
                batch_images,
                joint_ids,
                batch_joints,
                data_items,
                sm_size,
                target_size,
            ) = self.get_batch()

            pipeline = self.build_augmentation_pipeline(
                height=target_size[0], width=target_size[1], apply_prob=0.5
            )

            batch_images, batch_joints = pipeline(
                images=batch_images, keypoints=batch_joints
            )

            # If you would like to check the augmented images, script for saving
            # the images with joints on:
            if plotting:
                for i in range(self.batch_size):
                    joints = batch_joints[i]
                    kps = KeypointsOnImage(
                        [Keypoint(x=joint[0], y=joint[1]) for joint in joints],
                        shape=batch_images[i].shape,
                    )
                    im = kps.draw_on_image(batch_images[i])
                    # imageio.imwrite(data_items[i].im_path.split('/')[-1],im)
                    imageio.imwrite(
                        os.path.join(self.cfg["project_path"], str(i) + ".png"), im
                    )

            image_shape = arr(batch_images).shape[1:3]
            batch = {Batch.inputs: arr(batch_images).astype(np.float64)}
            if self.has_gt:
                targetmaps = self.get_targetmaps_update(
                    joint_ids, batch_joints, data_items, sm_size, image_shape
                )
                batch.update(targetmaps)

            # if returndata:
            #        return batch_images,batch_joints,targetmaps

            batch = {key: data_to_input_batch(data) for (key, data) in batch.items()}
            batch[Batch.data_item] = data_items
            return batch

    def set_test_mode(self, test_mode):
        self.has_gt = not test_mode

    def num_training_samples(self):
        num = self.num_images
        if self.cfg["mirror"]:
            num *= 2
        return num

    def get_scale(self):
        cfg = self.cfg
        scale = cfg["global_scale"]
        if hasattr(cfg, "scale_jitter_lo") and hasattr(cfg, "scale_jitter_up"):
            scale_jitter = rand.uniform(cfg["scale_jitter_lo"], cfg["scale_jitter_up"])
            scale *= scale_jitter
        return scale

    def is_valid_size(self, target_size):
        im_width = target_size[1]
        im_height = target_size[0]
        min_input_size = 100
        if im_height < min_input_size or im_width < min_input_size:
            return False
        if hasattr(self.cfg, "max_input_size"):
            max_input_size = self.cfg["max_input_size"]
            if im_width * im_height > max_input_size * max_input_size:
                return False
        return True

    def compute_scmap_weights(self, scmap_shape, joint_id, data_item):
        cfg = self.cfg
        if cfg["weigh_only_present_joints"]:
            weights = np.zeros(scmap_shape)
            for k, j_id in enumerate(
                np.concatenate(joint_id)
            ):  # looping over all animals
                weights[:, :, j_id] = 1.0
        else:
            weights = np.ones(scmap_shape)
        return weights

    def compute_target_part_scoremap_numpy(
        self, joint_id, coords, data_item, size, scale
    ):
        stride = self.cfg["stride"]
        dist_thresh = float(self.cfg["pos_dist_thresh"] * scale)
        num_idchannel = self.cfg.get("num_idchannel", 0)

        num_joints = self.cfg["num_joints"]
        half_stride = stride / 2

        scmap = np.zeros(cat([size, arr([num_joints + num_idchannel])]))
        locref_size = cat([size, arr([num_joints * 2])])

        locref_map = np.zeros(locref_size)
        locref_scale = 1.0 / self.cfg["locref_stdev"]
        dist_thresh_sq = dist_thresh ** 2

        partaffinityfield_shape = cat([size, arr([self.cfg["num_limbs"] * 2])])
        partaffinityfield_map = np.zeros(partaffinityfield_shape)
        if self.cfg["weigh_only_present_joints"]:
            partaffinityfield_mask = np.zeros(partaffinityfield_shape)
            locref_mask = np.zeros(locref_size)
        else:
            partaffinityfield_mask = np.ones(partaffinityfield_shape)
            locref_mask = np.ones(locref_size)

        width = size[1]
        height = size[0]
        grid = np.mgrid[:height, :width].transpose((1, 2, 0))

        # the animal id plays no role for scoremap + locref!
        # so let's just loop over all bpts.
        for k, j_id in enumerate(np.concatenate(joint_id)):
            joint_pt = coords[0][k, :]
            j_x = np.asscalar(joint_pt[0])
            j_x_sm = round((j_x - half_stride) / stride)
            j_y = np.asscalar(joint_pt[1])
            j_y_sm = round((j_y - half_stride) / stride)

            min_x = round(max(j_x_sm - dist_thresh - 1, 0))
            max_x = round(min(j_x_sm + dist_thresh + 1, width - 1))
            min_y = round(max(j_y_sm - dist_thresh - 1, 0))
            max_y = round(min(j_y_sm + dist_thresh + 1, height - 1))
            x = grid.copy()[:, :, 1]
            y = grid.copy()[:, :, 0]
            dx = j_x - x * stride - half_stride
            dy = j_y - y * stride - half_stride
            dist = dx ** 2 + dy ** 2
            mask1 = dist <= dist_thresh_sq
            mask2 = (x >= min_x) & (x <= max_x)
            mask3 = (y >= min_y) & (y <= max_y)
            mask = mask1 & mask2 & mask3
            scmap[mask, j_id] = 1
            if self.cfg["weigh_only_present_joints"]:
                locref_mask[mask, j_id * 2 + 0] = 1.0
                locref_mask[mask, j_id * 2 + 1] = 1.0
            locref_map[mask, j_id * 2 + 0] = (dx * locref_scale)[mask]
            locref_map[mask, j_id * 2 + 1] = (dy * locref_scale)[mask]

        if num_idchannel > 0:
            coordinateoffset = 0
            # Find indices of individuals in joint_id
            idx = [i for i, id_ in enumerate(data_item.joints)
                   if id_ < num_idchannel]
            for person_id in idx:
                joint_ids = joint_id[person_id].copy()
                if len(joint_ids) > 1:
                    for k, j_id in enumerate(joint_ids):
                        joint_pt = coords[0][k + coordinateoffset, :]
                        j_x = np.asscalar(joint_pt[0])
                        j_x_sm = round((j_x - half_stride) / stride)
                        j_y = np.asscalar(joint_pt[1])
                        j_y_sm = round((j_y - half_stride) / stride)

                        min_x = round(max(j_x_sm - dist_thresh - 1, 0))
                        max_x = round(min(j_x_sm + dist_thresh + 1, width - 1))
                        min_y = round(max(j_y_sm - dist_thresh - 1, 0))
                        max_y = round(min(j_y_sm + dist_thresh + 1, height - 1))
                        x = grid.copy()[:, :, 1]
                        y = grid.copy()[:, :, 0]
                        dx = j_x - x * stride - half_stride
                        dy = j_y - y * stride - half_stride
                        dist = dx ** 2 + dy ** 2
                        mask1 = dist <= dist_thresh_sq
                        mask2 = (x >= min_x) & (x <= max_x)
                        mask3 = (y >= min_y) & (y <= max_y)
                        mask = mask1 & mask2 & mask3
                        scmap[mask, person_id + num_joints] = 1
                coordinateoffset += len(joint_ids)

        x = grid.copy()[:, :, 1]
        y = grid.copy()[:, :, 0]

        # if self.cfg.partaffinityfield_predict:
        # print("hello",joint_id)
        # print(np.concatenate(joint_id)) #this is all joint_ids for all individuals!
        coordinateoffset = 0  # the offset based on
        for person_id in range(len(joint_id)):
            # for k, joint_ids in enumerate(joint_id[person_id]):
            joint_ids = joint_id[person_id].copy()
            if len(joint_ids) > 1:  # otherwise there cannot be a joint!
                # CONSIDER SMARTER SEARCHES here... (i.e. calculate the bpts beforehand?)
                for l in range(self.cfg["num_limbs"]):
                    bp1, bp2 = self.cfg["partaffinityfield_graph"][l]
                    I1 = np.where(np.array(joint_ids) == bp1)[0]
                    I2 = np.where(np.array(joint_ids) == bp2)[0]
                    if (len(I1) > 0) * (len(I2) > 0):
                        indbp1 = np.asscalar(I1[0])
                        indbp2 = np.asscalar(I2[0])
                        j_x = np.asscalar(coords[0][indbp1 + coordinateoffset, 0])
                        j_y = np.asscalar(coords[0][indbp1 + coordinateoffset, 1])

                        linkedj_x = np.asscalar(coords[0][indbp2 + coordinateoffset, 0])
                        linkedj_y = np.asscalar(coords[0][indbp2 + coordinateoffset, 1])

                        dist = np.sqrt((linkedj_x - j_x) ** 2 + (linkedj_y - j_y) ** 2)
                        if dist > 0:
                            Dx = (linkedj_x - j_x) * 1.0 / dist  # x-axis UNIT VECTOR
                            Dy = (linkedj_y - j_y) * 1.0 / dist

                            d1 = [
                                np.asscalar(Dx * j_x + Dy * j_y),
                                np.asscalar(Dx * linkedj_x + Dy * linkedj_y),
                            ]  # in-line with direct axis
                            d1lowerboundary = min(d1)
                            d1upperboundary = max(d1)
                            d2mid = np.asscalar(
                                j_y * Dx - j_x * Dy
                            )  # orthogonal direction

                            distance_along = Dx * (x * stride + half_stride) + Dy * (
                                y * stride + half_stride
                            )
                            distance_across = (
                                (
                                    (
                                        (y * stride + half_stride) * Dx
                                        - (x * stride + half_stride) * Dy
                                    )
                                    - d2mid
                                )
                                * 1.0
                                / self.cfg["pafwidth"]
                                * scale
                            )

                            mask1 = (distance_along >= d1lowerboundary) & (
                                distance_along <= d1upperboundary
                            )
                            mask2 = np.abs(distance_across) <= 1
                            # mask3 = ((x >= 0) & (x <= width-1))
                            # mask4 = ((y >= 0) & (y <= height-1))
                            mask = mask1 & mask2  # &mask3 &mask4
                            if self.cfg["weigh_only_present_joints"]:
                                partaffinityfield_mask[mask, l * 2 + 0] = 1.0
                                partaffinityfield_mask[mask, l * 2 + 1] = 1.0

                            partaffinityfield_map[mask, l * 2 + 0] = (
                                Dx * (1 - abs(distance_across))
                            )[mask]
                            partaffinityfield_map[mask, l * 2 + 1] = (
                                Dy * (1 - abs(distance_across))
                            )[mask]

            coordinateoffset += len(joint_ids)  # keeping track of the blocks

        weights = self.compute_scmap_weights(scmap.shape, joint_id, data_item)
        return (
            scmap,
            weights,
            locref_map,
            locref_mask,
            partaffinityfield_map,
            partaffinityfield_mask,
        )

    def gaussian_scmap(self, joint_id, coords, data_item, size, scale):
        # WIP!
        stride = self.cfg["stride"]
        dist_thresh = float(self.cfg["pos_dist_thresh"] * scale)
        num_idchannel = self.cfg.get("num_idchannel", 0)

        num_joints = self.cfg["num_joints"]
        half_stride = stride / 2
        scmap = np.zeros(cat([size, arr([num_joints])]))
        locref_size = cat([size, arr([num_joints * 2])])
        locref_mask = np.zeros(locref_size)
        locref_map = np.zeros(locref_size)

        locref_scale = 1.0 / self.cfg["locref_stdev"]
        dist_thresh_sq = dist_thresh ** 2

        partaffinityfield_shape = cat([size, arr([self.cfg["num_limbs"] * 2])])
        partaffinityfield_map = np.zeros(partaffinityfield_shape)
        if self.cfg["weigh_only_present_joints"]:
            partaffinityfield_mask = np.zeros(partaffinityfield_shape)
            locref_mask = np.zeros(locref_size)
        else:
            partaffinityfield_mask = np.ones(partaffinityfield_shape)
            locref_mask = np.ones(locref_size)

        # STD of gaussian is 1/4 of threshold
        std = dist_thresh / 4
        width = size[1]
        height = size[0]
        # Grid of coordinates
        grid = np.mgrid[:height, :width].transpose((1, 2, 0))
        grid = grid * stride + half_stride
        # the animal id plays no role for scoremap + locref!
        # so let's just loop over all bpts.
        for k, j_id in enumerate(np.concatenate(joint_id)):
            joint_pt = coords[0][k, :]
            j_x = np.asscalar(joint_pt[0])
            j_x_sm = round((j_x - half_stride) / stride)
            j_y = np.asscalar(joint_pt[1])
            j_y_sm = round((j_y - half_stride) / stride)

            map_j = grid.copy()
            # Distance between the joint point and each coordinate
            dist = np.linalg.norm(grid - (j_y, j_x), axis=2) ** 2
            scmap_j = np.exp(-dist / (2 * (std ** 2)))
            scmap[..., j_id] = scmap_j
            locref_mask[dist <= dist_thresh_sq, j_id * 2 + 0] = 1
            locref_mask[dist <= dist_thresh_sq, j_id * 2 + 1] = 1
            dx = j_x - grid.copy()[:, :, 1]
            dy = j_y - grid.copy()[:, :, 0]
            locref_map[..., j_id * 2 + 0] = dx * locref_scale
            locref_map[..., j_id * 2 + 1] = dy * locref_scale

        if num_idchannel > 0:
            # NEEDS TO BE DONE!
            assert 0 == 1

        coordinateoffset = 0  # the offset based on
        for person_id in range(len(joint_id)):
            # for k, joint_ids in enumerate(joint_id[person_id]):
            joint_ids = joint_id[person_id].copy()
            if len(joint_ids) > 1:  # otherwise there cannot be a joint!
                # CONSIDER SMARTER SEARCHES here... (i.e. calculate the bpts beforehand?)
                for l in range(self.cfg["num_limbs"]):
                    bp1, bp2 = self.cfg["partaffinityfield_graph"][l]
                    I1 = np.where(np.array(joint_ids) == bp1)[0]
                    I2 = np.where(np.array(joint_ids) == bp2)[0]
                    if (len(I1) > 0) * (len(I2) > 0):
                        indbp1 = np.asscalar(I1[0])
                        indbp2 = np.asscalar(I2[0])
                        j_x = np.asscalar(coords[0][indbp1 + coordinateoffset, 0])
                        j_y = np.asscalar(coords[0][indbp1 + coordinateoffset, 1])

                        linkedj_x = np.asscalar(coords[0][indbp2 + coordinateoffset, 0])
                        linkedj_y = np.asscalar(coords[0][indbp2 + coordinateoffset, 1])

                        dist = np.sqrt((linkedj_x - j_x) ** 2 + (linkedj_y - j_y) ** 2)
                        if dist > 0:
                            Dx = (linkedj_x - j_x) * 1.0 / dist  # x-axis UNIT VECTOR
                            Dy = (linkedj_y - j_y) * 1.0 / dist

                            d1 = [
                                np.asscalar(Dx * j_x + Dy * j_y),
                                np.asscalar(Dx * linkedj_x + Dy * linkedj_y),
                            ]  # in-line with direct axis
                            d1lowerboundary = min(d1)
                            d1upperboundary = max(d1)
                            d2mid = np.asscalar(
                                j_y * Dx - j_x * Dy
                            )  # orthogonal direction

                            distance_along = Dx * (x * stride + half_stride) + Dy * (
                                y * stride + half_stride
                            )
                            distance_across = (
                                (
                                    (
                                        (y * stride + half_stride) * Dx
                                        - (x * stride + half_stride) * Dy
                                    )
                                    - d2mid
                                )
                                * 1.0
                                / self.cfg["pafwidth"]
                                * scale
                            )
                            mask1 = (distance_along >= d1lowerboundary) & (
                                distance_along <= d1upperboundary
                            )
                            mask2 = np.abs(distance_across) <= 1
                            # mask3 = ((x >= 0) & (x <= width-1))
                            # mask4 = ((y >= 0) & (y <= height-1))
                            mask = mask1 & mask2  # &mask3 &mask4
                            if self.cfg["weigh_only_present_joints"]:
                                partaffinityfield_mask[mask, l * 2 + 0] = 1.0
                                partaffinityfield_mask[mask, l * 2 + 1] = 1.0

                            partaffinityfield_map[mask, l * 2 + 0] = (
                                Dx * (1 - abs(distance_across))
                            )[mask]
                            partaffinityfield_map[mask, l * 2 + 1] = (
                                Dy * (1 - abs(distance_across))
                            )[mask]

            coordinateoffset += len(joint_ids)  # keeping track of the blocks

        weights = self.compute_scmap_weights(scmap.shape, joint_id, data_item)
        return scmap, weights, locref_map, locref_mask
