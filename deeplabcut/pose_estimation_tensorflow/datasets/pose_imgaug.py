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
"""
Uses imgaug dataflow for flexible augmentation
Largely written by Mert Yüksekgönül during the summer in the Bethge lab -- Thanks!
https://imgaug.readthedocs.io/en/latest/
"""

import logging
import os
import pickle

import imgaug.augmenters as iaa
import numpy as np
import scipy.io as sio
from deeplabcut.pose_estimation_tensorflow.datasets import augmentation
from deeplabcut.utils.auxfun_videos import imread
from deeplabcut.utils.conversioncode import robust_split_path
from .factory import PoseDatasetFactory
from .pose_base import BasePoseDataset
from .utils import DataItem, Batch


@PoseDatasetFactory.register("default")
@PoseDatasetFactory.register("imgaug")
class ImgaugPoseDataset(BasePoseDataset):
    def __init__(self, cfg):
        super(ImgaugPoseDataset, self).__init__(cfg)
        self._n_kpts = len(cfg["all_joints_names"])
        self.data = self.load_dataset()
        self.batch_size = cfg.get("batch_size", 1)
        self.num_images = len(self.data)
        self.max_input_sizesquare = cfg.get("max_input_size", 1500) ** 2
        self.min_input_sizesquare = cfg.get("min_input_size", 64) ** 2

        self.locref_scale = 1.0 / cfg["locref_stdev"]
        self.stride = cfg["stride"]
        self.half_stride = self.stride / 2
        self.scale = cfg["global_scale"]

        # parameter initialization for augmentation pipeline:
        self.scale_jitter_lo = cfg.get("scale_jitter_lo", 0.75)
        self.scale_jitter_up = cfg.get("scale_jitter_up", 1.25)

        cfg["mirror"] = cfg.get("mirror", False)
        cfg["rotation"] = cfg.get("rotation", True)
        if cfg.get("rotation", True):  # i.e. pm 10 degrees
            opt = cfg.get("rotation", False)
            if type(opt) == int:
                cfg["rotation"] = cfg.get("rotation", 25)
            else:
                cfg["rotation"] = 25
            cfg["rotratio"] = cfg.get(
                "rotateratio", 0.4
            )  # what is the fraction of training samples with rotation augmentation?
        else:
            cfg["rotratio"] = 0.0
            cfg["rotation"] = 0

        cfg["covering"] = cfg.get("covering", True)
        cfg["elastic_transform"] = cfg.get("elastic_transform", True)

        cfg["motion_blur"] = cfg.get("motion_blur", True)
        if cfg["motion_blur"]:
            cfg["motion_blur_params"] = dict(
                cfg.get("motion_blur_params", {"k": 7, "angle": (-90, 90)})
            )

        print("Batch Size is %d" % self.batch_size)

    def load_dataset(self):
        cfg = self.cfg
        file_name = os.path.join(self.cfg["project_path"], cfg["dataset"])
        if ".mat" in file_name:  # legacy loader
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
                        assert (joint_id < cfg["num_joints"]).any()
                    joints[:, 0] = joint_id
                    item.joints = [joints]
                else:
                    has_gt = False
                data.append(item)

            self.has_gt = has_gt
            return data
        else:
            print("Loading pickle data with float coordinates!")
            file_name = cfg["dataset"].split(".")[0] + ".pickle"
            with open(os.path.join(self.cfg["project_path"], file_name), "rb") as f:
                pickledata = pickle.load(f)

            self.raw_data = pickledata
            num_images = len(pickledata)  # mlab.shape[1]
            data = []
            has_gt = True
            for i in range(num_images):
                sample = pickledata[i]  # mlab[0, i]
                item = DataItem()
                item.image_id = i
                item.im_path = os.path.join(*sample["image"])  # [0][0]
                item.im_size = sample["size"]  # sample[1][0]
                if len(sample) >= 3:
                    item.num_animals = len(sample["joints"])
                    item.joints = [sample["joints"]]

                else:
                    has_gt = False
                data.append(item)
            self.has_gt = has_gt
            return data

    def build_augmentation_pipeline(self, height=None, width=None, apply_prob=0.5):
        sometimes = lambda aug: iaa.Sometimes(apply_prob, aug)
        pipeline = iaa.Sequential(random_order=False)

        cfg = self.cfg
        if cfg["mirror"]:
            opt = cfg["mirror"]  # fliplr
            if type(opt) == int:
                pipeline.add(sometimes(iaa.Fliplr(opt)))
            else:
                pipeline.add(sometimes(iaa.Fliplr(0.5)))

        if cfg.get("fliplr", False) and cfg.get("symmetric_pairs"):
            opt = cfg.get("fliplr", False)
            if type(opt) == int:
                p = opt
            else:
                p = 0.5
            pipeline.add(
                sometimes(
                    augmentation.KeypointFliplr(
                        cfg["all_joints_names"],
                        symmetric_pairs=cfg["symmetric_pairs"],
                        p=p,
                    )
                )
            )

        if cfg["rotation"] > 0:
            pipeline.add(
                iaa.Sometimes(
                    cfg["rotratio"],
                    iaa.Affine(rotate=(-cfg["rotation"], cfg["rotation"])),
                )
            )

        if cfg["motion_blur"]:
            opts = cfg["motion_blur_params"]
            pipeline.add(sometimes(iaa.MotionBlur(**opts)))

        if cfg["covering"]:
            pipeline.add(
                sometimes(iaa.CoarseDropout(0.02, size_percent=0.3, per_channel=0.5))
            )

        if cfg["elastic_transform"]:
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
        if cfg.get("grayscale", False):
            pipeline.add(sometimes(iaa.Grayscale(alpha=(0.5, 1.0))))

        def get_aug_param(cfg_value):
            if isinstance(cfg_value, dict):
                opt = cfg_value
            else:
                opt = {}
            return opt

        cfg_cnt = cfg.get("contrast", {})
        cfg_cnv = cfg.get("convolution", {})

        contrast_aug = ["histeq", "clahe", "gamma", "sigmoid", "log", "linear"]
        for aug in contrast_aug:
            aug_val = cfg_cnt.get(aug, False)
            cfg_cnt[aug] = aug_val
            if aug_val:
                cfg_cnt[aug + "ratio"] = cfg_cnt.get(aug + "ratio", 0.1)

        convolution_aug = ["sharpen", "emboss", "edge"]
        for aug in convolution_aug:
            aug_val = cfg_cnv.get(aug, False)
            cfg_cnv[aug] = aug_val
            if aug_val:
                cfg_cnv[aug + "ratio"] = cfg_cnv.get(aug + "ratio", 0.1)

        if cfg_cnt["histeq"]:
            opt = get_aug_param(cfg_cnt["histeq"])
            pipeline.add(
                iaa.Sometimes(
                    cfg_cnt["histeqratio"], iaa.AllChannelsHistogramEqualization(**opt)
                )
            )

        if cfg_cnt["clahe"]:
            opt = get_aug_param(cfg_cnt["clahe"])
            pipeline.add(
                iaa.Sometimes(cfg_cnt["claheratio"], iaa.AllChannelsCLAHE(**opt))
            )

        if cfg_cnt["log"]:
            opt = get_aug_param(cfg_cnt["log"])
            pipeline.add(iaa.Sometimes(cfg_cnt["logratio"], iaa.LogContrast(**opt)))

        if cfg_cnt["linear"]:
            opt = get_aug_param(cfg_cnt["linear"])
            pipeline.add(
                iaa.Sometimes(cfg_cnt["linearratio"], iaa.LinearContrast(**opt))
            )

        if cfg_cnt["sigmoid"]:
            opt = get_aug_param(cfg_cnt["sigmoid"])
            pipeline.add(
                iaa.Sometimes(cfg_cnt["sigmoidratio"], iaa.SigmoidContrast(**opt))
            )

        if cfg_cnt["gamma"]:
            opt = get_aug_param(cfg_cnt["gamma"])
            pipeline.add(iaa.Sometimes(cfg_cnt["gammaratio"], iaa.GammaContrast(**opt)))

        if cfg_cnv["sharpen"]:
            opt = get_aug_param(cfg_cnv["sharpen"])
            pipeline.add(iaa.Sometimes(cfg_cnv["sharpenratio"], iaa.Sharpen(**opt)))

        if cfg_cnv["emboss"]:
            opt = get_aug_param(cfg_cnv["emboss"])
            pipeline.add(iaa.Sometimes(cfg_cnv["embossratio"], iaa.Emboss(**opt)))

        if cfg_cnv["edge"]:
            opt = get_aug_param(cfg_cnv["edge"])
            pipeline.add(iaa.Sometimes(cfg_cnv["edgeratio"], iaa.EdgeDetect(**opt)))

        if height is not None and width is not None:
            if not cfg.get("crop_by", False):
                crop_by = 0.15
            else:
                crop_by = cfg.get("crop_by", False)
            pipeline.add(
                iaa.Sometimes(
                    cfg.get("cropratio", 0.4),
                    iaa.CropAndPad(percent=(-crop_by, crop_by), keep_size=False),
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

        # Scale is sampled only once to transform all of the images of a batch into same size.
        scale = self.sample_scale()

        found_valid = False
        n_tries = 10
        while n_tries > 1:
            idx = np.random.choice(self.num_images)
            size = self.data[idx].im_size
            target_size = np.ceil(size[1:3] * scale).astype(int)
            if self.is_valid_size(target_size[1] * target_size[0]):
                found_valid = True
                break
            n_tries -= 1
        if not found_valid:
            if size[1] * size[2] > self.max_input_sizesquare:
                s = "large", "increasing `max_input_size`", "decreasing"
            else:
                s = "small", "decreasing `min_input_size`", "increasing"
            raise ValueError(
                f"Image size {size[1:3]} may be too {s[0]}. "
                f"Consider {s[1]} and/or {s[2]} `global_scale` "
                "in the train/pose_cfg.yaml."
            )

        stride = self.cfg["stride"]
        for i in range(self.batch_size):
            data_item = self.data[img_idx[i]]

            data_items.append(data_item)
            im_file = data_item.im_path

            logging.debug("image %s", im_file)
            image = imread(
                os.path.join(self.cfg["project_path"], im_file), mode="skimage"
            )

            if self.has_gt:
                joints = data_item.joints
                kpts = np.full((self._n_kpts, 2), np.nan)

                for n, x, y in joints[0]:
                    kpts[int(n)] = x, y

                joint_ids.append([np.arange(self._n_kpts)])
                batch_joints.append(kpts)

            batch_images.append(image)
        sm_size = np.ceil(target_size / (stride * 2)).astype(int) * 2
        assert len(batch_images) == self.batch_size
        return batch_images, joint_ids, batch_joints, data_items, sm_size, target_size

    def get_scmap_update(self, joint_ids, joints, data_items, sm_size, target_size):
        part_score_targets, part_score_weights, locref_targets, locref_masks = (
            [],
            [],
            [],
            [],
        )
        for i in range(len(data_items)):
            # Approximating the scale
            scale = min(
                target_size[0] / data_items[i].im_size[1],
                target_size[1] / data_items[i].im_size[2],
            )
            if self.cfg.get("scmap_type", None) == "gaussian":
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
                ) = self.compute_target_part_scoremap_numpy(
                    joint_ids[i], [joints[i]], data_items[i], sm_size, scale
                )
            part_score_targets.append(part_score_target)
            part_score_weights.append(part_score_weight)
            locref_targets.append(locref_target)
            locref_masks.append(locref_mask)

        return {
            Batch.part_score_targets: part_score_targets,
            Batch.part_score_weights: part_score_weights,
            Batch.locref_targets: locref_targets,
            Batch.locref_mask: locref_masks,
        }

    def next_batch(self):
        cfg = self.cfg
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
                height=target_size[0], width=target_size[1],
                apply_prob=cfg.get("apply_prob", 0.5),
            )

            batch_images, batch_joints = pipeline(
                images=batch_images, keypoints=batch_joints
            )

            image_shape = np.array(batch_images).shape[1:3]

            batch_joints_valid = []
            joint_ids_valid = []
            for joints, ids in zip(batch_joints, joint_ids):
                # invisible joints are represented by nans
                mask = ~np.isnan(joints[:, 0])
                joints = joints[mask, :]
                ids = ids[0][mask]
                inside = np.logical_and.reduce(
                    (
                        joints[:, 0] < image_shape[1],
                        joints[:, 0] > 0,
                        joints[:, 1] < image_shape[0],
                        joints[:, 1] > 0,
                    )
                )

                batch_joints_valid.append(joints[inside])
                joint_ids_valid.append([ids[inside]])

            # If you would like to check the augmented images, script for saving
            # the images with joints on:
            # import imageio
            # for i in range(self.batch_size):
            #    joints = batch_joints[i]
            #    kps = KeypointsOnImage([Keypoint(x=joint[0], y=joint[1]) for joint in joints], shape=batch_images[i].shape)
            #    im = kps.draw_on_image(batch_images[i])
            #    imageio.imwrite('some_location/augmented/'+str(i)+'.png', im)

            batch = {Batch.inputs: np.array(batch_images).astype(np.float64)}
            if self.has_gt:
                scmap_update = self.get_scmap_update(
                    joint_ids_valid,
                    batch_joints_valid,
                    data_items,
                    sm_size,
                    image_shape,
                )
                batch.update(scmap_update)

            batch = {key: np.asarray(data) for (key, data) in batch.items()}
            batch[Batch.data_item] = data_items
            return batch

    def set_test_mode(self, test_mode):
        self.has_gt = not test_mode

    def num_training_samples(self):
        num = self.num_images
        if self.cfg["mirror"]:
            num *= 2
        return num

    def is_valid_size(self, target_size_product):
        if target_size_product > self.max_input_sizesquare:
            return False

        if target_size_product < self.min_input_sizesquare:
            return False

        return True

    def gaussian_scmap(self, joint_id, coords, data_item, size, scale):
        # dist_thresh = float(self.cfg.pos_dist_thresh * scale)
        num_joints = self.cfg["num_joints"]
        scmap = np.zeros(np.concatenate([size, np.array([num_joints])]))
        locref_size = np.concatenate([size, np.array([num_joints * 2])])
        locref_mask = np.zeros(locref_size)
        locref_map = np.zeros(locref_size)

        width = size[1]
        height = size[0]
        dist_thresh = float((width + height) / 6)
        dist_thresh_sq = dist_thresh**2

        std = dist_thresh / 4
        # Grid of coordinates
        grid = np.mgrid[:height, :width].transpose((1, 2, 0))
        grid = grid * self.stride + self.half_stride
        for person_id in range(len(coords)):
            for k, j_id in enumerate(joint_id[person_id]):
                joint_pt = coords[person_id][k, :]
                j_x = np.asarray(joint_pt[0]).item()
                j_x_sm = round((j_x - self.half_stride) / self.stride)
                j_y = np.asarray(joint_pt[1]).item()
                j_y_sm = round((j_y - self.half_stride) / self.stride)
                map_j = grid.copy()
                # Distance between the joint point and each coordinate
                dist = np.linalg.norm(grid - (j_y, j_x), axis=2) ** 2
                scmap_j = np.exp(-dist / (2 * (std**2)))
                scmap[..., j_id] = scmap_j
                locref_mask[dist <= dist_thresh_sq, j_id * 2 + 0] = 1
                locref_mask[dist <= dist_thresh_sq, j_id * 2 + 1] = 1
                dx = j_x - grid.copy()[:, :, 1]
                dy = j_y - grid.copy()[:, :, 0]
                locref_map[..., j_id * 2 + 0] = dx * self.locref_scale
                locref_map[..., j_id * 2 + 1] = dy * self.locref_scale
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

    def compute_target_part_scoremap_numpy(
        self, joint_id, coords, data_item, size, scale
    ):
        dist_thresh = float(self.cfg["pos_dist_thresh"] * scale)
        dist_thresh_sq = dist_thresh**2
        num_joints = self.cfg["num_joints"]

        scmap = np.zeros(np.concatenate([size, np.array([num_joints])]))
        locref_size = np.concatenate([size, np.array([num_joints * 2])])
        locref_mask = np.zeros(locref_size)
        locref_map = np.zeros(locref_size)

        width = size[1]
        height = size[0]
        grid = np.mgrid[:height, :width].transpose((1, 2, 0))

        for person_id in range(len(coords)):
            for k, j_id in enumerate(joint_id[person_id]):
                joint_pt = coords[person_id][k, :]
                j_x = np.asarray(joint_pt[0]).item()
                j_x_sm = round((j_x - self.half_stride) / self.stride)
                j_y = np.asarray(joint_pt[1]).item()
                j_y_sm = round((j_y - self.half_stride) / self.stride)
                min_x = round(max(j_x_sm - dist_thresh - 1, 0))
                max_x = round(min(j_x_sm + dist_thresh + 1, width - 1))
                min_y = round(max(j_y_sm - dist_thresh - 1, 0))
                max_y = round(min(j_y_sm + dist_thresh + 1, height - 1))
                x = grid.copy()[:, :, 1]
                y = grid.copy()[:, :, 0]
                dx = j_x - x * self.stride - self.half_stride
                dy = j_y - y * self.stride - self.half_stride
                dist = dx**2 + dy**2
                mask1 = dist <= dist_thresh_sq
                mask2 = (x >= min_x) & (x <= max_x)
                mask3 = (y >= min_y) & (y <= max_y)
                mask = mask1 & mask2 & mask3
                scmap[mask, j_id] = 1
                locref_mask[mask, j_id * 2 + 0] = 1
                locref_mask[mask, j_id * 2 + 1] = 1
                locref_map[mask, j_id * 2 + 0] = (dx * self.locref_scale)[mask]
                locref_map[mask, j_id * 2 + 1] = (dy * self.locref_scale)[mask]

        weights = self.compute_scmap_weights(scmap.shape, joint_id, data_item)
        return scmap, weights, locref_map, locref_mask
