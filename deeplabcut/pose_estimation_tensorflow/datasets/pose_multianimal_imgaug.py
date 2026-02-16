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
import os
import pickle
import imageio
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
from imgaug.augmentables import Keypoint, KeypointsOnImage
from deeplabcut.generate_training_dataset import read_image_shape_fast
from deeplabcut.pose_estimation_tensorflow.datasets import augmentation
from deeplabcut.pose_estimation_tensorflow.datasets.factory import PoseDatasetFactory
from deeplabcut.pose_estimation_tensorflow.datasets.pose_base import BasePoseDataset
from deeplabcut.pose_estimation_tensorflow.datasets.utils import DataItem, Batch
from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal
from deeplabcut.utils.auxfun_videos import imread
from deeplabcut.utils.auxfun_videos import VideoReader
from deeplabcut.utils.conversioncode import robust_split_path
from pathlib import Path
from math import sqrt


@PoseDatasetFactory.register("multi-animal-imgaug")
class MAImgaugPoseDataset(BasePoseDataset):
    def __init__(self, cfg):
        super(MAImgaugPoseDataset, self).__init__(cfg)

        if cfg.get("pseudo_label", ""):
            self._n_kpts = len(cfg["all_joints_names"])
            self._n_animals = 1

        else:
            self.main_cfg = auxiliaryfunctions.read_config(
                os.path.join(self.cfg["project_path"], "config.yaml")
            )
            animals, unique, multi = auxfun_multianimal.extractindividualsandbodyparts(
                self.main_cfg
            )
            self._n_kpts = len(multi) + len(unique)
            self._n_animals = len(animals)

        if cfg.get("pseudo_label", "").endswith(".h5"):
            assert cfg["video_path"]
            print("loading video for image source", cfg["video_path"])
            self.vid = VideoReader(cfg["video_path"])
            self.video_image_size = (3, self.vid.height, self.vid.width)
        else:
            self.vid = None

        self.data = self.load_dataset()
        self.num_images = len(self.data)
        self.batch_size = cfg["batch_size"]
        print("Batch Size is %d" % self.batch_size)
        self._default_size = np.array(self.cfg.get("crop_size", (400, 400)))
        self.pipeline = self.build_augmentation_pipeline(
            apply_prob=cfg.get("apply_prob", 0.5),
        )

    @property
    def default_size(self):
        return self._default_size  # width, height

    @default_size.setter
    def default_size(self, size):
        self._default_size = np.array(size)

    def load_dataset(self):
        cfg = self.cfg

        if cfg.get("pseudo_label", ""):
            if cfg["pseudo_label"].endswith(".h5"):
                pseudo_threshold = cfg.get("pseudo_threshold", 0)
                print(f"Loading pseudo labels with threshold > {pseudo_threshold}")

                # for topview, it's safe to mask keypoints under threshold
                mask_kpts_below_thresh = "topview" in cfg.get("superanimal", "")
                return self._load_pseudo_data_from_h5(
                    cfg,
                    threshold=pseudo_threshold,
                    mask_kpts_below_thresh=mask_kpts_below_thresh,
                )

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
            im_path = sample["image"]
            if isinstance(im_path, str):
                im_path = robust_split_path(im_path)
            item.im_path = os.path.join(*im_path)
            item.im_size = sample["size"]
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

    def _load_pseudo_data_from_h5(
        self, cfg, threshold=0.5, mask_kpts_below_thresh=False
    ):
        gt_file = cfg["pseudo_label"]
        assert os.path.exists(gt_file)
        path_ = Path(gt_file)
        print("Using gt file:", path_.name)
        num_kpts = len(cfg["all_joints_names"])
        df = pd.read_hdf(gt_file)
        video_name = path_.name.split("DLC")[0]
        video_root = str(path_.parents[0] / video_name)

        itemlist = []
        for image_id, imagename in enumerate(df.index):
            item = DataItem()
            data = df.loc[imagename]
            # 3 for likelihood
            kpts = data.to_numpy().reshape(-1, 3)
            item.num_joints = kpts.shape[0]
            joint_ids = np.arange(item.num_joints)[..., np.newaxis]
            frame_name = "frame_" + str(int(imagename.split("frame")[1])) + ".png"
            item.im_path = os.path.join(video_root, frame_name)

            if self.vid:
                item.im_size = self.video_image_size
            else:
                item.im_size = read_image_shape_fast(
                    os.path.join(video_root, frame_name)
                )

            item.joints = {}

            if not mask_kpts_below_thresh:
                joints = np.concatenate([joint_ids, kpts], axis=1)
                joints = np.nan_to_num(joints, nan=0)
            else:
                for kpt_id, kpt in enumerate(kpts):
                    if kpt[-1] < threshold:
                        kpts[kpt_id][:-1] = -1
                    if np.isnan(kpt[0]):
                        kpts[kpt_id][:-1] = -1
                        kpts[kpt_id][-1] = 1
                joints = np.concatenate([joint_ids, kpts], axis=1)

            sparse_joints = []

            for coord in joints:
                if coord[1] != 0 and coord[3] > threshold:
                    sparse_joints.append(coord[:3])

            temp = np.array(sparse_joints)
            # we only do single animal here
            item.joints.update({0: temp})
            itemlist.append(item)

        self.has_gt = True
        return itemlist

    def build_augmentation_pipeline(self, apply_prob=0.5):
        cfg = self.cfg

        sometimes = lambda aug: iaa.Sometimes(apply_prob, aug)
        pipeline = iaa.Sequential(random_order=False)

        pre_resize = cfg.get("pre_resize")

        if cfg.get("traintime_resize", False):
            # let's hard code it
            print("using traintime resize")
            pipeline.add(iaa.Resize({"height": 400, "width": "keep-aspect-ratio"}))

        crop_sampling = cfg.get("crop_sampling", "hybrid")
        if pre_resize:
            width, height = pre_resize
            pipeline.add(iaa.Resize({"height": height, "width": width}))
            if crop_sampling == "none":
                self.default_size = width, height

        if crop_sampling != "none":
            # Add smart, keypoint-aware image cropping
            pipeline.add(iaa.PadToFixedSize(*self.default_size))
            pipeline.add(
                augmentation.KeypointAwareCropToFixedSize(
                    *self.default_size,
                    cfg.get("max_shift", 0.4),
                    crop_sampling,
                )
            )

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
        if cfg.get("grayscale", False):
            pipeline.add(sometimes(iaa.Grayscale(alpha=(0.5, 1.0))))

        def get_aug_param(cfg_value):
            if isinstance(cfg_value, dict):
                opt = cfg_value
            else:
                opt = {}
            return opt

        # TODO @deruyter92: This pattern should be refactored throughout the codebase
        # it is reading a config value that is supposed to be missing / None.
        cfg_cnt = cfg.get("contrast") or {}
        cfg_cnv = cfg.get("convolution") or {}

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

        return pipeline

    def get_batch_from_video(self):
        num_images = len(self.vid)
        batch_images = []
        batch_joints = []
        joint_ids = []
        data_items = []
        trim_ends = self.cfg.get("trim_ends", None)
        if trim_ends is None:
            trim_ends = 0
        # because of the existence of threshold, sampling population is adjusted to len(self.data)
        img_idx = np.random.choice(
            len(self.data) - trim_ends * 2, size=self.batch_size, replace=True
        )
        for i in range(self.batch_size):
            index = img_idx[i]
            offset = trim_ends
            data_item = self.data[index + offset]
            data_items.append(data_item)
            im_file = data_item.im_path

            logging.debug("image %s", im_file)
            self.vid.set_to_frame(index + offset)
            image = self.vid.read_frame()
            if self.has_gt:
                joints = data_item.joints
                if len(joints[0]) == 0:
                    # empty prediction for this frame
                    return None, None, None, None

                kpts = np.full((self._n_kpts * self._n_animals, 2), np.nan)
                for j in range(self._n_animals):
                    for n, x, y in joints.get(j, []):
                        kpts[j * self._n_kpts + int(n)] = x, y

                joint_id = [
                    np.array(list(range(self._n_kpts))) for _ in range(self._n_animals)
                ]
                joint_ids.append(joint_id)
                batch_joints.append(kpts)

            batch_images.append(image)

        return batch_images, joint_ids, batch_joints, data_items

    def get_batch(self):
        img_idx = np.random.choice(self.num_images, size=self.batch_size, replace=True)
        batch_images = []
        batch_joints = []
        joint_ids = []
        data_items = []
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
                kpts = np.full((self._n_kpts * self._n_animals, 2), np.nan)
                for j in range(self._n_animals):
                    for n, x, y in joints.get(j, []):
                        kpts[j * self._n_kpts + int(n)] = x, y
                joint_id = [
                    np.array(list(range(self._n_kpts))) for _ in range(self._n_animals)
                ]
                joint_ids.append(joint_id)
                batch_joints.append(kpts)

            batch_images.append(image)

        return batch_images, joint_ids, batch_joints, data_items

    def get_targetmaps_update(
        self,
        joint_ids,
        joints,
        data_items,
        sm_size,
        scale,
    ):
        part_score_targets = []
        part_score_weights = []
        locref_targets = []
        locref_masks = []
        partaffinityfield_targets = []
        partaffinityfield_masks = []
        for i in range(len(data_items)):
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
                    joint_ids[i], joints[i], data_items[i], sm_size, scale
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

    def calc_target_and_scoremap_sizes(self):
        target_size = self.default_size * self.sample_scale()
        target_size = np.ceil(target_size).astype(int)
        if not self.is_valid_size(target_size):
            target_size = self.default_size
        stride = self.cfg["stride"]
        sm_size = np.ceil(target_size / (stride * self.cfg.get("smfactor", 2))).astype(
            int
        ) * self.cfg.get("smfactor", 2)
        if stride == 2:
            sm_size = np.ceil(target_size / 16).astype(int)
            sm_size *= 8
        return target_size, sm_size

    def next_batch(self, plotting=False):
        while True:
            if self.vid:
                (
                    batch_images,
                    joint_ids,
                    batch_joints,
                    data_items,
                ) = self.get_batch_from_video()
            else:
                batch_images, joint_ids, batch_joints, data_items = self.get_batch()

            # in case it's empty prediction
            if batch_joints is None or batch_images is None:
                continue

            # Scale is sampled only once (per batch) to transform all of the images into same size.
            target_size, sm_size = self.calc_target_and_scoremap_sizes()
            scale = np.mean(target_size / self.default_size)
            augmentation.update_crop_size(self.pipeline, *target_size)
            batch_images, batch_joints = self.pipeline(
                images=batch_images, keypoints=batch_joints
            )
            batch_images = np.asarray(batch_images)
            image_shape = batch_images.shape[1:3]
            # Discard keypoints whose coordinates lie outside the cropped image
            batch_joints_valid = []
            joint_ids_valid = []

            for joints, ids in zip(batch_joints, joint_ids):
                # Invisible joints are represented by nans
                visible = ~np.isnan(joints[:, 0])
                inside = np.logical_and.reduce(
                    (
                        joints[:, 0] < image_shape[1],
                        joints[:, 0] > 0,
                        joints[:, 1] < image_shape[0],
                        joints[:, 1] > 0,
                    )
                )
                mask = visible & inside
                batch_joints_valid.append(joints[mask])

                temp = []
                start = 0
                for array in ids:
                    end = start + array.size
                    inds = np.arange(start, end)
                    temp.append(array[mask[inds]])
                    start = end
                joint_ids_valid.append(temp)

            # If you would like to check the augmented images, script for saving
            # the images with joints on:
            if plotting:
                for i in range(self.batch_size):
                    joints = batch_joints_valid[i]
                    kps = KeypointsOnImage(
                        [Keypoint(x=joint[0], y=joint[1]) for joint in joints],
                        shape=batch_images[i].shape,
                    )
                    im = kps.draw_on_image(batch_images[i])
                    imageio.imwrite(
                        os.path.join(self.cfg["project_path"], str(i) + ".png"), im
                    )

            batch = {Batch.inputs: batch_images.astype(np.float64)}
            if self.has_gt:
                targetmaps = self.get_targetmaps_update(
                    joint_ids_valid,
                    batch_joints_valid,
                    data_items,
                    (sm_size[1], sm_size[0]),
                    scale,
                )
                batch.update(targetmaps)

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

    def is_valid_size(self, target_size):
        im_width, im_height = target_size
        min_input_size = self.cfg.get("min_input_size", 100)
        if im_height < min_input_size or im_width < min_input_size:
            return False
        if "max_input_size" in self.cfg:
            max_input_size = self.cfg["max_input_size"]
            if im_width * im_height > max_input_size * max_input_size:
                return False
        return True

    def compute_scmap_weights(self, scmap_shape, joint_id):
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
        half_stride = stride // 2
        dist_thresh = float(self.cfg["pos_dist_thresh"] * scale)
        num_idchannel = self.cfg.get("num_idchannel", 0)

        num_joints = self.cfg["num_joints"]

        scmap = np.zeros((*size, num_joints + num_idchannel))
        locref_size = *size, num_joints * 2
        locref_map = np.zeros(locref_size)
        locref_scale = 1.0 / self.cfg["locref_stdev"]
        dist_thresh_sq = dist_thresh**2

        partaffinityfield_shape = *size, self.cfg["num_limbs"] * 2
        partaffinityfield_map = np.zeros(partaffinityfield_shape)
        if self.cfg["weigh_only_present_joints"]:
            partaffinityfield_mask = np.zeros(partaffinityfield_shape)
            locref_mask = np.zeros(locref_size)
        else:
            partaffinityfield_mask = np.ones(partaffinityfield_shape)
            locref_mask = np.ones(locref_size)

        height, width = size
        grid = np.mgrid[:height, :width].transpose((1, 2, 0))
        xx = np.expand_dims(grid[..., 1], axis=2)
        yy = np.expand_dims(grid[..., 0], axis=2)

        # Produce score maps and location refinement fields
        coords_sm = np.round((coords - half_stride) / stride).astype(int)
        mins = np.round(np.maximum(coords_sm - dist_thresh - 1, 0)).astype(int)
        maxs = np.round(
            np.minimum(coords_sm + dist_thresh + 1, [width - 1, height - 1])
        ).astype(int)
        dx = coords[:, 0] - xx * stride - half_stride
        dx_ = dx * locref_scale
        dy = coords[:, 1] - yy * stride - half_stride
        dy_ = dy * locref_scale
        dist = dx**2 + dy**2
        mask1 = dist <= dist_thresh_sq
        mask2 = (xx >= mins[:, 0]) & (xx <= maxs[:, 0])
        mask3 = (yy >= mins[:, 1]) & (yy <= maxs[:, 1])
        mask = mask1 & mask2 & mask3
        for n, ind in enumerate(np.concatenate(joint_id).tolist()):
            mask_ = mask[..., n]
            scmap[mask_, ind] = 1
            if self.cfg["weigh_only_present_joints"]:
                locref_mask[mask_, [ind * 2 + 0, ind * 2 + 1]] = 1.0
            locref_map[mask_, ind * 2 + 0] = dx_[mask_, n]
            locref_map[mask_, ind * 2 + 1] = dy_[mask_, n]

        if num_idchannel > 0:
            coordinateoffset = 0
            # Find indices of individuals in joint_id
            idx = [
                (i, id_)
                for i, id_ in enumerate(data_item.joints)
                if id_ < num_idchannel
            ]
            for i, person_id in idx:
                joint_ids = joint_id[i]
                n_joints = joint_ids.size
                if n_joints:
                    inds = np.arange(n_joints) + coordinateoffset
                    mask_ = mask[..., inds].any(axis=2)
                    scmap[mask_, person_id + num_joints] = 1
                coordinateoffset += n_joints

        coordinateoffset = 0  # the offset based on
        y, x = np.rollaxis(grid * stride + half_stride, 2)
        if self.cfg["partaffinityfield_graph"]:
            for person_id in range(len(joint_id)):
                joint_ids = joint_id[person_id].tolist()
                if len(joint_ids) >= 2:  # there is a possible edge
                    for l, (bp1, bp2) in enumerate(self.cfg["partaffinityfield_graph"]):
                        try:
                            ind1 = joint_ids.index(bp1)
                        except ValueError:
                            continue
                        try:
                            ind2 = joint_ids.index(bp2)
                        except ValueError:
                            continue
                        j_x, j_y = coords[ind1 + coordinateoffset]
                        linkedj_x, linkedj_y = coords[ind2 + coordinateoffset]
                        dist = sqrt((linkedj_x - j_x) ** 2 + (linkedj_y - j_y) ** 2)
                        if dist > 0:
                            Dx = (linkedj_x - j_x) / dist  # x-axis UNIT VECTOR
                            Dy = (linkedj_y - j_y) / dist
                            d1 = [
                                Dx * j_x + Dy * j_y,
                                Dx * linkedj_x + Dy * linkedj_y,
                            ]  # in-line with direct axis
                            d1lowerboundary = min(d1)
                            d1upperboundary = max(d1)
                            d2mid = j_y * Dx - j_x * Dy  # orthogonal direction

                            distance_along = Dx * x + Dy * y
                            distance_across = (
                                ((y * Dx - x * Dy) - d2mid)
                                * 1.0
                                / self.cfg["pafwidth"]
                                * scale
                            )

                            mask1 = (distance_along >= d1lowerboundary) & (
                                distance_along <= d1upperboundary
                            )
                            distance_across_abs = np.abs(distance_across)
                            mask2 = distance_across_abs <= 1
                            mask = mask1 & mask2
                            temp = 1 - distance_across_abs[mask]
                            if self.cfg["weigh_only_present_joints"]:
                                partaffinityfield_mask[mask, [l * 2 + 0, l * 2 + 1]] = (
                                    1.0
                                )
                            partaffinityfield_map[mask, l * 2 + 0] = Dx * temp
                            partaffinityfield_map[mask, l * 2 + 1] = Dy * temp

                coordinateoffset += len(joint_ids)  # keeping track of the blocks

        weights = self.compute_scmap_weights(scmap.shape, joint_id)
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
        scmap = np.zeros(np.concatenate([size, np.array([num_joints])]))
        locref_size = np.concatenate([size, np.array([num_joints * 2])])
        locref_mask = np.zeros(locref_size)
        locref_map = np.zeros(locref_size)

        locref_scale = 1.0 / self.cfg["locref_stdev"]
        dist_thresh_sq = dist_thresh**2

        partaffinityfield_shape = np.concatenate(
            [size, np.array([self.cfg["num_limbs"] * 2])]
        )
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
            j_x = joint_pt[0].item()
            j_x_sm = round((j_x - half_stride) / stride)
            j_y = joint_pt[1].item()
            j_y_sm = round((j_y - half_stride) / stride)

            map_j = grid.copy()
            # Distance between the joint point and each coordinate
            dist = np.linalg.norm(grid - (j_y, j_x), axis=2) ** 2
            scmap_j = np.exp(-dist / (2 * (std**2)))
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
                        indbp1 = I1[0].item()
                        indbp2 = I2[0].item()
                        j_x = (coords[0][indbp1 + coordinateoffset, 0]).item()
                        j_y = (coords[0][indbp1 + coordinateoffset, 1]).item()

                        linkedj_x = (coords[0][indbp2 + coordinateoffset, 0]).item()
                        linkedj_y = (coords[0][indbp2 + coordinateoffset, 1]).item()

                        dist = np.sqrt((linkedj_x - j_x) ** 2 + (linkedj_y - j_y) ** 2)
                        if dist > 0:
                            Dx = (linkedj_x - j_x) * 1.0 / dist  # x-axis UNIT VECTOR
                            Dy = (linkedj_y - j_y) * 1.0 / dist

                            d1 = [
                                Dx * j_x + Dy * j_y,
                                Dx * linkedj_x + Dy * linkedj_y,
                            ]  # in-line with direct axis
                            d1lowerboundary = min(d1)
                            d1upperboundary = max(d1)
                            d2mid = j_y * Dx - j_x * Dy  # orthogonal direction

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

        weights = self.compute_scmap_weights(scmap.shape, joint_id)
        return scmap, weights, locref_map, locref_mask
