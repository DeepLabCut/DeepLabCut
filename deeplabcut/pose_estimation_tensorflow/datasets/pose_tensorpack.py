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

See pull request:
https://github.com/DeepLabCut/DeepLabCut/pull/409
use tensorpack dataflow to improve augmentation #409 and #426
Written largely by Kate Rupp -- Thanks!

A Neural Net Training Interface on TensorFlow, with focus on speed + flexibility
https://github.com/tensorpack/tensorpack
"""


import multiprocessing
import os

import cv2
import numpy as np
import scipy.io as sio
from deeplabcut.utils.conversioncode import robust_split_path
from numpy import array as arr
from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.dataflow.common import MapData
from tensorpack.dataflow.imgaug import (
    Brightness,
    Contrast,
    RandomResize,
    Rotation,
    Saturation,
    GaussianNoise,
    GaussianBlur,
)
from tensorpack.dataflow.imgaug.crop import RandomCropRandomShape
from tensorpack.dataflow.imgaug.meta import RandomApplyAug
from tensorpack.dataflow.imgaug.transform import CropTransform
from tensorpack.dataflow.parallel import MultiProcessRunnerZMQ, MultiProcessRunner
from tensorpack.utils.utils import get_rng

from .factory import PoseDatasetFactory
from .pose_base import BasePoseDataset
from .utils import Batch, data_to_input


def img_to_bgr(im_path):
    img = cv2.imread(im_path)
    return img


class DataItem:
    def to_dict(self):
        return self.__dict__

    def from_dict(d):
        item = DataItem()
        for k, v in d.items():
            setattr(item, k, v)
        return item


class RandomCropping(RandomCropRandomShape):
    def __init__(self, wmin, hmin, wmax=None, hmax=None):
        self.rng = get_rng()
        super().__init__(wmin, hmin, wmax, hmax)

    def get_transform(self, img):
        hmax = self.hmax or img.shape[0]
        wmax = self.wmax or img.shape[1]
        hmin = min(self.hmin, img.shape[0])
        wmin = min(self.wmin, img.shape[1])
        hmax = min(hmax, img.shape[0])
        wmax = min(wmax, img.shape[1])
        h = self.rng.randint(hmin, hmax + 1)
        w = self.rng.randint(wmin, wmax + 1)
        diffh = img.shape[0] - h
        diffw = img.shape[1] - w
        assert diffh >= 0 and diffw >= 0
        y0 = 0 if diffh == 0 else self.rng.randint(diffh)
        x0 = 0 if diffw == 0 else self.rng.randint(diffw)
        crop_aug = CropTransform(y0, x0, h, w)

        return crop_aug


class Pose(RNGDataFlow):
    def __init__(self, cfg, shuffle=True, dir=None):
        self.shuffle = shuffle
        self.cfg = cfg
        self.data = self.load_dataset()
        self.has_gt = True

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
            base = str(self.cfg["project_path"])
            im_path = sample[0][0]
            if isinstance(im_path, str):
                im_path = robust_split_path(im_path)
            else:
                im_path = [s.strip() for s in im_path]
            item.im_path = os.path.join(base, *im_path)
            item.im_size = sample[1][0]
            if len(sample) >= 3:
                joints = sample[2][0][0]
                joint_id = joints[:, 0]
                # make sure joint ids are 0-indexed
                if joint_id.size != 0:
                    assert (joint_id < cfg["num_joints"]).any()
                joints[:, 0] = joint_id
                coords = [joint[1:] for joint in joints]
                coords = arr(coords)
                item.coords = coords
                item.joints = [joints]
                item.joint_id = [arr(joint_id)]
                # print(item.joints)
            else:
                has_gt = False
            # if cfg.crop:
            #    crop = sample[3][0] - 1
            #    item.crop = extend_crop(crop, cfg.crop_pad, item.im_size)
            data.append(item)

        self.has_gt = has_gt
        return data

    def __iter__(self):
        idxs = list(range(len(self.data)))
        while True:
            if self.shuffle:
                self.rng.shuffle(idxs)
            for k in idxs:
                data_item = self.data[k]
                yield data_item


@PoseDatasetFactory.register("tensorpack")
class TensorpackPoseDataset(BasePoseDataset):
    def __init__(self, cfg):
        # First, initializing variables (if they don't exist)
        # what is the fraction of training samples with scaling augmentation?
        cfg["scaleratio"] = cfg.get("scaleratio", 0.6)

        # loading defaults for rotation range!
        # Randomly rotates an image with respect to the image center within the
        # range [-rotate_max_deg_abs; rotate_max_deg_abs] to augment training data

        if cfg.get("rotation", True):  # i.e. pm 25 degrees
            if type(cfg.get("rotation", False)) == int:
                cfg["rotation"] = cfg.get("rotation", 25)
            else:
                cfg["rotation"] = 25

            # cfg["rotateratio"] = cfg.get(
            #    "rotratio", 0.4
            # )  # what is the fraction of training samples with rotation augmentation?
        else:
            cfg["rotratio"] = 0.0
            cfg["rotation"] = 0

        # Randomly adds brightness within the range [-brightness_dif, brightness_dif]
        # to augment training data
        cfg["brightness_dif"] = cfg.get("brightness_dif", 0.3)
        cfg["brightnessratio"] = cfg.get(
            "brightnessratio", 0.0
        )  # what is the fraction of training samples with brightness augmentation?

        # Randomly applies x = (x - mean) * contrast_factor + mean`` to each
        # color channel within the range [contrast_factor_lo, contrast_factor_up]
        # to augment training data
        cfg["contrast_factor_lo"] = cfg.get("contrast_factor_lo", 0.5)
        cfg["contrast_factor_up"] = cfg.get("contrast_factor_up", 2.0)
        cfg["contrastratio"] = cfg.get(
            "contrastratio", 0.2
        )  # what is the fraction of training samples with contrast augmentation?

        # Randomly adjusts saturation within range 1 + [-saturation_max_dif, saturation_max_dif]
        # to augment training data
        cfg["saturation_max_dif"] = cfg.get("saturation_max_dif", 0.5)
        cfg["saturationratio"] = cfg.get(
            "saturationratio", 0.0
        )  # what is the fraction of training samples with saturation augmentation?

        # Randomly applies gaussian noise N(0, noise_sigma^2) to an image
        # to augment training data
        cfg["noise_sigma"] = cfg.get("noise_sigma", 0.1)
        cfg["noiseratio"] = cfg.get(
            "noiseratio", 0.0
        )  # what is the fraction of training samples with noise augmentation?

        # Randomly applies gaussian blur to an image with a random window size
        # within the range [0, 2 * blur_max_window_size + 1] to augment training data
        cfg["blur_max_window_size"] = cfg.get("blur_max_window_size", 10)
        cfg["blurratio"] = cfg.get(
            "blurratio", 0.2
        )  # what is the fraction of training samples with blur augmentation?

        # Whether image is RGB  or RBG. If None, contrast augmentation uses the mean per-channel.
        cfg["is_rgb"] = cfg.get("is_rgb", True)

        # Clips image to [0, 255] even when data type is not uint8
        cfg["to_clip"] = cfg.get("to_clip", True)

        # Number of processes to use per core during training
        cfg["processratio"] = cfg.get("processratio", 1)
        # Number of datapoints to prefetch at a time during training
        cfg["num_prefetch"] = cfg.get("num_prefetch", 50)

        # Auto cropping is new (was not in Nature Neuroscience 2018 paper, but introduced in Nath et al. Nat. Protocols 2019)
        # and boosts performance by 2X, particularly on challenging datasets, like the cheetah in Nath et al.
        # Parameters for augmentation with regard to cropping:

        # what is the minimal frames size for cropping plus/minus ie.. [-100,100]^2 for an arb. joint
        cfg["minsize"] = cfg.get("minsize", 100)
        cfg["leftwidth"] = cfg.get("leftwidth", 400)
        cfg["rightwidth"] = cfg.get("rightwidth", 400)
        cfg["topheight"] = cfg.get("topheight", 400)
        cfg["bottomheight"] = cfg.get("bottomheight", 400)

        cfg["cropratio"] = cfg.get("cropratio", 0.4)

        super(TensorpackPoseDataset, self).__init__(cfg)
        self.scaling = RandomResize(
            xrange=(
                self.cfg["scale_jitter_lo"] * self.cfg["global_scale"],
                self.cfg["scale_jitter_up"] * self.cfg["global_scale"],
            ),
            aspect_ratio_thres=0.0,
        )
        self.scaling_apply = RandomApplyAug(self.scaling, self.cfg["scaleratio"])
        self.cropping = RandomCropping(
            wmin=self.cfg["minsize"],
            hmin=self.cfg["minsize"],
            wmax=self.cfg["leftwidth"] + self.cfg["rightwidth"] + self.cfg["minsize"],
            hmax=self.cfg["topheight"] + self.cfg["bottomheight"] + self.cfg["minsize"],
        )
        self.rotation = Rotation(max_deg=self.cfg["rotation"])
        self.brightness = Brightness(self.cfg["brightness_dif"])
        self.contrast = Contrast(
            (self.cfg["contrast_factor_lo"], self.cfg["contrast_factor_up"]),
            rgb=self.cfg["is_rgb"],
            clip=self.cfg["to_clip"],
        )
        self.saturation = Saturation(
            self.cfg["saturation_max_dif"], rgb=self.cfg["is_rgb"]
        )
        self.gaussian_noise = GaussianNoise(sigma=self.cfg["noise_sigma"])
        self.gaussian_blur = GaussianBlur(max_size=self.cfg["blur_max_window_size"])
        self.augmentors = [
            RandomApplyAug(self.cropping, self.cfg["cropratio"]),
            RandomApplyAug(self.rotation, self.cfg["rotratio"]),
            RandomApplyAug(self.brightness, self.cfg["brightnessratio"]),
            RandomApplyAug(self.contrast, self.cfg["contrastratio"]),
            RandomApplyAug(self.saturation, self.cfg["saturationratio"]),
            RandomApplyAug(self.gaussian_noise, self.cfg["noiseratio"]),
            RandomApplyAug(self.gaussian_blur, self.cfg["blurratio"]),
            self.scaling_apply,
        ]

        self.has_gt = True
        self.set_shuffle(cfg["shuffle"])
        self.data = self.load_dataset()
        self.num_images = len(self.data)
        df = self.get_dataflow(self.cfg)
        df.reset_state()
        self.aug = iter(df)

    def load_dataset(self):
        p = Pose(cfg=self.cfg, shuffle=self.shuffle)
        return p.load_dataset()

    def augment(self, data):
        img = img_to_bgr(data.im_path)
        coords = data.coords.astype("float64")
        scale = 1
        for aug in self.augmentors:
            tfm = aug.get_transform(img)
            aug_img = tfm.apply_image(img)
            aug_coords = tfm.apply_coords(coords)
            if aug is self.scaling_apply:
                scale = aug_img.shape[0] / img.shape[0]
            img = aug_img
            coords = aug_coords

        aug_img = img
        aug_coords = coords
        size = [aug_img.shape[0], aug_img.shape[1]]
        aug_coords = [
            aug_coords.reshape(int(len(aug_coords[~np.isnan(aug_coords)]) / 2), 2)
        ]
        joint_id = data.joint_id

        return [joint_id, aug_img, aug_coords, data, size, scale]

    def get_dataflow(self, cfg):

        df = Pose(cfg)
        df = MapData(df, self.augment)
        df = MapData(df, self.compute_target_part_scoremap)

        num_cores = multiprocessing.cpu_count()
        num_processes = int(num_cores * self.cfg["processratio"])
        if num_processes <= 1:
            num_processes = 2  # recommended to use more than one process for training
        if os.name == "nt":
            df2 = MultiProcessRunner(
                df, num_proc=num_processes, num_prefetch=self.cfg["num_prefetch"]
            )
        else:
            df2 = MultiProcessRunnerZMQ(
                df, num_proc=num_processes, hwm=self.cfg["num_prefetch"]
            )
        return df2

    def compute_target_part_scoremap(self, components):
        joint_id = components[0]
        aug_img = components[1]
        coords = components[2]
        data_item = components[3]
        img_size = components[4]
        scale = components[5]

        stride = self.cfg["stride"]
        dist_thresh = self.cfg["pos_dist_thresh"] * scale
        num_joints = self.cfg["num_joints"]
        half_stride = stride / 2
        size = np.ceil(arr(img_size) / (stride * 2)).astype(int) * 2
        scmap = np.zeros(np.append(size, num_joints))
        locref_size = np.append(size, num_joints * 2)
        locref_mask = np.zeros(locref_size)
        locref_map = np.zeros(locref_size)

        locref_scale = 1.0 / self.cfg["locref_stdev"]
        dist_thresh_sq = dist_thresh**2

        width = size[1]
        height = size[0]

        for person_id in range(len(coords)):
            for k, j_id in enumerate(joint_id[person_id]):
                joint_pt = coords[person_id][k, :]
                j_x = np.asarray(joint_pt[0]).item()
                j_y = np.asarray(joint_pt[1]).item()

                # don't loop over entire heatmap, but just relevant locations
                j_x_sm = round((j_x - half_stride) / stride)
                j_y_sm = round((j_y - half_stride) / stride)
                min_x = round(max(j_x_sm - dist_thresh - 1, 0))
                max_x = round(min(j_x_sm + dist_thresh + 1, width - 1))
                min_y = round(max(j_y_sm - dist_thresh - 1, 0))
                max_y = round(min(j_y_sm + dist_thresh + 1, height - 1))

                for j in range(min_y, max_y + 1):  # range(height):
                    pt_y = j * stride + half_stride
                    for i in range(min_x, max_x + 1):  # range(width):
                        pt_x = i * stride + half_stride
                        dx = j_x - pt_x
                        dy = j_y - pt_y
                        dist = dx**2 + dy**2
                        # print(la.norm(diff))
                        if dist <= dist_thresh_sq:
                            scmap[j, i, j_id] = 1
                            locref_mask[j, i, j_id * 2 + 0] = 1
                            locref_mask[j, i, j_id * 2 + 1] = 1
                            locref_map[j, i, j_id * 2 + 0] = dx * locref_scale
                            locref_map[j, i, j_id * 2 + 1] = dy * locref_scale

        weights = self.compute_scmap_weights(scmap.shape, joint_id)
        mirror = False
        d = data_item.to_dict()
        d["image"] = aug_img

        return d, scale, mirror, scmap, weights, locref_map, locref_mask

    def set_test_mode(self, test_mode):
        self.has_gt = not test_mode

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle
        if not shuffle:
            assert not self.cfg["mirror"]
            self.image_indices = np.arange(self.num_images)

    def shuffle_images(self):
        num_images = self.num_images
        if self.cfg["mirror"]:
            image_indices = np.random.permutation(num_images * 2)
            self.mirrored = image_indices >= num_images
            image_indices[self.mirrored] = image_indices[self.mirrored] - num_images
            self.image_indices = image_indices
        else:
            self.image_indices = np.random.permutation(num_images)

    def next_batch(self):
        next_batch = next(self.aug)
        return self.make_batch(next_batch)

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

    def make_batch(self, components):
        data_item = DataItem.from_dict(components[0])
        mirror = components[2]
        part_score_targets = components[3]
        part_score_weights = components[4]
        locref_targets = components[5]
        locref_mask = components[6]

        im_file = data_item.im_path
        # logging.debug('image %s', im_file)
        # print('image: {}'.format(im_file))
        # logging.debug('mirror %r', mirror)

        img = data_item.image  # augmented image

        batch = {Batch.inputs: img}

        if self.has_gt:
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

    def compute_scmap_weights(self, scmap_shape, joint_id):
        cfg = self.cfg
        if cfg["weigh_only_present_joints"]:
            weights = np.zeros(scmap_shape)
            for person_joint_id in joint_id:
                for j_id in person_joint_id:
                    weights[:, :, j_id] = 1.0
        else:
            weights = np.ones(scmap_shape)
        return weights
