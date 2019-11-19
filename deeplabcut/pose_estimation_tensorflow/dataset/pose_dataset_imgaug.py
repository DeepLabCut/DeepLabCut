"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Loader structure adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow

Uses imgaug dataflow for flexible augmentation
Largely written by Mert Yüksekgönül during the summer in the Bethge lab -- Thanks!
https://imgaug.readthedocs.io/en/latest/
"""

import os
import logging
import random as rand
import numpy as np
from numpy import array as arr
from numpy import concatenate as cat

import scipy.io as sio
from deeplabcut.utils.auxfun_videos import imread
from scipy.stats import truncnorm
from imgaug.augmentables import Keypoint, KeypointsOnImage
import imgaug.augmenters as iaa
import pickle

from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import Batch, data_to_input, DataItem, data_to_input_batch

class PoseDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data = self.load_dataset()
        self.batch_size = cfg.get('batch_size',1)
        self.num_images = len(self.data)
        self.max_input_sizesquare=cfg.get('max_input_size', 1500)**2
        self.min_input_sizesquare=cfg.get('min_input_size', 64)**2
        self.locref_scale = 1.0 / cfg.locref_stdev
        self.stride = cfg.stride
        self.half_stride = cfg.stride / 2
        self.scale = cfg.global_scale
        self.scale_jitter_lo=cfg.get('scale_jitter_lo',.75)
        self.scale_jitter_up=cfg.get('scale_jitter_up',1.25)
        print("Batch Size is %d" % self.batch_size)

    def load_dataset(self):
        cfg = self.cfg
        file_name = os.path.join(self.cfg.project_path,cfg.dataset)
        if '.mat' in file_name: #legacy loader
            mlab = sio.loadmat(file_name)
            self.raw_data = mlab
            mlab = mlab['dataset']

            num_images = mlab.shape[1]
            data = []
            has_gt = True

            for i in range(num_images):
                sample = mlab[0, i]

                item = DataItem()
                item.image_id = i
                item.im_path = sample[0][0]
                item.im_size = sample[1][0]
                if len(sample) >= 3:
                    joints = sample[2][0][0]
                    joint_id = joints[:, 0]
                    # make sure joint ids are 0-indexed
                    if joint_id.size != 0:
                        assert((joint_id < cfg.num_joints).any())
                    joints[:, 0] = joint_id
                    item.joints = [joints]
                else:
                    has_gt = False
                data.append(item)

            self.has_gt = has_gt
            return data
        else:
            print("Loading pickle data with float coordinates!")
            file_name = cfg.dataset.split(".")[0] + ".pickle"
            with open(os.path.join(self.cfg.project_path,file_name), 'rb') as f:
                pickledata=pickle.load(f)

            self.raw_data = pickledata
            num_images = len(pickledata) #mlab.shape[1]
            data = []
            has_gt = True
            for i in range(num_images):
                sample = pickledata[i] #mlab[0, i]
                item = DataItem()
                item.image_id = i
                item.im_path = sample['image'] #[0][0]
                item.im_size = sample['size'] #sample[1][0]
                if len(sample) >= 3:
                    item.num_animals=len(sample['joints'])
                    item.joints=[sample['joints']]

                else:
                    has_gt = False
                data.append(item)
            self.has_gt = has_gt
            return data

    def build_augmentation_pipeline(self,
                                    height=None,
                                    width=None,
                                    apply_prob=0.5
                                   ):
        sometimes = lambda aug: iaa.Sometimes(apply_prob, aug)
        pipeline = iaa.Sequential(random_order=False)
        cfg = self.cfg
        if cfg.get('fliplr', False):
            opt = cfg.get('fliplr',False)
            if type(opt) == int:
                pipeline.add(sometimes(iaa.Fliplr(opt)))
            else:
                pipeline.add(sometimes(iaa.Fliplr(0.5)))
        if cfg.get('rotation', False):
            opt = cfg.get('rotation',False)
            if type(opt) == int:
                pipeline.add(sometimes(iaa.Affine(rotate=(-opt,opt))))
            else:
                pipeline.add(sometimes(iaa.Affine(rotate=(-10,10))))
        if cfg.get('motion_blur', False):
            opts = cfg.get('motion_blur', False)
            if type(opts) == list:
                opts = dict(opts)
                pipeline.add(sometimes(iaa.MotionBlur(**opts)))
            else:
                pipeline.add(sometimes(iaa.MotionBlur(k=7, angle=(-90, 90))))
        if cfg.get('covering', False):
            pipeline.add(sometimes(iaa.CoarseDropout(0.02, size_percent=0.3, per_channel=0.5)))
        if cfg.get('elastic_transform', False):
            pipeline.add(sometimes(iaa.ElasticTransformation(sigma=5)))
        if cfg.get('gaussian_noise', False):
            opt = cfg.get('gaussian_noise', False)
            if type(opt) == int or type(opt)==float:
                pipeline.add(sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, opt), per_channel=0.5)))
            else:
                pipeline.add(sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)))
        if cfg.get('grayscale', False):
            pipeline.add(sometimes(iaa.Grayscale(alpha=(0.5, 1.0))))

        if cfg.get('hist_eq', False):
            pipeline.add(sometimes(iaa.AllChannelsHistogramEqualization()))
        if height is not None and width is not None:
            if not cfg.get('crop_by', False):
                crop_by = 0.15
            else:
                crop_by = cfg.get('crop_by', False)
            pipeline.add(iaa.Sometimes(cfg.cropratio,iaa.CropAndPad(percent=(-crop_by, crop_by),keep_size=False)))
            pipeline.add(iaa.Resize({"height": height, "width": width}))
        return pipeline

    def get_batch(self):
        img_idx = np.random.choice(self.num_images,
                                   size=self.batch_size,replace=True)
        batch_images = []
        batch_joints = []
        joint_ids = []
        data_items = []
        # Scale is sampled only once to transform all of the images into same size.
        scale = self.get_scale()
        while True:
            idx = np.random.choice(self.num_images)
            scale = self.get_scale()
            size = self.data[idx].im_size
            target_size = np.ceil(size[1:3]*scale).astype(int)
            if self.is_valid_size(target_size[1] * target_size[0]):
                break

        stride = self.cfg.stride
        for i in range(self.batch_size):
            data_item = self.data[img_idx[i]]

            data_items.append(data_item)
            im_file = data_item.im_path

            logging.debug('image %s', im_file)
            image = imread(os.path.join(self.cfg.project_path,im_file), mode='RGB')

            if self.has_gt:
                joints = np.copy(data_item.joints)
                joint_id = [person_joints[:, 0].astype(int) for person_joints in joints]
                joint_points = [person_joints[:, 1:3] for person_joints in joints]
                joint_ids.append(joint_id)
                batch_joints.append(arr(joint_points)[0])
            batch_images.append(image)
        sm_size = np.ceil(target_size / (stride * 2)).astype(int) * 2
        assert len(batch_images) == self.batch_size
        return batch_images, joint_ids, batch_joints, data_items, sm_size, target_size

    def get_scmap_update(self, joint_ids, joints, data_items, sm_size, target_size):
        part_score_targets, part_score_weights, locref_targets, locref_masks = [],[],[],[]
        for i in range(len(data_items)):
            # Approximating the scale
            scale = min(target_size[0]/data_items[i].im_size[1], target_size[1]/data_items[i].im_size[2])
            if self.cfg.get("scmap_type", None) == "gaussian":
                part_score_target, part_score_weight, locref_target, locref_mask = self.gaussian_scmap(
                    joint_ids[i], [joints[i]], data_items[i], sm_size, scale)
            else:
                part_score_target, part_score_weight, locref_target, locref_mask = self.compute_target_part_scoremap_numpy(
                    joint_ids[i], [joints[i]], data_items[i], sm_size, scale)
            part_score_targets.append(part_score_target)
            part_score_weights.append(part_score_weight)
            locref_targets.append(locref_target)
            locref_masks.append(locref_mask)

        return {
                    Batch.part_score_targets: part_score_targets,
                    Batch.part_score_weights: part_score_weights,
                    Batch.locref_targets: locref_targets,
                    Batch.locref_mask: locref_masks
                }

    def next_batch(self):
        while True:
            batch_images, joint_ids, batch_joints, data_items, sm_size, target_size = self.get_batch()
            pipeline = self.build_augmentation_pipeline(
                                    height=target_size[0],
                                    width=target_size[1],
                                    apply_prob=0.5)
            batch_images, batch_joints = pipeline(images=batch_images,
                                                  keypoints=batch_joints)
            #If you would like to check the augmented images, script for saving
            #the images with joints on:
            #import imageio
            #for i in range(self.batch_size):
            #    joints = batch_joints[i]
            #    kps = KeypointsOnImage([Keypoint(x=joint[0], y=joint[1]) for joint in joints], shape=batch_images[i].shape)
            #    im = kps.draw_on_image(batch_images[i])
            #    imageio.imwrite('some_location/augmented/'+str(i)+'.png', im)

            image_shape = arr(batch_images).shape[1:3]
            batch = {Batch.inputs: arr(batch_images).astype(np.float64)}
            if self.has_gt:
                scmap_update = self.get_scmap_update(joint_ids, batch_joints, data_items, sm_size, image_shape)
                batch.update(scmap_update)

            batch = {key: data_to_input_batch(data) for (key, data) in batch.items()}
            batch[Batch.data_item] = data_items
            return batch

    def set_test_mode(self, test_mode):
        self.has_gt = not test_mode

    def num_training_samples(self):
        num = self.num_images
        if self.cfg.mirror:
            num *= 2
        return num

    def get_scale(self):
        cfg = self.cfg
        scale = cfg.global_scale
        if hasattr(cfg, 'scale_jitter_lo') and hasattr(cfg, 'scale_jitter_up'):
            scale_jitter = rand.uniform(0.75*cfg.scale_jitter_lo, 1.25*cfg.scale_jitter_up)
            scale *= scale_jitter
        return scale

    def is_valid_size(self, target_size_product):
        if target_size_product > self.max_input_sizesquare:
            return False

        if target_size_product  < self.min_input_sizesquare:
            return False

        return True

    def gaussian_scmap(self, joint_id, coords, data_item, size, scale):
        #dist_thresh = float(self.cfg.pos_dist_thresh * scale)
        num_joints = self.cfg.num_joints
        scmap = np.zeros(cat([size, arr([num_joints])]))
        locref_size = cat([size, arr([num_joints * 2])])
        locref_mask = np.zeros(locref_size)
        locref_map = np.zeros(locref_size)

        width = size[1]
        height = size[0]
        dist_thresh = float((width+height)/6)
        dist_thresh_sq = dist_thresh ** 2

        std = dist_thresh/4
        # Grid of coordinates
        grid = np.mgrid[:height, :width].transpose((1,2,0))
        grid = grid*self.stride + self.half_stride
        for person_id in range(len(coords)):
            for k, j_id in enumerate(joint_id[person_id]):
                joint_pt = coords[person_id][k, :]
                j_x = np.asscalar(joint_pt[0])
                j_x_sm = round((j_x - self.half_stride) / self.stride)
                j_y = np.asscalar(joint_pt[1])
                j_y_sm = round((j_y - self.half_stride) / self.stride)
                map_j = grid.copy()
                # Distance between the joint point and each coordinate
                dist = np.linalg.norm(grid - (j_y, j_x), axis=2)**2
                scmap_j = np.exp(-dist/(2*(std**2)))
                scmap[..., j_id] = scmap_j
                locref_mask[dist<=dist_thresh_sq,j_id * 2 + 0]=1
                locref_mask[dist<=dist_thresh_sq,j_id * 2 + 1]=1
                dx = j_x - grid.copy()[:, :, 1 ]
                dy = j_y - grid.copy()[:, :, 0 ]
                locref_map[..., j_id * 2 + 0] = dx * self.locref_scale
                locref_map[..., j_id * 2 + 1] = dy * self.locref_scale
        weights = self.compute_scmap_weights(scmap.shape, joint_id, data_item)
        return scmap, weights, locref_map, locref_mask

    def compute_scmap_weights(self, scmap_shape, joint_id, data_item):
        if self.cfg.weigh_only_present_joints:
            weights = np.zeros(scmap_shape)
            for person_joint_id in joint_id:
                for j_id in person_joint_id:
                    weights[:, :, j_id] = 1.0
        else:
            weights = np.ones(scmap_shape)
        return weights

    def compute_target_part_scoremap_numpy(self, joint_id, coords, data_item, size, scale):
        dist_thresh = float(self.cfg.pos_dist_thresh * scale)
        dist_thresh_sq = dist_thresh ** 2
        num_joints = self.cfg.num_joints

        scmap = np.zeros(cat([size, arr([num_joints])]))
        locref_size = cat([size, arr([num_joints * 2])])
        locref_mask = np.zeros(locref_size)
        locref_map = np.zeros(locref_size)

        width = size[1]
        height = size[0]
        grid = np.mgrid[:height, :width].transpose((1,2,0))

        for person_id in range(len(coords)):
            for k, j_id in enumerate(joint_id[person_id]):
                joint_pt = coords[person_id][k, :]
                j_x = np.asscalar(joint_pt[0])
                j_x_sm = round((j_x - self.half_stride) / self.stride)
                j_y = np.asscalar(joint_pt[1])
                j_y_sm = round((j_y - self.half_stride) / self.stride)
                min_x = round(max(j_x_sm - dist_thresh - 1, 0))
                max_x = round(min(j_x_sm + dist_thresh + 1, width - 1))
                min_y = round(max(j_y_sm - dist_thresh - 1, 0))
                max_y = round(min(j_y_sm + dist_thresh + 1, height - 1))
                x = grid.copy()[:, :, 1]
                y = grid.copy()[:, :, 0]
                dx = j_x - x*self.stride - self.half_stride
                dy = j_y - y*self.stride - self.half_stride
                dist = dx**2 + dy**2
                mask1 = (dist <= dist_thresh_sq)
                mask2 = ((x >= min_x) & (x <= max_x))
                mask3 = ((y >= min_y) & (y <= max_y))
                mask = mask1 & mask2 & mask3
                scmap[mask, j_id] = 1
                locref_mask[mask, j_id*2+0] = 1
                locref_mask[mask, j_id*2+1] = 1
                locref_map[mask, j_id * 2 + 0] = (dx * self.locref_scale)[mask]
                locref_map[mask, j_id * 2 + 1] = (dy * self.locref_scale)[mask]

        weights = self.compute_scmap_weights(scmap.shape, joint_id, data_item)
        return scmap, weights, locref_map, locref_mask
