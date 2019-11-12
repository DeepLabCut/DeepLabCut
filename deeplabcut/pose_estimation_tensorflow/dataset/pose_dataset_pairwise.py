import logging, os, pickle
import random as rand
from enum import Enum

import numpy as np
from numpy import array as arr
from numpy import concatenate as cat

import scipy.io as sio
from scipy.misc import imread, imresize

#from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import Batch, data_to_input, mirror_joints_map, CropImage, DataItem
#from dataset.pose_dataset import Batch, data_to_input, mirror_joints_map, CropImage, DataItem
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import Batch, data_to_input, mirror_joints_map, CropImage, DataItem

def collect_pairwise_stats(joint_id, coords):
    pairwise_stats = {}
    for person_id in range(len(coords)):
        num_joints = len(joint_id[person_id])
        for k_start in range(num_joints):
            j_id_start = joint_id[person_id][k_start]
            joint_pt = coords[person_id][k_start, :]
            j_x_start = np.asscalar(joint_pt[0])
            j_y_start = np.asscalar(joint_pt[1])
            for k_end in range(num_joints):
                if k_start != k_end:
                    j_id_end = joint_id[person_id][k_end]
                    joint_pt = coords[person_id][k_end, :]
                    j_x_end = np.asscalar(joint_pt[0])
                    j_y_end = np.asscalar(joint_pt[1])
                    if (j_id_start, j_id_end) not in pairwise_stats:
                        pairwise_stats[(j_id_start, j_id_end)] = []
                    pairwise_stats[(j_id_start, j_id_end)].append([j_x_end - j_x_start, j_y_end - j_y_start])
    return pairwise_stats


def load_pairwise_stats(cfg):
    mat_stats = sio.loadmat(cfg.pairwise_stats_fn)
    pairwise_stats = {}
    for id in range(len(mat_stats['graph'])):
        pair = tuple(mat_stats['graph'][id])
        pairwise_stats[pair] = {"mean": mat_stats['means'][id], "std": mat_stats['std_devs'][id]}
    for pair in pairwise_stats:
        pairwise_stats[pair]["mean"] *= cfg.global_scale
        pairwise_stats[pair]["std"] *= cfg.global_scale
    return pairwise_stats


def get_pairwise_index(j_id, j_id_end, num_joints):
    return (num_joints - 1) * j_id + j_id_end - int(j_id < j_id_end)

class PoseDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data = self.load_dataset() if cfg.dataset else []
        self.num_images = len(self.data)
        if self.cfg.mirror:
            self.symmetric_joints = mirror_joints_map(cfg.all_joints, cfg.num_joints)

        self.curr_img = 0
        self.set_shuffle(cfg.shuffle)
        self.set_pairwise_stats_collect(cfg.pairwise_stats_collect)

        if self.cfg.pairwise_predict:
            self.pairwise_stats = load_pairwise_stats(self.cfg)


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
            # Load Matlab file dataset annotation
            #mlab = sio.loadmat(file_name)
            #mlab = sio.loadmat(os.path.join(self.cfg.project_path,file_name))
            with open(os.path.join(self.cfg.project_path,file_name), 'rb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickledata=pickle.load(f)

            self.raw_data = pickledata
            #mlab = mlab['dataset']
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

    def num_keypoints(self):
        return self.cfg.num_joints

    def set_test_mode(self, test_mode):
        self.has_gt = not test_mode

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle
        if not shuffle:
            assert not self.cfg.mirror
            self.image_indices = np.arange(self.num_images)

    def set_pairwise_stats_collect(self, pairwise_stats_collect):
        self.pairwise_stats_collect = pairwise_stats_collect
        if self.pairwise_stats_collect:
            assert self.get_scale() == 1.0


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
        num_images = self.num_images
        if self.cfg.mirror:
            image_indices = np.random.permutation(num_images * 2)
            self.mirrored = image_indices >= num_images
            image_indices[self.mirrored] = image_indices[self.mirrored] - num_images
            self.image_indices = image_indices
        else:
            self.image_indices = np.random.permutation(num_images)


    def num_training_samples(self):
        num = self.num_images
        if self.cfg.mirror:
            num *= 2
        return num


    def next_training_sample(self):
        if self.curr_img == 0 and self.shuffle:
            self.shuffle_images()

        curr_img = self.curr_img
        self.curr_img = (self.curr_img + 1) % self.num_training_samples()

        imidx = self.image_indices[curr_img]
        mirror = self.cfg.mirror and self.mirrored[curr_img]

        return imidx, mirror


    def get_training_sample(self, imidx):
        return self.data[imidx]


    def get_scale(self):
        cfg = self.cfg
        scale = cfg.global_scale
        if hasattr(cfg, 'scale_jitter_lo') and hasattr(cfg, 'scale_jitter_up'):
            scale_jitter = rand.uniform(cfg.scale_jitter_lo, cfg.scale_jitter_up)
            scale *= scale_jitter
        return scale


    def next_batch(self):
        while True:
            imidx, mirror = self.next_training_sample()
            data_item = self.get_training_sample(imidx)
            scale = self.get_scale()

            if not self.is_valid_size(data_item.im_size, scale):
                continue

            return self.make_batch(data_item, scale, mirror)


    def is_valid_size(self, image_size, scale):
        im_width = image_size[2]
        im_height = image_size[1]

        max_input_size = 100
        if im_height < max_input_size or im_width < max_input_size:
            return False

        if hasattr(self.cfg, 'max_input_size'):
            max_input_size = self.cfg.max_input_size
            input_width = im_width * scale
            input_height = im_height * scale
            if input_height * input_width > max_input_size * max_input_size:
                return False

        return True


    def make_batch(self, data_item, scale, mirror):
        im_file = data_item.im_path
        logging.debug('image %s', im_file)
        logging.debug('mirror %r', mirror)
        image = imread(os.path.join(self.cfg.project_path,im_file), mode='RGB')

        if self.has_gt:
            joints = np.copy(data_item.joints)

        if self.cfg.crop: #adapted cropping for DLC2
            if np.random.rand()<self.cfg.cropratio:
                #1. get center of joints
                j=np.random.randint(np.shape(joints)[1]) #pick a random joint
                # draw random crop dimensions & subtract joint points
                #print(joints,j,'ahah')
                joints,image=CropImage(joints,image,joints[0,j,1],joints[0,j,2],self.cfg)

                #if self.has_gt:
                #    joints[0,:, 1] -= x0
                #    joints[0,:, 2] -= y0
                '''
                print(joints)
                import matplotlib.pyplot as plt
                plt.clf()
                plt.imshow(image)
                plt.plot(joints[0,:,1],joints[0,:,2],'.')
                plt.savefig("abc"+str(np.random.randint(int(1e6)))+".png")
                '''
            else:
                pass #no cropping!


        img = imresize(image, scale) if scale != 1 else image
        scaled_img_size = arr(img.shape[0:2])

        if mirror:
            img = np.fliplr(img)

        batch = {Batch.inputs: img}

        if self.has_gt:
            stride = self.cfg.stride

            if mirror:
                joints = [self.mirror_joints(person_joints, self.symmetric_joints, image.shape[1]) for person_joints in
                          joints]

            sm_size = np.ceil(scaled_img_size / (stride * 2)).astype(int) * 2

            scaled_joints = [person_joints[:, 1:3] * scale for person_joints in joints]

            joint_id = [person_joints[:, 0].astype(int) for person_joints in joints]
            batch = self.compute_targets_and_weights(joint_id, scaled_joints, data_item, sm_size, scale, batch)

            if self.pairwise_stats_collect:
                data_item.pairwise_stats = collect_pairwise_stats(joint_id, scaled_joints)

        batch = {key: data_to_input(data) for (key, data) in batch.items()}

        batch[Batch.data_item] = data_item

        return batch


    def set_locref(self, locref_map, locref_mask, locref_scale, i, j, j_id, dx, dy):
        if if self.cfg.weigh_only_present_joints:
            locref_mask[j, i, j_id * 2 + 0] = 1
            locref_mask[j, i, j_id * 2 + 1] = 1
        locref_map[j, i, j_id * 2 + 0] = dx * locref_scale
        locref_map[j, i, j_id * 2 + 1] = dy * locref_scale


    def set_pairwise_map(self, pairwise_map, pairwise_mask, i, j, j_id, j_id_end, coords, pt_x, pt_y, person_id, k_end):
        num_joints = self.cfg.num_joints
        joint_pt = coords[person_id][k_end, :]
        j_x_end = np.asscalar(joint_pt[0])
        j_y_end = np.asscalar(joint_pt[1])
        pair_id = get_pairwise_index(j_id, j_id_end, num_joints)
        stats = self.pairwise_stats[(j_id, j_id_end)]
        dx = j_x_end - pt_x
        dy = j_y_end - pt_y
        if self.cfg.weigh_only_present_joints:
            pairwise_mask[j, i, pair_id * 2 + 0] = 1.
            pairwise_mask[j, i, pair_id * 2 + 1] = 1.
        pairwise_map[j, i, pair_id * 2 + 0] = (dx - stats["mean"][0]) / stats["std"][0]
        pairwise_map[j, i, pair_id * 2 + 1] = (dy - stats["mean"][1]) / stats["std"][1]


    def compute_targets_and_weights(self, joint_id, coords, data_item, size, scale, batch):
        stride = self.cfg.stride
        dist_thresh = self.cfg.pos_dist_thresh * scale
        num_joints = self.cfg.num_joints
        half_stride = stride / 2
        scmap = np.zeros(cat([size, arr([num_joints])]))

        locref_shape = cat([size, arr([num_joints * 2])])
        ############## ATTENTION: mask = all!
        if cfg.weigh_only_present_joints:
            locref_mask = np.zeros(locref_shape)
        else:
            locref_mask = np.ones(locref_shape)

        locref_map = np.zeros(locref_shape)

        pairwise_shape = cat([size, arr([num_joints * (num_joints - 1) * 2])])
        if cfg.weigh_only_present_joints:
            pairwise_mask = np.ones(pairwise_shape)
        else:
            pairwise_mask = np.zeros(pairwise_shape)

        pairwise_map = np.zeros(pairwise_shape)

        dist_thresh_sq = dist_thresh ** 2

        width = size[1]
        height = size[0]

        for person_id in range(len(coords)):
            for k, j_id in enumerate(joint_id[person_id]):
                joint_pt = coords[person_id][k, :]
                j_x = np.asscalar(joint_pt[0])
                j_y = np.asscalar(joint_pt[1])

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
                        # pt = arr([i*stride+half_stride, j*stride+half_stride])
                        # diff = joint_pt - pt
                        # The code above is too slow in python
                        pt_x = i * stride + half_stride
                        dx = j_x - pt_x
                        dy = j_y - pt_y
                        dist = dx ** 2 + dy ** 2
                        # print(la.norm(diff))

                        if dist <= dist_thresh_sq:
                            dist = dx ** 2 + dy ** 2
                            locref_scale = 1.0 / self.cfg.locref_stdev
                            current_normalized_dist = dist * locref_scale ** 2
                            prev_normalized_dist = locref_map[j, i, j_id * 2 + 0] ** 2 + \
                                                   locref_map[j, i, j_id * 2 + 1] ** 2
                            update_scores = (scmap[j, i, j_id] == 0) or prev_normalized_dist > current_normalized_dist
                            if self.cfg.location_refinement and update_scores:
                                self.set_locref(locref_map, locref_mask, locref_scale, i, j, j_id, dx, dy)
                            if self.cfg.pairwise_predict and update_scores:
                                for k_end, j_id_end in enumerate(joint_id[person_id]):
                                    if k != k_end:
                                        self.set_pairwise_map(pairwise_map, pairwise_mask, i, j, j_id, j_id_end,
                                                              coords, pt_x, pt_y, person_id, k_end)
                            scmap[j, i, j_id] = 1

        scmap_weights = self.compute_scmap_weights(scmap.shape, joint_id, data_item)

        # Update batch
        batch.update({
            Batch.part_score_targets: scmap,
            Batch.part_score_weights: scmap_weights
        })
        if self.cfg.location_refinement:
            batch.update({
                Batch.locref_targets: locref_map,
                Batch.locref_mask: locref_mask
            })
        if self.cfg.pairwise_predict:
            batch.update({
                Batch.pairwise_targets: pairwise_map,
                Batch.pairwise_mask: pairwise_mask
            })

        return batch


    def compute_scmap_weights(self, scmap_shape, joint_id, data_item):
        cfg = self.cfg
        if cfg.weigh_only_present_joints:
            weights = np.zeros(scmap_shape)
            for person_joint_id in joint_id:
                for j_id in person_joint_id:
                    weights[:, :, j_id] = 1.0
        else:
            weights = np.ones(scmap_shape)
        return weights
