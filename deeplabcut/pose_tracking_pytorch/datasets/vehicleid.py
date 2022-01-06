from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import os.path as osp
from .bases import BaseImageDataset
from collections import defaultdict
import pickle

class VehicleID(BaseImageDataset):
    """
    VehicleID
    Reference:
    Deep Relative Distance Learning: Tell the Difference Between Similar Vehicles
    
    Dataset statistics:
    # train_list: 13164 vehicles for model training
    # test_list_800: 800 vehicles for model testing(small test set in paper
    # test_list_1600: 1600 vehicles for model testing(medium test set in paper
    # test_list_2400: 2400 vehicles for model testing(large test set in paper
    # test_list_3200: 3200 vehicles for model testing
    # test_list_6000: 6000 vehicles for model testing
    # test_list_13164: 13164 vehicles for model testing
    """
    dataset_dir = 'VehicleID_V1.0'

    def __init__(self, root='', verbose=True, test_size=800, **kwargs):
        super(VehicleID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.img_dir = osp.join(self.dataset_dir, 'image')
        self.split_dir = osp.join(self.dataset_dir, 'train_test_split')
        self.train_list = osp.join(self.split_dir, 'train_list.txt')
        self.test_size = test_size

        if self.test_size == 800:
            self.test_list = osp.join(self.split_dir, 'test_list_800.txt')
        elif self.test_size == 1600:
            self.test_list = osp.join(self.split_dir, 'test_list_1600.txt')
        elif self.test_size == 2400:
            self.test_list = osp.join(self.split_dir, 'test_list_2400.txt')

        print(self.test_list)

        self.check_before_run()

        train, query, gallery = self.process_split(relabel=True)
        self.train = train
        self.query = query
        self.gallery = gallery

        if verbose:
            print('=> VehicleID loaded')
            self.print_dataset_statistics(train, query, gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError('"{}" is not available'.format(self.dataset_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError('"{}" is not available'.format(self.split_dir))
        if not osp.exists(self.train_list):
            raise RuntimeError('"{}" is not available'.format(self.train_list))
        if self.test_size not in [800, 1600, 2400]:
            raise RuntimeError('"{}" is not available'.format(self.test_size))
        if not osp.exists(self.test_list):
            raise RuntimeError('"{}" is not available'.format(self.test_list))

    def get_pid2label(self, pids):
        pid_container = set(pids)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        return pid2label


    def parse_img_pids(self, nl_pairs, pid2label=None, cam=0):
        # il_pair is the pairs of img name and label
        output = []
        for info in nl_pairs:
            name = info[0]
            pid = info[1]
            if pid2label is not None:
                pid = pid2label[pid]
            camid = cam  # use 0 or 1
            img_path = osp.join(self.img_dir, name+'.jpg')
            viewid = 1
            output.append((img_path, pid, camid, viewid))
        return output

    def process_split(self, relabel=False):
        # read train paths
        train_pid_dict = defaultdict(list)

        # 'train_list.txt' format:
        # the first number is the number of image
        # the second number is the id of vehicle
        with open(self.train_list) as f_train:
            train_data = f_train.readlines()
            for data in train_data:
                name, pid = data.strip().split(' ')

                pid = int(pid)
                train_pid_dict[pid].append([name, pid])
        train_pids = list(train_pid_dict.keys())
        num_train_pids = len(train_pids)
        assert num_train_pids == 13164, 'There should be 13164 vehicles for training,' \
                                        ' but but got {}, please check the data'\
                                        .format(num_train_pids)
        # print('num of train ids: {}'.format(num_train_pids))
        test_pid_dict = defaultdict(list)
        with open(self.test_list) as f_test:
            test_data = f_test.readlines()
            for data in test_data:
                name, pid = data.split(' ')
                pid = int(pid)
                test_pid_dict[pid].append([name, pid])
        test_pids = list(test_pid_dict.keys())
        num_test_pids = len(test_pids)
        assert num_test_pids == self.test_size, 'There should be {} vehicles for testing,' \
                                                ' but but got {}, please check the data'\
                                                .format(self.test_size, num_test_pids)

        train_data = []
        query_data = []
        gallery_data = []
        train_pids = sorted(train_pids)
        # for train ids, all images are used in the train set.
        for pid in train_pids:
            imginfo = train_pid_dict[pid]  # imginfo include image name and id
            train_data.extend(imginfo)

        # for each test id, random choose one image for gallery
        # and the other ones for query.
        for pid in test_pids:
            imginfo = test_pid_dict[pid]
            sample = random.choice(imginfo)
            imginfo.remove(sample)
            query_data.extend(imginfo)
            gallery_data.append(sample)

        if relabel:
            train_pid2label = self.get_pid2label(train_pids)
        else:
            train_pid2label = None

        train = self.parse_img_pids(train_data, train_pid2label)
        query = self.parse_img_pids(query_data, cam=0)
        gallery = self.parse_img_pids(gallery_data, cam=1)
        # attach different camera to prevent eval fail

        return train, query, gallery
