import glob
import re
import os.path as osp

from .bases import BaseImageDataset


class VeRi(BaseImageDataset):
    """
       VeRi-776
       Reference:
       Liu, Xinchen, et al. "Large-scale vehicle re-identification in urban surveillance videos." ICME 2016.

       URL:https://vehiclereid.github.io/VeRi/

       Dataset statistics:
       # identities: 776
       # images: 37778 (train) + 1678 (query) + 11579 (gallery)
       # cameras: 20
       """

    dataset_dir = 'VeRi'

    def __init__(self, root='', verbose=True, **kwargs):
        super(VeRi, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self._check_before_run()

        path_train = 'datasets/keypoint_train.txt'
        with open(path_train, 'r') as txt:
            lines = txt.readlines()
        self.image_map_view_train = {}
        for img_idx, img_info in enumerate(lines):
            content = img_info.split(' ')
            viewid = int(content[-1])
            self.image_map_view_train[osp.basename(content[0])] = viewid

        path_test = 'datasets/keypoint_test.txt'
        with open(path_test, 'r') as txt:
            lines = txt.readlines()
        self.image_map_view_test = {}
        for img_idx, img_info in enumerate(lines):
            content = img_info.split(' ')
            viewid = int(content[-1])
            self.image_map_view_test[osp.basename(content[0])] = viewid

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> VeRi-776 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        view_container = set()
        dataset = []
        count = 0
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 776  # pid == 0 means background
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            if osp.basename(img_path) not in self.image_map_view_train.keys():
                try:
                    viewid = self.image_map_view_test[osp.basename(img_path)]
                except:
                    count += 1
                    # print(img_path, 'img_path')
                    continue
            else:
                viewid = self.image_map_view_train[osp.basename(img_path)]
            view_container.add(viewid)
            dataset.append((img_path, pid, camid, viewid))
        print(view_container, 'view_container')
        print(count, 'samples without viewpoint annotations')
        return dataset

