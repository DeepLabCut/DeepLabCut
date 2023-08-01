# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class InstaBoost:
    r"""Data augmentation method in `InstaBoost: Boosting Instance
    Segmentation Via Probability Map Guided Copy-Pasting
    <https://arxiv.org/abs/1908.07801>`_.

    Refer to https://github.com/GothicAi/Instaboost for implementation details.

    Args:
        action_candidate (tuple): Action candidates. "normal", "horizontal", \
            "vertical", "skip" are supported. Default: ('normal', \
            'horizontal', 'skip').
        action_prob (tuple): Corresponding action probabilities. Should be \
            the same length as action_candidate. Default: (1, 0, 0).
        scale (tuple): (min scale, max scale). Default: (0.8, 1.2).
        dx (int): The maximum x-axis shift will be (instance width) / dx.
            Default 15.
        dy (int): The maximum y-axis shift will be (instance height) / dy.
            Default 15.
        theta (tuple): (min rotation degree, max rotation degree). \
            Default: (-1, 1).
        color_prob (float): Probability of images for color augmentation.
            Default 0.5.
        heatmap_flag (bool): Whether to use heatmap guided. Default False.
        aug_ratio (float): Probability of applying this transformation. \
            Default 0.5.
    """

    def __init__(self,
                 action_candidate=('normal', 'horizontal', 'skip'),
                 action_prob=(1, 0, 0),
                 scale=(0.8, 1.2),
                 dx=15,
                 dy=15,
                 theta=(-1, 1),
                 color_prob=0.5,
                 hflag=False,
                 aug_ratio=0.5):
        try:
            import instaboostfast as instaboost
        except ImportError:
            raise ImportError(
                'Please run "pip install instaboostfast" '
                'to install instaboostfast first for instaboost augmentation.')
        self.cfg = instaboost.InstaBoostConfig(action_candidate, action_prob,
                                               scale, dx, dy, theta,
                                               color_prob, hflag)
        self.aug_ratio = aug_ratio

    def _load_anns(self, results):
        labels = results['ann_info']['labels']
        masks = results['ann_info']['masks']
        bboxes = results['ann_info']['bboxes']
        n = len(labels)

        anns = []
        for i in range(n):
            label = labels[i]
            bbox = bboxes[i]
            mask = masks[i]
            x1, y1, x2, y2 = bbox
            # assert (x2 - x1) >= 1 and (y2 - y1) >= 1
            bbox = [x1, y1, x2 - x1, y2 - y1]
            anns.append({
                'category_id': label,
                'segmentation': mask,
                'bbox': bbox
            })

        return anns

    def _parse_anns(self, results, anns, img):
        gt_bboxes = []
        gt_labels = []
        gt_masks_ann = []
        for ann in anns:
            x1, y1, w, h = ann['bbox']
            # TODO: more essential bug need to be fixed in instaboost
            if w <= 0 or h <= 0:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            gt_bboxes.append(bbox)
            gt_labels.append(ann['category_id'])
            gt_masks_ann.append(ann['segmentation'])
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
        results['ann_info']['labels'] = gt_labels
        results['ann_info']['bboxes'] = gt_bboxes
        results['ann_info']['masks'] = gt_masks_ann
        results['img'] = img
        return results

    def __call__(self, results):
        img = results['img']
        orig_type = img.dtype
        anns = self._load_anns(results)
        if np.random.choice([0, 1], p=[1 - self.aug_ratio, self.aug_ratio]):
            try:
                import instaboostfast as instaboost
            except ImportError:
                raise ImportError('Please run "pip install instaboostfast" '
                                  'to install instaboostfast first.')
            anns, img = instaboost.get_new_data(
                anns, img.astype(np.uint8), self.cfg, background=None)

        results = self._parse_anns(results, anns, img.astype(orig_type))
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(cfg={self.cfg}, aug_ratio={self.aug_ratio})'
        return repr_str
