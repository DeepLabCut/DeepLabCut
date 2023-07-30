# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings
from argparse import ArgumentParser
from xtcocotools.coco import COCO
import mmcv
import json
import numpy as np

from mmpose.apis import (inference_bottom_up_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo


def main():
    """Visualize the demo images."""
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument('--plot_gt', action='store_true')
    parser.add_argument(
        '--json-file',
        type=str,
        default='',
        help='Json file containing image info.')    
    parser.add_argument(
        '--img-path',
        type=str,
        help='Path to an image file or a image folder.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='Root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--pose-nms-thr',
        type=float,
        default=0.9,
        help='OKS threshold for pose NMS')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')


    coco = COCO(args.json_file)
    
    # prepare image list
    '''
    if osp.isfile(args.img_path):
        image_list = [args.img_path]
    elif osp.isdir(args.img_path):
        image_list = [
            osp.join(args.img_path, fn) for fn in os.listdir(args.img_path)
            if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
        ]
    else:
        raise ValueError('Image path should be an image or image folder.'
                         f'Got invalid image path: {args.img_path}')
    '''
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        assert (dataset == 'BottomUpCocoDataset')
    else:

        try:
            dataset_info = DatasetInfo(dataset_info)
        except:
            with open(dataset_info, 'r') as f:
                dataset_info = json.load(f)['dataset_info']
                dataset_info = DatasetInfo(dataset_info)                
                
    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    img_keys = list(coco.imgs.keys())
    
    # process each image
    #for image_name in mmcv.track_iter_progress(image_list):
    for i in range(len(img_keys)):

        # test a single image, with a list of bboxes.

        image_id = img_keys[i]
        image = coco.loadImgs(image_id)[0]
        suffix_image_name = image['file_name']
        image_name = os.path.join(args.img_root, image['file_name'])
        ann_ids = coco.getAnnIds(image_id)        

        gt_kpts = []
        for ann_id in ann_ids:
            ann = coco.anns[ann_id]
            gt_kpts.append(np.array(ann['keypoints']).reshape(-1,3))
        
                
        pose_results, returned_outputs = inference_bottom_up_pose_model(
            pose_model,
            image_name,
            dataset=dataset,
            dataset_info=dataset_info,
            pose_nms_thr=args.pose_nms_thr,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(args.out_img_root, suffix_image_name)            
            #out_file = os.path.join(
            #    args.out_img_root,
            #    f'vis_{osp.splitext(osp.basename(image_name))[0]}.jpg')


        if not args.plot_gt:
            gt_kpts = []
        # show the results
        vis_pose_result(
            pose_model,
            image_name,
            pose_results,
            radius=args.radius,
            gt_kpts = gt_kpts,            
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=args.show,
            out_file=out_file)


if __name__ == '__main__':
    main()
