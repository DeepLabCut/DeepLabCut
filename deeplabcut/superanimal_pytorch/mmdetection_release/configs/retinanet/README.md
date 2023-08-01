# RetinaNet

> [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

<!-- [ALGORITHM] -->

## Abstract

The highest accuracy object detectors to date are based on a two-stage approach popularized by R-CNN, where a classifier is applied to a sparse set of candidate object locations. In contrast, one-stage detectors that are applied over a regular, dense sampling of possible object locations have the potential to be faster and simpler, but have trailed the accuracy of two-stage detectors thus far. In this paper, we investigate why this is the case. We discover that the extreme foreground-background class imbalance encountered during training of dense detectors is the central cause. We propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples. Our novel Focal Loss focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training. To evaluate the effectiveness of our loss, we design and train a simple dense detector we call RetinaNet. Our results show that when trained with the focal loss, RetinaNet is able to match the speed of previous one-stage detectors while surpassing the accuracy of all existing state-of-the-art two-stage detectors.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143973551-2b8e766a-1677-4f6d-953d-2e6d2a3c67b5.png" height="300"/>
</div>

## Results and Models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download  |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    R-50-FPN     |  caffe  |   1x    |   3.5    |      18.6      |  36.3  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r50_caffe_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_caffe_fpn_1x_coco/retinanet_r50_caffe_fpn_1x_coco_20200531-f11027c5.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_caffe_fpn_1x_coco/retinanet_r50_caffe_fpn_1x_coco_20200531_012518.log.json) |
|    R-50-FPN     | pytorch |   1x    |   3.8    |      19.0      |  36.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130_002941.log.json) |
|   R-50-FPN (FP16)      | pytorch |  1x      | 2.8      | 31.6           | 36.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r50_fpn_fp16_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/fp16/retinanet_r50_fpn_fp16_1x_coco/retinanet_r50_fpn_fp16_1x_coco_20200702-0dbfb212.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fp16/retinanet_r50_fpn_fp16_1x_coco/retinanet_r50_fpn_fp16_1x_coco_20200702_020127.log.json) |
|    R-50-FPN     | pytorch |   2x    |    -     |       -        |  37.4  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r50_fpn_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131_114738.log.json) |
|    R-101-FPN    |  caffe  |   1x    |   5.5    |      14.7      |  38.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r101_caffe_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_caffe_fpn_1x_coco/retinanet_r101_caffe_fpn_1x_coco_20200531-b428fa0f.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_caffe_fpn_1x_coco/retinanet_r101_caffe_fpn_1x_coco_20200531_012536.log.json) |
|    R-101-FPN    | pytorch |   1x    |   5.7    |      15.0      |  38.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r101_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_1x_coco/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_1x_coco/retinanet_r101_fpn_1x_coco_20200130_003055.log.json) |
|    R-101-FPN    | pytorch |   2x    |    -     |       -        |  38.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r101_fpn_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_20200131_114859.log.json) |
| X-101-32x4d-FPN | pytorch |   1x    |   7.0    |      12.1      |  39.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_x101_32x4d_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_1x_coco/retinanet_x101_32x4d_fpn_1x_coco_20200130-5c8b7ec4.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_1x_coco/retinanet_x101_32x4d_fpn_1x_coco_20200130_003004.log.json) |
| X-101-32x4d-FPN | pytorch |   2x    |    -     |       -        |  40.1  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_x101_32x4d_fpn_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_2x_coco/retinanet_x101_32x4d_fpn_2x_coco_20200131-237fc5e1.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_2x_coco/retinanet_x101_32x4d_fpn_2x_coco_20200131_114812.log.json) |
| X-101-64x4d-FPN | pytorch |   1x    |   10.0   |      8.7       |  41.0  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130_003008.log.json) |
| X-101-64x4d-FPN | pytorch |   2x    |    -     |       -        |  40.8  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_x101_64x4d_fpn_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_2x_coco/retinanet_x101_64x4d_fpn_2x_coco_20200131-bca068ab.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_2x_coco/retinanet_x101_64x4d_fpn_2x_coco_20200131_114833.log.json) |

## Pre-trained Models

We also train some models with longer schedules and multi-scale training. The users could finetune them for downstream tasks.

|    Backbone        |  Style  | Lr schd | Mem (GB) | box AP | Config | Download |
| :----------------: | :-----: | :-----: | :------: | :----: | :------: | :--------: |
|    R-50-FPN        |  pytorch|   3x    |   3.5    |  39.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r50_fpn_mstrain_640-800_3x_coco.py)     |  [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_mstrain_3x_coco/retinanet_r50_fpn_mstrain_3x_coco_20210718_220633-88476508.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_mstrain_3x_coco/retinanet_r50_fpn_mstrain_3x_coco_20210718_220633-88476508.log.json)
|    R-101-FPN       |  caffe  |   3x    |   5.4    |  40.7  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r101_caffe_fpn_mstrain_3x_coco.py)    |  [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_caffe_fpn_mstrain_3x_coco/retinanet_r101_caffe_fpn_mstrain_3x_coco_20210721_063439-88a8a944.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_caffe_fpn_mstrain_3x_coco/retinanet_r101_caffe_fpn_mstrain_3x_coco_20210721_063439-88a8a944.log.json)
|    R-101-FPN       |  pytorch|   3x    |   5.4    |  41    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r101_fpn_mstrain_640-800_3x_coco.py)      | [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_mstrain_3x_coco/retinanet_r101_fpn_mstrain_3x_coco_20210720_214650-7ee888e0.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_mstrain_3x_coco/retinanet_r101_fpn_mstrain_3x_coco_20210720_214650-7ee888e0.log.json)
|    X-101-64x4d-FPN |  pytorch|   3x    |   9.8   |  41.6  |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_x101_64x4d_fpn_mstrain_640-800_3x_coco.py)      |      [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_mstrain_3x_coco/retinanet_x101_64x4d_fpn_mstrain_3x_coco_20210719_051838-022c2187.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_mstrain_3x_coco/retinanet_x101_64x4d_fpn_mstrain_3x_coco_20210719_051838-022c2187.log.json)

## Citation

```latex
@inproceedings{lin2017focal,
  title={Focal loss for dense object detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2017}
}
```
