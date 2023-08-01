# PISA

> [Prime Sample Attention in Object Detection](https://arxiv.org/abs/1904.04821)

<!-- [ALGORITHM] -->

## Abstract

It is a common paradigm in object detection frameworks to treat all samples equally and target at maximizing the performance on average. In this work, we revisit this paradigm through a careful study on how different samples contribute to the overall performance measured in terms of mAP. Our study suggests that the samples in each mini-batch are neither independent nor equally important, and therefore a better classifier on average does not necessarily mean higher mAP. Motivated by this study, we propose the notion of Prime Samples, those that play a key role in driving the detection performance. We further develop a simple yet effective sampling and learning strategy called PrIme Sample Attention (PISA) that directs the focus of the training process towards such samples. Our experiments demonstrate that it is often more effective to focus on prime samples than hard samples when training a detector. Particularly, On the MSCOCO dataset, PISA outperforms the random sampling baseline and hard mining schemes, e.g., OHEM and Focal Loss, consistently by around 2% on both single-stage and two-stage detectors, even with a strong backbone ResNeXt-101.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143970710-5cfd5960-fcf9-4e32-860a-acd46ce5d274.png"/>
</div>

## Results and Models

| PISA | Network | Backbone            | Lr schd | box AP | mask AP | Config | Download |
|:----:|:-------:|:-------------------:|:-------:|:------:|:-------:|:------:|:--------:|
| ×    | Faster R-CNN | R-50-FPN       | 1x      | 36.4   |         | - |
| √    | Faster R-CNN | R-50-FPN       | 1x      | 38.4   |         | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pisa/pisa_faster_rcnn_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_faster_rcnn_r50_fpn_1x_coco/pisa_faster_rcnn_r50_fpn_1x_coco-dea93523.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_faster_rcnn_r50_fpn_1x_coco/pisa_faster_rcnn_r50_fpn_1x_coco_20200506_185619.log.json)  |
| ×    | Faster R-CNN | X101-32x4d-FPN | 1x      | 40.1   |         | - |
| √    | Faster R-CNN | X101-32x4d-FPN | 1x      | 41.9   |         | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pisa/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco-e4accec4.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco_20200505_181503.log.json) |
| ×    | Mask   R-CNN | R-50-FPN       | 1x      | 37.3   | 34.2    | - |
| √    | Mask   R-CNN | R-50-FPN       | 1x      | 39.1   | 35.2    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pisa/pisa_mask_rcnn_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_mask_rcnn_r50_fpn_1x_coco/pisa_mask_rcnn_r50_fpn_1x_coco-dfcedba6.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_mask_rcnn_r50_fpn_1x_coco/pisa_mask_rcnn_r50_fpn_1x_coco_20200508_150500.log.json) |
| ×    | Mask   R-CNN | X101-32x4d-FPN | 1x      | 41.1   | 37.1    | - |
| √    | Mask   R-CNN | X101-32x4d-FPN | 1x      |        |         |   |
| ×    | RetinaNet    | R-50-FPN       | 1x      | 35.6   |         | - |
| √    | RetinaNet    | R-50-FPN       | 1x      | 36.9   |         | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pisa/pisa_retinanet_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_retinanet_r50_fpn_1x_coco/pisa_retinanet_r50_fpn_1x_coco-76409952.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_retinanet_r50_fpn_1x_coco/pisa_retinanet_r50_fpn_1x_coco_20200504_014311.log.json) |
| ×    | RetinaNet    | X101-32x4d-FPN | 1x      | 39.0   |         | - |
| √    | RetinaNet    | X101-32x4d-FPN | 1x      | 40.7   |         | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pisa/pisa_retinanet_x101_32x4d_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_retinanet_x101_32x4d_fpn_1x_coco/pisa_retinanet_x101_32x4d_fpn_1x_coco-a0c13c73.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_retinanet_x101_32x4d_fpn_1x_coco/pisa_retinanet_x101_32x4d_fpn_1x_coco_20200505_001404.log.json) |
| ×    | SSD300       | VGG16          | 1x      | 25.6   |         | - |
| √    | SSD300       | VGG16          | 1x      | 27.6   |         | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pisa/pisa_ssd300_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_ssd300_coco/pisa_ssd300_coco-710e3ac9.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_ssd300_coco/pisa_ssd300_coco_20200504_144325.log.json) |
| ×    | SSD300       | VGG16          | 1x      | 29.3   |         | - |
| √    | SSD300       | VGG16          | 1x      | 31.8   |         | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pisa/pisa_ssd512_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_ssd512_coco/pisa_ssd512_coco-247addee.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_ssd512_coco/pisa_ssd512_coco_20200508_131030.log.json)  |

**Notes:**

- In the original paper, all models are trained and tested on mmdet v1.x, thus results may not be exactly the same with this release on v2.0.
- It is noted PISA only modifies the training pipeline so the inference time remains the same with the baseline.

## Citation

```latex
@inproceedings{cao2019prime,
  title={Prime sample attention in object detection},
  author={Cao, Yuhang and Chen, Kai and Loy, Chen Change and Lin, Dahua},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```
