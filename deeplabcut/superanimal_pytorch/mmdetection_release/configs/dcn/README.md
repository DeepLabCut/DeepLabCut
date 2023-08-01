# DCN

> [Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211)

<!-- [ALGORITHM] -->

## Abstract

Convolutional neural networks (CNNs) are inherently limited to model geometric transformations due to the fixed geometric structures in its building modules. In this work, we introduce two new modules to enhance the transformation modeling capacity of CNNs, namely, deformable convolution and deformable RoI pooling. Both are based on the idea of augmenting the spatial sampling locations in the modules with additional offsets and learning the offsets from target tasks, without additional supervision. The new modules can readily replace their plain counterparts in existing CNNs and can be easily trained end-to-end by standard back-propagation, giving rise to deformable convolutional networks. Extensive experiments validate the effectiveness of our approach on sophisticated vision tasks of object detection and semantic segmentation.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143876246-c4985e25-e286-4511-9b7c-97af2857461e.png"/>
</div>

## Results and Models

| Backbone         | Model        | Style   | Conv          | Pool   | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:----------------:|:------------:|:-------:|:-------------:|:------:|:-------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| R-50-FPN         | Faster       | pytorch | dconv(c3-c5)  | -      | 1x      | 4.0  | 17.8 | 41.3 |     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130_212941.log.json) |
| R-50-FPN         | Faster       | pytorch | -             | dpool  | 1x      | 5.0  | 17.2 | 38.9 |     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/faster_rcnn_r50_fpn_dpool_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dpool_1x_coco/faster_rcnn_r50_fpn_dpool_1x_coco_20200307-90d3c01d.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dpool_1x_coco/faster_rcnn_r50_fpn_dpool_1x_coco_20200307_203250.log.json) |
| R-101-FPN        | Faster       | pytorch | dconv(c3-c5)  | -      | 1x      | 6.0  | 12.5 | 42.7 |     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-1377f13d.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203_230019.log.json) |
| X-101-32x4d-FPN | Faster        | pytorch | dconv(c3-c5)  | -      | 1x      | 7.3  | 10.0  | 44.5 |     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco_20200203-4f85c69c.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco_20200203_001325.log.json) |
| R-50-FPN         | Mask         | pytorch | dconv(c3-c5)  | -      | 1x      | 4.5  | 15.4 | 41.8 | 37.4 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200203-4d9ad43b.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200203_061339.log.json) |
| R-101-FPN        | Mask         | pytorch | dconv(c3-c5)  | -      | 1x      | 6.5  | 11.7  | 43.5 | 38.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200216-a71f5bce.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200216_191601.log.json) |
| R-50-FPN         | Cascade      | pytorch | dconv(c3-c5)  | -      | 1x      | 4.5  | 14.6 | 43.8 |     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-2f1fca44.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130_220843.log.json) |
| R-101-FPN        | Cascade      | pytorch | dconv(c3-c5)  | -      | 1x      | 6.4  | 11.0 | 45.0 |     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-3b2f0594.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203_224829.log.json) |
| R-50-FPN         | Cascade Mask | pytorch | dconv(c3-c5)  | -      | 1x      | 6.0  | 10.0  | 44.4 | 38.6 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200202-42e767a2.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200202_010309.log.json) |
| R-101-FPN        | Cascade Mask | pytorch | dconv(c3-c5)  | -      | 1x      | 8.0  | 8.6  | 45.8 | 39.7 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200204-df0c5f10.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200204_134006.log.json) |
| X-101-32x4d-FPN        | Cascade Mask | pytorch | dconv(c3-c5)  | -      | 1x      | 9.2 |   | 47.3 | 41.1 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-e75f90c8.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-20200606_183737.log.json) |
| R-50-FPN (FP16)      | Mask         | pytorch | dconv(c3-c5) | -      | 1x      | 3.0      |                | 41.9   | 37.5    |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fp16/mask_rcnn_r50_fpn_fp16_dconv_c3-c5_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/fp16/mask_rcnn_r50_fpn_fp16_dconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_fp16_dconv_c3-c5_1x_coco_20210520_180247-c06429d2.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fp16/mask_rcnn_r50_fpn_fp16_dconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_fp16_dconv_c3-c5_1x_coco_20210520_180247.log.json)  |

**Notes:**

- `dconv` denotes deformable convolution, `c3-c5` means adding dconv in resnet stage 3 to 5. `dpool` denotes deformable roi pooling.
- The dcn ops are modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch, which should be more memory efficient and slightly faster.
- (*) For R-50-FPN (dg=4), dg is short for deformable_group. This model is trained and tested on Amazon EC2 p3dn.24xlarge instance.
- **Memory, Train/Inf time is outdated.**

## Citation

```latex
@inproceedings{dai2017deformable,
  title={Deformable Convolutional Networks},
  author={Dai, Jifeng and Qi, Haozhi and Xiong, Yuwen and Li, Yi and Zhang, Guodong and Hu, Han and Wei, Yichen},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2017}
}
```
