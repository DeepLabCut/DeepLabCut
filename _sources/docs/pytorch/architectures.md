(dlc3-architectures)=
# DeepLabCut 3.0 - PyTorch Model Architectures

## Introduction

You can see a list of supported architectures/variants by using:

```python
from deeplabcut.pose_estimation_pytorch import available_models
print(available_models())
```

You can see a list of supported object detection architectures/variants by using:

```python
from deeplabcut.pose_estimation_pytorch import available_detectors
print(available_detectors())
```

## Neural Networks Architectures

Several architectures are currently implemented in DeepLabCut PyTorch (more will come,
and you can add more easily in our new model registry). Also check out the explanations of bottom-up/top-down below. 

**ResNets**
- Adapted from [He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2016.](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) and [Insafutdinov, Eldar et al. "DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model". European Conference on Computer Vision (ECCV) 2016.]
- Current bottom-up variants are `resnet_50`, `resnet_101`
- Current top-down variants are `top_down_resnet_101`, `top_down_resnet_50`

**HRNet**
- Adapted from [Wang, Jingdong, et al. "Deep high-resolution representation learning for visual recognition." IEEE transactions on pattern analysis and machine intelligence 43.10 (2020): 3349-3364.](https://arxiv.org/abs/1908.07919)
- Current variants are `hrnet_w18`, `hrnet_w32`, `hrnet_w48`, 
- Current top-down variants are `top_down_hrnet_w18`, `top_down_hrnet_w32`, `top_down_hrnet_w48`
- Slower but typically more powerful than ResNets

**DEKR**
- Adapted from [Geng, Zigang et al. "Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression." Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2021.](https://openaccess.thecvf.com/content/CVPR2021/papers/Geng_Bottom-Up_Human_Pose_Estimation_via_Disentangled_Keypoint_Regression_CVPR_2021_paper.pdf)
- This model is a bottom-up model using HRNet as a backbone. It learns to predict the center of each animal, and predicts the offset between each animal center and their keypoints
- Current variants that are implemented (from smallest to largest): `dekr_w18`, `dekr_w32`, `dekr_w48`
- Note, this is a powerful multi-animal model but very heavy (slow)

**BUCTD**
- Adapted from [Zhou\*, Stoffl\*, Mathis, Mathis. "Rethinking Pose Estimation in Crowds: Overcoming the Detection Information Bottleneck and Ambiguity." Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_Rethinking_Pose_Estimation_in_Crowds_Overcoming_the_Detection_Information_Bottleneck_ICCV_2023_paper.pdf)
- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-pose-estimation-in-crowds/pose-estimation-on-crowdpose)](https://paperswithcode.com/sota/pose-estimation-on-crowdpose?p=rethinking-pose-estimation-in-crowds)
- This is a top-performing multi-animal method that combines the strengths of bottom-up and top-down approaches, and delivers exceptional performance on humans too (which are also animals)
- It can be used with a diverse set of architectures. Current variants are: `ctd_coam_w32`, `ctd_coam_w48`/`ctd_coam_w48_human`, `ctd_prenet_hrnet_w32`, `ctd_prenet_hrnet_w48`, `ctd_prenet_rtmpose_s`, `ctd_prenet_rtmpose_m`, `ctd_prenet_rtmpose_x`/`ctd_prenet_rtmpose_x_human`

**DLCRNet**
- From [Lauer, Zhou, et al. "Multi-animal pose estimation, identification and tracking with DeepLabCut." Nature Methods 19.4 (2022): 496-504.](https://www.nature.com/articles/s41592-022-01443-0)
- This model uses a multi-scale variant of a ResNet as a backbone, and part-affinity fields to assemble individuals
- Variants: `dlcrnet_stride16_ms5`, `dlcrnet_stride32_ms5`

**RTMPose**
- From [Jiang, Tao et al. "RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose"](https://arxiv.org/abs/2303.07399)
- Top-down pose estimation model using a fast CSPNeXt backbone with a SimCC-style head
- Variants: `rtmpose_s`, `rtmpose_m`, `rtmpose_x`

**AnimalTokenPose**
-  Adapted from [Li, Yanjie, et al. "Tokenpose: Learning keypoint tokens for human pose estimation." Proceedings of the IEEE/CVF International conference on computer vision. 2021.](https://arxiv.org/abs/2104.03516) as in Ye et al. "SuperAnimal pretrained pose estimation models for behavioral analysis." Nature Communications. 2024](https://arxiv.org/abs/2203.07436)
  - One variant is implemented as: `animal_tokenpose_base` for video inference only (we don't support directly training this within deeplabcut)


## Information on Single Animal Models

Single-animal models are composed of a backbone (encoder) and a head (decoder) 
predicting the position of keypoints. The default head contains a single deconvolutional
layer. To create the single animal model composed of a backbone and head, you can call
`deeplabcut.create_training_dataset` with `net_type` set to the backbone name (e.g. 
`resnet_50` or `hrnet_w32`).

If you want to add a second deconvolutional layer (which will make your model slower, 
but it might improve performance), you can simply edit your `pytorch_config.yaml` file.

Of course, any multi-animal model can also be used for single-animal projects!

## Approaches to Multi-Animal pose estimation

Single-animal pose estimation is quite straightforward: the model takes an image as 
input, and it outputs the predicted coordinate of each bodypart.

Multi-animal pose estimation is more complex. Not only do you need to localize bodyparts
in the image, but you also need to group bodyparts per individual. There are two main
approaches to multi-animal pose estimation.

### Bottom-up estimation

The first approach, **bottom-up** pose estimation, starts by detecting bodyparts in the
image before figuring out how they belong together (i.e., which keypoints belong to the
same animal).

![Schema representing the bottom-up approach to pose estimation](
assets/bottom-up-approach.png)

### Backbones with Part-Affinity Fields 

As in DeepLabCut 2.X, the base multi-animal model is composed of a backbone (encoder) 
and a head predicting keypoints and part-affinity fields (PAFs). These PAFs are used to 
assemble keypoints for individuals.

Passing a backbone as a net type (e.g., `resnet_50`, `hrnet_w32`) for a multi-animal 
project will create a model consisting of a backbone and a heatmap + PAF head.

### Top-down estimation

The second approach, **top-down** pose estimation, uses a two-step approach. A first 
model (an object detector) is used to localize every animal present in the image through
its bounding box. Then, the pose for each animal is determined by predicting bodyparts
in each bounding box. The pose estimation 

![Schema representing the top-down approach to pose estimation](
assets/top-down-approach.png)

The top-down approach tends to be more accurate in less crowded scenes, as the pose 
model only needs to process the pixels related to a single animal. However, in more 
crowded scenes, the pose estimation task becomes ambiguous. Multiple overlapping 
individuals will have very similar bounding boxes, and the pose model has no way of 
knowing which animal it is supposed to predict keypoints for.

The bottom-up approach does not have this ambiguïty, and also has the advantage of
only needing to run a pose estimation model, instead of needing to run an object 
detector first. However, grouping keypoints is a difficult problem.


Hence any single-animal model can be transformed into a top-down, multi-animal model. To
do so, simply prefix `top_down` to your single-animal model name. Currently, the 
following detectors are available: `ssdlite`, `fasterrcnn_mobilenet_v3_large_fpn`,
`fasterrcnn_resnet50_fpn_v2`.


### Hybrid, Bottom-up (BU) plus a ``conditioned" Top-down (CTD)

A new approach to pose estimation, named bottom-up conditioned top-down (or **BUCTD**), was
introduced in [Zhou, Stoffl, Mathis, Mathis. "Rethinking Pose Estimation in Crowds: 
Overcoming the Detection Information Bottleneck and Ambiguity." Proceedings of the 
IEEE/CVF International Conference on Computer Vision (ICCV). 2023](
https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_Rethinking_Pose_Estimation_in_Crowds_Overcoming_the_Detection_Information_Bottleneck_ICCV_2023_paper.pdf)
. It's a hybrid two-stage approach leveraging the strengths of the bottom-up and
top-down approaches to overcome the ambiguïty introduced through bounding boxes. Instead
of using an object detection model to localize individuals, it uses a bottom-up pose 
estimation model. The predictions made by the bottom-up model are given as proposals (or
_conditions_) to the pose estimation model. This is illustrated in the figure below. In modern language, one could state that CTD models are "pose-promptable". 


![BUCTD](https://github.com/amathislab/BUCTD/raw/main/media/BUCTD_fig1.png)
Zhou, Mu, et al. *"Rethinking pose estimation in crowds: overcoming the
detection information bottleneck and ambiguity."* Proceedings of the IEEE/CVF
International Conference on Computer Vision. 2023.
