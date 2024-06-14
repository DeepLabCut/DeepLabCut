(dlc3-architectures)=
# DeepLabCut 3.0 - PyTorch Model Architectures

## Introduction

You can see a list of supported architectures/variants by using:

```python
from deeplabcut.pose_estimation_pytorch import available_models
print(available_models())
```

## Backbones (neural networks)

Several  backbones are currently implemented in DeepLabCut PyTorch (more will come, and you can add more easily in our new model registry).

**ResNets**
- Adapted from [He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2016.](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) and [Insafutdinov, Eldar et al. "DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model". European Conference on Computer Vision (ECCV) 2016.]
- Current variants are `resnet_50`, `resnet_101`, `top_down_resnet_101`, `top_down_resnet_50`

**HRNet**
- Adapted from [Wang, Jingdong, et al. "Deep high-resolution representation learning for visual recognition." IEEE transactions on pattern analysis and machine intelligence 43.10 (2020): 3349-3364.](https://arxiv.org/abs/1908.07919)
- Current variants are `hrnet_w18`, `hrnet_w32`, `hrnet_w48`, `top_down_hrnet_w18`, `top_down_hrnet_w32`, `top_down_hrnet_w48`
- Slower but typically more powerful than ResNets

**DEKR**
- Adapted from [Geng, Zigang et al. "Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression." Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2021.](https://openaccess.thecvf.com/content/CVPR2021/papers/Geng_Bottom-Up_Human_Pose_Estimation_via_Disentangled_Keypoint_Regression_CVPR_2021_paper.pdf)
- This model uses HRNet as a backbone. It learns to predict the center of each animal, and predicts the offset between each animal center and their keypoints
- Current variants that are implemented (from smallest to largest): `dekr_w18`, `dekr_w32`, `dekr_w48`
- Note, this is a powerful multi-animal model but very heavy (slow)

**BUTCTD**
- Adapted from [Zhou, Stoffl, Mathis, Mathis. "Rethinking Pose Estimation in Crowds: Overcoming the Detection Information Bottleneck and Ambiguity." Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_Rethinking_Pose_Estimation_in_Crowds_Overcoming_the_Detection_Information_Bottleneck_ICCV_2023_paper.pdf)
- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-pose-estimation-in-crowds/pose-estimation-on-crowdpose)](https://paperswithcode.com/sota/pose-estimation-on-crowdpose?p=rethinking-pose-estimation-in-crowds)
- Current variants are `BUCTD-hrnet_w32` and `BUCTD-hrnet_w48`
- This is a top-performing mutli-animal (and for humans, which are also animals) method that can be used with other architectures

**DLCRNet**
- From [Lauer, Zhou, et al. "Multi-animal pose estimation, identification and tracking with DeepLabCut." Nature Methods 19.4 (2022): 496-504.](https://www.nature.com/articles/s41592-022-01443-0)
- This model uses a multi-scale variant of a ResNet as a backbone, and part-affinity fields to assemble individuals
- Variants: `dlcrnet_stride16_ms5`, `dlcrnet_stride32_ms5`


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

## Information on Multi-Animal Models

### Backbones with Part-Affinity Fields

As in DeepLabCut 2.X, the base multi-animal model is composed of a backbone (encoder) 
and a head predicting keypoints and part-affinity fields (PAFs). These PAFs are used to 
assemble keypoints for individuals.

Passing a backbone as a net type (e.g., `resnet_50`, `hrnet_w32`) for a multi-animal 
project will create a model consisting of a backbone and a heatmap + PAF head.

### Top-Down Models

Top-down pose estimation models split the task into two distinct parts: individual 
localization (through an object detector), followed by pose estimation (for each 
individual). As localization of individuals is handled by the detector, this simplifies
the pose task to single-animal pose estimation!

Hence any single-animal model can be transformed into a top-down, multi-animal model. To
do so, simply prefix `top_down` to your single-animal model name. Currently only a 
single FasterRCNN variant is available as a detector. Other variants will be added soon! 

The pose model for top-down nets is simply the backbone followed by a single convolution
for pose estimation. It's also possible to add deconvolutional layers to top-down model
heads.

Example top-down models would be `top_down_resnet_50` and `top_down_hrnet_w32`.