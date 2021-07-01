## A development roadmap for DeepLabCut


:loudspeaker: :hourglass_flowing_sand: :construction:

**deeplabcut 2.2: multi-animal pose estimation and tracking with DeepLabCut**
- [X] alpha testing complete (early May 2020)
- [X] beta release: 2.2.b5 on 5 / 22 / 20 :smile:
- [X] beta release: 2.2b8 released 9/2020 :smile:
- [X] beta release 2.2b9 (rolled into 2.1.9 --> candidate release, slotted for Oct 2020)
- [X] 2.2rc1
- [ ] 2.2rc2
- [ ] 2.2rc3
- [X] Manuscript Lauer et al 2021 https://www.biorxiv.org/content/10.1101/2021.04.30.442096v1
- [ ] full 2.2 stable release

**real-time module with DEMO for how to set up on your camera system, integration with our [Camera Control Software]**(https://github.com/AdaptiveMotorControlLab/Camera_Control)
- [X] Integration with Bonsai completed! See: https://github.com/bonsai-rx/deeplabcut
- [X] DeepLabCut-live! released Aug 5th, 2020: preprint & code: https://www.biorxiv.org/content/10.1101/2020.08.04.236422v1

**DeepLabCut Model Zoo: a collection of pretrained models for plug-in-play DLC and community crowd-sourcing.**
- [X] BETA release with 2.1.8b0: http://www.mousemotorlab.org/dlc-modelzoo
- [X] full release with 2.1.8.1 http://www.mousemotorlab.org/dlc-modelzoo
- [ ] Manuscript forthcoming!
- [X] new models added with 2.2b8!
- [ ] contribution module

**DeepLabCut GUI and DeepLabCut-core:**
- [X] to make DLC more modular, we will move core functions to https://github.com/DeepLabCut/DeepLabCut-core
- [ ] new GUI for DeepLabCut; due to extended issues with wxPython, we will be moving to napari https://github.com/napari/napari
- [X] tensorflow 2.2 support in DeepLabCut-core: https://github.com/DeepLabCut/DeepLabCut/issues/601
- [ ] DeepLabCut-Core to be depreciated; TF2 will go into main repo.
- [ ] Web-based GUI for labeling --> Colab training pipeline

**General Improvements:**
- [X] EfficientNet backbones added (currently SOTA on ImageNet). https://openaccess.thecvf.com/content/WACV2021/html/Mathis_Pretraining_Boosts_Out-of-Domain_Robustness_for_Pose_Estimation_WACV_2021_paper.html https://github.com/DeepLabCut/DeepLabCut/commit/96da2cacf837a9b84ecdeafb50dfb4a93b402f33
- [X] New multi-fusion multi-scale networks; DLCRNet_ms5

**General Enhancements:**
- [ ] DeepLabCut PyTorch model zoo
- [ ] DLC-CookBook v0.1
- [X] DLC BLog for releases and user-highlights
