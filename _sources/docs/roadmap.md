(dev-roadmap)=
## A development roadmap for DeepLabCut


📢 ⏳ 🚧

**General Enhancements:**
- [ ] DeepLabCut PyTorch & Model Zoo --> DLC 3.0 🔥
- [X] DLC-CookBook v0.1
- [X] DLC BLog for releases and user-highlights
- [X] New Docker containers into main repo / linked to Docker hub and repo(s)
- [ ] 3D >2 camera support --> better 3D in PyTorch version 🔥

**General NN Improvements:**
- [X] EfficientNet backbones added (currently SOTA on ImageNet). https://openaccess.thecvf.com/content/WACV2021/html/Mathis_Pretraining_Boosts_Out-of-Domain_Robustness_for_Pose_Estimation_WACV_2021_paper.html https://github.com/DeepLabCut/DeepLabCut/commit/96da2cacf837a9b84ecdeafb50dfb4a93b402f33
- [X] New multi-fusion multi-scale networks; DLCRNet_ms5
- [ ] BUCTD Integration, see ICCV 2023 paper at https://arxiv.org/abs/2306.07879

**deeplabcut 2.2: multi-animal pose estimation and tracking with DeepLabCut**
- [X] alpha testing complete (early May 2020)
- [X] beta release: 2.2.b5 on 5 / 22 / 20 :smile:
- [X] beta release: 2.2b8 released 9/2020 :smile:
- [X] beta release 2.2b9 (rolled into 2.1.9 --> candidate release, slotted for Oct 2020)
- [X] 2.2rc1
- [X] 2.2rc2
- [X] 2.2rc3
- [X] Manuscript Lauer et al 2021 https://www.biorxiv.org/content/10.1101/2021.04.30.442096v1
- [X] full 2.2 stable release

**real-time module with DEMO for how to set up on your camera system, integration with our [Camera Control Software]**(https://github.com/AdaptiveMotorControlLab/Camera_Control)
- [X] Integration with Bonsai completed! See: https://github.com/bonsai-rx/deeplabcut
- [X] Integration with Auto-pi-lot. See: https://auto-pi-lot.com/
- [X] DeepLabCut-live! released Aug 5th, 2020: preprint & code: https://www.biorxiv.org/content/10.1101/2020.08.04.236422v1
- [X] DeepLabCut-live! published in eLife

**DeepLabCut Model Zoo: a collection of pretrained models for plug-in-play DLC and community crowd-sourcing.**
- [X] BETA release with 2.1.8b0: https://www.mackenziemathislab.org/deeplabcut
- [X] full release with 2.1.8.1 https://www.mackenziemathislab.org/deeplabcut
- [X] Manuscript forthcoming! --> see arXiv https://arxiv.org/abs/2203.07436
- [X] new models added; horse, cheetah
- [X] TopView_Mouse model
- [X] Quadruped model
- [ ] contribution module
- [ ] PyTorch Model zoo code

**DeepLabCut GUI and DeepLabCut-core:**
- [X] to make DLC more modular, we will move core functions to https://github.com/DeepLabCut/DeepLabCut-core
- [X] DLC-core depreciated, and core is now simply `pip install deeplabcut` GUI is with `pip install deeplabcut[gui]`
- [X] new GUI for DeepLabCut; due to extended issues with wxPython, we will be moving to release a napari plugin https://github.com/napari/napari
- [X] New project management GUI
- [X] tensorflow 2.2 support in DeepLabCut-core: https://github.com/DeepLabCut/DeepLabCut/issues/601
- [X] DeepLabCut-Core to be depreciated; TF2 will go into main repo.
- [X] TF2 support while also maintaining TF1 support until 2022.
- [ ] Web-based GUI for labeling --> Colab training pipeline for users (full no-install DLC)
