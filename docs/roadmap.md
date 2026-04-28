---
deeplabcut:
  last_content_updated: '2025-02-28'
  last_metadata_updated: '2026-03-06'
  ignore: false
---

(dev-roadmap)=

## A development roadmap for DeepLabCut

📢 ⏳ 🚧

**General Enhancements:**

- [ ] DeepLabCut PyTorch & Model Zoo --> DLC 3.0 🔥
- [x] DLC-CookBook v0.1
- [x] DLC BLog for releases and user-highlights
- [x] New Docker containers into main repo / linked to Docker hub and repo(s)
- [ ] 3D >2 camera support --> better 3D in PyTorch version 🔥

**General NN Improvements:**

- [x] EfficientNet backbones added (currently SOTA on ImageNet). https://openaccess.thecvf.com/content/WACV2021/html/Mathis_Pretraining_Boosts_Out-of-Domain_Robustness_for_Pose_Estimation_WACV_2021_paper.html https://github.com/DeepLabCut/DeepLabCut/commit/96da2cacf837a9b84ecdeafb50dfb4a93b402f33
- [x] New multi-fusion multi-scale networks; DLCRNet_ms5
- [ ] BUCTD Integration, see ICCV 2023 paper at https://arxiv.org/abs/2306.07879

**deeplabcut 2.2: multi-animal pose estimation and tracking with DeepLabCut**

- [x] alpha testing complete (early May 2020)
- [x] beta release: 2.2.b5 on 5 / 22 / 20 :smile:
- [x] beta release: 2.2b8 released 9/2020 :smile:
- [x] beta release 2.2b9 (rolled into 2.1.9 --> candidate release, slotted for Oct 2020)
- [x] 2.2rc1
- [x] 2.2rc2
- [x] 2.2rc3
- [x] Manuscript Lauer et al 2021 https://www.biorxiv.org/content/10.1101/2021.04.30.442096v1
- [x] full 2.2 stable release

**real-time module with DEMO for how to set up on your camera system, integration with our [Camera Control Software]**(https://github.com/AdaptiveMotorControlLab/Camera_Control)

- [x] Integration with Bonsai completed! See: https://github.com/bonsai-rx/deeplabcut
- [x] Integration with Auto-pi-lot. See: https://auto-pi-lot.com/
- [x] DeepLabCut-live! released Aug 5th, 2020: preprint & code: https://www.biorxiv.org/content/10.1101/2020.08.04.236422v1
- [x] DeepLabCut-live! published in eLife

**DeepLabCut Model Zoo: a collection of pretrained models for plug-in-play DLC and community crowd-sourcing.**

- [x] BETA release with 2.1.8b0: https://www.mackenziemathislab.org/deeplabcut
- [x] full release with 2.1.8.1 https://www.mackenziemathislab.org/deeplabcut
- [x] Manuscript forthcoming! --> see arXiv https://arxiv.org/abs/2203.07436
- [x] new models added; horse, cheetah
- [x] TopView_Mouse model
- [x] Quadruped model
- [ ] contribution module
- [ ] PyTorch Model zoo code

**DeepLabCut GUI and DeepLabCut-core:**

- [x] to make DLC more modular, we will move core functions to https://github.com/DeepLabCut/DeepLabCut-core
- [x] DLC-core depreciated, and core is now simply `pip install deeplabcut` GUI is with `pip install deeplabcut[gui]`
- [x] new GUI for DeepLabCut; due to extended issues with wxPython, we will be moving to release a napari plugin https://github.com/napari/napari
- [x] New project management GUI
- [x] tensorflow 2.2 support in DeepLabCut-core: https://github.com/DeepLabCut/DeepLabCut/issues/601
- [x] DeepLabCut-Core to be depreciated; TF2 will go into main repo.
- [x] TF2 support while also maintaining TF1 support until 2022.
- [ ] Web-based GUI for labeling --> Colab training pipeline for users (full no-install DLC)
