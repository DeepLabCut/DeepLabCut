[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Krihelimeter](http://krihelinator.xyz/badge/AlexEMG/DeepLabCut)](http://krihelinator.xyz/repositories/AlexEMG/DeepLabCut)
[![GitHub stars](https://img.shields.io/github/stars/AlexEMG/DeepLabCut.svg?style=social&label=Star)](https://github.com/AlexEMG/DeepLabCut)
[![GitHub forks](https://img.shields.io/github/forks/AlexEMG/DeepLabCut.svg?style=social&label=Fork)](https://github.com/AlexEMG/DeepLabCut)

## DeepLabCut

DeepLabCut is a toolbox for markerless pose estimation of animals performing various tasks, like [trail tracking](https://vnmurthylab.org/), [reaching in mice](http://www.mousemotorlab.org/) and various Drosophila behaviors during egg-laying (see [Mathis et al.](https://www.nature.com/articles/s41593-018-0209-y) for details). There is, however, nothing specific that makes the toolbox only applicable to these tasks and/or species. The toolbox has also already been successfully applied (by us and others) to [rats](http://www.mousemotorlab.org/deeplabcut), humans, various fish species, bacteria, leeches, various robots, cheetahs, [mouse whiskers](http://www.mousemotorlab.org/deeplabcut) and [race horses](http://www.mousemotorlab.org/deeplabcut). This work utilizes the feature detectors (ResNets + readout layers) of one of the state-of-the-art algorithms for human pose estimation by Insafutdinov et al., called DeeperCut, which inspired the name for our toolbox (see references below).

VERSION 2.0: This is the **python package** of [DeepLabCut](https://www.nature.com/articles/s41593-018-0209-y).
This package includes graphical user interfaces to label your data, and take you from data set creation to automatic behavioral analysis. It also introduces an active learning framework to efficiently use DeepLabCut on large experimental projects.

VERSION 1.0: The initial, Nature Neuroscience version of **DeepLabCut** can be found in the history of git, or the latest version here: https://github.com/AlexEMG/DeepLabCut/releases/tag/1.11

<p align="center">
<img src="http://www.people.fas.harvard.edu/~amathis/dlc/MATHIS_2018_odortrail.gif" height="220">
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/5b0ca2b503ce64b895556b67/5b0ca8cd6d2a73781c50cf78/1527556316005/R0228_20180207_11-33-42_011_direct.gif?format=300w"  height="220">
<img src="http://www.people.fas.harvard.edu/~amathis/dlc/MATHIS_2018_fly.gif" height="220">
</p>

Please check out [www.mousemotorlab.org/deeplabcut](https://www.mousemotorlab.org/deeplabcut/) for more video demonstrations of automated tracking. Above: courtesy of the Murthy (mouse), Leventhal (rat), and Axel (fly) labs!


# [Installation](docs/installation.md)
# [Overview of the work-flow and how to use DeepLabCut](docs/UseOverviewGuide.md)

<p align="center">
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c3e47454fa51a420fa8ecdf/1547585367234/flowfig.png?format=750w" width="50%">
</p>

# [DEMO the code](/examples)
We provide several Jupyter Notebooks: one that walks you through a demo dataset to test your installation, and another Notebook to run DeepLabCut from the begining on your own data. We also show you how to use the code in Docker, and on Google Colab. Please also read the [user-guide](https://www.biorxiv.org/content/early/2018/11/24/476531).

# News (and in the news):

- Dec 2018: There will be a workshop on DeepLabCut at the Rowland Institute, Harvard University in January 2019! [Sign up until Dec 28](https://twitter.com/TrackingActions/status/1067253667315806211/photo/1)
- Nov 2018: We posted a detailed guide for DeepLabCut 2.0 on [BioRxiv](https://www.biorxiv.org/content/early/2018/11/24/476531). It also contains a case study for 3D pose estimation in cheetahs.
- Nov 2018: Various (post-hoc) analysis scripts contributed by users (and us) will be gathered at [DLCutils](https://github.com/AlexEMG/DLCutils). Feel free to contribute! In particular, there is a script guiding you through
importing a project into the new data format for DLC 2.0
- Oct 2018: new pre-print on the speed video-compression and robustness of DeepLabCut on [BioRxiv](https://www.biorxiv.org/content/early/2018/10/30/457242)
- Sept 2018: Nature Lab Animal covers DeepLabCut: [Behavior tracking cuts deep](https://www.nature.com/articles/s41684-018-0164-y)
- Kunlin Wei & Konrad Kording write a very nice News & Views on our paper: [Behavioral Tracking Gets Real](https://www.nature.com/articles/s41593-018-0215-0)
- August 2018: Our [preprint](https://arxiv.org/abs/1804.03142) appeared in [Nature Neuroscience](https://www.nature.com/articles/s41593-018-0209-y)
- August 2018: NVIDIA AI Developer News: [AI Enables Markerless Animal Tracking](https://news.developer.nvidia.com/ai-enables-markerless-animal-tracking/)
- July 2018: Ed Yong covered DeepLabCut and interviewed several users for the [Atlantic](https://www.theatlantic.com/science/archive/2018/07/deeplabcut-tracking-animal-movements/564338).
- April 2018: first DeepLabCut preprint on [arXiv.org](https://arxiv.org/abs/1804.03142)

# Why use DeepLabCut?

- Top left: Due to transfer learning it requires **little training data** for multiple, challenging behaviors (see [Mathis et al.](https://www.nature.com/articles/s41593-018-0209-y) for details).

- Top Right: Video anlysis is fast (see [Mathis/Warren](https://www.biorxiv.org/content/early/2018/10/30/457242) for details)

- Mid Left: The feature detectors are robust to video compression (see [Mathis/Warren](https://www.biorxiv.org/content/early/2018/10/30/457242) for details)

- Mid Right: It allows 3D pose estimation with a single network and camera (see [Mathis/Warren](https://www.biorxiv.org/content/early/2018/10/30/457242) for details)

- Bottom: It allows 3D pose estimation with a single network trained on data from multple cameras together with standard triangulation methods (see [Nath* and Mathis* et al.](https://www.biorxiv.org/content/early/2018/11/24/476531) for details)

<p align="center">
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c3e47258a922d548c483247/1547585339819/ErrorvsTrainingsetSize.png?format=750w" width="50%">
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c3e477170a6adda176dcaa4/1547585409727/inferencespeed.png?format=500w" width="30%">  
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c3e469d8a922d548c4828fa/1547585194560/compressionrobustness.png?format=750w" width="40%">
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5b1aa6a088251b08dc647c0b/1528473288768/MouseLocomotion_warren.gif?format=500w" width="30%">  
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c3fc1c6758d46950ce7eec7/1547682383595/cheetah.png?format=750w" width="75%">
</p>

## Code contributors:

[Alexander Mathis](https://github.com/AlexEMG), [Tanmay Nath](http://www.mousemotorlab.org/team), [Mackenzie Mathis](https://github.com/MMathisLab), and especially the authors of DeeperCut authors for the feature detector code. The feature detector code is based on Eldar Insafutdinov's TensorFlow implementation of [DeeperCut](https://github.com/eldar/pose-tensorflow). DeepLabCut is an open-source tool and has benefited from suggestions and edits by many individuals including Richard Warren, Ronny Eichler, Jonas Rauber, Hao Wu, Federico Claudi, Taiga Abe, and Jonny Saunders as well as the [contributors](https://github.com/AlexEMG/DeepLabCut/graphs/contributors). In particular, the authors thank Ronny Eichler for input on the modularized version. We are also grateful to all the beta testers!

This is an actively developed package and we welcome community development and involvement!

For **help and questions that don't fit a GitHub code issue,** we ask you to post in https://forum.image.sc/

If you would like to join the [development community](https://deeplabcut.slack.com), please drop us a note to be invited by emailing: mackenzie@post.harvard.edu

Please check out the following references for more details:

## References:

    @article{Mathisetal2018,
        title={DeepLabCut: markerless pose estimation of user-defined body parts with deep learning},
        author = {Alexander Mathis and Pranav Mamidanna and Kevin M. Cury and Taiga Abe  and Venkatesh N. Murthy and Mackenzie W. Mathis and Matthias Bethge},
        journal={Nature Neuroscience},
        year={2018},
        url={https://www.nature.com/articles/s41593-018-0209-y}
    }

    @article{insafutdinov2016eccv,
        title = {DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model},
        author = {Eldar Insafutdinov and Leonid Pishchulin and Bjoern Andres and Mykhaylo Andriluka and Bernt Schiele},
        booktitle = {ECCV'16},
        url = {http://arxiv.org/abs/1605.03170}
    }

Our open-access pre-prints:

    @article{mathis2018markerless,
        title={Markerless tracking of user-defined features with deep learning},
        author={Mathis, Alexander and Mamidanna, Pranav and Abe, Taiga and Cury, Kevin M and Murthy, Venkatesh N and Mathis, Mackenzie W and Bethge, Matthias},
        journal={arXiv preprint arXiv:1804.03142},
        year={2018}
    }

    @article {NathMathis2018,
        author = {Nath*, Tanmay and Mathis*, Alexander and Chen, An Chi and Patel, Amir and Bethge, Matthias and Mathis, Mackenzie W},
        title = {Using DeepLabCut for 3D markerless pose estimation across species and behaviors},
        year = {2018},
        doi = {10.1101/476531},
        publisher = {Cold Spring Harbor Laboratory},
        URL = {https://www.biorxiv.org/content/early/2018/11/24/476531},
        eprint = {https://www.biorxiv.org/content/early/2018/11/24/476531.full.pdf},
        journal = {bioRxiv}
    }

    @article {MathisWarren2018speed,
        author = {Mathis, Alexander and Warren, Richard A.},
        title = {On the inference speed and video-compression robustness of DeepLabCut},
        year = {2018},
        doi = {10.1101/457242},
        publisher = {Cold Spring Harbor Laboratory},
        URL = {https://www.biorxiv.org/content/early/2018/10/30/457242},
        eprint = {https://www.biorxiv.org/content/early/2018/10/30/457242.full.pdf},
        journal = {bioRxiv}
    }

## License:

This project is licensed under the GNU Lesser General Public License v3.0. Note that the software is provided "as is", without warranty of any kind, express or implied. If you use this code, please [cite us!](https://www.nature.com/articles/s41593-018-0209-y).
