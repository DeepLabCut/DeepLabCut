<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
![Python package](https://github.com/DeepLabCut/DeepLabCut/workflows/Python%20package/badge.svg)
[![PyPI version](https://badge.fury.io/py/deeplabcut.svg)](https://badge.fury.io/py/deeplabcut)
[![Downloads](https://static.pepy.tech/personalized-badge/deeplabcut?period=total&units=international_system&left_color=black&right_color=black&left_text=Downloads)](https://pepy.tech/project/deeplabcut)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/deeplabcut.svg?color=purple&label=PyPi)](https://pypistats.org/packages/deeplabcut)
[![GitHub stars](https://img.shields.io/github/stars/AlexEMG/DeepLabCut.svg?style=social&label=Star)](https://github.com/AlexEMG/DeepLabCut)

[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Image.sc forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&amp;url=https%3A%2F%2Fforum.image.sc%2Ftag%2Fdeeplabcut.json&amp;query=%24.topic_list.tags.0.topic_count&amp;colorB=brightgreen&amp;&amp;suffix=%20topics&amp;logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC)](https://forum.image.sc/tag/deeplabcut)
[![Gitter](https://badges.gitter.im/DeepLabCut/community.svg)](https://gitter.im/DeepLabCut/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Twitter Follow](https://img.shields.io/twitter/follow/DeepLabCut.svg?label=DeepLabCut&style=social)](https://twitter.com/DeepLabCut)
[![GitHub forks](https://img.shields.io/github/forks/AlexEMG/DeepLabCut.svg?style=social&label=Fork)](https://github.com/AlexEMG/DeepLabCut)

<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1609244670293-I4G1WXUK4R4ELSG20YWN/ke17ZwdGBToddI8pDm48kMSuf4e2vyXWwmoX99kS6G0UqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYxCRW4BPu10St3TBAUQYVKcB2zTWnNYNsbaeRctoqYAz_Z2hBxiefVgk3iqj7SAtv40MC358Orr9N-QLax1vImS/dlc_logo_whiteBkgnd.png?format=2500w" width="95%">
</p>
<p align="center">
    <a href="https://www.mousemotorlab.org/deeplabcut/">www.deeplabcut.org</a>
</p>

<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/5d90cb67886fb8184560c3ef/1569773279570-XQAKFA299I0YGMZI129U/ke17ZwdGBToddI8pDm48kAsfLZj9Z3cwME2xx-GVPktZw-zPPgdn4jUwVcJE1ZvWEtT5uBSRWt4vQZAgTJucoTqqXjS3CfNDSuuf31e0tVFduml12xzze87D3uxh5wWTU2EfgWtpp0j_eVfs7Ce7Qib8BodarTVrzIWCp72ioWw/MATHIS_2018_odortrail.gif?format=300w" height="150">


<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1534797521117-EIEUED03C68241QZ4KCK/ke17ZwdGBToddI8pDm48kAx9qLOWpcHWRGxWsJQSczRZw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZUJFbgE-7XRK3dMEBRBhUpwdr4GYy30vFzf31Oe7KAPZKkqgaiEgc5jBNdhZmDPlzxdkDSclo6ofuXZm6YCEhUo/MATHIS_2018_fly.gif?format=300w" height="150">

<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1588014385061-6HA5RSD4ETBFVSD8R547/ke17ZwdGBToddI8pDm48kLyW1liujvcx7mp5osyaH61Zw-zPPgdn4jUwVcJE1ZvWhcwhEtWJXoshNdA9f1qD7Sn5EHp5ffw70kWDv5IdoCwQFuuJNKK_dG37osgdmxm5k0Yq7uiJIG_-lMb52Fc-iw/ezgif.com-resize.gif?format=300w" height="150">

<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c3fbd0c898583417a040dfc/1547681053201/rat-grasp.gif?format=300w" height="150">
</p>

DeepLabCut is a toolbox for markerless pose estimation of animals performing various tasks. [Read a short development and application summary below](https://github.com/AlexEMG/DeepLabCut#why-use-deeplabcut). As long as you can see (label) what you want to track, you can use this toolbox, as it is animal and object agnostic.

**Latest updates:**

:purple_heart: DeepLabCut supports multi-animal pose estimation (BETA release, plese give us your feedback! `pip install deeplabcut==2.2b8`).

:purple_heart: We have a real-time package available! http://DLClive.deeplabcut.org

# [Installation: how to install DeepLabCut](docs/installation.md)

Very quick start: `pip install deeplabcut`
* you also need tensorflow and wxPython see [here](https://github.com/DeepLabCut/DeepLabCut/blob/master/conda-environments/README.md#creating-your-own-customized-conda-env-recommended-route-for-linux-ubuntu-centos-mint-etc)

# [Documentation: The DeepLabCut Process](docs/UseOverviewGuide.md)

An overview of the pipeline and workflow for project management. For a step-by-step user guide, please also read the [Nature Protocols paper](https://doi.org/10.1038/s41596-019-0176-0)!

For a deeper understanding and more resources for you to get started with Python and DeepLabCut, please check out our free online course! http://DLCcourse.deeplabcut.org

<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1609244903687-US1SN063QIFJS4BP4IJD/ke17ZwdGBToddI8pDm48kFG9xAYub2PPnmh56PTVg7gUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYxCRW4BPu10St3TBAUQYVKcAju5e7u9RZJEVbVQPZRu9xb_m-kUO2M3I1IeDqD4l8YcGqu2nZPx1UhKV8wc1ELN/dlc_overview_whitebkgrnd.png?format=2500w" width="95%">
</p>

# [DEMO the code](/examples)

We provide several Jupyter Notebooks: one that walks you through a demo dataset to test your installation, and another Notebook to run DeepLabCut from the beginning on your own data. We also show you how to use the code in Docker, and on Google Colab.

# Why use DeepLabCut?

In 2018, we demonstrated the capabilities for [trail tracking](https://vnmurthylab.org/), [reaching in mice](http://www.mousemotorlab.org/) and various Drosophila behaviors during egg-laying (see [Mathis et al.](https://www.nature.com/articles/s41593-018-0209-y) for details). There is, however, nothing specific that makes the toolbox only applicable to these tasks and/or species. The toolbox has already been successfully applied (by us and others) to [rats](http://www.mousemotorlab.org/deeplabcut), humans, various fish species, bacteria, leeches, various robots, cheetahs, [mouse whiskers](http://www.mousemotorlab.org/deeplabcut) and [race horses](http://www.mousemotorlab.org/deeplabcut). DeepLabCut utilizes the feature detectors (ResNets + readout layers) of one of the state-of-the-art algorithms for human pose estimation by Insafutdinov et al., called DeeperCut, which inspired the name for our toolbox (see references below). Furthermore, we have added faster variants with MobileNetV2 backbones (see [Pretraining boosts out-of-domain robustness for pose estimation](https://arxiv.org/abs/1909.11229)). Additionally, we have improved the inference speed and provided additional augmentation methods (via tensorpack and imgaug), and added real-time and mutli-animal support in a beta release (more to come ...)

<p align="center">
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c3e47258a922d548c483247/1547585339819/ErrorvsTrainingsetSize.png?format=750w" height="160">
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c3e469d8a922d548c4828fa/1547585194560/compressionrobustness.png?format=750w" height="160">
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c3fbed74fa51acecd63deeb/1547681534736/MouseLocomotion_warren.gif?format=500w" height="160">  
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c3fc1c6758d46950ce7eec7/1547682383595/cheetah.png?format=750w" height="160">
</p>

**Left:** Due to transfer learning it requires **little training data** for multiple, challenging behaviors (see [Mathis et al. 2018](https://www.nature.com/articles/s41593-018-0209-y) for details). **Mid Left:** The feature detectors are robust to video compression (see [Mathis/Warren](https://www.biorxiv.org/content/early/2018/10/30/457242) for details). **Mid Right:** It allows 3D pose estimation with a single network and camera (see [Mathis/Warren](https://www.biorxiv.org/content/early/2018/10/30/457242)). **Right:** It allows 3D pose estimation with a single network trained on data from multiple cameras together with standard triangulation methods (see [Nath* and Mathis* et al. 2019](https://doi.org/10.1038/s41596-019-0176-0)).

**DeepLabCut** is embedding in a larger open-source eco-system, providing behavioral tracking for neuroscience, ecology, medical, and technical applications. Moreover, many new tools are being actively developed. See [DLC-Utils](https://github.com/DeepLabCut/DLCutils) for some helper code.

<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1588292233203-FD1DVKAQYNV2TU91CO7R/ke17ZwdGBToddI8pDm48kIX24IsDPzy6M4KUaihfICJZw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZamWLI2zvYWH8K3-s_4yszcp2ryTI0HqTOaaUohrI8PIxtGUdkzp028KVNnpOijF3PweOM5su6FUQHO6Wkh72Nw/dlc_eco.gif?format=1000w" width="80%">
</p>

## Code contributors:

DLC code was originally developed by [Alexander Mathis](https://github.com/AlexEMG) & [Mackenzie Mathis](https://github.com/MMathisLab), and was extended in 2.0 with [Tanmay Nath](http://www.mousemotorlab.org/team), and currently actively developed with [Jessy Lauer](https://github.com/jeylau). The feature detector code is based on Eldar Insafutdinov's TensorFlow implementation of [DeeperCut](https://github.com/eldar/pose-tensorflow). DeepLabCut is an open-source tool and has benefited from suggestions and edits by many individuals including  Mert Yuksekgonul, Tom Biasi, Richard Warren, Ronny Eichler, Hao Wu, Federico Claudi, Gary Kane and Jonny Saunders as well as the [contributors](https://github.com/AlexEMG/DeepLabCut/graphs/contributors). Please see [AUTHORS](https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS) for more details!

This is an actively developed package and we welcome community development and involvement.

## Community Support, Developers, & Help:

- We are a community partner on the [![Image.sc forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&amp;url=https%3A%2F%2Fforum.image.sc%2Ftag%2Fdeeplabcut.json&amp;query=%24.topic_list.tags.0.topic_count&amp;colorB=brightgreen&amp;&amp;suffix=%20topics&amp;logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC)](https://forum.image.sc/tag/deeplabcut). Please post help and support questions on the forum with the tag DeepLabCut. Check out their mission statement [Scientific Community Image Forum: A discussion forum for scientific image software](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000340).

- If you encounter a previously unreported bug/code issue, please post here (we encourage you to search issues first): https://github.com/DeepLabCut/DeepLabCut/issues

- For quick discussions amongst users, please see here: [![Gitter](https://badges.gitter.im/DeepLabCut/community.svg)](https://gitter.im/DeepLabCut/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

- If you want to contribute to the code, please read our guide [here!](CONTRIBUTING.md)

- The project [road map](docs/roadmap.md). Get in touch with us if you want to help!

## References:

If you use this code or data we kindly as that you please [cite Mathis et al, 2018](https://www.nature.com/articles/s41593-018-0209-y) and, if you use the Python package (DeepLabCut2.x) please also cite [Nath, Mathis et al, 2019](https://doi.org/10.1038/s41596-019-0176-0). If you utilize the MobileNetV2s or EfficientNets please cite [Mathis, Biasi et al. 2020](https://arxiv.org/abs/1909.11229).

DOIs (#ProTip, for helping you find citations for software, check out [CiteAs.org](http://citeas.org/)!):

- Mathis et al 2018: [10.1038/s41593-018-0209-y](https://doi.org/10.1038/s41593-018-0209-y)
- Nath, Mathis et al 2019: [10.1038/s41596-019-0176-0](https://doi.org/10.1038/s41596-019-0176-0)

Please check out the following references for more details:

    @article{Mathisetal2018,
        title={DeepLabCut: markerless pose estimation of user-defined body parts with deep learning},
        author = {Alexander Mathis and Pranav Mamidanna and Kevin M. Cury and Taiga Abe  and Venkatesh N. Murthy and Mackenzie W. Mathis and Matthias Bethge},
        journal={Nature Neuroscience},
        year={2018},
        url={https://www.nature.com/articles/s41593-018-0209-y}}

     @article{NathMathisetal2019,
        title={Using DeepLabCut for 3D markerless pose estimation across species and behaviors},
        author = {Nath*, Tanmay and Mathis*, Alexander and Chen, An Chi and Patel, Amir and Bethge, Matthias and Mathis, Mackenzie W},
        journal={Nature Protocols},
        year={2019},
        url={https://doi.org/10.1038/s41596-019-0176-0}}

    @article{insafutdinov2016eccv,
        title = {DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model},
        author = {Eldar Insafutdinov and Leonid Pishchulin and Bjoern Andres and Mykhaylo Andriluka and Bernt Schiele},
        booktitle = {ECCV'16},
        url = {http://arxiv.org/abs/1605.03170}}

    @article{Mathis2020DeepLT,
        title={Deep learning tools for the measurement of animal behavior in neuroscience},
        author={Mackenzie W. Mathis and Alexander Mathis},
        journal={Current Opinion in Neurobiology},
        year={2020},
        volume={60},
        pages={1-11}}

    @article{Mathis2020Primer,
        title={A Primer on Motion Capture with Deep Learning: Principles, Pitfalls, and Perspectives},
        author={Alexander Mathis and Steffen Schneider and Jessy Lauer and Mackenzie W. Mathis},
        journal={Neuron},
        year={2020},
        volume={108},
        pages={44-65}}

Our open-access pre-prints:

    @misc{Mathis2019_pretraining,
        title={Pretraining boosts out-of-domain robustness for pose estimation},
        author={Alexander Mathis and Mert Y\"uksekg\"on\"ul and Byron Rogers and Matthias Bethge and Mackenzie W. Mathis},
        year={2019},
        eprint={1909.11229},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
        }

    @article{NathMathis2018,
        author = {Nath*, Tanmay and Mathis*, Alexander and Chen, An Chi and Patel, Amir and Bethge, Matthias and Mathis, Mackenzie W},
        title = {Using DeepLabCut for 3D markerless pose estimation across species and behaviors},
        year = {2018},
        doi = {10.1101/476531},
        publisher = {Cold Spring Harbor Laboratory},
        URL = {https://www.biorxiv.org/content/early/2018/11/24/476531},
        eprint = {https://www.biorxiv.org/content/early/2018/11/24/476531.full.pdf},
        journal = {bioRxiv}
        }

    @article{mathis2018markerless,
        title={Markerless tracking of user-defined features with deep learning},
        author={Mathis, Alexander and Mamidanna, Pranav and Abe, Taiga and Cury, Kevin M and Murthy, Venkatesh N and Mathis, Mackenzie W and Bethge, Matthias},
        journal={arXiv preprint arXiv:1804.03142},
        year={2018}
        }

    @article{MathisWarren2018speed,
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

This project is licensed under the GNU Lesser General Public License v3.0. Note that the software is provided "as is", without warranty of any kind, express or implied. If you use the code or data, please cite us!. Note, artwork and images are copyrighted; please do not take or use these images without written permission.

## Versions:

VERSION 2.2: Multi-animal pose estimation and tracking with DeepLabCut.

VERSION 2.0-2.1: This is the **Python package** of [DeepLabCut](https://www.nature.com/articles/s41593-018-0209-y) that was originally released with our [Nature Protocols](https://doi.org/10.1038/s41596-019-0176-0) paper (preprint [here](https://www.biorxiv.org/content/10.1101/476531v1)).
This package includes graphical user interfaces to label your data, and take you from data set creation to automatic behavioral analysis. It also introduces an active learning framework to efficiently use DeepLabCut on large experimental projects, and data augmentation tools that improve network performance, especially in challenging cases (see [panel b](https://camo.githubusercontent.com/77c92f6b89d44ca758d815bdd7e801247437060b/68747470733a2f2f737461746963312e73717561726573706163652e636f6d2f7374617469632f3537663664353163396637343536366635356563663237312f742f3563336663316336373538643436393530636537656563372f313534373638323338333539352f636865657461682e706e673f666f726d61743d37353077)).

VERSION 1.0: The initial, Nature Neuroscience version of [DeepLabCut](https://www.nature.com/articles/s41593-018-0209-y) can be found in the history of git, or here: https://github.com/AlexEMG/DeepLabCut/releases/tag/1.11

## News (and in the news):

- Jan 2021: [Pretraining boosts out-of-domain robustness for pose estimation](https://openaccess.thecvf.com/content/WACV2021/html/Mathis_Pretraining_Boosts_Out-of-Domain_Robustness_for_Pose_Estimation_WACV_2021_paper.html) published in the IEEE Winter Conference on Applications of Computer Vision. We also added EfficientNet backbones to DeepLabCut, those are best trained with cosine decay (see paper). To use them, just pass "efficientnet-b0" to "efficientnet-b6" when creating the trainingset!
- Dec 2020: We released a real-time package that allows for online pose estimation and real-time feedback. See [DLClive.deeplabcut.org](http://DLClive.deeplabcut.org).
- 5/22 2020: We released 2.2beta5. This beta release has some of the features of DeepLabCut 2.2, whose major goal is to integrate multi-animal pose estimation to DeepLabCut.
- Mar 2020: Inspired by suggestions we heard at this weeks CZI's Essential Open Source Software meeting in Berkeley, CA we updated our [docs](docs/UseOverviewGuide.md). Let us know what you think!
- Feb 2020: Our [review on animal pose estimation is published!](https://www.sciencedirect.com/science/article/pii/S0959438819301151)
- Nov 2019: DeepLabCut was recognized by the Chan Zuckerberg Initiative (CZI) with funding to support this project. Read more in the [Harvard Gazette](https://news.harvard.edu/gazette/story/newsplus/harvard-researchers-awarded-czi-open-source-award/), on [CZI's Essential Open Source Software for Science site](https://chanzuckerberg.com/eoss/proposals/) and in their [Medium post](https://medium.com/@cziscience/how-open-source-software-contributors-are-accelerating-biomedicine-1a5f50f6846a)
- Oct 2019: DLC 2.1 released with lots of updates. In particular, a Project Manager GUI, MobileNetsV2, and augmentation packages (Imgaug and Tensorpack). For detailed updates see [releases](https://github.com/AlexEMG/DeepLabCut/releases)
- Sept 2019: We published two preprints. One showing that [ImageNet pretraining contributes to robustness](https://arxiv.org/abs/1909.11229) and a [review on animal pose estimation](https://arxiv.org/abs/1909.13868). Check them out!
- Jun 2019: DLC 2.0.7 released with lots of updates. For updates see [releases](https://github.com/AlexEMG/DeepLabCut/releases)
- Feb 2019: DeepLabCut joined [twitter](https://twitter.com/deeplabcut) [![Twitter Follow](https://img.shields.io/twitter/follow/DeepLabCut.svg?label=DeepLabCut&style=social)](https://twitter.com/DeepLabCut)
- Jan 2019: We hosted workshops for DLC in Warsaw, Munich and Cambridge. The materials are available [here](https://github.com/AlexEMG/DeepLabCut-Workshop-Materials)
- Jan 2019: We joined the Image Source Forum for user help: [![Image.sc forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&amp;url=https%3A%2F%2Fforum.image.sc%2Ftag%2Fdeeplabcut.json&amp;query=%24.topic_list.tags.0.topic_count&amp;colorB=brightgreen&amp;&amp;suffix=%20topics&amp;logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC)](https://forum.image.sc/tag/deeplabcut)

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
