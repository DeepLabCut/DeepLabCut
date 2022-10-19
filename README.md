<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
![Python package](https://github.com/DeepLabCut/DeepLabCut/workflows/Python%20package/badge.svg)
[![PyPI version](https://badge.fury.io/py/deeplabcut.svg)](https://badge.fury.io/py/deeplabcut)
[![Downloads](https://pepy.tech/badge/deeplabcut)](https://pepy.tech/project/deeplabcut)
[![Downloads](https://pepy.tech/badge/deeplabcut/month)](https://pepy.tech/project/deeplabcut)
[![GitHub stars](https://img.shields.io/github/stars/DeepLabCut/DeepLabCut.svg?style=social&label=Star)](https://github.com/DeepLabCut/DeepLabCut)

[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Image.sc forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&amp;url=https%3A%2F%2Fforum.image.sc%2Ftag%2Fdeeplabcut.json&amp;query=%24.topic_list.tags.0.topic_count&amp;colorB=brightgreen&amp;&amp;suffix=%20topics&amp;logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC)](https://forum.image.sc/tag/deeplabcut)
[![Gitter](https://badges.gitter.im/DeepLabCut/community.svg)](https://gitter.im/DeepLabCut/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Twitter Follow](https://img.shields.io/twitter/follow/DeepLabCut.svg?label=DeepLabCut&style=social)](https://twitter.com/DeepLabCut)
[![GitHub forks](https://img.shields.io/github/forks/DeepLabCut/DeepLabCut.svg?style=social&label=Fork)](https://github.com/DeepLabCut/DeepLabCut)

<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1628250004229-KVYD7JJVHYEFDJ32L9VJ/DLClogo2021.jpg?format=1000w" width="95%">
</p>
<p align="center">
    <a href="https://www.mousemotorlab.org/deeplabcut/">www.deeplabcut.org</a>
</p>

<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/5d90cb67886fb8184560c3ef/1569773279570-XQAKFA299I0YGMZI129U/ke17ZwdGBToddI8pDm48kAsfLZj9Z3cwME2xx-GVPktZw-zPPgdn4jUwVcJE1ZvWEtT5uBSRWt4vQZAgTJucoTqqXjS3CfNDSuuf31e0tVFduml12xzze87D3uxh5wWTU2EfgWtpp0j_eVfs7Ce7Qib8BodarTVrzIWCp72ioWw/MATHIS_2018_odortrail.gif?format=300w" height="150">


<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1534797521117-EIEUED03C68241QZ4KCK/ke17ZwdGBToddI8pDm48kAx9qLOWpcHWRGxWsJQSczRZw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZUJFbgE-7XRK3dMEBRBhUpwdr4GYy30vFzf31Oe7KAPZKkqgaiEgc5jBNdhZmDPlzxdkDSclo6ofuXZm6YCEhUo/MATHIS_2018_fly.gif?format=300w" height="150">

<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1619609897110-TKSTWKEM6HTGXID9D489/ke17ZwdGBToddI8pDm48kAvjv6tW_eojYQmNU0ncbllZw-zPPgdn4jUwVcJE1ZvWEtT5uBSRWt4vQZAgTJucoTqqXjS3CfNDSuuf31e0tVHBSTXHtjUKlhRtWJ1Vo6l1B2bxJtByvWSjL6Vz3amc5yb8BodarTVrzIWCp72ioWw/triMouseDLC.gif?format=300w" height="150">

<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c3fbd0c898583417a040dfc/1547681053201/rat-grasp.gif?format=300w" height="150">
</p>

DeepLabCut is a toolbox for markerless pose estimation of animals performing various tasks. [Read a short development and application summary below](https://github.com/DeepLabCut/DeepLabCut#why-use-deeplabcut). As long as you can see (label) what you want to track, you can use this toolbox, as it is animal and object agnostic.

**Latest updates:**

:purple_heart: **DeepLabCut supports multi-animal pose estimation!** maDLC is out of beta/rc mode and beta is deprecated, thanks to the testers out there for feedback! Your labeled data will be backwards compatible, but not all other steps. Please see the [new `2.2+` releases](https://github.com/DeepLabCut/DeepLabCut/releases) for what's new & how to install it, please see our new [paper, Lauer et al 2022](https://www.nature.com/articles/s41592-022-01443-0), and the [new docs]( https://deeplabcut.github.io/DeepLabCut) on how to use it!

:purple_heart: We support mulit-animal re-identification, see [Lauer et al 2022](https://www.nature.com/articles/s41592-022-01443-0).

:purple_heart: We have a **real-time** package available! http://DLClive.deeplabcut.org

# [Installation: how to install DeepLabCut](https://deeplabcut.github.io/DeepLabCut/docs/installation.html)

Very quick start: `pip install "deeplabcut[gui]"` that includes all GUI functions, or `pip install deeplabcut` (headless version).
* We recommend using our conda file, see [here](https://github.com/DeepLabCut/DeepLabCut/blob/master/conda-environments/README.md) or the new [`deeplabcut-docker` package](https://github.com/DeepLabCut/DeepLabCut/tree/master/docker).

# [Documentation: The DeepLabCut Process](https://deeplabcut.github.io/DeepLabCut)

Our docs walk you through using DeepLabCut, and key API points. For an overview of the toolbox and workflow for project management, see our step-by-step at [Nature Protocols paper](https://doi.org/10.1038/s41596-019-0176-0).

For a deeper understanding and more resources for you to get started with Python and DeepLabCut, please check out our free online course! http://DLCcourse.deeplabcut.org

<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1609244903687-US1SN063QIFJS4BP4IJD/ke17ZwdGBToddI8pDm48kFG9xAYub2PPnmh56PTVg7gUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYxCRW4BPu10St3TBAUQYVKcAju5e7u9RZJEVbVQPZRu9xb_m-kUO2M3I1IeDqD4l8YcGqu2nZPx1UhKV8wc1ELN/dlc_overview_whitebkgrnd.png?format=2500w" width="95%">
</p>

# [DEMO the code](/examples)

We provide data and several Jupyter Notebooks: one that walks you through a demo dataset to test your installation, and another Notebook to run DeepLabCut from the beginning on your own data. We also show you how to use the code in Docker, and on Google Colab.

# Why use DeepLabCut?

In 2018, we demonstrated the capabilities for [trail tracking](https://vnmurthylab.org/), [reaching in mice](http://www.mousemotorlab.org/) and various Drosophila behaviors during egg-laying (see [Mathis et al.](https://www.nature.com/articles/s41593-018-0209-y) for details). There is, however, nothing specific that makes the toolbox only applicable to these tasks and/or species. The toolbox has already been successfully applied (by us and others) to [rats](http://www.mousemotorlab.org/deeplabcut), humans, various fish species, bacteria, leeches, various robots, cheetahs, [mouse whiskers](http://www.mousemotorlab.org/deeplabcut) and [race horses](http://www.mousemotorlab.org/deeplabcut). DeepLabCut utilized the feature detectors (ResNets + readout layers) of one of the state-of-the-art algorithms for human pose estimation by Insafutdinov et al., called DeeperCut, which inspired the name for our toolbox (see references below). Since this time, the package has changed substantially.  The code has been re-tooled and re-factored since 2.1+: We have added faster and higher performance variants with MobileNetV2s, EfficientNets, and our own DLCRNet backbones (see [Pretraining boosts out-of-domain robustness for pose estimation](https://arxiv.org/abs/1909.11229) and [Lauer et al 2021](https://www.biorxiv.org/content/10.1101/2021.04.30.442096v1)). Additionally, we have improved the inference speed and provided both additional and novel augmentation methods, added real-time, and multi-animal support. We currently provide state-of-the-art performance for animal pose estimation.

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

DLC code was originally developed by [Alexander Mathis](https://github.com/AlexEMG) & [Mackenzie Mathis](https://github.com/MMathisLab), and was extended in 2.0 with [Tanmay Nath](http://www.mousemotorlab.org/team), and currently (2.1+) actively developed with our CZI DLC Fellow, [Jessy Lauer](https://github.com/jeylau). DeepLabCut is an open-source tool and has benefited from suggestions and edits by many individuals including  Mert Yuksekgonul, Tom Biasi, Richard Warren, Ronny Eichler, Hao Wu, Federico Claudi, Gary Kane and Jonny Saunders as well as the [contributors](https://github.com/DeepLabCut/DeepLabCut/graphs/contributors). Please see [AUTHORS](https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS) for more details!

This is an actively developed package and we welcome community development and involvement.

## Be part of the DLC Community✨:

| 🚉 Platform                                                 | 🎯 Goal                                                                      | ⏱️ Estimated Response Time | 📢 Support Squad                        |
|------------------------------------------------------------|-----------------------------------------------------------------------------|---------------------------|----------------------------------------|
| [![Image.sc forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&amp;url=https%3A%2F%2Fforum.image.sc%2Ftag%2Fdeeplabcut.json&amp;query=%24.topic_list.tags.0.topic_count&amp;colorB=brightgreen&amp;&amp;suffix=%20topics&amp;logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC)](https://forum.image.sc/tag/deeplabcut) <br /> 🐭Tag: DeepLabCut | To ask help and support questions👋                                          | Promptly🔥                 | DLC Team and The DLC Community |
| GitHub DeepLabCut/[Issues](https://github.com/DeepLabCut/DeepLabCut/issues)                                  | To report bugs and code issues🐛   (we encourage you to search issues first) | 2-3 days                  | DLC Team                               |
|[![Gitter](https://badges.gitter.im/DeepLabCut/community.svg)](https://gitter.im/DeepLabCut/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)                               | To discuss with other users,  share ideas and collaborate💡                  | 2 days                    | The DLC Community                      |
| GitHub DeepLabCut/[Contributing](https://github.com/DeepLabCut/DeepLabCut/blob/master/CONTRIBUTING.md)                          | To contribute your expertise and experience🙏💯                               | Promptly🔥                 | DLC Team                               |
| 🚧 GitHub DeepLabCut/[Roadmap](https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/roadmap.md)                             | To learn more about our journey✈️                                            | N/A                       | N/A                                    |
| [![Twitter Follow](https://img.shields.io/twitter/follow/DeepLabCut.svg?label=DeepLabCut&style=social)](https://twitter.com/DeepLabCut)                                                   | To keep up with our latest news and updates 📢                               | Daily                     | DLC Team                               |
| The DeepLabCut [AI Residency Program](https://www.deeplabcutairesidency.org/)                        | To come and work with us next summer👏                                       | Annually                  | DLC Team                               |


## References:

If you use this code or data we kindly as that you please [cite Mathis et al, 2018](https://www.nature.com/articles/s41593-018-0209-y) and, if you use the Python package (DeepLabCut2.x) please also cite [Nath, Mathis et al, 2019](https://doi.org/10.1038/s41596-019-0176-0). If you utilize the MobileNetV2s or EfficientNets please cite [Mathis, Biasi et al. 2021](https://openaccess.thecvf.com/content/WACV2021/papers/Mathis_Pretraining_Boosts_Out-of-Domain_Robustness_for_Pose_Estimation_WACV_2021_paper.pdf). If you use multi-animal in beta mode, please contact us; if you use the 2.2rc1+, please cite [Lauer et al. 2021](https://www.biorxiv.org/content/10.1101/2021.04.30.442096v1).

DOIs (#ProTip, for helping you find citations for software, check out [CiteAs.org](http://citeas.org/)!):

- Mathis et al 2018: [10.1038/s41593-018-0209-y](https://doi.org/10.1038/s41593-018-0209-y)
- Nath, Mathis et al 2019: [10.1038/s41596-019-0176-0](https://doi.org/10.1038/s41596-019-0176-0)


Please check out the following references for more details:

    @article{Mathisetal2018,
        title = {DeepLabCut: markerless pose estimation of user-defined body parts with deep learning},
        author = {Alexander Mathis and Pranav Mamidanna and Kevin M. Cury and Taiga Abe  and Venkatesh N. Murthy and Mackenzie W. Mathis and Matthias Bethge},
        journal = {Nature Neuroscience},
        year = {2018},
        url = {https://www.nature.com/articles/s41593-018-0209-y}}

     @article{NathMathisetal2019,
        title = {Using DeepLabCut for 3D markerless pose estimation across species and behaviors},
        author = {Nath*, Tanmay and Mathis*, Alexander and Chen, An Chi and Patel, Amir and Bethge, Matthias and Mathis, Mackenzie W},
        journal = {Nature Protocols},
        year = {2019},
        url = {https://doi.org/10.1038/s41596-019-0176-0}}
        
    @InProceedings{Mathis_2021_WACV,
        author    = {Mathis, Alexander and Biasi, Thomas and Schneider, Steffen and Yuksekgonul, Mert and Rogers, Byron and Bethge, Matthias and Mathis, Mackenzie W.},
        title     = {Pretraining Boosts Out-of-Domain Robustness for Pose Estimation},
        booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
        month     = {January},
        year      = {2021},
        pages     = {1859-1868}}
        
    @article{Lauer2022MultianimalPE,
        title={Multi-animal pose estimation, identification and tracking with DeepLabCut},
        author={Jessy Lauer and Mu Zhou and Shaokai Ye and William Menegas and Steffen Schneider and Tanmay Nath and Mohammed Mostafizur Rahman and     Valentina Di Santo and Daniel Soberanes and Guoping Feng and Venkatesh N. Murthy and George Lauder and Catherine Dulac and M. Mathis and Alexander Mathis},
        journal={Nature Methods},
        year={2022},
        volume={19},
        pages={496 - 504}}

    @article{insafutdinov2016eccv,
        title = {DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model},
        author = {Eldar Insafutdinov and Leonid Pishchulin and Bjoern Andres and Mykhaylo Andriluka and Bernt Schiele},
        booktitle = {ECCV'16},
        url = {http://arxiv.org/abs/1605.03170}}
        
Review articles:

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

Other open-access pre-prints related to our work on DeepLabCut:

    @article{MathisWarren2018speed,
        author = {Mathis, Alexander and Warren, Richard A.},
        title = {On the inference speed and video-compression robustness of DeepLabCut},
        year = {2018},
        doi = {10.1101/457242},
        publisher = {Cold Spring Harbor Laboratory},
        URL = {https://www.biorxiv.org/content/early/2018/10/30/457242},
        eprint = {https://www.biorxiv.org/content/early/2018/10/30/457242.full.pdf},
        journal = {bioRxiv}}

## License:

This project is licensed under the GNU Lesser General Public License v3.0. Note that the software is provided "as is", without warranty of any kind, express or implied. If you use the code or data, please cite us! Note, artwork (DeepLabCut logo) and images are copyrighted; please do not take or use these images without written permission.

## Versions:

VERSION 2.2: Multi-animal pose estimation and tracking with DeepLabCut is supported (as well as single-animal projects).

VERSION 2.0-2.1: This is the **Python package** of [DeepLabCut](https://www.nature.com/articles/s41593-018-0209-y) that was originally released with our [Nature Protocols](https://doi.org/10.1038/s41596-019-0176-0) paper (preprint [here](https://www.biorxiv.org/content/10.1101/476531v1)).
This package includes graphical user interfaces to label your data, and take you from data set creation to automatic behavioral analysis. It also introduces an active learning framework to efficiently use DeepLabCut on large experimental projects, and data augmentation tools that improve network performance, especially in challenging cases (see [panel b](https://camo.githubusercontent.com/77c92f6b89d44ca758d815bdd7e801247437060b/68747470733a2f2f737461746963312e73717561726573706163652e636f6d2f7374617469632f3537663664353163396637343536366635356563663237312f742f3563336663316336373538643436393530636537656563372f313534373638323338333539352f636865657461682e706e673f666f726d61743d37353077)).

VERSION 1.0: The initial, Nature Neuroscience version of [DeepLabCut](https://www.nature.com/articles/s41593-018-0209-y) can be found in the history of git, or here: https://github.com/DeepLabCut/DeepLabCut/releases/tag/1.11

## News (and in the news):

- August 2021: 2.2 becomes the new stable release for DeepLabCut.

- July 2021: Docs are now at https://deeplabcut.github.io/DeepLabCut and we now include TensorFlow 2 support!

- May 2021: DeepLabCut hit 200,000 downloads! Also, Our preprint on 2.2, multi-animal DeepLabCut is released!

- Jan 2021: [Pretraining boosts out-of-domain robustness for pose estimation](https://openaccess.thecvf.com/content/WACV2021/html/Mathis_Pretraining_Boosts_Out-of-Domain_Robustness_for_Pose_Estimation_WACV_2021_paper.html) published in the IEEE Winter Conference on Applications of Computer Vision. We also added EfficientNet backbones to DeepLabCut, those are best trained with cosine decay (see paper). To use them, just pass "`efficientnet-b0`" to "`efficientnet-b6`" when creating the trainingset!
- Dec 2020: We released a real-time package that allows for online pose estimation and real-time feedback. See [DLClive.deeplabcut.org](http://DLClive.deeplabcut.org).
- 5/22 2020: We released 2.2beta5. This beta release has some of the features of DeepLabCut 2.2, whose major goal is to integrate multi-animal pose estimation to DeepLabCut.
- Mar 2020: Inspired by suggestions we heard at this weeks CZI's Essential Open Source Software meeting in Berkeley, CA we updated our [docs](docs/UseOverviewGuide.md). Let us know what you think!
- Feb 2020: Our [review on animal pose estimation is published!](https://www.sciencedirect.com/science/article/pii/S0959438819301151)
- Nov 2019: DeepLabCut was recognized by the Chan Zuckerberg Initiative (CZI) with funding to support this project. Read more in the [Harvard Gazette](https://news.harvard.edu/gazette/story/newsplus/harvard-researchers-awarded-czi-open-source-award/), on [CZI's Essential Open Source Software for Science site](https://chanzuckerberg.com/eoss/proposals/) and in their [Medium post](https://medium.com/@cziscience/how-open-source-software-contributors-are-accelerating-biomedicine-1a5f50f6846a)
- Oct 2019: DLC 2.1 released with lots of updates. In particular, a Project Manager GUI, MobileNetsV2, and augmentation packages (Imgaug and Tensorpack). For detailed updates see [releases](https://github.com/DeepLabCut/DeepLabCut/releases)
- Sept 2019: We published two preprints. One showing that [ImageNet pretraining contributes to robustness](https://arxiv.org/abs/1909.11229) and a [review on animal pose estimation](https://arxiv.org/abs/1909.13868). Check them out!
- Jun 2019: DLC 2.0.7 released with lots of updates. For updates see [releases](https://github.com/DeepLabCut/DeepLabCut/releases)
- Feb 2019: DeepLabCut joined [twitter](https://twitter.com/deeplabcut) [![Twitter Follow](https://img.shields.io/twitter/follow/DeepLabCut.svg?label=DeepLabCut&style=social)](https://twitter.com/DeepLabCut)
- Jan 2019: We hosted workshops for DLC in Warsaw, Munich and Cambridge. The materials are available [here](https://github.com/DeepLabCut/DeepLabCut-Workshop-Materials)
- Jan 2019: We joined the Image Source Forum for user help: [![Image.sc forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&amp;url=https%3A%2F%2Fforum.image.sc%2Ftag%2Fdeeplabcut.json&amp;query=%24.topic_list.tags.0.topic_count&amp;colorB=brightgreen&amp;&amp;suffix=%20topics&amp;logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC)](https://forum.image.sc/tag/deeplabcut)

- Nov 2018: We posted a detailed guide for DeepLabCut 2.0 on [BioRxiv](https://www.biorxiv.org/content/early/2018/11/24/476531). It also contains a case study for 3D pose estimation in cheetahs.
- Nov 2018: Various (post-hoc) analysis scripts contributed by users (and us) will be gathered at [DLCutils](https://github.com/DeepLabCut/DLCutils). Feel free to contribute! In particular, there is a script guiding you through
importing a project into the new data format for DLC 2.0
- Oct 2018: new pre-print on the speed video-compression and robustness of DeepLabCut on [BioRxiv](https://www.biorxiv.org/content/early/2018/10/30/457242)
- Sept 2018: Nature Lab Animal covers DeepLabCut: [Behavior tracking cuts deep](https://www.nature.com/articles/s41684-018-0164-y)
- Kunlin Wei & Konrad Kording write a very nice News & Views on our paper: [Behavioral Tracking Gets Real](https://www.nature.com/articles/s41593-018-0215-0)
- August 2018: Our [preprint](https://arxiv.org/abs/1804.03142) appeared in [Nature Neuroscience](https://www.nature.com/articles/s41593-018-0209-y)
- August 2018: NVIDIA AI Developer News: [AI Enables Markerless Animal Tracking](https://news.developer.nvidia.com/ai-enables-markerless-animal-tracking/)
- July 2018: Ed Yong covered DeepLabCut and interviewed several users for the [Atlantic](https://www.theatlantic.com/science/archive/2018/07/deeplabcut-tracking-animal-movements/564338).
- April 2018: first DeepLabCut preprint on [arXiv.org](https://arxiv.org/abs/1804.03142)
