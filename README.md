[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Image.sc forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&amp;url=https%3A%2F%2Fforum.image.sc%2Ftags%2Fdeeplabcut.json&amp;query=%24.topic_list.tags.0.topic_count&amp;colorB=brightgreen&amp;&amp;suffix=%20topics&amp;logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC)](https://forum.image.sc/tags/deeplabcut)
[![Krihelimeter](http://krihelinator.xyz/badge/AlexEMG/DeepLabCut)](http://krihelinator.xyz/repositories/AlexEMG/DeepLabCut)
[![GitHub stars](https://img.shields.io/github/stars/AlexEMG/DeepLabCut.svg?style=social&label=Star)](https://github.com/AlexEMG/DeepLabCut)
[![GitHub forks](https://img.shields.io/github/forks/AlexEMG/DeepLabCut.svg?style=social&label=Fork)](https://github.com/AlexEMG/DeepLabCut)

<p align="center">
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c489e83aa4a992e80059d8c/1548263081887/DLCheader.png?format=1000w" width="100%">
</p>

DeepLabCut is a toolbox for markerless pose estimation of animals performing various tasks. Originally, we demonstrated the capabilities for [trail tracking](https://vnmurthylab.org/), [reaching in mice](http://www.mousemotorlab.org/) and various Drosophila behaviors during egg-laying (see [Mathis et al.](https://www.nature.com/articles/s41593-018-0209-y) for details). There is, however, nothing specific that makes the toolbox only applicable to these tasks and/or species. The toolbox has also already been successfully applied (by us and others) to [rats](http://www.mousemotorlab.org/deeplabcut), humans, various fish species, bacteria, leeches, various robots, cheetahs, [mouse whiskers](http://www.mousemotorlab.org/deeplabcut) and [race horses](http://www.mousemotorlab.org/deeplabcut). This work utilizes the feature detectors (ResNets + readout layers) of one of the state-of-the-art algorithms for human pose estimation by Insafutdinov et al., called DeeperCut, which inspired the name for our toolbox (see references below).

VERSION 2.0: This is the **Python package** of [DeepLabCut](https://www.nature.com/articles/s41593-018-0209-y) that is released with our Protocols paper (in press, preprint [here](https://www.biorxiv.org/content/10.1101/476531v1)).
This package includes graphical user interfaces to label your data, and take you from data set creation to automatic behavioral analysis. It also introduces an active learning framework to efficiently use DeepLabCut on large experimental projects, and new data augmentation that improves network performance, especially in challenging cases (see [panel b](https://camo.githubusercontent.com/77c92f6b89d44ca758d815bdd7e801247437060b/68747470733a2f2f737461746963312e73717561726573706163652e636f6d2f7374617469632f3537663664353163396637343536366635356563663237312f742f3563336663316336373538643436393530636537656563372f313534373638323338333539352f636865657461682e706e673f666f726d61743d37353077)).

VERSION 1.0: The initial, Nature Neuroscience version of [DeepLabCut](https://www.nature.com/articles/s41593-018-0209-y) can be found in the history of git, or here: https://github.com/AlexEMG/DeepLabCut/releases/tag/1.11

<p align="center">
<img src="http://www.people.fas.harvard.edu/~amathis/dlc/MATHIS_2018_odortrail.gif" height="220">
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c3fbd0c898583417a040dfc/1547681053201/rat-grasp.gif?format=300w"  height="220">
<img src="http://www.people.fas.harvard.edu/~amathis/dlc/MATHIS_2018_fly.gif" height="220">
</p>

Please check out [www.mousemotorlab.org/deeplabcut](https://www.mousemotorlab.org/deeplabcut/) for more video demonstrations of automated tracking. Above: courtesy of the Murthy (mouse), Leventhal (rat), and Axel (fly) labs!


# [Installation](docs/installation.md) 

How to [install DeeplabCut](docs/installation.md)
# [The DeepLabCut Process](docs/UseOverviewGuide.md)

An overview of the pipeline and workflow for project management. Please also read the [user-guide preprint!](https://www.biorxiv.org/content/early/2018/11/24/476531)

<p align="center">
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5cca1d519b747a750d680de5/1556749676166/dlc_overview-01.png?format=1000w" width="95%">
</p>

# [DEMO the code](/examples)

We provide several Jupyter Notebooks: one that walks you through a demo dataset to test your installation, and another Notebook to run DeepLabCut from the begining on your own data. We also show you how to use the code in Docker, and on Google Colab. Please also read the [user-guide preprint!](https://www.biorxiv.org/content/early/2018/11/24/476531)

# News (and in the news):

- March 2019: DeepLabCut joined [twitter](https://twitter.com/deeplabcut)
- Jan 2019: We hosted workshops for DLC in Warsaw, Munich and Cambridge. The materials are available [here](https://github.com/AlexEMG/DeepLabCut-Workshop-Materials)
- Jan 2019: We joined the Image Source Forum for user help: [![Image.sc forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&amp;url=https%3A%2F%2Fforum.image.sc%2Ftags%2Fdeeplabcut.json&amp;query=%24.topic_list.tags.0.topic_count&amp;colorB=brightgreen&amp;&amp;suffix=%20topics&amp;logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC)](https://forum.image.sc/tags/deeplabcut)

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
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c3fbed74fa51acecd63deeb/1547681534736/MouseLocomotion_warren.gif?format=500w" width="30%">  
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c3fc1c6758d46950ce7eec7/1547682383595/cheetah.png?format=750w" width="75%">
</p>

## Code contributors:

[Alexander Mathis](https://github.com/AlexEMG), [Tanmay Nath](http://www.mousemotorlab.org/team), [Mackenzie Mathis](https://github.com/MMathisLab). The feature detector code is based on Eldar Insafutdinov's TensorFlow implementation of [DeeperCut](https://github.com/eldar/pose-tensorflow). DeepLabCut is an open-source tool and has benefited from suggestions and edits by many individuals including Richard Warren, Ronny Eichler, Jonas Rauber, Hao Wu, Federico Claudi, Gary Kane, Taiga Abe, and Jonny Saunders as well as the [contributors](https://github.com/AlexEMG/DeepLabCut/graphs/contributors). In particular, the authors thank Ronny Eichler for input on the modularized version. 

This is an actively developed package and we welcome community development and involvement.

## Support and help:

For **help and questions that don't fit a GitHub code issue,** we ask you to post them here: https://forum.image.sc/ [![Image.sc forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&amp;url=https%3A%2F%2Fforum.image.sc%2Ftags%2Fdeeplabcut.json&amp;query=%24.topic_list.tags.0.topic_count&amp;colorB=brightgreen&amp;&amp;suffix=%20topics&amp;logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC)](https://forum.image.sc/tags/deeplabcut)

## References:

If you use this code or data please [cite Mathis et al, 2018](https://www.nature.com/articles/s41593-018-0209-y) and if you use the Python package (DeepLabCut2.0) please also cite [Nath, Mathis et al, 2019](https://www.biorxiv.org/content/10.1101/476531v1)

Please check out the following references for more details:

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

Our open-access pre-prints:

    @article{mathis2018markerless,
        title={Markerless tracking of user-defined features with deep learning},
        author={Mathis, Alexander and Mamidanna, Pranav and Abe, Taiga and Cury, Kevin M and Murthy, Venkatesh N and Mathis, Mackenzie W and Bethge, Matthias},
        journal={arXiv preprint arXiv:1804.03142},
        year={2018}
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

This project is licensed under the GNU Lesser General Public License v3.0. Note that the software is provided "as is", without warranty of any kind, express or implied. If you use this code or data, please [cite us!](https://www.nature.com/articles/s41593-018-0209-y).
