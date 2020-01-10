# DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS

The code in this folder is based of Eldar's DeeperCut code: https://github.com/eldar/pose-tensorflow

We adopted it to be integrated in the pip package of DeepLabCut, and among other things added additional networks (MobileNets), faster inference code, additional augmentation code.

Check out the following references for details:

@article{Mathisetal2018,
        title={DeepLabCut: markerless pose estimation of user-defined body parts with deep learning},
        author = {Alexander Mathis and Pranav Mamidanna and Kevin M. Cury and Taiga Abe  and Venkatesh N. Murthy and Mackenzie W. Mathis and Matthias Bethge},
        journal={Nature Neuroscience},
        year={2018},
        url={https://www.nature.com/articles/s41593-018-0209-y}
    }

@article{mathis2019pretraining,
    title={Pretraining boosts out-of-domain robustness for pose estimation},
    author={Alexander Mathis and Mert Yüksekgönül and Byron Rogers and Matthias Bethge and Mackenzie W. Mathis},
    year={2019},
    eprint={1909.11229},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@article{insafutdinov2016deepercut,
    author = {Eldar Insafutdinov and Leonid Pishchulin and Bjoern Andres and Mykhaylo Andriluka and Bernt Schiele},
    url = {http://arxiv.org/abs/1605.03170}
    title = {DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model},
    year = {2016}
}

@inproceedings{pishchulin16cvpr,
    title = {DeepCut: Joint Subset Partition and Labeling for Multi Person Pose Estimation},
    booktitle = {CVPR'16},
    url = {https://arxiv.org/abs/1511.06645},
    author = {Leonid Pishchulin and Eldar Insafutdinov and Siyu Tang and Bjoern Andres and Mykhaylo Andriluka and Peter Gehler and Bernt Schiele}
}

# License:

This project (DeepLabCut and DeeperCut) is licensed under the GNU Lesser General Public License v3.0.
