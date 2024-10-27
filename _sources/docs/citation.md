# How to Cite DeepLabCut

Thank you for using DeepLabCut! Here are our recommendations for citing and documenting your use of DeepLabCut in your Methods section:


If you use this code or data we kindly ask that you please [cite Mathis et al, 2018](https://www.nature.com/articles/s41593-018-0209-y)
and, if you use the Python package (DeepLabCut2.x+) please also cite [Nath, Mathis et al, 2019](https://doi.org/10.1038/s41596-019-0176-0).
If you utilize the MobileNetV2s or EfficientNets please cite [Mathis, Biasi et al. 2021](https://openaccess.thecvf.com/content/WACV2021/papers/Mathis_Pretraining_Boosts_Out-of-Domain_Robustness_for_Pose_Estimation_WACV_2021_paper.pdf).
If you use multi-animal versions 2.2beta+ or 2.2rc1+, please cite [Lauer et al. 2022](https://www.nature.com/articles/s41592-022-01443-0).
If you use our SuperAnimal models, please cite [Ye et al. 2024](https://www.nature.com/articles/s41467-024-48792-2).

DOIs (#ProTip, for helping you find citations for software, check out [CiteAs.org](http://citeas.org/)!):

- Mathis et al 2018: [10.1038/s41593-018-0209-y](https://doi.org/10.1038/s41593-018-0209-y)
- Nath, Mathis et al 2019: [10.1038/s41596-019-0176-0](https://doi.org/10.1038/s41596-019-0176-0)
- Lauer et al 2022: [10.1038/s41592-022-01443-0](https://doi.org/10.1038/s41592-022-01443-0)
- Ye et al 2024: [10.1038/s41467-024-48792-2](https://www.nature.com/articles/s41467-024-48792-2)

## Formatted citations:

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

    @article{Ye2024SuperAnimal,
        title={SuperAnimal pretrained pose estimation models for behavioral analysis},
        author={Shaokai Ye and Anastasiia Filippova and Jessy Lauer and Steffen Schneider and Maxime Vidal and and Tian Qiu and Alexander Mathis and Mackenzie W. Mathis},
        journal={Nature Communications},
        year={2024},
        volume={15}}


### Review & Educational articles:

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

### Other open-access pre-prints related to our work on DeepLabCut:

    @article{MathisWarren2018speed,
        author = {Mathis, Alexander and Warren, Richard A.},
        title = {On the inference speed and video-compression robustness of DeepLabCut},
        year = {2018},
        doi = {10.1101/457242},
        publisher = {Cold Spring Harbor Laboratory},
        URL = {https://www.biorxiv.org/content/early/2018/10/30/457242},
        eprint = {https://www.biorxiv.org/content/early/2018/10/30/457242.full.pdf},
        journal = {bioRxiv}}



## Methods Suggestion:

For body part tracking we used DeepLabCut (version 2.X.X)* [Mathis et al, 2018, Nath et al, 2019, Lauer et al. 2022]. Specifically, we labeled X number of frames taken from X videos/animals (then X% was used for training (default is 95%). We used a X-based neural network (i.e. X = ResNet-50, ResNet-101, MobileNetV2-0.35, MobileNetV2-0.5, MobileNetV2-0.75, MobileNetV2-1***) with default parameters* for X number of training iterations. We validated with X number of shuffles, and found the test error was: X pixels, train: X pixels (image size was X by X). We then used a p-cutoff of X (i.e. 0.9) to condition the X,Y coordinates for future analysis. This network was then used to analyze videos from similar experimental settings. 

> Mathis, A. et al. Deeplabcut: markerless pose estimation
> of user-defined body parts with deep learning. Nature
> Neuroscience 21, 1281–1289 (2018).

> Nath, T. et al. Using deeplabcut for 3d markerless pose
> estimation across species and behaviors. Nature Protocols
> 14, 2152–2176 (2019).

*If any defaults were changed in *`pose_config.yaml`*, mention them here. 

i.e. common things one might change: 
* the loader (options are `default`, `imgaug`, `tensorpack`, `deterministic`). 
* the `post_dist_threshold` (default is 17 and determines training resolution).
* optimizer: do you use the default `SGD` or `ADAM`? 

*** here, you could add additional citations. 
If you use ResNets, consider citing Insafutdinov et al 2016 & He et al 2016. If you use the MobileNetV2s consider citing Mathis et al 2019, and Sandler et al, 2018.


> Mathis, A. et al. Pretraining boosts out-of-domain robustness for pose estimation
> arXiv 1909.11229 (2019)

> Insafutdinov, E., Pishchulin, L., Andres, B., Andriluka,
> M. & Schiele, B. DeeperCut: A deeper, stronger, and
> faster multi-person pose estimation model. In European
> Conference on Computer Vision, 34–50 (Springer, 2016).

> Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. &
> Chen, L.-C. Mobilenetv2: Inverted residuals and linear
> bottlenecks. In Proceedings of the IEEE Conference
> on Computer Vision and Pattern Recognition, 4510–4520
> (2018).

> He, K., Zhang, X., Ren, S. & Sun, J. Deep residual
> learning for image recognition. In Proceedings of the
> IEEE conference on computer vision and pattern recognition,
> 770–778 (2016). URL https://arxiv.org/abs/
> 1512.03385.

## Graphics 

We also have the network graphic freely available on SciDraw.io if you'd like to use it! https://scidraw.io/drawing/290

You are welcome to use our logo in your works as well.
