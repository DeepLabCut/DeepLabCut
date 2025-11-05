# How to write a DLC Methods Section

**Pose estimation using DeepLabCut**

For body part tracking we used DeepLabCut (version 3.X.X) [Mathis et al, 2018, Nath et al, 2019]. Specifically, we
labeled X number of frames taken from X videos/animals (then X% was used for training (default is 95%). We used a
X-based neural network (i.e., X = ResNet-50, ResNet-101, MobileNetV2-0.35, MobileNetV2-0.5, MobileNetV2-0.75,
MobileNetV2-1, EfficientNet ..X, dlcrnet_ms5, cspnext_s, dekr_w32, rtmpose_s, etc.)*** with default parameters* for X
number of training iterations. We validated with X number of shuffles, and found the test error was: X pixels, train:
X pixels (image size was X by X). We then used a p-cutoff of X (i.e. 0.9) to condition the X,Y coordinates for future
analysis. This network was then used to analyze videos from similar experimental settings.

*If any defaults were changed in *`pose_config.yaml`*, mention them.

i.e. common things one might change:
* the loader (options are `default`, `imgaug`, `tensorpack`, `deterministic`).
* the `post_dist_threshold` (default is 17 and determines training resolution).
* optimizer: do you use the default `SGD` or `ADAM`?

*** here, you could add additional citations.
If you use ResNets, consider citing Insafutdinov et al 2016 & He et al 2016. If you use the MobileNetV2s consider citing Mathis et al 2021, and Sandler et al, 2018. If you use DLCRNet, please cite Lauer et al, 2021.

> Mathis, A. et al. Deeplabcut: markerless pose estimation
> of user-defined body parts with deep learning. Nature
> Neuroscience 21, 1281–1289 (2018).

> Nath, T. et al. Using deeplabcut for 3d markerless pose
> estimation across species and behaviors. Nature Protocols
> 14, 2152–2176 (2019).

> Mathis, A. Biasi, T. et al. Pretraining boosts out-of-domain robustness for pose estimation. WACV (2021).

> Lauer et al. Multi-animal pose estimation and tracking with DeepLabCut. BioRxiv (2021).

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

We also have the network graphic freely available on SciDraw.io if you'd like to use it! https://scidraw.io/drawing/290.
If you use our DLC logo, please include the TM symbol, thank you!
