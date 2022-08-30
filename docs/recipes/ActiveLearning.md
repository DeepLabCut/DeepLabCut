# Exploring active learning with DeepLabCut

<!-- % PROBLEM -->
One of the main disadvantages of applying supervised learning approaches in real-life scenarios is the need for relatively large amounts of good quality labeled data. For a pose estimation model to achieve good performance, we need to provide to it sufficient labelled examples during training [[1]](cit_primer). 

In DeepLabCut, we do this by manually annotating a subset of frames in the GUI. We know that with enough training samples, that are representative of the data the model will see at inference time, our model will be able to generalise well to new data [[2]](cit_horse10). Although in DeepLabCut typically a few hundred labelled frames provide a good performance on videos that are similar to the training set, we are always interested in approaches that reduce the labelling effort.

<!-- % FIGURE? -->

<!-- % CONTEXT FOR ACTIVE LEARNING -->
Let's consider the case of a DeepLabCut user who investigates gait biomechanics in horses. They collected data on a first experimental season, labelled a fraction of the recorded frames, and trained a DeepLabCut model that performs very well on the data that was not seen in training. 

```{image} ../images/horse_imgs_crop.png
:alt: active_learning_problem
:class: bg-primary mb-1
:width: 800px
:align: center
```
<p align = "center">
Modified from <a href="url">http://horse10.deeplabcut.org/</a>
</p>

Suppose the following year, in a subsequent round of experiments, they collect new video data that differs slightly from the previous set. For example, they may record horses of very different coats, in different backgrounds, or seen from a very different perspective. 

When the researcher runs their model in this fresh new data, they find the model doesn't generalise well to the new data. They know that if they invest time and effort in labelling, they will achieve good performance as they did before... but our researcher would like to focus their efforts in the analysis bit of the project, the one they are really excited about! How can they select the most informative subset of frames to label, such that they improve the ability to generalise of their model, with minimal labelling effort?

<!-- % WHAT IS ACTIVE LEARNING -->
This problem is the main focus of **Active Learning** approaches. In them, the goal is to identify which unlabelled frames are more worthwhile to label. The user is then requested to provide annotations for these frames, which are then used to re-train or fine-tune the model. In that sense, active learning can be seen as an iterative supervised learning approach, in which at every iteration an algorithm queries the user to label new frames. The assumption is that we can improve performance of the model in the new data by selecting a few specific samples to label, rather than a large amount of frames sampled at random. 

<!-- Unlabelled data is often much more abundant than labelled one. For example, for a typical DeepLabCut user collecting video data from animal behaviour experiments in the lab, it may be relatively easy to collect new videos, but less so to carefully label all of their frames.
This is often the case for DeepLabCut users, who typically count . In DeepLabCut, we prepare the training dataset by manually annotating keypoints in a subset of frames from the videos of interest
 -->


## How to select the 'best' frames to label?
<!-- % OVERVIEW -->
How do we define which unlabelled frames are most worthwile to label? It is not straight-forward to identify *a priori* which frames will be the most informative, or most useful for the model to generalise better to the new data. 

For example, should we select the frames in which the model is least certain about the estimated pose? [[3]](cit_liu2017) Are these the most likely to be wrongly predicted? Or should we identify the frames that are most likely to yield a high loss? [[4]](cit_yoo2019) [[5]](cit_shukla2021). We could also consider selecting the frames that are most dissimilar to what the model has already seen during training [[6]](cit_sener2018) [[7]](cit_kim2022) [[8]](cit_caramalau2021). Or maybe, labelling the frames that are most representative of the unlabelled data, so that the information is more efficiently propagated?[[3]](cit_liu2017) One could even define the active learning method as a multi-agent reinforcement learning problem [[9]](cit_gong2022)! Also, can these criteria be combined, and would that be useful? All these are strategies that have been explored in the literature, and we can see them all as attempts to locate the new unlabelled data in the space of what the model currently knows.


```{image} ../images/active_learning_with_RL.png
:alt: active_learning_with_RL
:class: bg-primary mb-1
:width: 800px
:align: center
```
<p align = "justify">
Active learning as a reinforcement learning approach, from Gong et al. 2022. Image from the corresponding oral presentation at <a href="url">https://www.youtube.com/watch?v=oVGb7i5MCe8</a>
</p>

In the current DeepLabCut implementation, similar strategies are used, for example when sampling from k-means clusters during the frame extraction step, or when doing refinement [[10]](cit_protocols). In this DLC AI Residency project, we explored further whether active learning approaches can be useful to improve the performance of a model on new 'out-of-domain' data. 

We focused on the work by Liu *et al.*, 2017 [[3]](cit_liu2017), in which the authors explored sampling the frames in two ways: first, they use an uncertainty score that assesses how uncertain are the model's predictions in a given image. Next, they consider an 'influence' metric, that measures how representative an image is within the set of unlabelled frames. They also consider dynamically combining these two criteria over several active learning iterations, and they find that this improved the model's performance on unseen data, and was more efficient than randomly sampling images from the unlabelled set. In this project, we explored the implementation of these two approaches in DeepLabCut.

We used the [Horse-10 dataset](http://horse10.deeplabcut.org/), which contains 8114 frames of 30 diverse thoroughbred horses, with 22 body parts labeled by an expert. We defined three shuffles of the data with the following structure: one horse made up the base training set, 9 other horses made up the active learning set (from which we sample frames with different criteria), and the 20 remaining horses constituted the evaluation set.

```{image} ../images/active_learning_dataset.png
:alt: active_learning_dataset
:class: bg-primary mb-1
:width: 400px
:align: center
```
<p align = "center">
Schematic of the train/test dataset split used. The base training indices ('base'), correspond to 1 horse, while the active learning train set (Train AL) covers 9 horses.
</p>

## Most uncertain frames
We used the Multiple Peak Entropy (MPE) metric from Liu *et al.*, 2017 [[3]](cit_liu2017) to assess the uncertainty of the model on the unlabelled frames. Like in DeepLabCut, the model they use in the paper also outputs confidence scoremaps for each estimated bodypart. However, rather than using directly the model's output confidence, they define a metric based on the distribution of local maxima in each bodypart's scoremap. This is because neural networks tend to overestimate the confidence of predictions in new data. An estimated bodypart will have a high MPE value if their corresponding scoremap has multiple weak peaks. For example, we would expect a high MPE value for the following scoremaps:

```{image} ../images/active_learning_uncert_nearhindfetlock.png
:alt: active_learning_uncert_nearhindfetlock
:class: bg-primary mb-1
:width: 400px
:align: center
```
```{image} ../images/active_learning_uncert_nearhindfoot.png
:alt: active_learning_uncert_nearhindfoot
:class: bg-primary mb-1
:width: 400px
:align: center
```
<p align = "justify">
The scoremaps shown were computed with a not fully-trained model. The overlaid colormap represents the model's confidence on the specific bodypart's location in the image, and the groundtruth location is shown with a red marker. 
</p>

Mathematically, the MPE metric per bodypart is defined as:
```{image} ../images/active_learning_uncertainty_eq_1.png
:alt: active_learning_uncertainty_eq_1
:class: bg-primary mb-1
:width: 300px
:align: center
```

where *C<sub>MPE</sub>* is the Multiple Peak Entropy metric for a given image and bodypart, and *Prob* is the normalised probability for a given image *I<sub>i</sub>*, local maxima *m* and bodypart *p* (see [the paper](cit_liu2017) for further details). We defined the MPE value per image as the maximum value across all the bodyparts shown in an image (note that in the original paper, they use the mean value instead). 

When ranking the images in the active learning train set for each shuffle by MPE score, we found that the top scoring images showed horses as they entered or exited the scene. In contrast, the lowest scoring images showed horses in which all body joints were easily distinguishable. We computed these scores using a model trained on one horse only (i.e., the base training set per shuffle).

```{image} ../images/active_learning_top_mpe.png
:alt: active_learning_top_mpe
:class: bg-primary mb-1
:width: 800px
:align: center
```
<p align = "center">
Images with top MPE scores for each shuffle
</p>

```{image} ../images/active_learning_bottom_mpe.png
:alt: active_learning_bottom_mpe
:class: bg-primary mb-1
:width: 800px
:align: center
```
<p align = "center">
Images with the lowest MPE scores for each shuffle
</p>


## Most influential frames
We also explored the possibility of sampling the most influential frames from the set of unlabelled images, following the definition from Liu *et al.*, 2017 [[3]](cit_liu2017). The idea is that labelling the images that are most representative of the unlabelled set may be a particularly efficient option, because the labelling information will 'propagate' to the largest number of frames. 

```{image} ../images/active_learning_alexnet.png
:alt: active_learning_alexnet
:class: bg-primary mb-1
:width: 600px
:align: center
```
<p align = "center">
AlexNet architecture. Image modified from <a href="ttps://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7869276">Bui et al., 2016 </a>.
</p>


Following the paper's approach, we used AlexNet as a feature extractor and encoded each of the images in the active learning train set as the output obtained at the *fc6* layer (see figure above). We then computed the pairwise distance between feature vectors as a function of their cosine similarity:

```{image} ../images/active_learning_infl_metric.png
:alt: active_learning_infl_metric
:class: bg-primary mb-1
:width: 300px
:align: center
```
where *u* and *v* are feature vectors corresponding to two images. 

We again inspected for each shuffle, which images ranked at the top and at the bottom of the influential scale. The results are shown below.

```{image} ../images/active_learning_top_infl.png
:alt: active_learning_top_infl
:class: bg-primary mb-1
:width: 800px
:align: center
```
<p align = "center">
Images with top influence scores for each shuffle
</p>

```{image} ../images/active_learning_bottom_infl.png
:alt: active_learning_bottom_infl
:class: bg-primary mb-1
:width: 800px
:align: center
```
<p align = "center">
Images with the lowest influence scores for each shuffle
</p>



## Preliminary results
We carried out experiments to inspect whether these active learning strategies allowed us to achieve good performance on the test set (20 unseen horses), with a reduced number of additionally labelled frames.

We considered models trained with 0%, 25%, 50%, 75% and 100% of the active learning frames (9 horses) added to the base training set (1 horse). We sampled the active learning frames based on three criteria: (1) by uniformly sampling across the active learning set, (2) by selecting the ones with highest MPE score, and (3) by selecting the ones with highest influence score. Note that if 100% of the active learning frames are included in the training set, all approaches should be produce very similar results. 

```{image} ../images/active_learning_biased_results.png
:alt: active_learning_biased_results
:class: bg-primary mb-1
:width: 500px
:align: center
```

The figure shows the results from sampling in the active learning set uniformly (blue), by MPE score (green) and by influence score (orange). The model's performance is assessed with the normalised root-mean-square error (RMSE) between a bodypart's predicted location and its groundtruth location. The normalisation is done by dividing the pixel error with the median eye-to-nose distance per video. Results obtained using the same dataset shuffle are connected with a line. 

Looking at the uniformly sampled results (blue), we find that just uniformly sampling 25% of the frames from the active learning set we achieve a performance comparable to incorporating 100% of the data. However, neither the uncertainty or the influence approaches produce more efficient results (i.e., better performance with less labelled frames). We also observe that for a given fraction of active learning frames sampled, there is larger spread in the normalised RMSE in the uncertainty or influence based approaches, compared to the uniform sampling. 

We hypothesised these two effects are due to the active learning approaches being biased, compared to the uniform sampling approach. Whereas the uniform sampling will by definition sample across all 9 horses in the active learning train set, the other two approaches may be biased to a reduced number of horses.

To address this, we carried a new set of experiments. This time, to reduce the sampling bias and increase diversity, we computed AlexNet feature vectors for all images in the active learning set, and then clustered them using k-means.  When sampling them based on MPE or influence score, we now ensured we sampled alternatively from each of the clusters. 

```{image} ../images/active_learning_biased_results.png
:alt: active_learning_biased_results
:class: bg-primary mb-1
:width: 500px
:align: center
```
<p align = "center">
<span style="color:red">TEMP FIGURE, RESULTS COMING SOON!</span>.
</p>

## Conclusions and next steps
In this project, we carried out a short study of active learning approaches using DeepLabCut and the Horse-10 dataset. We focused on two main strategies to identify frames to label: one was based on the model's uncertainty on the estimated bodyparts per image (the MPE metric), and the other one was based on the images' appearance, and how representative they are of the whole unlabelled dataset (the influence metric). <span style="color:red">We found that... (results coming soon)</span>.

Regarding next steps, one aspect that may be relevant to explore is sampling images that show novel poses. We started some work in this direction, inspecting metrics to assess how novel a pose is, compared to the already seen data. We encoded poses as vectors holding the pairwise distances between bodyparts, and then computed how far a given 'novel' pose was from a distribution of already seen poses, using the Mahalanobis distance. 

The encoding of images as feature vectors could also be further explored. For example, only the features around the estimated bodyparts could be considered. We also explored the possibility of using a self-supervised vision transformer model [DINO](https://github.com/facebookresearch/dino) to compute pairs of very dissimilar images. A min-max approach, rather than the ranking approach we use here could be followed: images that are furthest from their nearest neighbour could be the first one to be labelled, for example. 

This work from this project is aligned with the data-efficient efforts already carried out in DeepLabCut. We hope it inspires new contributions to some key steps in the processing pipeline, as are the frame extraction or the refinement steps.

<!-- Features at keypoints
Novel poses
Min-max approach
Produce pairs of dissimilar frames (with [DINO](https://github.com/facebookresearch/dino)) 
Applications to feature extraction or refinement steps -->

<!-- ## Acknowledgements --rephrase a bit?
This work was carried out by Sofia Minano, Sabrina Benas and Alex Mathis. We would like to thank Steffen Schneider for very useful discussions regarding DINO, and Jessy Lauer for the help when exploring the computation of novel poses. We would also like to thank the rest of the the DeepLabCut AI residents from 2022 for providing a fantastic and welcoming working environment. -->

## References
(cit_primer)=
[1] Mathis, A., Schneider, S., Lauer, J., & Mathis, M. W. (2020). *A primer on motion capture with deep learning: principles, pitfalls, and perspectives*. Neuron, 108(1), 44-65 [[link]](https://www.sciencedirect.com/science/article/pii/S0896627320307170)

(cit_horse10)=
[2] Mathis, A., Biasi, T., Schneider, S., Yuksekgonul, M., Rogers, B., Bethge, M., & Mathis, M. W. (2021). *Pretraining boosts out-of-domain robustness for pose estimation.* In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 1859-1868). [[link]](https://openaccess.thecvf.com/content/WACV2021/papers/Mathis_Pretraining_Boosts_Out-of-Domain_Robustness_for_Pose_Estimation_WACV_2021_paper.pdf)

(cit_liu2017)=
[3] Liu, B., & Ferrari, V. (2017). *Active learning for human pose estimation.* In Proceedings of the IEEE International Conference on Computer Vision (pp. 4363-4372). [[link]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Active_Learning_for_ICCV_2017_paper.pdf)

(cit_yoo2019)=
[4] Yoo, D., & Kweon, I. S. (2019). *Learning loss for active learning.* In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 93-102). [[link]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yoo_Learning_Loss_for_Active_Learning_CVPR_2019_paper.pdf)

(cit_shukla2021)=
[5] Shukla, M., & Ahmed, S. (2021). *A mathematical analysis of learning loss for active learning in regression.* In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 3320-3328). [[link]](https://arxiv.org/pdf/2104.09315.pdf)

(cit_sener2018)=
[6] Sener, O., & Savarese, S. (2017). *Active learning for convolutional neural networks: A core-set approach.* arXiv preprint arXiv:1708.00489.
[[link]](https://arxiv.org/abs/1708.00489)

(cit_kim2022)=
[7] Kim, Y., & Shin, B. (2022, August). *In Defense of Core-set: A Density-aware Core-set Selection for Active Learning. *In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 804-812).[[link]](https://arxiv.org/abs/2206.04838)

(cit_caramalau2021)=
[8] Caramalau, R., Bhattarai, B., & Kim, T. K. (2021). *Active learning for bayesian 3d hand pose estimation.* In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 3419-3428). [[link]](https://openaccess.thecvf.com/content/WACV2021/papers/Caramalau_Active_Learning_for_Bayesian_3D_Hand_Pose_Estimation_WACV_2021_paper.pdf)

(cit_gong2022)=
[9] Gong, J., Fan, Z., Ke, Q., Rahmani, H., & Liu, J. (2022). *Meta agent teaming active learning for pose estimation.* In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 11079-11089).
[[link]](https://openaccess.thecvf.com/content/CVPR2022/papers/Gong_Meta_Agent_Teaming_Active_Learning_for_Pose_Estimation_CVPR_2022_paper.pdf)

(cit_protocols)=
[10] Nath, T., Mathis, A., Chen, A. C., Patel, A., Bethge, M., & Mathis, M. W. (2019). *Using DeepLabCut for 3D markerless pose estimation across species and behaviors.* Nature protocols, 14(7), 2152-2176. [[link]](https://www.nature.com/articles/s41596-019-0176-0)