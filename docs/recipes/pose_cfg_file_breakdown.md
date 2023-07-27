<a id="hyperparam"></a>
## 3.1 Training Hyperparameters 

<a id="input_size"></a>
### 3.1.A `max_input_size` and `min_input_size`
The default values are `1500` and `64`, respectively. 

💡Pro-tip:💡
- change `max_input_size` when the resolution of the video is higher than 1500x1500 or when `scale_jitter_up` will possibly go over that value
- change `min_input_size` when the resolution of the video is smaller than 64x64 or when `scale_jitter_lo` will possibly go below that value

<a id="global_scale"></a>
### 3.1.B `global_scale`
The default value is `0.8`. It's the most basic, first scaling that happens to all images in the training queue.

💡Pro-tip:💡
- With images that are low resolution or lack detail, it may be beneficial to increase the `global_scale` to 1, to keep the original size and retain as much information as possible.

### 3.1.C `batch_size`
<a id="batch_size"></a>

The default for single animal projects is 1, and for maDLC projects it's `8`. It's the number of frames used per training iteration.

In both cases, you can increase the batchsize up to the limit of your GPU memory and train for a lower number of iterations. The relationship between the number of iterations and `batch_size` is not linear so `batch_size: 8` doesn't mean you can train for 8x less iterations, but like with every training, plateauing loss can be treated as an indicator of reaching optimal performance.

💡Pro-tip:💡
- Having a higher `batch_size` can be beneficial in terms of models' generalization

___________________________________________________________________________________

Values mentioned above and the augmentation parameters are often intuitive, and knowing our own data, we are able to decide on what will and won't be beneficial. Unfortunately, not all hyperparameters are this simple or intuitive. Two parameters that might require some tuning on challenging datasets are `pafwidth` and `pos_dist_thresh`. 

<a id="pos"></a>
### 3.1.D `pos_dist_thresh`
The default value is `17`. It's the size of a window within which detections are considered positive training samples, meaning they tell the model that it's going in the right direction. 

<a id="paf"></a>
### 3.1.E `pafwidth`
The default value is `20`. PAF stands for part affinity fields. It is a method of learning associations between pairs of bodyparts by preserving the location and orientation of the limb (the connection between two keypoints). This learned part affinity helps in proper animal assembly, making the model less prone to associating bodyparts of one individual with those of another. [1](#ref1)
