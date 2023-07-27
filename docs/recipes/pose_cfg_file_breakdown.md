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

<a id ="crop_size"></a>
 ### 3.2.5 `crop_size`
 Cropping consists of removing unwanted pixels from the image, thus selecting a part of the image and discarding the rest, reducing the size of the input. 

 In DeepLabCut *pose_config.yaml* file, by default, `crop_size` is set to (`400,400`), width, and height, respectively. This means it will cut out parts of an image of this size.

 💡Pro-tip:💡
  - If your images are very large, you could consider increasing the crop size. However, be aware that you'll need a strong GPU, or you will hit memory errors!
  - If your images are very small, you could consider decreasing the crop size. 

 <a id ="cropratio"></a>
 ### 3.2.6 `crop_ratio`
  Also, the number of frames to be cropped is defined by the variable `cropratio`, which is set to `0.4` by default. That means that there is a $40\%$ the images within the current batch will be cropped. By default, this value works well. 

  <a id ="max_shift"></a>
 ### 3.2.7 `max_shift`

  The crop shift between each cropped image is defined by `max_shift` variable, which explains the max relative shift to the position of the crop centre. By default is set to `0.4`, which means it will be displaced 40% max from the center to not apply identical cropping each time the same image is encountered during training - this is especially important for `density` and `hybrid` cropping methods.

 The image below is modified from 
 [2](#references). 
 ![cropping.png](attachment:cropping.png)

 <a id ="crop_sampling"></a>
 ### 3.2.8 `crop_sampling`
 Likewise, there are different cropping sampling methods (`crop_sampling`), we can use depending on how our image looks like. 

 💡Pro-tips💡
 - For highly crowded scenes, `hybrid` and `density` approaches will work best. 
 - `uniform` will take out random parts of the image, disregarding the annotations completely
 - 'keypoint' centers on a random keypoint and crops based on that location - might be best in preserving the whole animal (if reasonable `crop_size` is used)

 <a id ="kernel"></a>
 ### Kernel transformations 
 Kernel filters are very popular in image processing to sharpen and blur images. Intuitively, blurring an image might increase the motion blur resistance during testing. Otherwise, sharpening for data enhancement could result in capturing more detail on objects of interest.

 <a id ="sharp"></a>
 ### 3.2.9 `sharpening` and `sharpenratio`
 In DeepLabCut *pose_config.yaml* file, by default, `sharpening` is set to `False`, but if we want to use this type of data augmentation, we can set it `True` and specify a value for `sharpenratio`, which by default is set to `0.3`. Blurring is not defined in the *pose_config.yaml*, but if the user finds it convenient, it can be added to the data augmentation pipeline. 

 The image below is modified from 
 [2](#references). 
 ![kernelfilter.png](attachment:kernelfilter.png)

 <a id ="edge"></a>
 ### 3.2.10 `edge`
 Concerning sharpness, we have an additional parameter, `edge` enhancement, which enhances the edge contrast of an image to improve its apparent sharpness. Likewise, by default, this parameter is set `False`, but if you want to include it, you just need to set it `True`.

