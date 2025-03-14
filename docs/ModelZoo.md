# The DeepLabCut Model Zoo! 

![image](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/8957c690-4f27-4430-8581-4161fd58d052/68747470733a2f2f696d616765732e73717561726573706163652d63646e2e636f6d2f636f6e74656e742f76312f3537663664353163396637343536366635356563663237312f313631363439323337333730302d50474f41433732494f4236415545343756544a582f6b6531375a77644742546f646449.png?format=450w)


## 🏠 [Home page](http://modelzoo.deeplabcut.org/)


Started in 2020, expanded in 2022 with PhD student [Shaokai Ye et al.](https://arxiv.org/abs/2203.07436v1), and the
first proper [SuperAnimal Foundation Models](#about-the-superanimal-models) published in 2024 🔥, the Model Zoo is four things:

- (1) a collection of models that are trained on diverse data across (typically) large datasets, which means you do not need to train models yourself, rather you can use them in your research applications.
- (2) a contribution website for community crowd sourcing of expertly labeled keypoints to improve models! You can get involved here: [contrib.deeplabcut.org](https://contrib.deeplabcut.org/).
- (3) a no-install DeepLabCut that you can use on ♾[Google Colab](https://github.com/DeepLabCut/DeepLabCut/blob/main/examples/COLAB/COLAB_DEMO_SuperAnimal.ipynb), 
test our models in 🕸[the browser](https://contrib.deeplabcut.org/), or on our 🤗[HuggingFace](https://huggingface.co/spaces/DeepLabCut/DeepLabCutModelZoo-SuperAnimals) app!
- (4) new methods to make SuperAnimal Foundation Models that combine data across different labs/datasets, keypoints, animals/species, and use on your data!

## Quick Start:
```
pip install deeplabcut[gui,modelzoo]
```

## About the SuperAnimal Models

Animal pose estimation is critical in applications ranging from neuroscience to veterinary medicine. However, reliable inference of animal poses currently requires domain knowledge and labeling effort. To ease access to high-performance animal pose estimation models across diverse environments and species, we present a new paradigm for pre-training and fine-tuning that provides excellent zero-shot (no training required) performance on two major classes of animal pose data: quadrupeds and lab mice. 

To provide the community with easy access to such high performance models across diverse environments and species, we present a new paradigm for building pre-trained animal pose models -- which we call SuperAnimal models -- and the ability to use them for transfer learning (e.g., fine-tune them if needed).

## SuperAnimal members:
- Models are based on what they are trained on, for example `superanimal_quadruped_x` is trained on [SuperAnimal-Quadruped-80K](https://zenodo.org/records/10619173). Each model class is described below:



### SuperAnimal-Quadruped: 


- `superanimal_quadruped_x` models aim to work across a large range of quadruped animals, from horses, dogs, sheep, rodents, to elephants. The camera perspective is orthogonal to the animal ("side view"), and most of the data includes the animals face (thus the front and side of the animal). You will note we have several variants that differ in speed vs. performance, so please do test them out on your data to see which is best suited for your application. Also note we have a "video adaptation" feature, which lets you adapt your data to the model in a self-supervised way. No labeling needed!
- [Please see the full datasheet here](https://zenodo.org/records/10619173)
- [More details on the models (detector, pose estimators)](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped)
- We provide several models:
    - `superanimal_quadruped_hrnetw32` (pytorch engine)
        - `superanimal_quadruped_hrnetw32` is a top-down model that is paired with a detector. That means it takes a cropped image from an object detector and predicts the keypoints. The object detector is currently a trained [ResNet50-based Faster-RCNN](https://pytorch.org/vision/stable/models/faster_rcnn.html).
    - `superanimal_quadruped_dlcrnet` (tensorflow engine)
        - `superanimal_quadruped_dlcrnet` is a bottom-up model that predicts all keypoints, then groups them into individuals. This can be faster, but more error prone.
    - `superanimal_quadruped` -> This is the same as `superanimal_quadruped_dlcrnet`, this was the old naming and being depreciated.
    - For all models, they are automatically downloaded to modelzoo/checkpoints when used.

- Here are example images of what the model is trained on:
![SA_Q](https://user-images.githubusercontent.com/28102185/209957688-954fb616-7750-4521-bb52-20a51c3a7718.png)



### SuperAnimal-TopViewMouse:


-  `superanimal_topviewmouse_x` aims to work across lab mice in different lab settings from a top-view perspective; this is very polar in many behavioral assays in freely moving mice.
- [Please see the full datasheet here](https://zenodo.org/records/10618947)
- [More details on the models (detector, pose estimators)](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)
- We provide several models:
    - `superanimal_topviewmouse_hrnetw32` (pytorch engine)
        - `superanimal_topviewmouse_hrnetw32` is a top-down model that is paired with a detector. That means it takes a cropped image from an object detector and predicts the keypoints. The object detector is currently a trained [ResNet50-based Faster-RCNN](https://pytorch.org/vision/stable/models/faster_rcnn.html).
    - `superanimal_topviewmouse_dlcrnet` (tensorflow engine)
        - `superanimal_topviewmouse_dlcrnet` is a bottom-up model that predicts all keypoints then groups them into individuals. This can be faster, but more error prone.
    - `superanimal_topviewmouse` -> This is the same as `superanimal_topviewmouse_dlcrnet`, this was the old naming and being depreciated.
    - For all models, they are automatically downloaded to modelzoo/checkpoints when used.
    
-  Here are example images of what the model is trained on:
![SA-TVM](https://user-images.githubusercontent.com/28102185/209957260-c0db72e0-4fdf-434c-8579-34bc5f27f907.png)


#### Practical example: Using SuperAnimal models for inference without training.

You can simply call the model and run video inference. 

To note, a good step is typically to use our self-supervised video adaptation method to reduce jitter. In the `deeplabcut.video_inference_superanimal` simply function set the `video_adapt` option to __True__. Be aware, that enabling this option will (minimally) extend the processing time. 

```python
import deeplabcut
video_path = "demo-video.mp4"
superanimal_name = "superanimal_quadruped"

deeplabcut.video_inference_superanimal([video_path],
                                        superanimal_name,
                                        model_name="hrnet_w32",
                                        detector_name="fasterrcnn_resnet50_fpn_v2",
                                        video_adapt = False)
```


#### Practical example: Using SuperAnimal model bottom up, considering video/animal size.

In our work we introduced a spatial-pyramid for smartly rescaling images. Imagine if you frames are much larger than what we trained on, it would be hard for the model to find the animal! Here, you can simply guide the model with the `scale_list`:

```python
import deeplabcut
video_path = "demo-video.mp4"
superanimal_name = "superanimal_quadruped"

# The purpose of the scale list is to aggregate predictions from various image sizes. We anticipate the appearance size of the animal in the images to be approximately 400 pixels.
scale_list = range(200, 600, 50)

deeplabcut.video_inference_superanimal([video_path],
                                        superanimal_name,
                                        model_name="hrnet_w32",
                                        detector_name="fasterrcnn_resnet50_fpn_v2",
                                        scale_list=scale_list,
                                        video_adapt = False)
```

#### Practical example: Using transfer learning with superanimal weights.
In the `deeplabcut.train_network` function, the `superanimal_transfer_learning` option plays a pivotal role. If it's set to __True__, it uses a new decoding layer and allows you to use superanimal weights in any project, no matter the number of keypoints. However, if it's set to __False__, you are doing fine-tuning. So, make sure your dataset has the right number of keypoints.  

Specifically:
* `superanimal_quadruped_x` uses 39 keypoints and,
* `superanimal_topviewmouse_x` uses 27 keypoints

```python
import os
import deeplabcut
from deeplabcut.modelzoo import build_weight_init

superanimal_name = "superanimal_topviewmouse"

config_path = os.path.join(os.getcwd(), "openfield-Pranav-2018-10-30", "config.yaml")

weight_init = build_weight_init(
    cfg=config_path,
    super_animal=superanimal_name,
    model_name="hrnet_w32",
    detector_name="fasterrcnn_resnet50_fpn_v2",
    with_decoder=False,
)

deeplabcut.create_training_dataset(config_path, weight_init = weight_init)

deeplabcut.train_network(config_path,
                         epochs=10,
                         superanimal_name = superanimal_name,
                         superanimal_transfer_learning = True)
```

### Potential failure modes for SuperAnimal Models and how to fix it.

Spatial domain shift: typical DNN models suffer from the spatial resolution shift between training datasets and test
videos. To help find the proper resolution for our model, please try a range of `scale_list` in the API (details in the
API docs). For `superanimal_quadruped`, we empirically observe that if your video is larger than 1500 pixels, it is
better to pass `scale_list` in the range within 1000.

Pixel statistics domain shift: The brightness of your video might look very different from our training datasets.
This might either result in jittering predictions in the video or fail modes for lab mice videos (if the brightness of
the mice is unusual compared to our training dataset). You can use our "video adaptation" model to counter this.



### Our longer term perspective ...

Via DeepLabCut Model Zoo, we aim to provide plug and play models that do not need any labeling and will just work
decently on novel videos. If the predictions are not great enough due to failure modes described below, please give us
feedback! We are rapidly improving our models and adaptation methods. We will also continue to expand this project to
new model/data classes. Please do get in touch is you have data or ideas: modelzoo@deeplabcut.org

## Publication:

To see the first preprint on the work, click [here](https://arxiv.org/abs/2203.07436v1).

Our first [publication](https://www.nature.com/articles/s41467-024-48792-2) on this project is now published at Nature
Communications:

```{hint}
Here is the citation:
@article{Ye2024,
  title={SuperAnimal pretrained pose estimation models for behavioral analysis},
  author={Shaokai Ye and Anastasiia Filippova and Jessy Lauer and Steffen Schneider and Maxime Vidal and Tian Qiu and Alexander Mathis and Mackenzie Weygandt Mathis},
  journal={Nature Communications},
  year={2024},
  preprint={abs/2203.07436}
}
```
