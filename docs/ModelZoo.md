# The DeepLabCut Model Zoo! 

🦒 🐈 🐕‍🦺 🐀 🐁 🦡 🦦 🐏 🐫 🐆 🦓 🐖 🐄 🐂 🦖 🐿 🦍 🦥

## 🏠 [Home page](http://modelzoo.deeplabcut.org/)

Started in 2020 and expanded in 2022, the model zoo is four things:

- (1) a collection of models that are trained on diverse data across (typically) large datasets, which means you do not need to train models yourself
- (2) a contribution website for community crowd sourcing of expertly labeled keypoints to improve models in part 1!
- (3) a no-install DeepLabCut that you can use on ♾[Google Colab](https://colab.research.google.com/github/DeepLabCut/DeepLabCut/blob/master/examples/COLAB/COLAB_DLC_ModelZoo.ipynb), 
test our models in 🕸[the browser](https://contrib.deeplabcut.org/), or on our 🤗[HuggingFace](https://huggingface.co/spaces/DeepLabCut/MegaDetector_DeepLabCut) app!
- (4) new methods to make SuperAnimal Models that combine data across different labs/datasets, keypoints, animals/species, and use on your data!

## Quick Start:
```
pip install deeplabcut[tf,gui,modelzoo]
```

## About the SuperAnimal Models

Animal pose estimation is critical in applications ranging from neuroscience to veterinary medicine. However, reliable inference of animal poses currently requires domain knowledge and labeling effort. To ease access to high-performance animal pose estimation models across diverse environments and species, we present a new paradigm for pre-training and fine-tuning that provides excellent zero-shot (no training required) performance on two major classes of animal pose data: quadrupeds and lab mice. 

To provide the community with easy access to such high performance models across diverse environments and species, we present a new paradigm for building pre-trained animal pose models -- which we call SuperAnimal models -- and the ability to use them for transfer learning (e.g., fine-tune them if needed).

### We now  introduce two SuperAnimal members, namely, `superanimal_quadruped` and `superanimal_topviewmouse`.

#### `superanimal_quadruped` model aim to work across a large range of quadruped animals, from horses, dogs, sheep, rodents, to elephants. The camera perspective is orthogonal to the animal ("side view"), and most of the data includes the animals face (thus the front and side of the animal). Here are example images of what the model is trained on:

![SA_Q](https://user-images.githubusercontent.com/28102185/209957688-954fb616-7750-4521-bb52-20a51c3a7718.png)

#### `superanimal_topviewmouse` aims to work across lab mice in different lab settings from a top-view perspective; this is very polar in many behavioral assays in freely moving mice. Here are example images of what the model is trained on:

![SA-TVM](https://user-images.githubusercontent.com/28102185/209957260-c0db72e0-4fdf-434c-8579-34bc5f27f907.png)


IMPORTANT:  we currently only support single animal scenarios

### Our perspective.

Via DeepLabCut Model Zoo, we aim to provide plug and play models that do not need any labeling and will just work decently on novel videos. If the predictions are not great enough due to failure modes described below, please give us feedback! We are rapidly improving our models and adaptation methods.


### To use our models in DeepLabCut (versions 2.3+), please use the following API

```
pip install deeplabcut[tf,modelzoo]
```

#### Practical example: Using SuperAnimal models for inference without training.
In the `deeplabcut.video_inference_superanimal` function, if the output video appears to be jittery, consider setting the `video_adapt` option to __True__. Be aware, that enabling this option might extend the processing time. 

```python
video_path = 'demo-video.mp4'
superanimal_name = 'superanimal_quadruped'

# The purpose of the scale list is to aggregate predictions from various image sizes. We anticipate the appearance size of the animal in the images to be approximately 400 pixels.
scale_list = range(200, 600, 50)

deeplabcut.video_inference_superanimal([video_path], superanimal_name, scale_list=scale_list, video_adapt = False)
```

#### Practical example: Using transfer learning with superanimal weights.
In the `deeplabcut.train_network` function, the `superanimal_transfer_learning` option plays a pivotal role. If it's set to __True__, it uses a new decoding layer and allows you to use superanimal weights in any project, no matter the number of keypoints. However, if it's set to __False__, you are doing fine-tuning. So, make sure your dataset has the right number of keypoints.  
  Specifically:
* `superquadruped` uses 39 keypoints and,
* `supertopview` uses 27 keypoints

```python
superanimal_name = "superanimal_topviewmouse"
config_path = os.path.join(os.getcwd(), "openfield-Pranav-2018-10-30", "config.yaml")

deeplabcut.create_training_dataset(config_path, superanimal_name = superanimal_name)

deeplabcut.train_network(config_path,
                         maxiters=10,
                         superanimal_name = superanimal_name,
                         superanimal_transfer_learning = True)
```



### To see the list of available models, check out the [Home page](http://modelzoo.deeplabcut.org/). 

**Coming soon:** The DeepLabCut Project Manager GUI will allow you to use the SuperAnimal Models. You can run the model and do ``active learning" to improve performance on your data. 
Specifically, we have *new* video adaptation methods to make your tracking extra smooth and robust!

### Potential failure modes for SuperAnimal Models and how to fix it.

Spatial domain shift: typical DNN models suffer from the spatial resolution shift between training datasets and test videos. To help find the proper resolution for our model, please try a range of `scale_list` in the API (details in the API docs). For `superanimal_quadruped`, we empirically observe that if your video is larger than 1500 pixels, it is better to pass `scale_list` in the range within 1000.

Pixel statistics domain shift: The brightness of your video might look very different from our training datasets. This might either result in jittering predictions in the video or fail modes for lab mice videos (if the brightness of the mice is unusual compared to our training dataset). You can use our "video adaptation" model (released soon) to counter this.
### To see our first preprint on the work, check out [our paper](https://arxiv.org/abs/2203.07436v1):

```{hint}
Here is the citation:
@article{Ye2022PanopticAP,
  title={Panoptic animal pose estimators are zero-shot performers},
  author={Shaokai Ye and Alexander Mathis and Mackenzie W. Mathis},
  journal={ArXiv},
  year={2022},
  volume={abs/2203.07436}
}
```
