# The DeepLabCut Model Zoo! 

🦒 🐈 🐕‍🦺 🐀 🐁 🦡 🦦 🐏 🐫 🐆 🦓 🐖 🐄 🐂 🦖

## 🏠 [Home page](http://modelzoo.deeplabcut.org/)

Started in 2020, the model zoo is four things: 
- (1) a collection of models that are trained on diverse data across (typically) large datsets, which means you do not need to train models yourself
- (2) a contribution website for community crowd sourcing of expertly labeled keypoints to improve models in part 1!
- (3) a no-install DeepLabCut that you can use on ♾[Google Colab](https://colab.research.google.com/github/DeepLabCut/DeepLabCut/blob/master/examples/COLAB/COLAB_DLC_ModelZoo.ipynb), 
test our models in 🕸[the browser](https://contrib.deeplabcut.org/), or on our 🤗[HuggingFace](https://huggingface.co/spaces/DeepLabCut/MegaDetector_DeepLabCut) app!
- (4) new methods to make SuperAnimal Models that combine data across different labs/datasets, keypoints, animals/species, and use on your data!

## Quick Start:

```
pip install deeplabcut[tf,gui,modelzoo]
```


### About SuperAnimal Models.

Our newest generation models act as a paradigm shift of using pre-trained model. It aims to provide a plug and play solution that works without training.

IMPORTANT:  we currently only support single animal scenarios

We now introduce two SuperAnimal members, namely, superquadruped and supertopview.

- superquadruped model aim to work across a large range of quadruped animals. Note since quadrupeds are mostly side viewed, it is important to tune the pcutoff to help model remove keypoints are occluded.

- supertopview model aims to work across labmice in different cage settings.


### Our perspective.

Via DeepLabCut Model Zoo, we aim to provide plug and play models that do not need any labeling and will just work decently on novel videos. If the predictions are not great enough due to failure modes described below, please give us feedback! We are rapidly improving our models and adaptation methods.


### To use our models in DeepLabCut, please use the following API

```python
video_path = 'demo-video.mp4'
superanimal_name = 'superquadruped'
scale_list = range(200, 600, 50)  # image height pixel size range and increment

deeplabcut.video_inference_superanimal([video_path], superanimal_name, scale_list=scale_list)
```


### To see the list of available models, check out the [Home page](http://modelzoo.deeplabcut.org/). 

**Coming soon:** The DeepLabCut Project Manager GUI will allow you to use the SuperAnimal Models. You can run the model and do ``active learning" to improve performance on your data. 
Specifically, we have *new* video adaptation methods to make your tracking extra smooth and robust!

### Potential failure modes for SuperAnimal Models.

Spatial domain shift: typical DNN models suffer from the spatial resolution shift between training datasets and test videos. To help find the proper resolution for our model, please try a range of scale_list in the API (details in the API docs). For superquadruped, we empirically observe that if your video is larger than 1500 pixels, it is better to pass `scale_list` in the range within 1000.

Pixel statistics domain shift: The brightness of your video might look very different from our training datasets. This might either result in jittering predictions in the video or
fail modes for lab mice videos (if the brightness of the mice is unusual compared to our training dataset). We are currently developing new models and new methods to counter that.

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
