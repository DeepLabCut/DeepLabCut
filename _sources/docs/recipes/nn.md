(tf-training-tips-and-tricks)=
# Model training tips & tricks

## TensorFlow Engine: Limiting a GPU's memory consumption

With TensorFlow, all GPU memory is allocated to training by default, preventing
other Tensorflow processes from being run on the same machine.

A flexible solution to limiting memory usage is to call 
`deeplabcut.train(..., allow_growth=True)`, which dynamically grows the GPU memory
region as it is needed. Another, stricter option is to explicitly cap GPU usage to only
a fraction of the available memory. For example, allocating a maximum of 1/4 of the
total memory could be done as follows:

```python
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
```

(tf-custom-image-augmentation)=
## Using custom image augmentation

Image augmentation is the process of artificially expanding the training set
by applying various transformations to images (e.g., rotation or rescaling)
in order to make models more robust and more accurate (read our
[primer](https://www.sciencedirect.com/science/article/pii/S0896627320307170) for
more information). Although data augmentation is automatically accomplished
by DeepLabCut, default values can be readily overwritten prior to training. See the
augmentation variables defined in the:

- PyTorch Engine: [docs for the `pytorch_config.yaml` file](dlc3-pytorch-config)
- TensorFlow Engine: [default pose_cfg.yaml file](
https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/pose_cfg.yaml#L23-L74)

For the single-animal TensorFlow models, [you have several options](
https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html#f-create-training-dataset-s-and-selection-of-your-neural-network)
for image augmentation when calling `create_training_dataset`

An in-depth tutorial on image augmentation and training hyperparameters can be found [
here](
https://deeplabcut.github.io/DeepLabCut/docs/recipes/pose_cfg_file_breakdown.html).

## Evaluating intermediate (and all) snapshots

The latest snapshot stored during training may not necessarily be the one that yields
the highest performance. Therefore, you should analyze ALL snapshots, and select the
best. Put 'all' in the snapshots section of the `config.yaml` to do this.

(what-neural-network-should-i-use)=
## What neural network should I use? (Trade offs, speed performance, and considerations)

You always select the network type when you create a training data set: i.e., standard
dlc: `deeplabcut.create_training_dataset(config, net_type=resnet_50)` , or maDLC: 
`deeplabcut.create_multianimaltraining_dataset(config, net_type=dlcrnet_ms5)`. There is
nothing else you should change.

### PyTorch Engine

The different architectures available are described in the [PyTorch model architectures
](dlc3-architectures) page.

### TensorFlow Engine

With the release of even more network options, you now have to decide what to use! This
additionally flexibility is hopefully helpful, but we want to give you some guidance on
where to start.

**TL;DR - your best performance for most everything is ResNet-50; MobileNetV2-1 is much
faster, needs less memory on your GPU to train and nearly as accurate.**

***
### ResNets:

In Mathis et al. 2018 we benchmarked three networks: **ResNet-50, ResNet-101, and
ResNet-101ws**. For ALL lab applications, ResNet-50 was enough. For all the demo videos
on [www.deeplabcut.org](http://www.mousemotorlab.org/deeplabcut) the backbones are
ResNet-50's. Thus, we recommend making this your go-to workhorse for data analysis. Here
is a figure from the paper, see panel "B" (they are all within a few pixels of each
other on the open-field dataset):

<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1548558406678-S32H6T3M3U7BWVS4IGYD/ke17ZwdGBToddI8pDm48kD4CqqHoJgLzZVYacqX5G8QUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYy7Mythp_T-mtop-vrsUOmeInPi9iDjx9w8K4ZfjXt2dqTB9h4P9po3-YSCqzKkit0PccqviqYX7RTAdOBUgXwbCjLISwBs8eEdxAxTptZAUg/SupplFig2-01.png?format=1000w" width="80%">
</p>

This is also one of the main result figures, generated with ResNet-50. BLUE is 
training - RED is testing - BLACK is our best human-level performance, and 10 pixels is
the width - of the mouse nose -so anything under that is good performance for us on this
task!

<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1547585317499-0QWTWL5KVPK8ZWINQ30U/ke17ZwdGBToddI8pDm48kH23KVWagbNOYpajbj_MQLNZw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZamWLI2zvYWH8K3-s_4yszcp2ryTI0HqTOaaUohrI8PI-4DGGLi3WdhIPQDa6khDzRWGU5SknjCO3Yd6rloU2Zw/ErrorvsTrainingsetSize.png?format=1000w" width="60%">
</p>

Here are also some speed stats for analyzing videos with ResNet-50, see 
https://www.biorxiv.org/content/early/2018/10/30/457242 for more details:

<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1547585393723-8BQ6RGSPUUEQ1NNGUQDZ/ke17ZwdGBToddI8pDm48kCebzxgICDi_Bmgq_409OyxZw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZamWLI2zvYWH8K3-s_4yszcp2ryTI0HqTOaaUohrI8PICdDqlshOygx3FUsifuoze123Z0BWMsGmyODBJYiFvQc/inferencespeed.png?format=1000w" width="60%">
</p>

**So, why use a ResNet-101 or even 152?** if you have a much more challenging problem,
like multiple humans dancing, this is a good option. You should then also set
`intermediate_supervision=True` for best performance in the `pose_config.yaml` of that
shuffle folder (before you train). Note, for ResNet-50 this does NOT help, and can
hurt.

### When should I use a MobileNet?

MobileNets are fast to run, fast to train, more memory efficient, and faster for
analysis (inference) - e.g. on CPUs they are 4 times faster, on GPUs up to 2x! So, if
you don't have a GPU (or a GPU with little memory), and don't want to use Google COLAB,
etc, then these are a great starting point.

They are smaller/shallower networks though, so you don't want to be pushing in very
large images. So, be sure to use `deeplabcut.DownSampleVideo` on your data (which is
frankly never a bad idea).

Additionally, these are good options for running on "live" videos, i.e. if you want to
give real-time feedback in an experiment, you can run a video around a smaller cropped
area, and run this rather fast!

**So, how fast are they?**

Here are comparisons of 4 MobileNetV2 variants to ResNet-50 and ResNet-101 (darkest
red - read more here: https://arxiv.org/abs/1909.11229)

<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1570054128042-51HCY1Y9GV7GAQTZ5BMB/ke17ZwdGBToddI8pDm48kKr5oWkDv6XTQOpQfQOqjiAUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYxCRW4BPu10St3TBAUQYVKchrM-h1v5jGhVgANO1xgMJaHKhYxZ0-Cf0LQLHXkOaBUlIOyXFtu3PNQa47ngsqiu/mbnetv2speed.png?format=1000w" width="100%">
</p>

<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1570054117297-YA8WOYG50EK55WM6Y8ZI/ke17ZwdGBToddI8pDm48kAWg0301pwdoqO-Bo48aILYUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYxCRW4BPu10St3TBAUQYVKcZbh5EzlyubXk7Q3qHw5ayJHISnXwMOq8Pp90__8eMJefaZFcnumpU7B4DHTHEFkQ/speedtables.png?format=1000w" width="100%">
</p>

### When should I use an EfficientNet?

Built with inverse residual blocks like MobileNets, but more powerful than ResNets, due
to optimal depth/width/resolution scaling, [EfficientNet](
https://arxiv.org/abs/1905.11946) are an excellent choice if you want speed and
performance. They do require more careful handling though! Especially for small
datasets, you will need to tune the batch size and learning rates. So, we suggest these
for more advanced users, or those willing to run experiments to find the best settings.
Here is the speed comparison, and for performance see our latest work at: 
http://horse10.deeplabcut.org

<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1615891029784-87JAZJN1C5S4HS62F752/ke17ZwdGBToddI8pDm48kLId9V2zDiOqQ5EIZz4b_S0UqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYxCRW4BPu10St3TBAUQYVKctCpCjeabgTq1Hv_G9BIks_zjAnmEpAaVGioFPvsrieXDegXGHA0z-h8QeHOQDokM/speedTest.png?format=1000w" width="100%">
</p>

### How can I compare them?

Great question! So, the best way to do this is to use the **same** test/train split (
that is generated in create_training_dataset) with different models. Here, as of 2.1+,
we have a **new** function that lets you do this easily. Instead of using
`create_training_dataset` you will run `create_training_model_comparison` (see the
docstrings by `deeplabcut.create_training_model_comparison?` or run the Project Manager
GUI - `deeplabcut.launch_dlc()`-  for assistance.
