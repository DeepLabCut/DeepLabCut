# Using ModelZoo models on your own datasets

<p style='text-align: justify;'>Animal behavior has to be analyzed with painstaking accuracy. Therefore, animal pose estimation has been
an important tool to study animal behavior precisely.

Beside providing an open source toolbox for researchers to develop customized deep neural networks for markerless pose
estimation, we at DeepLabCut also aim to build robust, generalizable models. Part of this effort is via the
[DeeplabCut ModelZoo](http://modelzoo.deeplabcut.org/).

The Zoo hosts user-contributed and DLC-team developed models that are trained on specific animals and scenarios. You can
analyze your videos directly with these models without training. The models have strong zero-shot performance on unseen
out-of-domain data which can be further improved via pseudo-labeling. Please check the first
[ModelZoo manuscript](https://arxiv.org/abs/2203.07436v1) for further details.

This recipe aims to show a usecase of the **mouse_pupil_vclose** and is contributed by 2022 DLC AI Resident
[Neslihan Wittek](https://github.com/neslihanedes) 💜.

## `mouse_pupil_vclose` model

This model was contributed by Jim McBurney-Lin at University of California Riverside, USA.
The model was trained on images of C57/B6J mice eyes, and also then augmented with mouse eye data from the Mathis Lab at
EPFL.


 <img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1661439618442-RAACYCYD4RWEND4X1UFU/pupil_one.png?format=500w" width="250" title="DLC" alt="DLC" align="left" vspace = "50">

  <img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1661439618750-97KC2HW8HH6VOJHLMO46/pupil_two.png?format=300w" width="250" title="DLC" alt="DLC" align="right" vspace = "50">

| Landmark_Number  | Landmark_Name  | Description|
| --- | --- | --- |
| 1 | Lpupil  | Left aspect of pupil |
| 2 | LDpupil | Left/dorsal aspect of pupil |
| 3 | Dpupil  | Dorsal aspect of pupil |
| 4 | DRpupil  | Dorsal/Right aspect of pupil |
| 5 | Rpupil  | Right aspect of pupil |
| 6 | RVpupil  | Right/ventral aspect of pupil |
| 7 | Vpupil  | Ventral aspect of pupil |
| 8 | VLpupil  | Ventral/left aspect of pupil |


Since we would like to evaluate the models performance on out-of-domain data, we will analyze pigeon pupils. For more
discussions and work on so-called out-of-domain data, see
[Mathis, Biasi 2020](https://paperswithcode.com/dataset/horse-10).

## Pigeon Pupil

The eye pupil admits and regulates the amount of light entering the retina in order to enable image perception. Beside
this curicial role, the pupil also reflects the state of the brain. The systemic behavior of the pupil has not been
vastly studied in birds, although researchers from
<a href="https://www.sciencedirect.com/science/article/pii/S0960982221013166?via%3Dihub" target="_blank">Max Planck Institute for Ornithology in Seewiesen</a>
have shed light on pupil behaviors in pigeons.

The pupils of male pigeons get smaller during courtship behavior. This is in contrast to mammals, for which the pupil
size dilates in response to an increase in arousal. In addition, the pupil size of pigeons dilates during non-REM sleep,
while they rapidly constrict during REM sleep. Examining these differences and the reason behind them, might be helpful
to understand the pupillary behavior in general.

In light of these findings, we wanted to show whether the **mouse_pupil_vclose** model give us an accurate tracking
performance for the pigeon pupil as well.

### Jupyter & Google Colab Notebook

DeepLabCut provides a Google Colab Notebook to analyze your video with a pretrained networks from the ModelZoo. No need
for local installation of DeepLabCut!

Since we are interested in the accuracy of the **mouse_pupil_vclose** on pigeon pupil data, we will use a video which
consists of 7 recordings of pigeon pupils.

Check the
[ModelZoo Colab page](https://github.com/DeepLabCut/DeepLabCut/blob/main/examples/COLAB/COLAB_DLC_ModelZoo.ipynb)
and a video tutorial on how to use the ModelZoo on Google Colab.

<div align="center">
  <a href="https://www.youtube.com/watch?v=twHBa1ZvXM8" target= "_blank"><img src="http://img.youtube.com/vi/twHBa1ZvXM8/0.jpg" alt="IMAGE ALT TEXT"></a>
</div>

```{hint}
You are happy with the model and want to go on analyzing further videos on your local machine or you want to refine the model for your specific usecase?
```html
!zip -r /content/file.zip /content/pigeon_modelZoo-nessi-2022-08-22
from google.colab import files
files.download("/content/file.zip")

```

### Analyze Videos at Your Local Machine

DeepLabCut host models from the [DeepLabCut ModelZoo Project](http://modelzoo.deeplabcut.org/).

The `create_pretrained_project` function will create a new project directory with the necessary sub-directories and a basic configuration file.
It will also initialize your project with a pre-trained model from the DeepLabCut ModelZoo.

The rest of the code should be run within your DeepLabCut environment.
Check [here](how-to-install) for the instructions for the DeepLabCut installation.

To initialize a new project directory with a pre-trained model from the DeepLabCut ModelZoo, run the code below.

::::{warning}
This method is currently implemented for Tensorflow only, Pytorch compatibility is coming soon.
::::

```python
import deeplabcut

deeplabcut.create_pretrained_project(
    "projectname",
    "experimenter",
    [r"path_for_the_videos"],
    model="mouse_pupil_vclose",
    working_directory=r"project_directory",
    copy_videos=True,
    videotype=".mp4 or .avi?",
    analyzevideo=True,
    filtered=True,
    createlabeledvideo=True,
    trainFraction=None,
    engine=deeplabcut.Engine.TF,
)
```

::::{important}
Your videos should be cropped around the eye for better model accuracy! 👁🐭
::::

Excitingly, 6 out of the 7 pigeon pupils were tracked nicely:

<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/44858d34-dca7-4bb5-a6e5-cd8078b50bec/Screen+Shot+2022-08-25+at+5.39.33+PM.png?format=1500w" width="500" title="DLC" alt="DLC" align="center" vspace = "50">

When we further evaluate the model accuracy by checking the likelihood of tracked points, we see that the tracking is
low confidience when the pigeons close their eyelid (which is of course expected, and can be leveraged to measure
blinking 👁).

<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1661439615047-OVOOMU1Z5NJIWJ1HHNFD/likelihood.png?format=500w" width="600" title="DLC" alt="6 pigeion eyes tracked with deeplabcut" align="center" vspace = "50">

But you also might encounter larger problems than small tracking glitches:

<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1661439618037-4GNTZD476MJMQX19N0Z4/pigeon_7.png?format=500w" width="250" title="DLC" alt="1eye" align="center" vspace = "5">

To deal with this, you can extract poorly tracked outlier frames, refine them and feed the training data set with them for re-training.
Be sure that you set the number of frames to label in the `config.yaml` file of your project folder.
The more problems you encounter, the higher the number of frames you might want to label.
You should also add the path of the video(s) into the `config.yaml` file, or run the following command to add the videos to your project:

```python
deeplabcut.add_new_videos(
    "/pathofproject/config.yaml",
    ["/pathofvideos/pigeon.mp4"],
    copy_videos=False,
    coords=None,
    extract_frames=False
)
```
The `deeplabcut.extract_outlier_frames` function will check for outliers and ask your feedback on whether to extract these outliers frames.

```python
deeplabcut.analyze_videos(
    "/pathofproject/config.yaml",
    ["/pathofvideos/pigeon.mp4"]
)
deeplabcut.extract_outlier_frames(
    "/pathofproject/config.yaml",
    ["/pathofvideos/pigeon.mp4"],
    automatic=True
)
```
The `deeplabcut.refine_labels` function starts the GUI which allows you to refine the outlier frames manually.
You should load the outlier frames directory and corresponding `.h5` file from the previous model.
It will ask you to define the `likelihood` threshold: labels under the threshold should be refined at this stage.

After refining, you should combine these data with your previous model's data set and create a new training data set.
```python
deeplabcut.refine_labels("/pathofproject/config.yaml")
deeplabcut.merge_datasets("/pathofproject/config.yaml")
deeplabcut.create_training_dataset("/pathofproject/config.yaml")
```
Before starting the training of your model, there is one last step left: editing the `init_weights` parameter in your `pose_cfg.yaml` file.
Go to your project and check the latest snapshot (e.g., `snapshot-610000`) of your model in `dlc-models/train` directory.
Edit the value of the `init_weights` key in the `pose_cfg.yaml` file and start to re-train your model!


`init_weights: pathofyourproject\dlc-models\iteration-0\DLCFeb31-trainset95shuffle1\train\snapshot-610000`

```python
deeplabcut.train_network("/pathofproject/config.yaml", shuffle=1, saveiters=25000)
```
```{hint}
Check this video for model refining!
<div align="center">
  <a href="https://www.youtube.com/watch?v=bgfnz1wtlpo" target="_blank"><img src="http://img.youtube.com/vi/bgfnz1wtlpo/0.jpg" alt="IMAGE ALT TEXT"></a>
</div>
```
