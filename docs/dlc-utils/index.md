---
deeplabcut:
  last_metadata_updated: '2026-05-06'
  last_verified: '2026-05-06'
  verified_for: 3.0.0rc14
  ignore: false
---

# DeepLabCut-Utils - Community contributions

[![Image.sc forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&url=https%3A%2F%2Fforum.image.sc%2Ftags%2Fdeeplabcut.json&query=%24.topic_list.tags.0.topic_count&colorB=brightgreen&suffix=%20topics&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC)](https://forum.image.sc/tags/deeplabcut)

```{image} https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1572296495650-Y4ZTJ2XP2Z9XF1AD74VW/ke17ZwdGBToddI8pDm48kMulEJPOrz9Y8HeI7oJuXxR7gQa3H78H3Y0txjaiv_0fDoOvxcdMmMKkDsyUqMSsMWxHk725yiiHCCLfrh8O1z5QPOohDIaIeljMHgDF5CVlOqpeNLcJ80NK65_fV7S1UZiU3J6AN9rgO1lHw9nGbkYQrCLTag1XBHRgOrY8YAdXW07ycm2Trb21kYhaLJjddA/DLC_logo_blk-01.png?format=1000w
---
alt: DLC Utils
width: 350px
align: right
---
```

The DeepLabCut-Utils repository contains various scripts as well as links to other packages related to [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut). Feel free to contribute your own analysis methods, and perhaps some short notebook of how to use it. Thanks!

```{admonition} DLC-Utils
---
class: tip
---
[Link to repository](https://github.com/DeepLabCut/DLCutils)
```

```{caution}
Please direct inquiries to the **contributors/code maintainers of that code**. Note that the software(s) are provided "as is", without warranty of any kind, express or implied.
The DeepLabCut team is not responsible for the maintenance of these packages, and cannot guarantee that they will work with present & future versions of DeepLabCut.
```

## Example scripts for scaling up your DLC analysis & training

These two scripts illustrate how to train, test, and analyze videos for multiple projects automatically (`scale_training_and_evaluation.py`) and how to analyze videos that are organized in subfolders automatically (`scale_analysis_oversubfolders.py`). Feel free to adjust them for your needs!

- [Code: `scale_analysis_oversubfolders.py`](https://github.com/DeepLabCut/DLCutils/tree/master/SCALE_YOUR_ANALYSIS/scale_analysis_oversubfolders.py)
- [Code: `scale_training_and_evaluation.py`](https://github.com/DeepLabCut/DLCutils/blob/master/SCALE_YOUR_ANALYSIS/scale_training_and_evaluation.py)

Contributed by [Alexander Mathis](https://github.com/AlexEMG)

## Using DLC + XROMM_DLCTools + xmalab

> Contributed by [@homfunc](https://github.com/homfunc)

The DeepLabCut repository can also be used as a sibling checkout together with:

- `../XROMM_DLCTools`
- `../xmalab`

In that local layout, `XROMM_DLCTools` uses the optional `dlc` dependency group to import this checkout directly, and its baseline harness runs both a DeepLabCut smoke scenario and a broader end-to-end local workflow integration scenario.

See {ref}`file:xamalab-dlc-integration` for the local integration notes and exact commands.

## Using your DLC outputs, loading, simple ROI analysis, visualization examples

### Time spent of a body part in a particular region of interest (ROI)

You can compute time spent in particular ROIs in frames. This demo Jupyter Notebook shows you how to load the outputs of DLC and perform the analysis (plus other plotting functions):

- [Code: `Demo_loadandanalyzeDLCdata.ipynb`](https://github.com/DeepLabCut/DLCutils/blob/master/Demo_loadandanalyzeDLCdata.ipynb)
- [Code: `time_in_each_roi.py`](https://github.com/DeepLabCut/DLCutils/blob/master/time_in_each_roi.py)

Contributed by [Federico Claudi](https://github.com/FedeClaudi) and Jupyter Notebook from [Alexander Mathis](https://github.com/AlexEMG)

### DeepLabCut-Display GUI

Open and view data to understand pose estimation errors and trends. Filter data by likelihood threshold.

- [Code: `DeepLabCut-Display`](https://github.com/jakeshirey/DeepLabCut-Display)

Contributed by [Jacob Shirey](https://github.com/jakeshirey)

### A GUI-based ROI tool for time spent of a body part in a defined region of interest

- [Code: `DLC_ROI_tool`](https://github.com/PolarBean/DLC_ROI_tool)

Contributed by [Harry Carey](https://github.com/PolarBean)

### Linear transformation and scaling of DLC output data (`transform_and_scale`)

This package is designed for anyone who wants to know where a tracked marker is within a reference frame (i.e. behavioral context). DeepLabCut outputs coordinates in relation to the field of view of the recorded video. With this tool, these coordinates can be linearly transformed and scaled to the reference frame of the behavioral context, meaning that the output coordinates are distances [cm] to a corner of the behavioral context, instead of distances [px] to a corner of the video field of view.

- [Code: `transform_and_scale`](https://github.com/DeepLabCut/DLCutils/tree/master/transform_and_scale/)
- [Tutorial: `transform_and_scale_tutorial.ipynb`](https://github.com/DeepLabCut/DLCutils/tree/master/transform_and_scale/transform_and_scale_tutorial.ipynb)

Contributed by [Michael Schellenberger](https://github.com/MSchellenberger)

## Clustering tools (using the output of DLC)

### Identifying Behavioral Structure from Deep Variational Embeddings of Animal Motion

- [Paper](https://www.biorxiv.org/content/10.1101/2020.05.14.095430)
- [Code: `VAME`](https://github.com/LINCellularNeuroscience/VAME)

### Behavior clustering with MotionMapper

- Adapted from [MotionMapper](https://github.com/gordonberman/MotionMapper)
- [Code: `DLC_2_MotionMapper`](https://github.com/DeepLabCut/DLCutils/tree/master/DLC_2_MotionMapper)

Contributed by [Mackenzie Mathis](https://github.com/MMathisLab)

### Behavior clustering with B-SOiD

B-SOiD is an open source unsupervised algorithm for discovery of spontaneous behaviors, and you can use the outputs of DLC to feed directly into B-SOiD in MATLAB.

- [Paper](https://www.biorxiv.org/content/10.1101/770271v1.abstract)
- [Code: `B-SOiD`](https://github.com/YttriLab/B-SOiD)

## Machine-learning helper packages (using the output of DLC)

### Behavior analysis with machine learning in R (`ETH-DLCAnalyzer`)

Deep learning based behavioral analysis enables high precision rodent tracking and is capable of outperforming commercial solutions. Oliver Sturman, Lukas von Ziegler, Christa Schläppi, Furkan Akyol, Benjamin Grewe, Johannes Bohacek

- [Paper](https://www.biorxiv.org/content/10.1101/2020.01.21.913624v1)
- [Code: `DLCAnalyzer`](https://github.com/ETHZ-INS/DLCAnalyzer)

### Behavior analysis with machine learning classifiers (SimBA)

A pipeline for using pose estimation (i.e. DeepLabCut) then behavioral annotation and generation of supervised machine-learning-based classifiers. \<-- you can use the outputs of DLC to feed directly into SimBA (in Python).

Code written by: [Simon Nilsson](https://github.com/sronilsson) (please direct use questions to Simon).

- [Paper](https://www.biorxiv.org/content/10.1101/2020.04.19.049452v2)
- [Code: `simba`](https://github.com/sgoldenlab/simba)

## 3D DeepLabCut helper packages

### A wrapper package for DeepLabCut 2.0 for 3D videos (`anipose`)

- [Code: `anipose`](https://github.com/lambdaloop/anipose)

Maintainer: [Pierre Karashchuk](https://github.com/lambdaloop)

### 3D reconstruction with EasyWand/Argus DLT system with DeepLabCut data

Written by [Brandon Jackson](https://github.com/haliaetus13), post our DLC workshop in Jan 2020:

A small set of utilities that allow conversion between the data storage formats of DeepLabCut (DLC) and one of the DLT-based 3D tracking systems: either Ty Hedrick's DigitizingTools in MATLAB, or the Python-based Argus. These functions should allow you to use data previously digitized in a DLT system to create the files needed to train a DLC model, and to import DLC-tracked points back into a DLT 3D calibration to reconstruct 3D points.

- [Code: `DLCconverterDLT`](https://github.com/haliaetus13/DLCconverterDLT)

### Pupil Tracking

- From Tom Vaissie - tvaissie@scripps.edu
- Please see the [README.txt file](https://github.com/DeepLabCut/DLCutils/tree/master/pupilTracking) for details; this code makes the video in case study 7 [http://www.mousemotorlab.org/deeplabcut/](http://www.mousemotorlab.org/deeplabcut/).

### Using DeepLabCut for USB-CGPIO feedback

- [Paper](https://www.biorxiv.org/content/early/2018/11/28/482349)
- [Code: `DeepCutRealTime`](https://github.com/bf777/DeepCutRealTime)

Maintainer: [Brandon Forys](https://github.com/bf777)

## Legacy utility functions (no longer required in DLC 2+)

```{warning}
These utilities are marked as legacy and are no longer required in DLC 2+.
```

### DLC 1 to DLC 2 conversion code

This code allows you to import the labeled data from DLC 1 to DLC 2 projects. Note, it is not streamlined and should be used with care.

- [Conversion scripts (`conversion_scripts_LEGACY`)](https://github.com/DeepLabCut/DLCutils/tree/master/conversion_scripts_LEGACY)

Contributed by [Alexander Mathis](https://github.com/AlexEMG)

### Running project created on Windows on Colaboratory

```{note}
**UPDATE:** as of DeepLabCut 2.0.4 onwards you no longer need to use this code! You can simply create the training set on the cloud and it will automatically convert your project for you.
```

- This solves a path problem when creating a project and annotating data on Windows (see [issue #172](https://github.com/AlexEMG/DeepLabCut/issues/172)). This functionality will be included in a later version of DLC 2 (DONE!)
- [Conversion scripts (`conversion_scripts_LEGACY`)](https://github.com/DeepLabCut/DLCutils/tree/master/conversion_scripts_LEGACY)

*Usage:* change in lines 70 and 71 of [`convertWin2Unix.py`](https://github.com/DeepLabCut/DLCutils/tree/master/conversion_scripts_LEGACY/convertWin2Unix.py)

```python
basepath='/content/drive/My Drive/DeepLabCut/examples/'
projectname='Reaching-Mackenzie-2018-08-30'
```

then run this script on Colaboratory after uploading your labeled data to the drive. Thereby it will be converted to Unix format, then create a training set (with DeepLabCut) and proceed as usual...

Contributed by [Alexander Mathis](https://github.com/AlexEMG)
