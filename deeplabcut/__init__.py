"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os
import platform

# Supress tensorflow warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DEBUG = True and 'DEBUG' in os.environ and os.environ['DEBUG']
from deeplabcut import DEBUG

# DLClight version does not support GUIs. Importing accordingly
import matplotlib as mpl
if os.environ.get('DLClight', default=False) == 'True':
    print("DLC loaded in light mode; you cannot use any GUI (labeling, relabeling and standalone GUI)")
    mpl.use('AGG') #anti-grain geometry engine #https://matplotlib.org/faq/usage_faq.html
    pass
else: #standard use [wxpython supported]
    if platform.system() == 'Darwin': #for OSX use WXAgg
        mpl.use('WXAgg')
    else:
        mpl.use('Agg')
    from deeplabcut import generate_training_dataset
    from deeplabcut import refine_training_dataset
    from deeplabcut.generate_training_dataset import label_frames, dropannotationfileentriesduetodeletedimages, comparevideolistsanddatafolders, dropimagesduetolackofannotation
    from deeplabcut.generate_training_dataset import multiple_individuals_labeling_toolbox
    from deeplabcut.generate_training_dataset import adddatasetstovideolistandviceversa,  dropduplicatesinannotatinfiles
    from deeplabcut.gui.launch_script import launch_dlc

    from deeplabcut.refine_training_dataset import refine_labels
    from deeplabcut.utils import select_crop_parameters

if os.environ.get('Colab', default=False) == 'True':
    print("Project loaded in colab-mode. Apparently Colab has trouble loading statsmodels, so the smoothing & outlier frame extraction is disabled. Sorry!")
else:
    from deeplabcut.refine_training_dataset import extract_outlier_frames, merge_datasets
    from deeplabcut.post_processing import filterpredictions, analyzeskeleton


# Train, evaluate & predict functions / require TF
from deeplabcut.pose_estimation_tensorflow import train_network, return_train_network_path
from deeplabcut.pose_estimation_tensorflow import evaluate_network, return_evaluate_network_data
from deeplabcut.pose_estimation_tensorflow import analyze_videos, analyze_time_lapse_frames

from deeplabcut.pose_estimation_3d import calibrate_cameras,check_undistortion,triangulate,create_labeled_video_3d

from deeplabcut.create_project import create_new_project, create_new_project_3d, add_new_videos, load_demo_data, create_pretrained_human_project
from deeplabcut.generate_training_dataset import extract_frames, select_cropping_area
from deeplabcut.generate_training_dataset import check_labels,create_training_dataset, mergeandsplit, create_training_model_comparison
from deeplabcut.utils import create_labeled_video,plot_trajectories, auxiliaryfunctions, convertcsv2h5, convertannotationdata_fromwindows2unixstyle, analyze_videos_converth5_to_csv, auxfun_videos
from deeplabcut.utils.auxfun_videos import ShortenVideo, DownSampleVideo

from deeplabcut.version import __version__, VERSION
