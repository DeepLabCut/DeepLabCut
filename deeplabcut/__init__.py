"""
DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu


"""

import os
import platform
# Supress tensorflow warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DEBUG = True and 'DEBUG' in os.environ and os.environ['DEBUG']
from deeplabcut import DEBUG
import os

import matplotlib as mpl
if os.environ.get('DLClight', default=False) == 'True':
    mpl.use('AGG') #anti-grain geometry engine #https://matplotlib.org/faq/usage_faq.html
    pass
elif platform.system() == 'Darwin':
    mpl.use('WXAgg')
    from deeplabcut import generate_training_dataset
    from deeplabcut import refine_training_dataset
# Direct import for convenience
    from deeplabcut.generate_training_dataset import label_frames, dropannotationfileentriesduetodeletedimages, comparevideolistsanddatafolders, adddatasetstovideolistandviceversa,  dropduplicatesinannotatinfiles
    from deeplabcut.refine_training_dataset import refine_labels
else:
    mpl.use('TkAgg')
    from deeplabcut import generate_training_dataset
    from deeplabcut import refine_training_dataset

    #Direct import for convenience
    from deeplabcut.generate_training_dataset import label_frames, dropannotationfileentriesduetodeletedimages, comparevideolistsanddatafolders, adddatasetstovideolistandviceversa,  dropduplicatesinannotatinfiles
    from deeplabcut.refine_training_dataset import refine_labels
    from deeplabcut.utils import select_crop_parameters

#from deeplabcut import create_project
from deeplabcut import pose_estimation_tensorflow
from deeplabcut import utils
from deeplabcut.create_project import create_new_project, add_new_videos, load_demo_data
from deeplabcut.generate_training_dataset import extract_frames
from deeplabcut.generate_training_dataset import check_labels,create_training_dataset, mergeandsplit

if os.environ.get('Colab', default=False) == 'True':
    print("Project loaded in colab-mode. Apparently Colab has trouble loading statsmodels, so the smooting & outlier frame extraction is disabled. Sorry!")
else:
    from deeplabcut.refine_training_dataset import extract_outlier_frames, merge_datasets, filterpredictions

#Direct import for convenience
from deeplabcut.pose_estimation_tensorflow import train_network
from deeplabcut.pose_estimation_tensorflow import evaluate_network
from deeplabcut.pose_estimation_tensorflow import analyze_videos, analyze_time_lapse_frames

from deeplabcut.utils import create_labeled_video, plot_trajectories, auxiliaryfunctions, convertcsv2h5, analyze_videos_converth5_to_csv
from deeplabcut.version import __version__, VERSION
