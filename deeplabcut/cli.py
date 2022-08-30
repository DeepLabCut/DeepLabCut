"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
from pathlib import Path

import click

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(invoke_without_command=True)
# @click.version_option()
@click.option("-v", "--verbose", is_flag=True, help="Verbose printing")
@click.pass_context
def main(ctx, verbose):
    if ctx.invoked_subcommand is None:
        click.echo("deeplabcut v0.0.")
        click.echo(main.get_help(ctx))


###########################################################################################################################
@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument("project")
@click.argument("experimenter")
@click.argument("videos", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-d",
    "--wd",
    "working_directory",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=Path.cwd(),
    help="Directory to create project in. Default is cwd().",
)
@click.option(
    "--copy_videos/--dont_copy_videos",
    is_flag=True,
    default=True,
    help="Specify if you need to create the symlinks of the video and store in the videos directory. Default is True.",
)
#              type=click.Path(exists=True, file_okay=False, resolve_path=True), default=Path.cwd(),
#              help='Directory to create project in. Default is cwd().')
@click.pass_context
def create_new_project(_, *args, **kwargs):
    """Create a new project directory, sub-directories and a basic configuration file. The configuration file is loaded with default values. Change its parameters to your projects need.\n

    Options \n
    ---------- \n
    project : string \n
    \tString containing the name of the project.\n
    experimenter : string \n
    \tString containing the name of the experimenter. \n
    videos : list \n
    \tA list of string containing the full paths of the videos to include in the project.\n
    working_directory : string, optional \n
    \tThe directory where the project will be created. The default is the ``current working directory``; if provided, it must be a string\n
    copy_videos : bool, optional \n
    If this is set to True, the symlink of the videos are copied to the project/videos directory. The default is ``True``; if provided it must be either ``True`` or ``False`` \n

    Example \n
    -------- \n
    To create the project in the current working directory \n
    python3 dlc.py create_new_project reaching-task Tanmay /data/videos/mouse1.avi /data/videos/mouse2.avi /data/videos/mouse3.avi /analysis/project/

    To create the project in the current working directory but do not want to create the symlinks \n
    python3 dlc.py create_new_project reaching-task Tanmay /data/videos/mouse1.avi /data/videos/mouse2.avi /data/videos/mouse3.avi /analysis/project/ -c False

    To create the project in another directory \n
    python3 dlc.py create_new_project reaching-task Tanmay /data/vies/mouse1.avi /data/videos/mouse2.avi /data/videos/mouse3.avi analysis/project -d home/project

    """
    from deeplabcut.create_project import new

    new.create_new_project(*args, **kwargs)


###########################################################################################################################


@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument("config")
@click.argument("videos", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--copy_videos/--dont_copy_videos",
    is_flag=True,
    default=True,
    help="Specify if you need to create the symlinks of the video and store in the videos directory. Default is True.",
)
@click.pass_context
def add_new_videos(_, *args, **kwargs):
    """
    Add new videos to the config file at any stage of the project.\n

    Options\n
    ----------\n
    config : string\n
        String containing the full path of the config file in the project.

    videos : list \n
        A list of string containing the full paths of the videos to include in the project.

    copy_videos : bool, optional\n
        If this is set to True, the symlink of the videos are copied to the project/videos directory. The default is
        ``True``; if provided it must be either ``True`` or ``False``

    Examples\n
    --------\n
    >>> python3 dlc.py add_new_videos /home/project/reaching-task-Tanmay-2018-08-23/config.yaml /data/videos/mouse5.avi

    """
    from deeplabcut.create_project import add

    add.add_new_videos(*args, **kwargs)


###########################################################################################################################
@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument("config")
@click.argument("mode")
@click.option(
    "-a",
    "--algo",
    "algo",
    default="uniform",
    help='For automatic extraction, specify the algorithm- "kmeans" or "uniform". Default is uniform.',
)
@click.option(
    "--crop",
    is_flag=True,
    default=False,
    help="Specify if you need to crop the image. Default is True.",
)
@click.pass_context
def extract_frames(_, *args, **kwargs):
    """
    Extracts frames from the videos in the config.yaml file. Only the videos in the config.yaml will be used to select the frames.\n
    Use the function ``add_new_videos`` at any stage of the project to add new videos to the config file and extract their frames.\n

    CONFIG : string \n
        Full path of the config.yaml file as a string.  \n \n \n
    MODE : string \n \n
        String containing the mode of extraction. It must be either ``automatic`` or ``manual``.  \n

    Examples \n
    -------- \n
    for selecting frames automatically with 'kmeans' and do not want to crop the frames \n
    >>> python3 dlc.py extract_frames /analysis/project/reaching-task/config.yaml automatic --algo kmeans \n
    -------- \n
    for selecting frames automatically with 'uniform' and want to crop the frames based on the ``crop`` parameters in config.yaml \n
    >>> python3 dlc.py extract_frames /analysis/project/reaching-task/config.yaml automatic --crop
    -------- \n
    for selecting frames manually, \n
    >>> deeplabcut.extract_frames /analysis/project/reaching-task/config.yaml manual \n
    While selecting the frames manually, you do not need to specify the cropping parameters. Rather, you will get a prompt in the graphic user interface to choose if you need to crop or not. \n
    -------- \n

    """
    from deeplabcut.generate_training_dataset import frameExtraction

    frameExtraction.extract_frames(*args, **kwargs)


###########################################################################################################################
@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument("config")
@click.pass_context
def label_frames(_, config):
    """Manually label/annotate the extracted frames. Update the list of body parts you want to localize in the config.yaml file first.\n
    Example\n
    --------\n
    python3 dlc.py label_frames /analysis/project/reaching-task/config.yaml
    """
    from deeplabcut.generate_training_dataset import labelFrames

    labelFrames.label_frames(config)


###########################################################################################################################
@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument("config")
@click.pass_context
def check_labels(_, config):
    """Check if labels were stored correctly by plotting annotations and inspect them visually. If some are wrong, then use the refine_labels to correct the labels.\n

    """
    from deeplabcut.generate_training_dataset import labelFrames

    labelFrames.check_labels(config)


###########################################################################################################################
@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument("config")
@click.option(
    "-num",
    "--num_shuffles",
    "num_shuffles",
    default=1,
    help="Number of shuffles of training dataset to create. Default is set to 1.",
)
@click.pass_context
def create_training_dataset(_, *args, **kwargs):
    """Combine frame and label information into a an array. Create training and test sets. Update parameters TrainFraction, iteration in config.yaml
	Also update parameters for pose_config.yaml as wanted.\n
    CONFIG: Full path of the config.yaml file in the train directory of a project.\n
    Example \n
    --------\n
    To create a training dataset with only 1 shuffle
    python3 dlc.py create_training_dataset /analysis/project/reaching-task/config.yaml

    To create a training dataset with only 2 shuffles
    python3 dlc.py create_training_dataset /analysis/project/reaching-task/config.yaml num_shuffles 2
    """
    from deeplabcut.generate_training_dataset import labelFrames

    labelFrames.create_training_dataset(*args, **kwargs)


###########################################################################################################################
@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument("config")
@click.option(
    "-num",
    "--num_shuffles",
    "shuffle",
    default=1,
    help="Shuffle index of the training dataset. Default is set to 1.",
)
@click.pass_context
def train_network(_, *args, **kwargs):
    """Train a trained Feature detector with a specific training data set.\n
	Provide path to the pose_config file.
        CONFIG: Full path of the config.yaml file in the train directory of a project.\n

    e.g. run the script like this:
    python3 dlc.py step7_train  /home/project/reaching/config.yaml

    """
    from deeplabcut.pose_estimation_tensorflow import training

    training.train_network(*args, **kwargs)


###########################################################################################################################
@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument("config")
@click.option(
    "-num",
    "--num_shuffles",
    "shuffle",
    default=[1],
    help="Shuffle index of the training dataset. Default is set to 1.",
)
@click.option(
    "-p", "--plot", "plotting", is_flag=True, help="Make plots. Default is False."
)
@click.pass_context
def evaluate_network(_, config, **kwargs):
    """Evaluates a trained Feature detector model.\n
        CONFIG: Full path of the "pose_config.yaml" file in the train directory of a project.\n

    Example\n
    ----------
    Evalaute the network
    python3 dlc.py evaluate_network  /home/project/reaching/config.yaml

    """
    from deeplabcut.pose_estimation_tensorflow import evaluate

    evaluate.evaluate_network(config, **kwargs)


###########################################################################################################################


@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument("config")
@click.argument("videos", nargs=-1)
@click.option(
    "-num",
    "--num_shuffles",
    "shuffle",
    default=1,
    help="Shuffle index of the training dataset. Default is set to 1.",
)
@click.option(
    "-vtype",
    "--video_type",
    "videotype",
    default=".avi",
    help="The extension of video in case the input is a directory",
)
@click.option(
    "-c",
    "--save",
    "save_as_csv",
    is_flag=True,
    help="Saves as a .csv file. Default is False.",
)
@click.pass_context
def analyze_videos(_, *args, **kwargs):

    """Makes prediction.\n
        CONFIG: Full path of the "config.yaml" file in the train directory of a project.\n
        VIDEOS: Full path to video.\n

    Example\n
    ----------

    python3 dlc.py analyze_videos /home/project/reaching/config.yaml /home/project/reaching/newVideo/1.avi

    """
    from deeplabcut.pose_estimation_tensorflow import predict_videos

    predict_videos.analyze_videos(*args, **kwargs)

    # for video in videos:
    #     predict.predict_video(config, video,**kwargs)


###########################################################################################################################


@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument("config")
@click.argument("videos")
@click.option(
    "-num",
    "--num_shuffles",
    "shuffle",
    default=1,
    help="The shuffle index of training dataset. The extracted frames will be stored in the labeled-dataset for the corresponding shuffle of training dataset. Default is set to 1",
)
@click.option(
    "-outlier",
    "--outlier_algo",
    "outlieralgorithm",
    default="fitting",
    help="String specifying the algorithm used to detect the outliers. Currently, deeplabcut supports only sarimax (this will be updated). \
              This method fits a Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors model to data and computes confidence interval. \
              Based on the fraction of data points outside the confidence interval and the average distance (compared to delta) \
              the user can identify potential outlier frames. The default is set to ``fitting``. Other choices: `fitting`, `jump`, `uncertain`",
)
@click.option(
    "-compare",
    "--comparisonbodyparts",
    "comparisonbodyparts",
    default="all",
    help="This select the body parts for which the comparisons with the outliers are carried out. Either ``all``, \
              then all body parts from config.yaml are used orr a list of strings that are a subset of the full list.\
               E.g. [`hand`,`Joystick`] for the demo Reaching-Mackenzie-2018-08-30/config.yaml to select only these two body parts.",
)
@click.option(
    "-e",
    "--epsilon",
    "epsilon",
    default=20,
    help="Meaning depends on outlieralgoritm. The default is set to 20 pixels.For outlieralgorithm `fitting`: \
              Float bound according to which frames are picked when the (average) body part estimate deviates from model fit. \
              For outlieralgorithm `jump`: Float bound specifying the distance by which body points jump from one frame to next (Euclidean distance)",
)
@click.option(
    "-p",
    "--p_bound",
    "p_bound",
    default=0.01,
    help="For outlieralgorithm `uncertain` this parameter defines the likelihood below, below which a body part will be flagged as a putative outlier.",
)
@click.option(
    "-ard",
    "--ar_degree",
    "ARdegree",
    default=7,
    help="For outlieralgorithm `fitting`: Autoregressive degree of Sarimax model degree. \
              See https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html",
)
@click.option(
    "-mad",
    "--ma_degree",
    "MAdegree",
    default=1,
    help="Int value. For outlieralgorithm `fitting`: MovingAvarage degree of Sarimax model degree.\
               See https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html",
)
@click.option(
    "-a",
    "--alpha",
    "alpha",
    default=0.01,
    help="Significance level for detecting outliers based on confidence interval of fitted SARIMAX model.",
)
@click.option(
    "-extract",
    "--extraction_algo",
    "extractionalgorithm",
    default="uniform",
    help="String specifying the algorithm to use for selecting the frames from the identified outliers. \
              Currently, deeplabcut supports either ``kmeans`` or ``uniform`` based selection (same logic as for extract_frames).\
              The default is set to``uniform``, if provided it must be either ``uniform`` or ``kmeans``.",
)
@click.pass_context
def extract_outlier_frames(_, *args, **kwargs):
    """
    Extracts the outlier frames in case, the predictions are not correct for a certain video from the cropped video running from
    start to stop as defined in config.yaml.

    Another crucial parameter in config.yaml is how many frames to extract 'numframes2extract'.

    CONFIG : string \n
    Full path of the config.yaml file as a string.  \n
    VIDEO : Full path of the video to extract the frame from. Make sure that this video is already analyzed.


    Example \n
    --------\n
    for extracting the frames with default settings\n
    >>> python3 dlc.py extract_outlier_frames /analysis/project/reaching-task/config.yaml /analysis/project/video/reachinvideo1.avi \n
    --------\n
    for extracting the frames with kmeans\n
    >>> python3 dlc.py extract_outlier_frames /analysis/project/reaching-task/config.yaml /analysis/project/video/reachinvideo1.avi --extractionalgorithm 'kmeans' \n
    --------\n
    for extracting the frames with kmeans and epsilon = 5 pixels.\n
    >>> python3 dlc.py extract_outlier_frames /analysis/project/reaching-task/config.yaml /analysis/project/video/reachinvideo1.avi --epsilon 5 --extractionalgorithm kmeans \n
    --------\n

    """
    from deeplabcut.refine_training_dataset import outlier_frames

    outlier_frames.extract_outlier_frames(*args, **kwargs)


###########################################################################################################################
@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument("config")
@click.pass_context
def refine_labels(_, config):
    """Refines the labels of the outlier frames extracted from the analyzed videos.\n Helps in augmenting the training dataset.
    Use the function ``analyze_video`` to analyze a video and extracts the outlier frames using the function
    ``extract_outlier_frames`` before refining the labels.\n

    Examples \n
    --------\n
    >>> python3 dlc.py refine_labels /analysis/project/reaching-task/config.yaml \n
    --------\n
    """
    from deeplabcut.refine_training_dataset import outlier_frames

    outlier_frames.refine_labels(config)


###########################################################################################################################
@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument("config")
@click.argument("videos", nargs=-1)
@click.option(
    "-num",
    "--num_shuffles",
    "shuffle",
    default=1,
    help="Number of shuffles of training dataset. Default is set to 1.",
)
@click.option(
    "-v",
    "--video_type",
    "videotype",
    default=".avi",
    help="Checks for the extension of the video in case the input is a directory.\
              Only videos with this extension are analyzed. The default is ``.avi``",
)
@click.option(
    "-s",
    "--save_frames",
    "save_frames",
    is_flag=True,
    default=False,
    help="If true creates each frame individual and then combines into a video. \
              This variant is relatively slow as it stores all individual frames. However, it \
              uses matplotlib to create the frames and is therefore much more flexible \
              (one can set transparency of markers, crop, and easily customize.",
)
@click.option(
    "-d",
    "--delete",
    "delete",
    is_flag=True,
    default=False,
    help="If true then the individual frames created during the video generation will be deleted.\
              Only the video will be left.",
)
@click.pass_context
def create_labeled_video(_, *args, **kwargs):
    """
    Labels the bodyparts in a video. Make sure the video is already analyzed by the function 'analyze_video'
    """
    from deeplabcut.utils import make_labeled_video

    make_labeled_video.create_labeled_video(*args, **kwargs)


###########################################################################################################################
@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument("config")
@click.argument("videos", nargs=-1)
@click.option(
    "-num",
    "--num_shuffles",
    "shuffle",
    default=1,
    help="Number of shuffles of training dataset. Default is set to 1.",
)
@click.option(
    "-v",
    "--video_type",
    "videotype",
    default=".avi",
    help="Checks for the extension of the video in case the input is a directory.\
              Only videos with this extension are analyzed. The default is ``.avi``",
)
@click.option(
    "-s",
    "--show",
    "showfigures",
    is_flag=True,
    default=False,
    help="If true then plots are also displayed simultaneously.",
)
@click.pass_context
def plot_trajectories(_, *args, **kwargs):
    """
    Plots the trajectories of various bodyparts across the video.\n

    Example\n
    --------\n
    for labeling the frames\n
    >>> python3 dlc.py plot_trajectories /analysis/project/reaching-task/config.yaml /analysis/project/videos/reachingvideo1.avi  \n
    --------\n
    """
    from deeplabcut.utils import plotting

    plotting.plot_trajectories(*args, **kwargs)


###########################################################################################################################
@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument("cfg-path", nargs=1, type=click.STRING)
@click.option(
    "-i",
    "--iteration",
    "iteration",
    default=None,
    required=False,
    type=int,
    help="the model iteration you wish to export. If None, uses the iteration listed in the config file",
)
@click.option(
    "-s",
    "--shuffle",
    "shuffle",
    default=1,
    required=False,
    type=int,
    help="the shuffle of the model to export. Default is set to 1.",
)
@click.option(
    "-t",
    "--trainingsetindex",
    "trainingsetindex",
    default=0,
    required=False,
    type=int,
    help="the index of the training fraction for the model you wish to export. default = 0",
)
@click.option(
    "-n",
    "--snapshotindex",
    "snapshotindex",
    default=None,
    required=False,
    type=int,
    help="the snapshot index for the weights you wish to export",
)
@click.option(
    "--TFGPUinference/--NPinference",
    "TFGPUinference",
    default=True,
    required=False,
    help="use the tensorflow inference model? Default = True",
)
@click.option(
    "--overwrite",
    "-o",
    is_flag=True,
    required=False,
    help="if the model you wish to export has already been exported, whether to overwrite. default = False",
)
@click.option(
    "--make-tar/--no-tar",
    "make_tar",
    default=True,
    required=False,
    help="Do you want to compress the exported directory to a tar file? Default = True",
)
@click.pass_context
def export_model(_, *args, **kwargs):
    """
    Export DLC models for the model zoo or for live inference.\n
    Saves the pose configuration, snapshot files, and frozen graph of the model to a directory named exported-models within the project directory

    Parameters
    -----------

    cfg_path : string\n
    \tpath to the DLC Project config.yaml file

    iteration : int, optional\n
    \tthe model iteration you wish to export.\n
    \tIf None, uses the iteration listed in the config file

    shuffle : int, optional\n
    \tthe shuffle of the model to export. default = 1

    trainingsetindex : int, optional\n
    \tthe index of the training fraction for the model you wish to export. default = 1

    snapshotindex : int, optional\n
    \tthe snapshot index for the weights you wish to export.\n
    \tIf None, uses the snapshotindex as defined in 'config.yaml'. Default = None

    TFGPUinference : bool, optional\n
    \tuse the tensorflow inference model? Default = True\n
    \tFor inference using DeepLabCut-live, it is recommended to set TFGPIinference=False

    overwrite : bool, optional\n
    \tif the model you wish to export has already been exported, whether to overwrite. default = False

    make_tar : bool, optional\n
    \tDo you want to compress the exported directory to a tar file? Default = True\n
    \tThis is necessary to export to the model zoo, but not for live inference.
    """

    from deeplabcut import export_model

    export_model(*args, **kwargs)


###########################################################################################################################
