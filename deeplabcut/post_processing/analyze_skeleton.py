import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import argparse
from scipy.spatial import distance
from math import factorial, atan2, degrees, acos, sqrt, pi

from deeplabcut.utils import auxiliaryfunctions

# utility functions
def calc_distance_between_points_two_vectors_2d(v1, v2):
    '''calc_distance_between_points_two_vectors_2d [pairwise distance between vectors points]

    Arguments:
        v1 {[np.array]} -- [description]
        v2 {[type]} -- [description]

    Raises:
        ValueError -- [description]
        ValueError -- [description]
        ValueError -- [description]

    Returns:
        [type] -- [description]

    testing:
    >>> v1 = np.zeros((2, 5))
    >>> v2 = np.zeros((2, 5))
    >>> v2[1, :]  = [0, 10, 25, 50, 100]
    >>> d = calc_distance_between_points_two_vectors_2d(v1.T, v2.T)
    '''

    # Check dataformats
    if not isinstance(v1, np.ndarray) or not isinstance(v2, np.ndarray):
        raise ValueError('Invalid argument data format')
    if not v1.shape[1] == 2 or not v2.shape[1] == 2:
        raise ValueError('Invalid shape for input arrays')
    if not v1.shape[0] == v2.shape[0]:
        raise ValueError('Error: input arrays should have the same length')

    # Calculate distance
    dist = [distance.euclidean(p1, p2) for p1, p2 in zip(v1, v2)]
    return dist


def angle_between_points_2d_anticlockwise(p1, p2):
    '''angle_between_points_2d_clockwise [Determines the angle of a straight line drawn between point one and two.
        The number returned, which is a double in degrees, tells us how much we have to rotate
        a horizontal line anti-clockwise for it to match the line between the two points.]

    Arguments:
        p1 {[np.ndarray, list]} -- np.array or list [ with the X and Y coordinates of the point]
        p2 {[np.ndarray, list]} -- np.array or list [ with the X and Y coordinates of the point]

    Returns:
        [int] -- [clockwise angle between p1, p2 using the inner product and the deterinant of the two vectors]

    Testing:  - to check:     print(zero, ninety, oneeighty, twoseventy)
        >>> zero = angle_between_points_2d_clockwise([0, 1], [0, 1])
        >>> ninety = angle_between_points_2d_clockwise([1, 0], [0, 1])
        >>> oneeighty = angle_between_points_2d_clockwise([0, -1], [0, 1])
        >>> twoseventy = angle_between_points_2d_clockwise([-1, 0], [0, 1])
        >>> ninety2 = angle_between_points_2d_clockwise([10, 0], [10, 1])
        >>> print(ninety2)
    '''

    """
        Determines the angle of a straight line drawn between point one and two.
        The number returned, which is a double in degrees, tells us how much we have to rotate
        a horizontal line anit-clockwise for it to match the line between the two points.
    """

    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    ang = degrees(atan2(yDiff, xDiff))
    if ang < 0: ang += 360
    # if not 0 <= ang <+ 360: raise ValueError('Ang was not computed correctly')
    return ang



def calc_angle_between_vectors_of_points_2d(v1, v2):
    '''calc_angle_between_vectors_of_points_2d [calculates the clockwise angle between each set of point for two 2d arrays of points]

    Arguments:
        v1 {[np.ndarray]} -- [2d array with X,Y position at each timepoint]
        v2 {[np.ndarray]} -- [2d array with X,Y position at each timepoint]

    Returns:
        [np.ndarray] -- [1d array with clockwise angle between pairwise points in v1,v2]

    Testing:
    >>> v1 = np.zeros((2, 4))
    >>> v1[1, :] = [1, 1, 1, 1, ]
    >>> v2 = np.zeros((2, 4))
    >>> v2[0, :] = [0, 1, 0, -1]
    >>> v2[1, :] = [1, 0, -1, 0]
    >>> a = calc_angle_between_vectors_of_points_2d(v2, v1)
    '''

    # Check data format
    if v1 is None or v2 is None or not isinstance(v1, np.ndarray) or not isinstance(v2, np.ndarray):
        raise ValueError('Invalid format for input arguments')
    if len(v1) != len(v2):
        raise ValueError('Input arrays should have the same length, instead: ', len(v1), len(v2))
    if not v1.shape[0] == 2 or not v2.shape[0] == 2:
        raise ValueError('Invalid shape for input arrays: ', v1.shape, v2.shape)

    # Calculate
    n_points = v1.shape[1]
    angs = np.zeros(n_points)
    for i in range(v1.shape[1]):
        p1, p2 = v1[:, i], v2[:, i]
        angs[i] = angle_between_points_2d_anticlockwise(p1, p2)

    return angs


# Process single bone
def analyzebone(bp1, bp2):
    """[Computes length and orientation of the bone at each frame]

    Arguments:
        bp1 {[type]} -- [description]
        bp2 {[type]} -- [description]
    """
    bp1_pos = np.vstack([bp1.x.values, bp1.y.values]).T
    bp2_pos = np.vstack([bp2.x.values, bp2.y.values]).T

    # get bone length and orientation
    bone_length = calc_distance_between_points_two_vectors_2d(bp1_pos, bp2_pos)
    bone_orientation = calc_angle_between_vectors_of_points_2d(bp1_pos.T, bp2_pos.T)

    # keep the smallest of the two likelihoods
    likelihoods = np.vstack([bp2.likelihood.values, bp2.likelihood.values]).T
    likelihood = np.min(likelihoods, 1)

    # Create dataframe and return
    df = pd.DataFrame.from_dict(dict(
                                    length=bone_length,
                                    orientation=bone_orientation,
                                    likelihood=likelihood,
                                    ))
    # df.index.name=name

    return df

# MAIN FUNC
def analyzeskeleton(config, videos, videotype='avi', shuffle=1, trainingsetindex=0, save_as_csv=False, destfolder=None):
    """
    Extracts length and orientation of each "bone" of the skeleton as defined in the config file.

    Parameter
    ----------
    config : string
        Full path of the config.yaml file as a string.

    videos : list
        A list of strings containing the full paths to videos for analysis or a path to the directory, where all the videos with same extension are stored.

    shuffle : int, optional
        The shufle index of training dataset. The extracted frames will be stored in the labeled-dataset for
        the corresponding shuffle of training dataset. Default is set to 1

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    save_as_csv: bool, optional
        Saves the predictions in a .csv file. The default is ``False``; if provided it must be either ``True`` or ``False``

    destfolder: string, optional
        Specifies the destination folder for analysis data (default is the path of the video). Note that for subsequent analysis this
        folder also needs to be passed.
    """
    # Load config file, scorer and videos
    cfg = auxiliaryfunctions.read_config(config)
    DLCscorer,DLCscorerlegacy=auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction = cfg['TrainingFraction'][trainingsetindex])

    Videos=auxiliaryfunctions.Getlistofvideos(videos,videotype)
    for video in Videos:
        print("Processing %s"%(video))
        if destfolder is None:
            destfolder= str(Path(video).parents[0])

        vname=Path(video).stem
        notanalyzed,outdataname,sourcedataname,scorer=auxiliaryfunctions.CheckifPostProcessing(destfolder,vname,DLCscorer,DLCscorerlegacy,suffix='_skeleton')
        if notanalyzed:
                Dataframe = pd.read_hdf(sourcedataname,'df_with_missing')
                # Process skeleton
                bones = {}
                for bp1, bp2 in cfg['skeleton']:
                    name = "{}_{}".format(bp1, bp2)
                    bones[name] = analyzebone(Dataframe[scorer][bp1], Dataframe[scorer][bp2])

                skeleton = pd.concat(bones, axis=1)
                # save
                skeleton.to_hdf(outdataname, 'df_with_missing', format='table', mode='w')
                if save_as_csv:
                    skeleton.to_csv(outdataname.split('.h5')[0]+'.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('videos')
    cli_args = parser.parse_args()
