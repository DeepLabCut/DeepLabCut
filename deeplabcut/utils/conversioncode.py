"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os
import pandas as pd
from deeplabcut.utils import auxiliaryfunctions
from pathlib import Path


def convertcsv2h5(config, userfeedback=True, scorer=None):
    """
    Convert (image) annotation files in folder labeled-data from csv to h5.
    This function allows the user to manually edit the csv (e.g. to correct the scorer name and then convert it into hdf format).
    WARNING: conversion might corrupt the data.

    config : string
        Full path of the config.yaml file as a string.

    userfeedback: bool, optional
        If true the user will be asked specifically for each folder in labeled-data if the containing csv shall be converted to hdf format.

    scorer: string, optional
        If a string is given, then the scorer/annotator in all csv and hdf files that are changed, will be overwritten with this name.

    Examples
    --------
    Convert csv annotation files for reaching-task project into hdf.
    >>> deeplabcut.convertcsv2h5('/analysis/project/reaching-task/config.yaml')

    --------
    Convert csv annotation files for reaching-task project into hdf while changing the scorer/annotator in all annotation files to Albert!
    >>> deeplabcut.convertcsv2h5('/analysis/project/reaching-task/config.yaml',scorer='Albert')
    --------
    """
    cfg = auxiliaryfunctions.read_config(config)
    videos = cfg["video_sets"].keys()
    video_names = [Path(i).stem for i in videos]
    folders = [Path(config).parent / "labeled-data" / Path(i) for i in video_names]
    if not scorer:
        scorer = cfg["scorer"]

    for folder in folders:
        try:
            if userfeedback:
                print("Do you want to convert the csv file in folder:", folder, "?")
                askuser = input("yes/no")
            else:
                askuser = "yes"

            if askuser in ("y", "yes", "Ja", "ha", "oui"):  # multilanguage support :)
                fn = os.path.join(
                    str(folder), "CollectedData_" + cfg["scorer"] + ".csv"
                )
                # Determine whether the data are single- or multi-animal without loading into memory
                # simply by checking whether 'individuals' is in the second line of the CSV.
                with open(fn) as datafile:
                    next(datafile)
                    if "individuals" in next(datafile):
                        header = list(range(4))
                    else:
                        header = list(range(3))
                data = pd.read_csv(fn, index_col=0, header=header)
                data.columns = data.columns.set_levels([scorer], level="scorer")
                data.to_hdf(fn.replace(".csv", ".h5"), key="df_with_missing", mode="w")
                data.to_csv(fn)
        except FileNotFoundError:
            print("Attention:", folder, "does not appear to have labeled data!")


def analyze_videos_converth5_to_csv(video_folder, videotype=".mp4"):
    """
    By default the output poses (when running analyze_videos) are stored as MultiIndex Pandas Array, which contains the name of the network, body part name, (x, y) label position \n
    in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient Hierarchical Data Format (HDF) \n
    in the same directory, where the video is stored. If the flag save_as_csv is set to True, the data is also exported as comma-separated value file. However,
    if the flag was *not* set, then this function allows the conversion of all h5 files to csv files (without having to analyze the videos again)!

    This functions converts hdf (h5) files to the comma-separated values format (.csv), which in turn can be imported in many programs, such as MATLAB, R, Prism, etc.

    Parameters
    ----------

    video_folder : string
        Absolute path of a folder containing videos and the corresponding h5 data files.

    videotype: string, optional (default=.mp4)
        Only videos with this extension are screened.

    Examples
    --------

    Converts all pose-output files belonging to mp4 videos in the folder '/media/alex/experimentaldata/cheetahvideos' to csv files.
    deeplabcut.analyze_videos_converth5_to_csv('/media/alex/experimentaldata/cheetahvideos','.mp4')

    """
    h5_files = list(
        auxiliaryfunctions.grab_files_in_folder(video_folder, "h5", relative=False)
    )
    videos = auxiliaryfunctions.grab_files_in_folder(
        video_folder, videotype, relative=False
    )
    for video in videos:
        if "_labeled" in video:
            continue
        vname = Path(video).stem
        for file in h5_files:
            if vname in file:
                scorer = file.split(vname)[1].split(".h5")[0]
                if "DLC" in scorer or "DeepCut" in scorer:
                    print("Found output file for scorer:", scorer)
                    print(f"Converting {file}...")
                    df = pd.read_hdf(file)
                    df.to_csv(file.replace(".h5", ".csv"))
    print("All pose files were converted.")


def merge_windowsannotationdataONlinuxsystem(cfg):
    """ If a project was created on Windows (and labeled there,) but ran on unix then the data folders
    corresponding in the keys in cfg['video_sets'] are not found. This function gets them directly by
    looping over all folders in labeled-data """

    AnnotationData = []
    data_path = Path(cfg["project_path"], "labeled-data")
    annotationfolders = []
    for elem in auxiliaryfunctions.grab_files_in_folder(data_path, relative=False):
        if os.path.isdir(elem):
            annotationfolders.append(elem)
    print("The following folders were found:", annotationfolders)
    for folder in annotationfolders:
        filename = os.path.join(folder, "CollectedData_" + cfg["scorer"] + ".h5")
        try:
            data = pd.read_hdf(filename)
            AnnotationData.append(data)
        except FileNotFoundError:
            print(filename, " not found (perhaps not annotated)")

    return AnnotationData


def guarantee_multiindex_rows(df):
    # Make paths platform-agnostic if they are not already
    if not isinstance(df.index, pd.MultiIndex):  # Backwards compatibility
        path = df.index[0]
        sep = "/" if "/" in path else "\\"
        splits = tuple(df.index.str.split(sep))
        df.index = pd.MultiIndex.from_tuples(splits)


def robust_split_path(s):
    sep = "/" if "/" in s else "\\"
    return tuple(s.split(sep))
