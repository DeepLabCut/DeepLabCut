#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
import os
import pandas as pd
from deeplabcut.utils import auxiliaryfunctions
from itertools import islice
from pathlib import Path


SUPPORTED_FILETYPES = "csv", "nwb"


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
                    head = list(islice(datafile, 0, 5))
                if "individuals" in head[1]:
                    header = list(range(4))
                else:
                    header = list(range(3))
                if head[-1].split(",")[0] == "labeled-data":
                    index_col = [0, 1, 2]
                else:
                    index_col = 0
                data = pd.read_csv(fn, index_col=index_col, header=header)
                data.columns = data.columns.set_levels([scorer], level="scorer")
                guarantee_multiindex_rows(data)
                data.to_hdf(fn.replace(".csv", ".h5"), key="df_with_missing", mode="w")
                data.to_csv(fn)
        except FileNotFoundError:
            print("Attention:", folder, "does not appear to have labeled data!")


def analyze_videos_converth5_to_csv(video_folder, videotype=".mp4", listofvideos=False):
    """
    By default the output poses (when running analyze_videos) are stored as MultiIndex Pandas Array, which contains the name of the network, body part name, (x, y) label position \n
    in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient Hierarchical Data Format (HDF) \n
    in the same directory, where the video is stored. This functions converts hdf (h5) files to the comma-separated values format (.csv),
    which in turn can be imported in many programs, such as MATLAB, R, Prism, etc.

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

    if listofvideos:  # can also be called with a list of videos (from GUI)
        videos = video_folder  # GUI gives a list of videos
        if len(videos) > 0:
            h5_files = list(
                auxiliaryfunctions.grab_files_in_folder(
                    Path(videos[0]).parent, "h5", relative=False
                )
            )
        else:
            h5_files = []
    else:
        h5_files = list(
            auxiliaryfunctions.grab_files_in_folder(video_folder, "h5", relative=False)
        )
        videos = auxiliaryfunctions.grab_files_in_folder(
            video_folder, videotype, relative=False
        )

    _convert_h5_files_to("csv", None, h5_files, videos)


def analyze_videos_converth5_to_nwb(
    config,
    video_folder,
    videotype=".mp4",
    listofvideos=False,
):
    """
    Convert all h5 output data files in `video_folder` to NWB format.

    Parameters
    ----------
    config : string
        Absolute path to the project YAML config file.

    video_folder : string
        Absolute path of a folder containing videos and the corresponding h5 data files.

    videotype: string, optional (default=.mp4)
        Only videos with this extension are screened.

    Examples
    --------

    Converts all pose-output files belonging to mp4 videos in the folder '/media/alex/experimentaldata/cheetahvideos' to csv files.
    deeplabcut.analyze_videos_converth5_to_csv('/media/alex/experimentaldata/cheetahvideos','.mp4')

    """
    if listofvideos:  # can also be called with a list of videos (from GUI)
        videos = video_folder  # GUI gives a list of videos
        if len(videos) > 0:
            h5_files = list(
                auxiliaryfunctions.grab_files_in_folder(
                    Path(videos[0]).parent, "h5", relative=False
                )
            )
        else:
            h5_files = []
    else:
        h5_files = list(
            auxiliaryfunctions.grab_files_in_folder(video_folder, "h5", relative=False)
        )
        videos = auxiliaryfunctions.grab_files_in_folder(
            video_folder, videotype, relative=False
        )

    _convert_h5_files_to("nwb", config, h5_files, videos)


def _convert_h5_files_to(filetype, config, h5_files, videos):
    filetype = filetype.lower()
    if filetype not in SUPPORTED_FILETYPES:
        raise ValueError(
            f"""Unsupported destination format {filetype}.
            Must be one of {SUPPORTED_FILETYPES}."""
        )

    if filetype == "nwb":
        try:
            from dlc2nwb.utils import convert_h5_to_nwb
        except ImportError:
            raise ImportError(
                "The package `dlc2nwb` is missing. Please run `pip install dlc2nwb`."
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
                    if filetype == "csv":
                        df = pd.read_hdf(file)
                        df.to_csv(file.replace(".h5", ".csv"))
                    else:
                        convert_h5_to_nwb(config, file)

    print(f"All H5 files were converted to {filetype.upper()}.")


def merge_windowsannotationdataONlinuxsystem(cfg):
    """If a project was created on Windows (and labeled there,) but ran on unix then the data folders
    corresponding in the keys in cfg['video_sets'] are not found. This function gets them directly by
    looping over all folders in labeled-data"""

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
            guarantee_multiindex_rows(data)
            AnnotationData.append(data)
        except FileNotFoundError:
            print(filename, " not found (perhaps not annotated)")

    return AnnotationData


def guarantee_multiindex_rows(df):
    # Make paths platform-agnostic if they are not already
    if not isinstance(df.index, pd.MultiIndex):  # Backwards compatibility
        path = df.index[0]
        try:
            sep = "/" if "/" in path else "\\"
            splits = tuple(df.index.str.split(sep))
            df.index = pd.MultiIndex.from_tuples(splits)
        except TypeError:  #  Ignore numerical index of frame indices
            pass

    # Ensure folder names are strings
    try:
        df.index = df.index.set_levels(df.index.levels[1].astype(str), level=1)
    except AttributeError:
        pass


def robust_split_path(s):
    sep = "/" if "/" in s else "\\"
    return tuple(s.split(sep))
