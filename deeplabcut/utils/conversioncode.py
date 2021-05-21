"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os
from pathlib import Path

import pandas as pd

from deeplabcut.generate_training_dataset import trainingsetmanipulation
from deeplabcut.utils import auxiliaryfunctions
import warnings


def convertannotationdata_fromwindows2unixstyle(
    config, userfeedback=True, win2linux=True
):
    """
    Converts paths in annotation file (CollectedData_*user*.h5) in labeled-data/videofolder1, etc.

    from windows to linux format. This is important when one e.g. labeling on Windows, but
    wants to re-label/check_labels/ on a Linux computer (and vice versa).

    Note for training data annotated on Windows in Linux this is not necessary, as the data
    gets converted during training set creation.

    config : string
        Full path of the config.yaml file as a string.

    userfeedback: bool, optional
        If true the user will be asked specifically for each folder in labeled-data if the containing csv shall be converted to hdf format.

    win2linux: bool, optional.
        By default converts from windows to linux. If false, converts from unix to windows.
    """
    cfg = auxiliaryfunctions.read_config(config)
    folders = [
        Path(config).parent / "labeled-data" / trainingsetmanipulation._robust_path_split(vid)[1]
        for vid in cfg["video_sets"]
    ]

    for folder in folders:
        if userfeedback:
            print("Do you want to convert the annotationdata in folder:", folder, "?")
            askuser = input("yes/no")
        else:
            askuser = "yes"

        if askuser == "y" or askuser == "yes" or askuser == "Ja" or askuser == "ha":
            fn = os.path.join(str(folder), "CollectedData_" + cfg["scorer"])
            if os.path.exists(fn + ".h5"):
                Data = pd.read_hdf(fn + ".h5")
                if win2linux:
                    convertpaths_to_unixstyle(Data, fn)
                else:
                    convertpaths_to_windowsstyle(Data, fn)
            else:
                warnings.warn(f"Could not find '{fn+'.h5'}'. skipping")


def convertpaths_to_unixstyle(Data, fn):
    """ auxiliary function that converts paths in annotation files:
        labeled-data\\video\\imgXXX.png to labeled-data/video/imgXXX.png """
    Data.to_csv(fn + "windows" + ".csv")
    Data.to_hdf(fn + "windows" + ".h5", "df_with_missing", format="table", mode="w")
    Data.index = Data.index.str.replace("\\", "/")
    Data.to_csv(fn + ".csv")
    Data.to_hdf(fn + ".h5", "df_with_missing", format="table", mode="w")
    return Data


def convertpaths_to_windowsstyle(Data, fn):
    """ auxiliary function that converts paths in annotation files:
        labeled-data/video/imgXXX.png to labeled-data\\video\\imgXXX.png """
    Data.to_csv(fn + "unix" + ".csv")
    Data.to_hdf(fn + "unix" + ".h5", "df_with_missing", format="table", mode="w")
    Data.index = Data.index.str.replace("/", "\\")
    Data.to_csv(fn + ".csv")
    Data.to_hdf(fn + ".h5", "df_with_missing", format="table", mode="w")
    return Data


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
    use_cropped = cfg.get("croppedtraining", False)
    annotationfolders = []
    for elem in auxiliaryfunctions.grab_files_in_folder(data_path, relative=False):
        if os.path.isdir(elem) and (
            (use_cropped and elem.endswith("_cropped"))
            or not (use_cropped or "_cropped" in elem)
        ):
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
