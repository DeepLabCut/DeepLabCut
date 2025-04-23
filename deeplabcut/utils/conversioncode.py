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

import io
import json
import yaml
import time

import h5py
import numpy as np

import PIL.Image as Image
import cv2


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


def adapt_labeled_data_to_new_project(
    config_path, remove_old_bodyparts=False, other_scorer=False, userfeedback=False
):
    """Given the config.yaml file, this function will convert the labels of an ancient project to a new project.
        For this, the labeled data must be in the project folder, under the labeled-data folder and with the same configuration as all deeplabcut projects.

    Parameters
    ----------
    config_path : str
        The path to the config.yaml file.
    remove_old_bodyparts : bool (default = False)
        If True, the old bodyparts that are not in the new project will be removed from the dataframe.
    other_scorer : bool (default = False)
        If True, the labels will be converted to the new scorer.
    userfeedback : bool (default = True)
        If true the user will be asked specifically for each folder in labeled-data if the containing csv shall be converted to hdf format.
    """

    # Load the config file
    cfg = dlc.auxiliaryfunctions.read_config(config_path)

    # Get the Project path
    project_path = cfg["project_path"]

    # Get the bodyparts
    bodyparts = cfg["multianimalbodyparts"]
    print("New Bodyparts:", bodyparts)

    # Iterate over each labeled data video

    # Use tqdm for a progress bar
    for video in tqdm.tqdm(cfg["video_sets"]):
        print("Video:", video)

        video_name = video.split("\\")[-1]
        # discard the file extension
        video_name = video_name.split(".")[0]
        # Load the csv file
        label_path = os.path.join(project_path, "labeled-data", video_name)
        csv_files = [file for file in os.listdir(label_path) if file.endswith(".csv")]
        if not csv_files:
            print("No csv file in the folder:", label_path)
        else:
            csv_path = os.path.join(label_path, csv_files[0])
            df = pd.read_csv(csv_path, header=None)

            # get the scorer
            if other_scorer:
                scorer = cfg["scorer"]
                # Change the scorer in the dataframe
                df.iloc[0, 3:] = pd.Series([scorer] * len(df.columns[3:]))

            else:
                scorer = df.iloc[0, 3]

            # Get the individuals
            individuals = np.unique(df.iloc[1, 3:])

            # Get the old bodyparts
            old_bodyparts = np.unique(df.iloc[2, 3:])
            print("Old bodyparts:", old_bodyparts)

            # Get the unmber of old bodyparts
            num_of_old_bodyparts = len(old_bodyparts)

            # Bodyparts to add
            print("Bodyparts to add:", set(bodyparts) - set(old_bodyparts))

            # If a bodypart is missing, add it to the dataframe
            for index, bodypart in enumerate(bodyparts):
                if bodypart not in old_bodyparts:
                    num_of_old_bodyparts += 1
                    for i, individual in enumerate(individuals):
                        # create the columns for the bodypart, concatenate, the individual, the bodypart, and nan values
                        x_column = pd.concat(
                            [
                                pd.Series(scorer),
                                pd.Series(individual),
                                pd.Series(bodypart),
                                pd.Series("x"),
                                pd.Series(np.nan, index=df.index),
                            ],
                            axis=0,
                            ignore_index=True,
                        )
                        y_column = pd.concat(
                            [
                                pd.Series(scorer),
                                pd.Series(individual),
                                pd.Series(bodypart),
                                pd.Series("y"),
                                pd.Series(np.nan, index=df.index),
                            ],
                            axis=0,
                            ignore_index=True,
                        )
                        # Insert the columns in the dataframe
                        df.insert(
                            i * 2 * num_of_old_bodyparts + index * 2 + 3,
                            "insert_" + bodypart + "_x" + individual,
                            x_column,
                        )
                        df.insert(
                            i * 2 * num_of_old_bodyparts + index * 2 + 4,
                            "insert" + bodypart + "_y" + individual,
                            y_column,
                        )

            # If the old bodyparts are not in the new project, remove them
            if remove_old_bodyparts:
                for bodypart in old_bodyparts:
                    if bodypart not in bodyparts:
                        df = df.drop(df.columns[df.iloc[2, :] == bodypart], axis=1)

            # Save the dataframe
            df.to_csv(csv_path, index=False, header=False)

    # Create/Update the h5 file
    convertcsv2h5(config_path, userfeedback=userfeedback)


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

def extract_frames_from_pkg_sleap(file_path, base_output_dir, scorer='me'):
    """
    Convert a SLEAP project into a DeepLabCut project
    WARNING: conversion might corrupt the data.

    file_path : string
        Path to the .slp file.

    base_output_dir: string
        Output directory.

    scorer: string, optional
        Name of scorer.

    Once conversion is complete, you will need to manually create a DeepLabCut config.yaml file based 
    on the keypoint names in the generated .csv files.

    Finally, run deeplabcut.convertcsv2h5('config.yaml')
    """
    config = {'scorer': scorer,
              'date': time.strftime("%Y-%m-%d"),
              'identity': None,
              'project_path': base_output_dir,
              'engine': 'pytorch',
              'video_sets': {},
              'start': 0,
              'stop': 1,
              'numframes2pick': 20,
              'skeleton_color': 'black',
              'pcutoff': 0.6,
              'dotsize': 12,
              'alphavalue': 0.6,
              'colormap': 'rainbow',
              'TrainingFraction': [0.95],
              'iteration': 0,
              'default_net_type': 'resnet_50',
              'default_augmenter': 'default',
              'snapshotindex': -1,
              'detector_snapshotindex': -1,
              'batch_size': 8,
              'detector_batch_size': 1,
              'cropping': False,
              'x1': 0,
              'x2': 640,
              'y1': 277,
              'y2': 624,
              'corner2move2': [50, 50],
              'move2corner': True,
              'SuperAnimalConversionTables': None
              }
    
    
    #create directories
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        os.makedirs(os.path.join(base_output_dir,"labeled-data"))
        os.makedirs(os.path.join(base_output_dir,"videos"))
        
    #parse .slp file
    with h5py.File(file_path, 'r') as hdf_file:
        # Identify video names
        video_names = {}
        for video_group_name in hdf_file.keys():
            if video_group_name.startswith('video'):
                source_video_path = f'{video_group_name}/source_video'
                if source_video_path in hdf_file:
                    source_video_json = hdf_file[source_video_path].attrs['json']
                    source_video_dict = json.loads(source_video_json)
                    video_filename = source_video_dict['backend']['filename']
                    video_names[video_group_name] = video_filename
                    
        # Extract and save images for each video
        for video_group, video_filename in video_names.items():
            data_frames = []
            scorer_row, bodyparts_row, coords_row = None, None, None
            output_dir = os.path.join(
                base_output_dir, "labeled-data", os.path.basename(video_filename).split('.')[0]
            )
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # extract labeled frames and save them in a separate directory for each video
            if video_group in hdf_file and 'video' in hdf_file[video_group]:
                video_data = hdf_file[f'{video_group}/video'][:]
                frame_numbers = hdf_file[f'{video_group}/frame_numbers'][:]
                frame_names = []
                for i, (img_bytes, frame_number) in enumerate(zip(video_data, frame_numbers)):
                    img = Image.open(io.BytesIO(np.array(img_bytes, dtype=np.uint8)))
                    img = np.array(img)
                    if i==0:
                        video_path = os.path.join(base_output_dir,'videos',video_names[video_group].split('/')[-1])
                        config['video_sets'][video_path] = {'crop': f'0, {img.shape[1]}, 0, {img.shape[0]}'}
                    frame_name = f"img{str(frame_number).zfill(8)}.png"
                    cv2.imwrite(f"{output_dir}/{frame_name}", img)
                    frame_names.append(frame_name)
                    print(f"Saved frame {frame_number} as {frame_name}")
                    
                    
            # extract coordinates and save them in a separate directory for each video
            if video_group in hdf_file and 'frames' in hdf_file:
                frames_dataset = hdf_file['frames']
                frame_references = {
                    frame['frame_id']: frame['frame_idx']
                    for frame in frames_dataset
                    if frame['video'] == int(video_group.replace('video', ''))
                }


                # Extract instances and points
                points_dataset = hdf_file['points']
                instances_dataset = hdf_file['instances']
                
                data = []
                for frame_id in frame_references.keys():
                    for i in range(len(instances_dataset)):
                        if frame_id==instances_dataset[i]['frame_id']:
                            point_id_start = instances_dataset[i+1]['point_id_start']
                            point_id_end = instances_dataset[i+1]['point_id_end']
                            break
                        
                    points = points_dataset[point_id_start:point_id_end]

                    keypoints_flat = []
                    for kp in points:
                        x, y, vis = kp['x'], kp['y'], kp['visible']
                        if np.isnan(x) or np.isnan(y) or vis==False:
                            x, y = None, None
                        keypoints_flat.extend([x, y])

                    frame_idx = frame_references[frame_id]
                    data.append([frame_idx] + keypoints_flat)
                        
                    
                # parse data
                metadata_json = hdf_file['metadata'].attrs['json']
                metadata_dict = json.loads(metadata_json)
                nodes = metadata_dict['nodes']
                links = metadata_dict['skeletons'][0]['links']
                
                keypoints = [node['name'] for node in nodes]
                skeleton = [[keypoints[l['source']], keypoints[l['target']]] for l in links]
                config['skeleton'] = skeleton
                
                keypoints_ids = [n['id'] for n in metadata_dict['skeletons'][0]['nodes']]
                keypoints_ordered = [keypoints[idx] for idx in keypoints_ids]
                config['bodyparts'] = keypoints_ordered

                columns = [
                    'frame'
                ] + [
                    f'{kp}_x' for kp in keypoints_ordered
                ] + [
                    f'{kp}_y' for kp in keypoints_ordered
                ]
                scorer_row = ['scorer'] + [f'{scorer}'] * (len(columns) - 1)
                bodyparts_row = ['bodyparts'] + [f'{kp}' for kp in keypoints_ordered for _ in (0, 1)]
                coords_row = ['coords'] + ['x', 'y'] * len(keypoints_ordered)

                labels_df = pd.DataFrame(data, columns=columns)
                video_base_name = os.path.basename(video_filename).split('.')[0]
                labels_df['frame'] = labels_df['frame'].apply(
                    lambda x: (
                        f"labeled-data/{video_base_name}/"
                        f"img{str(int(x)).zfill(8)}.png"
                    )
                )
                labels_df = labels_df.groupby('frame', as_index=False).first()
                data_frames.append(labels_df)
                    
                # Combine all data frames into a single DataFrame
                combined_df = pd.concat(data_frames, ignore_index=True)
                    
                header_df = pd.DataFrame(
                    [scorer_row, bodyparts_row, coords_row],
                    columns=combined_df.columns
                )
                final_df = pd.concat([header_df, combined_df], ignore_index=True)
                final_df.columns = [None] * len(final_df.columns)  # Set header to None
                
                # Save concatenated labels
                final_df.to_csv(os.path.join(output_dir, f"CollectedData_{scorer}.csv"), index=False, header=None)
                
    with open(os.path.join(base_output_dir, 'config.yaml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False) 
