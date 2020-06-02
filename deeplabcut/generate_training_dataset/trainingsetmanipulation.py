"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
import logging
import os
import os.path
import shutil

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from skimage import io

from deeplabcut.pose_estimation_tensorflow import training
from deeplabcut.utils import (
    auxiliaryfunctions,
    conversioncode,
    auxfun_models,
    auxfun_multianimal,
)


def comparevideolistsanddatafolders(config):
    """
    Auxiliary function that compares the folders in labeled-data and the ones listed under video_sets (in the config file).

    Parameter
    ----------
    config : string
        String containing the full path of the config file in the project.

    """
    cfg = auxiliaryfunctions.read_config(config)
    videos = cfg["video_sets"].keys()
    video_names = [Path(i).stem for i in videos]
    alldatafolders = [
        fn
        for fn in os.listdir(Path(config).parent / "labeled-data")
        if "_labeled" not in fn
    ]

    print("Config file contains:", len(video_names))
    print("Labeled-data contains:", len(alldatafolders))

    for vn in video_names:
        if vn not in alldatafolders:
            print(vn, " is missing as a folder!")

    for vn in alldatafolders:
        if vn not in video_names:
            print(vn, " is missing in config file!")


def adddatasetstovideolistandviceversa(config):
    """
    First run comparevideolistsanddatafolders(config) to compare the folders in labeled-data and the ones listed under video_sets (in the config file).
    If you detect differences this function can be used to maker sure each folder has a video entry & vice versa.

    It corrects this problem in the following way:

    If a video entry in the config file does not contain a folder in labeled-data, then the entry is removed.
    If a folder in labeled-data does not contain a video entry in the config file then the prefix path will be added in front of the name of the labeled-data folder and combined
    with the suffix variable as an ending. Width and height will be added as cropping variables as passed on.

    Handle with care!

    Parameter
    ----------
    config : string
        String containing the full path of the config file in the project.
    """
    cfg = auxiliaryfunctions.read_config(config)
    videos = cfg["video_sets"]
    video_names = [Path(i).stem for i in videos]

    alldatafolders = [
        fn
        for fn in os.listdir(Path(config).parent / "labeled-data")
        if "_labeled" not in fn and not fn.startswith(".")
    ]

    print("Config file contains:", len(video_names))
    print("Labeled-data contains:", len(alldatafolders))

    toberemoved = []
    for vn in video_names:
        if vn not in alldatafolders:
            print(vn, " is missing as a labeled folder >> removing key!")
            for fullvideo in videos:
                if vn in fullvideo:
                    toberemoved.append(fullvideo)

    for vid in toberemoved:
        del videos[vid]

    # Load updated lists:
    video_names = [Path(i).stem for i in videos]
    for vn in alldatafolders:
        if vn not in video_names:
            print(vn, " is missing in config file >> adding it!")
            # Find the corresponding video file
            found = False
            for file in os.listdir(os.path.join(cfg["project_path"], "videos")):
                if os.path.splitext(file)[0] == vn:
                    found = True
                    break
            if found:
                video_path = os.path.join(cfg["project_path"], "videos", file)
                clip = cv2.VideoCapture(video_path)
                width = int(clip.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(clip.get(cv2.CAP_PROP_FRAME_HEIGHT))
                videos.update(
                    {video_path: {"crop": ", ".join(map(str, [0, width, 0, height]))}}
                )

    auxiliaryfunctions.write_config(config, cfg)


def dropduplicatesinannotatinfiles(config):
    """

    Drop duplicate entries (of images) in annotation files (this should no longer happen, but might be useful).

    Parameter
    ----------
    config : string
        String containing the full path of the config file in the project.

    """
    cfg = auxiliaryfunctions.read_config(config)
    videos = cfg["video_sets"].keys()
    video_names = [Path(i).stem for i in videos]
    folders = [Path(config).parent / "labeled-data" / Path(i) for i in video_names]

    for folder in folders:
        try:
            fn = os.path.join(str(folder), "CollectedData_" + cfg["scorer"] + ".h5")
            DC = pd.read_hdf(fn, "df_with_missing")
            numimages = len(DC.index)
            DC = DC[~DC.index.duplicated(keep="first")]
            if len(DC.index) < numimages:
                print("Dropped", numimages - len(DC.index))
                DC.to_hdf(fn, key="df_with_missing", mode="w")
                DC.to_csv(
                    os.path.join(str(folder), "CollectedData_" + cfg["scorer"] + ".csv")
                )

        except FileNotFoundError:
            print("Attention:", folder, "does not appear to have labeled data!")


def dropannotationfileentriesduetodeletedimages(config):
    """
    Drop entries for all deleted images in annotation files, i.e. for folders of the type: /labeled-data/*folder*/CollectedData_*scorer*.h5
    Will be carried out iteratively for all *folders* in labeled-data.

    Parameter
    ----------
    config : string
        String containing the full path of the config file in the project.

    """
    cfg = auxiliaryfunctions.read_config(config)
    videos = cfg["video_sets"].keys()
    video_names = [Path(i).stem for i in videos]
    folders = [Path(config).parent / "labeled-data" / Path(i) for i in video_names]

    for folder in folders:
        fn = os.path.join(str(folder), "CollectedData_" + cfg["scorer"] + ".h5")
        DC = pd.read_hdf(fn, "df_with_missing")
        dropped = False
        for imagename in DC.index:
            if os.path.isfile(os.path.join(cfg["project_path"], imagename)):
                pass
            else:
                print("Dropping...", imagename)
                DC = DC.drop(imagename)
                dropped = True
        if dropped == True:
            DC.to_hdf(fn, key="df_with_missing", mode="w")
            DC.to_csv(
                os.path.join(str(folder), "CollectedData_" + cfg["scorer"] + ".csv")
            )


def dropimagesduetolackofannotation(config):
    """
    Drop images from corresponding folder for not annotated images: /labeled-data/*folder*/CollectedData_*scorer*.h5
    Will be carried out iteratively for all *folders* in labeled-data.

    Parameter
    ----------
    config : string
        String containing the full path of the config file in the project.
    """
    cfg = auxiliaryfunctions.read_config(config)
    videos = cfg["video_sets"].keys()
    video_names = [Path(i).stem for i in videos]
    folders = [Path(config).parent / "labeled-data" / Path(i) for i in video_names]

    for folder in folders:
        fn = os.path.join(str(folder), "CollectedData_" + cfg["scorer"] + ".h5")
        DC = pd.read_hdf(fn, "df_with_missing")
        dropped = False
        annotatedimages = [fn.split(os.sep)[-1] for fn in DC.index]
        imagelist = [fns for fns in os.listdir(str(folder)) if ".png" in fns]
        print("Annotated images: ", len(annotatedimages), " In folder:", len(imagelist))
        for imagename in imagelist:
            if imagename in annotatedimages:
                pass
            else:
                fullpath = os.path.join(
                    cfg["project_path"], "labeled-data", folder, imagename
                )
                if os.path.isfile(fullpath):
                    print("Deleting", fullpath)
                    os.remove(fullpath)

        annotatedimages = [fn.split(os.sep)[-1] for fn in DC.index]
        imagelist = [fns for fns in os.listdir(str(folder)) if ".png" in fns]
        print(
            "PROCESSED:",
            folder,
            " now # of annotated images: ",
            len(annotatedimages),
            " in folder:",
            len(imagelist),
        )


def cropimagesandlabels(
    config,
    numcrops=10,
    size=(400, 400),
    userfeedback=True,
    cropdata=True,
    excludealreadycropped=True,
    updatevideoentries=True,
):
    """
    Crop images into multiple random crops (defined by numcrops) of size dimensions. If cropdata=True then the
    annotation data is loaded and labels for cropped images are inherited.
    If false, then one can make crops for unlabeled folders.

    This can be helpul for large frames with multiple animals. Then a smaller set of equally sized images is created.

    Parameters
    ----------
    config : string
        String containing the full path of the config file in the project.

    numcrops: number of random crops (around random bodypart)

    size: height x width in pixels

    userfeedback: bool, optional
        If this is set to false, then all requested train/test splits are created (no matter if they already exist). If you
        want to assure that previous splits etc. are not overwritten, then set this to True and you will be asked for each split.

    cropdata: bool, default True:
        If true creates corresponding annotation data (from ground truth)

    excludealreadycropped: bool, def true
        If true excludes folders that already contain _cropped in their name.

    updatevideoentries, bool, default true
        If true updates video_list entries to refer to cropped frames instead. This makes sense for subsequent processing.

    Example
    --------
    for labeling the frames
    >>> deeplabcut.cropimagesandlabels('/analysis/project/reaching-task/config.yaml')

    --------
    """
    from tqdm import trange

    indexlength = int(np.ceil(np.log10(numcrops)))
    project_path = os.path.dirname(config)
    cfg = auxiliaryfunctions.read_config(config)
    videos = cfg["video_sets"].keys()
    video_names = []
    for video in videos:
        parent, filename, ext = _robust_path_split(video)
        if excludealreadycropped and "_cropped" in filename:
            continue
        video_names.append([parent, filename, ext])

    if (
        "video_sets_original" not in cfg.keys() and updatevideoentries
    ):  # this dict is kept for storing links to original full-sized videos
        cfg["video_sets_original"] = {}

    for vidpath, vidname, videotype in video_names:
        folder = os.path.join(project_path, "labeled-data", vidname)
        if userfeedback:
            print("Do you want to crop frames for folder: ", folder, "?")
            askuser = input("(yes/no):")
        else:
            askuser = "y"
        if askuser == "y" or askuser == "yes" or askuser == "Y" or askuser == "Yes":
            new_vidname = vidname + "_cropped"
            new_folder = folder.replace(vidname, new_vidname)
            auxiliaryfunctions.attempttomakefolder(new_folder)

            AnnotationData = []
            pd_index = []

            fn = os.path.join(folder, f"CollectedData_{cfg['scorer']}.h5")
            df = pd.read_hdf(fn, "df_with_missing")
            data = df.values.reshape((df.shape[0], -1, 2))
            sep = "/" if "/" in df.index[0] else "\\"
            if sep != os.path.sep:
                df.index = df.index.str.replace(sep, os.path.sep)
            images = project_path + os.path.sep + df.index
            # Avoid cropping already cropped images
            cropped_images = auxiliaryfunctions.grab_files_in_folder(new_folder, "png")
            cropped_names = set(map(lambda x: x.split("c")[0], cropped_images))
            imnames = [
                im for im in images.to_list() if Path(im).stem not in cropped_names
            ]
            ic = io.imread_collection(imnames)
            for i in trange(len(ic)):
                frame = ic[i]
                h, w = np.shape(frame)[:2]
                if size[0] >= h or size[1] >= w:
                    shutil.rmtree(new_folder, ignore_errors=True)
                    raise ValueError("Crop dimensions are larger than image size")

                imagename = os.path.relpath(ic.files[i], project_path)
                ind = np.flatnonzero(df.index == imagename)[0]
                cropindex = 0
                attempts = -1
                while cropindex < numcrops:
                    dd = np.array(data[ind].copy(), dtype=float)
                    y0, x0 = (
                        np.random.randint(h - size[0]),
                        np.random.randint(w - size[1]),
                    )
                    y1 = y0 + size[0]
                    x1 = x0 + size[1]
                    with np.errstate(invalid="ignore"):
                        within = np.all((dd >= [x0, y0]) & (dd < [x1, y1]), axis=1,)
                    if cropdata:
                        dd[within] -= [x0, y0]
                        dd[~within] = np.nan
                    attempts += 1
                    if within.any() or attempts > 10:
                        newimname = str(
                            Path(imagename).stem
                            + "c"
                            + str(cropindex).zfill(indexlength)
                            + ".png"
                        )
                        cropppedimgname = os.path.join(new_folder, newimname)
                        io.imsave(cropppedimgname, frame[y0:y1, x0:x1])
                        cropindex += 1
                        pd_index.append(
                            os.path.join("labeled-data", new_vidname, newimname)
                        )
                        AnnotationData.append(dd.flatten())

            if cropdata:
                df = pd.DataFrame(AnnotationData, index=pd_index, columns=df.columns)
                fn_new = fn.replace(folder, new_folder)
                df.to_hdf(fn_new, key="df_with_missing", mode="w")
                df.to_csv(fn_new.replace(".h5", ".csv"))

            if updatevideoentries and cropdata:
                # moving old entry to _original, dropping it from video_set and update crop parameters
                video_orig = sep.join((vidpath, vidname + "." + videotype))
                cfg["video_sets_original"][video_orig] = cfg["video_sets"][video_orig]
                cfg["video_sets"].pop(video_orig)
                cfg["video_sets"][video_orig.replace(vidname, new_vidname)] = {
                    "crop": ", ".join(map(str, [0, size[1], 0, size[0]]))
                }

    cfg["croppedtraining"] = True
    auxiliaryfunctions.write_config(config, cfg)


def label_frames(config, multiple_individualsGUI=False, imtypes=["*.png"]):
    """
    Manually label/annotate the extracted frames. Update the list of body parts you want to localize in the config.yaml file first.

    Parameter
    ----------
    config : string
        String containing the full path of the config file in the project.

    multiple_individualsGUI: bool, optional
          If this is set to True, a user can label multiple individuals. Note for "multianimalproject=True" this is automatically used.
          The default is ``False``; if provided it must be either ``True`` or ``False``.

    imtypes: list of imagetypes to look for in folder to be labeled. By default only png images are considered.

    Example
    --------
    Standard use case:
    >>> deeplabcut.label_frames('/myawesomeproject/reaching4thestars/config.yaml')

    To label multiple individuals (without having a multiple individuals project); otherwise this GUI is loaded automatically
    >>> deeplabcut.label_frames('/analysis/project/reaching-task/config.yaml',multiple_individualsGUI=True)

    To label other image types
    >>> label_frames(config,multiple=False,imtypes=['*.jpg','*.jpeg'])
    --------

    """
    startpath = os.getcwd()
    wd = Path(config).resolve().parents[0]
    os.chdir(str(wd))
    cfg = auxiliaryfunctions.read_config(config)
    if cfg.get("multianimalproject", False) or multiple_individualsGUI:
        from deeplabcut.generate_training_dataset import (
            multiple_individuals_labeling_toolbox,
        )

        multiple_individuals_labeling_toolbox.show(config)
    else:
        from deeplabcut.generate_training_dataset import labeling_toolbox

        labeling_toolbox.show(config, imtypes=imtypes)

    os.chdir(startpath)


def check_labels(
    config,
    Labels=["+", ".", "x"],
    scale=1,
    dpi=100,
    draw_skeleton=True,
    visualizeindividuals=True,
):
    """
    Double check if the labels were at correct locations and stored in a proper file format.\n
    This creates a new subdirectory for each video under the 'labeled-data' and all the frames are plotted with the labels.\n
    Make sure that these labels are fine.

    Parameter
    ----------
    config : string
        Full path of the config.yaml file as a string.

    Labels: List of at least 3 matplotlib markers. The first one will be used to indicate the human ground truth location (Default: +)

    scale : float, default =1
        Change the relative size of the output images.

    dpi : int, optional
        Output resolution. 100 dpi by default.

    draw_skeleton: bool, default True.
        Plot skeleton overlaid over body parts.

    visualizeindividuals: bool, default True:
        For a multianimal project the different individuals have different colors (and all bodyparts the same).
        If False, the colors change over bodyparts rather than individuals.

    Example
    --------
    for labeling the frames
    >>> deeplabcut.check_labels('/analysis/project/reaching-task/config.yaml')
    --------
    """

    from deeplabcut.utils import visualization

    cfg = auxiliaryfunctions.read_config(config)
    videos = cfg["video_sets"].keys()
    video_names = [Path(i).stem for i in videos]

    folders = [
        os.path.join(cfg["project_path"], "labeled-data", str(Path(i)))
        for i in video_names
    ]
    print("Creating images with labels by %s." % cfg["scorer"])
    for folder in folders:
        try:
            DataCombined = pd.read_hdf(
                os.path.join(str(folder), "CollectedData_" + cfg["scorer"] + ".h5"),
                "df_with_missing",
            )
            if cfg.get("multianimalproject", False):
                color_by = "individual" if visualizeindividuals else "bodypart"
            else:  # for single animal projects
                color_by = "bodypart"

            visualization.make_labeled_images_from_dataframe(
                DataCombined,
                cfg,
                folder,
                scale,
                dpi=dpi,
                keypoint=Labels[0],
                draw_skeleton=draw_skeleton,
                color_by=color_by,
            )
        except FileNotFoundError:
            print("Attention:", folder, "does not appear to have labeled data!")

    print(
        "If all the labels are ok, then use the function 'create_training_dataset' to create the training dataset!"
    )


def boxitintoacell(joints):
    """ Auxiliary function for creating matfile."""
    outer = np.array([[None]], dtype=object)
    outer[0, 0] = np.array(joints, dtype="int64")
    return outer


def ParseYaml(configfile):
    raw = open(configfile).read()
    docs = []
    for raw_doc in raw.split("\n---"):
        try:
            docs.append(yaml.load(raw_doc, Loader=yaml.SafeLoader))
        except SyntaxError:
            docs.append(raw_doc)
    return docs


def MakeTrain_pose_yaml(itemstochange, saveasconfigfile, defaultconfigfile):
    docs = ParseYaml(defaultconfigfile)
    for key in itemstochange.keys():
        docs[0][key] = itemstochange[key]

    with open(saveasconfigfile, "w") as f:
        yaml.dump(docs[0], f)
    return docs[0]


def MakeTest_pose_yaml(
    dictionary, keys2save, saveasfile, nmsradius=None, minconfidence=None
):
    dict_test = {}
    for key in keys2save:
        dict_test[key] = dictionary[key]

    # adding important values for multianiaml project:
    if nmsradius is not None:
        dict_test["nmsradius"] = nmsradius
    if minconfidence is not None:
        dict_test["minconfidence"] = minconfidence

    dict_test["scoremap_dir"] = "test"
    with open(saveasfile, "w") as f:
        yaml.dump(dict_test, f)


def MakeInference_yaml(itemstochange, saveasconfigfile, defaultconfigfile):
    docs = ParseYaml(defaultconfigfile)
    for key in itemstochange.keys():
        docs[0][key] = itemstochange[key]

    with open(saveasconfigfile, "w") as f:
        yaml.dump(docs[0], f)
    return docs[0]


def _robust_path_split(path):
    sep = "\\" if "\\" in path else "/"
    parent, file = path.rsplit(sep, 1)
    filename, ext = file.split(".")
    return parent, filename, ext


def merge_annotateddatasets(cfg, trainingsetfolder_full, windows2linux):
    """
    Merges all the h5 files for all labeled-datasets (from individual videos).

    This is a bit of a mess because of cross platform compatibility.

    Within platform comp. is straightforward. But if someone labels on windows and wants to train on a unix cluster or colab...
    """
    AnnotationData = []
    data_path = Path(os.path.join(cfg["project_path"], "labeled-data"))
    videos = cfg["video_sets"].keys()
    for video in videos:
        _, filename, _ = _robust_path_split(video)
        if cfg.get("croppedtraining", False):
            filename += "_cropped"
        file_path = os.path.join(
            data_path / filename, f'CollectedData_{cfg["scorer"]}.h5'
        )
        try:
            data = pd.read_hdf(file_path, "df_with_missing")
            AnnotationData.append(data)
        except FileNotFoundError:
            print(
                file_path,
                " not found (perhaps not annotated). If training on cropped data, "
                "make sure to call `cropimagesandlabels` prior to creating the dataset.",
            )

    if not len(AnnotationData):
        print(
            "Annotation data was not found by splitting video paths (from config['video_sets']). An alternative route is taken..."
        )
        AnnotationData = conversioncode.merge_windowsannotationdataONlinuxsystem(cfg)
        if not len(AnnotationData):
            print("No data was found!")
            return

    AnnotationData = pd.concat(AnnotationData).sort_index()
    # When concatenating DataFrames with misaligned column labels,
    # all sorts of reordering may happen (mainly depending on 'sort' and 'join')
    # Ensure the 'bodyparts' level agrees with the order in the config file.
    if cfg.get("multianimalproject", False):
        (
            _,
            uniquebodyparts,
            multianimalbodyparts,
        ) = auxfun_multianimal.extractindividualsandbodyparts(cfg)
        bodyparts = multianimalbodyparts + uniquebodyparts
    else:
        bodyparts = cfg["bodyparts"]
    AnnotationData = AnnotationData.reindex(
        bodyparts, axis=1, level=AnnotationData.columns.names.index("bodyparts")
    )

    # Let's check if the code is *not* run on windows (Source: #https://stackoverflow.com/questions/1325581/how-do-i-check-if-im-running-on-windows-in-python)
    # but the paths are in windows format...
    windowspath = "\\" in AnnotationData.index[0]
    if os.name != "nt" and windowspath and not windows2linux:
        print(
            "It appears that the images were labeled on a Windows system, but you are currently trying to create a training set on a Unix system. \n In this case the paths should be converted. Do you want to proceed with the conversion?"
        )
        askuser = input("yes/no")
    else:
        askuser = "no"

    filename = os.path.join(trainingsetfolder_full, f'CollectedData_{cfg["scorer"]}')
    if (
        windows2linux or askuser == "yes" or askuser == "y" or askuser == "Ja"
    ):  # convert windows path in pandas array \\ to unix / !
        AnnotationData = conversioncode.convertpaths_to_unixstyle(
            AnnotationData, filename
        )
        print("Annotation data converted to unix format...")
    else:  # store as is
        AnnotationData.to_hdf(filename + ".h5", key="df_with_missing", mode="w")
        AnnotationData.to_csv(filename + ".csv")  # human readable.

    return AnnotationData


def SplitTrials(trialindex, trainFraction=0.8):
    """ Split a trial index into train and test sets. Also checks that the trainFraction is a two digit number between 0 an 1. The reason
    is that the folders contain the trainfraction as int(100*trainFraction). """
    if trainFraction > 1 or trainFraction < 0:
        print(
            "The training fraction should be a two digit number between 0 and 1; i.e. 0.95. Please change accordingly."
        )
        return ([], [])

    if abs(trainFraction - round(trainFraction, 2)) > 0:
        print(
            "The training fraction should be a two digit number between 0 and 1; i.e. 0.95. Please change accordingly."
        )
        return ([], [])
    else:
        trainsetsize = int(len(trialindex) * round(trainFraction, 2))
        shuffle = np.random.permutation(trialindex)
        testIndices = shuffle[trainsetsize:]
        trainIndices = shuffle[:trainsetsize]

        return (trainIndices, testIndices)


def mergeandsplit(config, trainindex=0, uniform=True, windows2linux=False):
    """
    This function allows additional control over "create_training_dataset".

    Merge annotated data sets (from different folders) and split data in a specific way, returns the split variables (train/test indices).
    Importantly, this allows one to freeze a split.

    One can also either create a uniform split (uniform = True; thereby indexing TrainingFraction in config file) or leave-one-folder out split
    by passing the index of the corrensponding video from the config.yaml file as variable trainindex.

    Parameter
    ----------
    config : string
        Full path of the config.yaml file as a string.

    trainindex: int, optional
        Either (in case uniform = True) indexes which element of TrainingFraction in the config file should be used (note it is a list!).
        Alternatively (uniform = False) indexes which folder is dropped, i.e. the first if trainindex=0, the second if trainindex =1, etc.

    uniform: bool, optional
        Perform uniform split (disregarding folder structure in labeled data), or (if False) leave one folder out.

    windows2linux: bool.
        The annotation files contain path formated according to your operating system. If you label on windows
        but train & evaluate on a unix system (e.g. ubunt, colab, Mac) set this variable to True to convert the paths.

    Examples
    --------
    To create a leave-one-folder-out model:
    >>> trainIndices, testIndices=deeplabcut.mergeandsplit(config,trainindex=0,uniform=False)
    returns the indices for the first video folder (as defined in config file) as testIndices and all others as trainIndices.
    You can then create the training set by calling (e.g. defining it as Shuffle 3):
    >>> deeplabcut.create_training_dataset(config,Shuffles=[3],trainIndices=trainIndices,testIndices=testIndices)

    To freeze a (uniform) split:
    >>> trainIndices, testIndices=deeplabcut.mergeandsplit(config,trainindex=0,uniform=True)

    You can then create two model instances that have the identical trainingset. Thereby you can assess the role of various parameters on the performance of DLC.
    >>> deeplabcut.create_training_dataset(config,Shuffles=[0,1],trainIndices=[trainIndices, trainIndices],testIndices=[testIndices, testIndices])
    --------

    """
    # Loading metadata from config file:
    cfg = auxiliaryfunctions.read_config(config)
    scorer = cfg["scorer"]
    project_path = cfg["project_path"]
    # Create path for training sets & store data there
    trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(
        cfg
    )  # Path concatenation OS platform independent
    auxiliaryfunctions.attempttomakefolder(
        Path(os.path.join(project_path, str(trainingsetfolder))), recursive=True
    )
    fn = os.path.join(project_path, trainingsetfolder, "CollectedData_" + cfg["scorer"])

    try:
        Data = pd.read_hdf(fn + ".h5", "df_with_missing")
    except FileNotFoundError:
        Data = merge_annotateddatasets(
            cfg,
            Path(os.path.join(project_path, trainingsetfolder)),
            windows2linux=windows2linux,
        )
        if Data is None:
            return [], []

    Data = Data[scorer]  # extract labeled data

    if uniform == True:
        TrainingFraction = cfg["TrainingFraction"]
        trainFraction = TrainingFraction[trainindex]
        trainIndices, testIndices = SplitTrials(range(len(Data.index)), trainFraction)
    else:  # leave one folder out split
        videos = cfg["video_sets"].keys()
        test_video_name = [Path(i).stem for i in videos][trainindex]
        print("Excluding the following folder (from training):", test_video_name)
        trainIndices, testIndices = [], []
        for index, name in enumerate(Data.index):
            # print(index,name.split(os.sep)[1])
            if test_video_name == name.split(os.sep)[1]:  # this is the video name
                # print(name,test_video_name)
                testIndices.append(index)
            else:
                trainIndices.append(index)

    return trainIndices, testIndices


@lru_cache(maxsize=None)
def _read_image_shape_fast(path):
    return io.imread(path).shape


def format_training_data(df, train_inds, nbodyparts, project_path):
    train_data = []
    matlab_data = []

    def to_matlab_cell(array):
        outer = np.array([[None]], dtype=object)
        outer[0, 0] = array.astype("int64")
        return outer

    for i in train_inds:
        data = dict()
        filename = df.index[i]
        data["image"] = filename
        img_shape = _read_image_shape_fast(os.path.join(project_path, filename))
        try:
            data["size"] = img_shape[2], img_shape[0], img_shape[1]
        except IndexError:
            data["size"] = 1, img_shape[0], img_shape[1]
        temp = df.iloc[i].values.reshape(-1, 2)
        joints = np.c_[range(nbodyparts), temp]
        joints = joints[~np.isnan(joints).any(axis=1)].astype(int)
        # Check that points lie within the image
        inside = np.logical_and(
            np.logical_and(joints[:, 1] < img_shape[1], joints[:, 1] > 0),
            np.logical_and(joints[:, 2] < img_shape[0], joints[:, 2] > 0),
        )
        if not all(inside):
            joints = joints[inside]
        if joints.size:  # Exclude images without labels
            data["joints"] = joints
            train_data.append(data)
            matlab_data.append(
                (
                    np.array([data["image"]], dtype="U"),
                    np.array([data["size"]]),
                    to_matlab_cell(data["joints"]),
                )
            )
    matlab_data = np.asarray(
        matlab_data, dtype=[("image", "O"), ("size", "O"), ("joints", "O")]
    )
    return train_data, matlab_data


def create_training_dataset(
    config,
    num_shuffles=1,
    Shuffles=None,
    windows2linux=False,
    userfeedback=False,
    trainIndices=None,
    testIndices=None,
    net_type=None,
    augmenter_type=None,
):
    """
    Creates a training dataset. Labels from all the extracted frames are merged into a single .h5 file.\n
    Only the videos included in the config file are used to create this dataset.\n

    [OPTIONAL] Use the function 'add_new_video' at any stage of the project to add more videos to the project.

    Parameter
    ----------
    config : string
        Full path of the config.yaml file as a string.

    num_shuffles : int, optional
        Number of shuffles of training dataset to create, i.e. [1,2,3] for num_shuffles=3. Default is set to 1.

    Shuffles: list of shuffles.
        Alternatively the user can also give a list of shuffles (integers!).

    windows2linux: bool.
        The annotation files contain path formated according to your operating system. If you label on windows
        but train & evaluate on a unix system (e.g. ubunt, colab, Mac) set this variable to True to convert the paths.

    userfeedback: bool, optional
        If this is set to false, then all requested train/test splits are created (no matter if they already exist). If you
        want to assure that previous splits etc. are not overwritten, then set this to True and you will be asked for each split.

    trainIndices: list of lists, optional (default=None)
        List of one or multiple lists containing train indexes.
        A list containing two lists of training indexes will produce two splits.

    testIndices: list of lists, optional (default=None)
        List of one or multiple lists containing test indexes.

    net_type: string
        Type of networks. Currently resnet_50, resnet_101, resnet_152, mobilenet_v2_1.0,mobilenet_v2_0.75, mobilenet_v2_0.5, and mobilenet_v2_0.35 are supported.

    augmenter_type: string
        Type of augmenter. Currently default, imgaug, tensorpack, and deterministic are supported.

    Example
    --------
    >>> deeplabcut.create_training_dataset('/analysis/project/reaching-task/config.yaml',num_shuffles=1)
    Windows:
    >>> deeplabcut.create_training_dataset('C:\\Users\\Ulf\\looming-task\\config.yaml',Shuffles=[3,17,5])
    --------
    """
    import scipy.io as sio

    # Loading metadata from config file:
    cfg = auxiliaryfunctions.read_config(config)
    if cfg.get("multianimalproject", False):
        from deeplabcut.generate_training_dataset.multiple_individuals_trainingsetmanipulation import (
            create_multianimaltraining_dataset,
        )

        create_multianimaltraining_dataset(
            config, num_shuffles, Shuffles, windows2linux, net_type
        )
    else:
        scorer = cfg["scorer"]
        project_path = cfg["project_path"]
        # Create path for training sets & store data there
        trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(
            cfg
        )  # Path concatenation OS platform independent
        auxiliaryfunctions.attempttomakefolder(
            Path(os.path.join(project_path, str(trainingsetfolder))), recursive=True
        )

        Data = merge_annotateddatasets(
            cfg, Path(os.path.join(project_path, trainingsetfolder)), windows2linux
        )
        if Data is None:
            return
        Data = Data[scorer]  # extract labeled data

        # loading & linking pretrained models
        if net_type is None:  # loading & linking pretrained models
            net_type = cfg.get("default_net_type", "resnet_50")
        else:
            if "resnet" in net_type or "mobilenet" in net_type:
                pass
            else:
                raise ValueError("Invalid network type:", net_type)

        if augmenter_type is None:
            augmenter_type = cfg.get("default_augmenter", "default")
            if augmenter_type is None:  # this could be in config.yaml for old projects!
                # updating variable if null/None! #backwardscompatability
                auxiliaryfunctions.edit_config(config, {"default_augmenter": "default"})
                augmenter_type = "default"
        else:
            if augmenter_type in ["default", "imgaug", "tensorpack", "deterministic"]:
                pass
            else:
                raise ValueError("Invalid augmenter type:", augmenter_type)

        # Loading the encoder (if necessary downloading from TF)
        dlcparent_path = auxiliaryfunctions.get_deeplabcut_path()
        defaultconfigfile = os.path.join(dlcparent_path, "pose_cfg.yaml")
        model_path, num_shuffles = auxfun_models.Check4weights(
            net_type, Path(dlcparent_path), num_shuffles
        )

        if Shuffles is None:
            Shuffles = range(1, num_shuffles + 1)
        else:
            Shuffles = [i for i in Shuffles if isinstance(i, int)]

        # print(trainIndices,testIndices, Shuffles, augmenter_type,net_type)
        if trainIndices is None and testIndices is None:
            splits = [
                (
                    trainFraction,
                    shuffle,
                    SplitTrials(range(len(Data.index)), trainFraction),
                )
                for trainFraction in cfg["TrainingFraction"]
                for shuffle in Shuffles
            ]
        else:
            if len(trainIndices) != len(testIndices) != len(Shuffles):
                raise ValueError(
                    "Number of Shuffles and train and test indexes should be equal."
                )
            splits = []
            for shuffle, (train_inds, test_inds) in enumerate(
                zip(trainIndices, testIndices)
            ):
                trainFraction = round(
                    len(train_inds) * 1.0 / (len(train_inds) + len(test_inds)), 2
                )
                print(
                    f"You passed a split with the following fraction: {int(100 * trainFraction)}%"
                )
                splits.append(
                    (trainFraction, Shuffles[shuffle], (train_inds, test_inds))
                )

        bodyparts = cfg["bodyparts"]
        nbodyparts = len(bodyparts)
        for trainFraction, shuffle, (trainIndices, testIndices) in splits:
            if len(trainIndices) > 0:
                if userfeedback:
                    trainposeconfigfile, _, _ = training.return_train_network_path(
                        config,
                        shuffle=shuffle,
                        trainingsetindex=cfg["TrainingFraction"].index(trainFraction),
                    )
                    if trainposeconfigfile.is_file():
                        askuser = input(
                            "The model folder is already present. If you continue, it will overwrite the existing model (split). Do you want to continue?(yes/no): "
                        )
                        if (
                            askuser == "no"
                            or askuser == "No"
                            or askuser == "N"
                            or askuser == "No"
                        ):
                            raise Exception(
                                "Use the Shuffles argument as a list to specify a different shuffle index. Check out the help for more details."
                            )

                ####################################################
                # Generating data structure with labeled information & frame metadata (for deep cut)
                ####################################################
                # Make training file!
                (
                    datafilename,
                    metadatafilename,
                ) = auxiliaryfunctions.GetDataandMetaDataFilenames(
                    trainingsetfolder, trainFraction, shuffle, cfg
                )

                ################################################################################
                # Saving data file (convert to training file for deeper cut (*.mat))
                ################################################################################
                data, MatlabData = format_training_data(
                    Data, trainIndices, nbodyparts, project_path
                )
                sio.savemat(
                    os.path.join(project_path, datafilename), {"dataset": MatlabData}
                )

                ################################################################################
                # Saving metadata (Pickle file)
                ################################################################################
                auxiliaryfunctions.SaveMetadata(
                    os.path.join(project_path, metadatafilename),
                    data,
                    trainIndices,
                    testIndices,
                    trainFraction,
                )

                ################################################################################
                # Creating file structure for training &
                # Test files as well as pose_yaml files (containing training and testing information)
                #################################################################################
                modelfoldername = auxiliaryfunctions.GetModelFolder(
                    trainFraction, shuffle, cfg
                )
                auxiliaryfunctions.attempttomakefolder(
                    Path(config).parents[0] / modelfoldername, recursive=True
                )
                auxiliaryfunctions.attempttomakefolder(
                    str(Path(config).parents[0] / modelfoldername) + "/train"
                )
                auxiliaryfunctions.attempttomakefolder(
                    str(Path(config).parents[0] / modelfoldername) + "/test"
                )

                path_train_config = str(
                    os.path.join(
                        cfg["project_path"],
                        Path(modelfoldername),
                        "train",
                        "pose_cfg.yaml",
                    )
                )
                path_test_config = str(
                    os.path.join(
                        cfg["project_path"],
                        Path(modelfoldername),
                        "test",
                        "pose_cfg.yaml",
                    )
                )
                # str(cfg['proj_path']+'/'+Path(modelfoldername) / 'test'  /  'pose_cfg.yaml')
                items2change = {
                    "dataset": datafilename,
                    "metadataset": metadatafilename,
                    "num_joints": len(bodyparts),
                    "all_joints": [[i] for i in range(len(bodyparts))],
                    "all_joints_names": [str(bpt) for bpt in bodyparts],
                    "init_weights": model_path,
                    "project_path": str(cfg["project_path"]),
                    "net_type": net_type,
                    "dataset_type": augmenter_type,
                }
                trainingdata = MakeTrain_pose_yaml(
                    items2change, path_train_config, defaultconfigfile
                )
                keys2save = [
                    "dataset",
                    "num_joints",
                    "all_joints",
                    "all_joints_names",
                    "net_type",
                    "init_weights",
                    "global_scale",
                    "location_refinement",
                    "locref_stdev",
                ]
                MakeTest_pose_yaml(trainingdata, keys2save, path_test_config)
                print(
                    "The training dataset is successfully created. Use the function 'train_network' to start training. Happy training!"
                )
        return splits


def get_largestshuffle_index(config):
    """ Returns the largest shuffle for all dlc-models in the current iteration."""
    cfg = auxiliaryfunctions.read_config(config)
    project_path = cfg["project_path"]
    iterate = "iteration-" + str(cfg["iteration"])
    dlc_model_path = os.path.join(project_path, "dlc-models", iterate)
    if os.path.isdir(dlc_model_path):
        models = os.listdir(dlc_model_path)
        # sort the models directories
        models.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        # get the shuffle index
        max_shuffle_index = int(models[-1].split("shuffle")[-1])
    else:
        max_shuffle_index = 0
    return max_shuffle_index


def create_training_model_comparison(
    config,
    trainindex=0,
    num_shuffles=1,
    net_types=["resnet_50"],
    augmenter_types=["default"],
    userfeedback=False,
    windows2linux=False,
):
    """
    Creates a training dataset with different networks and augmentation types (dataset_loader) so that the shuffles
    have same training and testing indices.

    Therefore, this function is useful for benchmarking the performance of different network and augmentation types on the same training/testdata.\n

    Parameter
    ----------
    config : string
        Full path of the config.yaml file as a string.

    trainindex: int, optional
        Either (in case uniform = True) indexes which element of TrainingFraction in the config file should be used (note it is a list!).
        Alternatively (uniform = False) indexes which folder is dropped, i.e. the first if trainindex=0, the second if trainindex =1, etc.

    num_shuffles : int, optional
        Number of shuffles of training dataset to create, i.e. [1,2,3] for num_shuffles=3. Default is set to 1.

    net_types: list
        Type of networks. Currently resnet_50, resnet_101, resnet_152, mobilenet_v2_1.0,mobilenet_v2_0.75, mobilenet_v2_0.5, and mobilenet_v2_0.35 are supported.

    augmenter_types: list
        Type of augmenters. Currently "default", "imgaug", "tensorpack", and "deterministic" are supported.

    userfeedback: bool, optional
        If this is set to false, then all requested train/test splits are created (no matter if they already exist). If you
        want to assure that previous splits etc. are not overwritten, then set this to True and you will be asked for each split.

    windows2linux: bool.
        The annotation files contain path formated according to your operating system. If you label on windows
        but train & evaluate on a unix system (e.g. ubunt, colab, Mac) set this variable to True to convert the paths.

    Example
    --------
    >>> deeplabcut.create_training_model_comparison('/analysis/project/reaching-task/config.yaml',num_shuffles=1,net_types=['resnet_50','resnet_152'],augmenter_types=['tensorpack','deterministic'])

    Windows:
    >>> deeplabcut.create_training_model_comparison('C:\\Users\\Ulf\\looming-task\\config.yaml',num_shuffles=1,net_types=['resnet_50','resnet_152'],augmenter_types=['tensorpack','deterministic'])

    --------
    """
    # read cfg file
    cfg = auxiliaryfunctions.read_config(config)

    # create log file
    log_file_name = os.path.join(cfg["project_path"], "training_model_comparison.log")
    logger = logging.getLogger("training_model_comparison")
    if not logger.handlers:
        logger = logging.getLogger("training_model_comparison")
        hdlr = logging.FileHandler(log_file_name)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)
    else:
        pass

    largestshuffleindex = get_largestshuffle_index(config)

    for shuffle in range(num_shuffles):
        trainIndices, testIndices = mergeandsplit(
            config, trainindex=trainindex, uniform=True
        )
        for idx_net, net in enumerate(net_types):
            for idx_aug, aug in enumerate(augmenter_types):
                get_max_shuffle_idx = (
                    largestshuffleindex
                    + idx_aug
                    + idx_net * len(augmenter_types)
                    + shuffle * len(augmenter_types) * len(net_types)
                )
                log_info = str(
                    "Shuffle index:"
                    + str(get_max_shuffle_idx)
                    + ", net_type:"
                    + net
                    + ", augmenter_type:"
                    + aug
                    + ", trainsetindex:"
                    + str(trainindex)
                )
                create_training_dataset(
                    config,
                    Shuffles=[get_max_shuffle_idx],
                    net_type=net,
                    trainIndices=[trainIndices],
                    testIndices=[testIndices],
                    augmenter_type=aug,
                    userfeedback=userfeedback,
                    windows2linux=windows2linux,
                )
                logger.info(log_info)
