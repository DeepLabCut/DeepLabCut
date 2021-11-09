"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.generate_training_dataset import trainingsetmanipulation

def extractindividualsandbodyparts(cfg):
    individuals = cfg["individuals"].copy()
    if len(cfg["uniquebodyparts"]) > 0:
        individuals.append("single")
    return individuals, cfg["uniquebodyparts"], cfg["multianimalbodyparts"]


def IntersectionofIndividualsandOnesGivenbyUser(cfg, individuals):
    """ Returns all individuals when set to 'all', otherwise all bpts that are in the intersection of comparisonbodyparts and the actual bodyparts """
    if "individuals" not in cfg:  # Not a multi-animal project...
        return [""]
    all_indivs = extractindividualsandbodyparts(cfg)[0]
    if individuals == "all":
        return all_indivs
    else:  # take only items in list that are actually bodyparts...
        return [ind for ind in individuals if ind in all_indivs]


def filter_unwanted_paf_connections(cfg, paf_graph):
    """Get rid of skeleton connections between multi and unique body parts."""
    multi = extractindividualsandbodyparts(cfg)[2]
    desired = list(combinations(range(len(multi)), 2))
    return [i for i, edge in enumerate(paf_graph) if tuple(edge) not in desired]


def validate_paf_graph(cfg, paf_graph):
    multianimalbodyparts = extractindividualsandbodyparts(cfg)[2]
    connected = set()
    for bpt1, bpt2 in paf_graph:
        connected.add(bpt1)
        connected.add(bpt2)
    unconnected = set(range(len(multianimalbodyparts))).difference(connected)
    if unconnected and len(multianimalbodyparts) > 1:  # for single bpt not important!
        raise ValueError(
            f'Unconnected {", ".join(multianimalbodyparts[i] for i in unconnected)}. '
            f"For multi-animal projects, all multianimalbodyparts should be connected. "
            f"Ideally there should be at least one (multinode) path from each multianimalbodyparts to each other multianimalbodyparts. "
        )


def getpafgraph(cfg, printnames=True):
    """ Auxiliary function that turns skeleton (list of connected bodypart pairs)
        into a list of corresponding indices (with regard to the stacked multianimal/uniquebodyparts)

        Convention: multianimalbodyparts go first!
    """
    individuals, uniquebodyparts, multianimalbodyparts = extractindividualsandbodyparts(
        cfg
    )
    # Attention this order has to be consistent (for training set creation, training, inference etc.)

    bodypartnames = multianimalbodyparts + uniquebodyparts
    lookupdict = {bodypartnames[j]: j for j in range(len(bodypartnames))}

    if cfg["skeleton"] is None:
        cfg["skeleton"] = []

    connected = set()
    partaffinityfield_graph = []
    for link in cfg["skeleton"]:
        if link[0] in bodypartnames and link[1] in bodypartnames:
            bp1 = int(lookupdict[link[0]])
            bp2 = int(lookupdict[link[1]])
            connected.add(bp1)
            connected.add(bp2)
            partaffinityfield_graph.append([bp1, bp2])
        else:
            print("Attention, parts do not exist!", link)

    if printnames:
        graph2names(cfg, partaffinityfield_graph)

    return partaffinityfield_graph


def graph2names(cfg, partaffinityfield_graph):
    individuals, uniquebodyparts, multianimalbodyparts = extractindividualsandbodyparts(
        cfg
    )
    bodypartnames = multianimalbodyparts + uniquebodyparts
    for pair in partaffinityfield_graph:
        print(pair, bodypartnames[pair[0]], bodypartnames[pair[1]])


def SaveFullMultiAnimalData(data, metadata, dataname, suffix="_full"):
    """ Save predicted data as h5 file and metadata as pickle file; created by predict_videos.py """
    data_path = dataname.split(".h5")[0] + suffix + ".pickle"
    metadata_path = dataname.split(".h5")[0] + "_meta.pickle"
    with open(data_path, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)
    return data_path, metadata_path


def LoadFullMultiAnimalData(dataname):
    """ Save predicted data as h5 file and metadata as pickle file; created by predict_videos.py """
    with open(dataname.split(".h5")[0] + "_full.pickle", "rb") as handle:
        data = pickle.load(handle)
    with open(dataname.split(".h5")[0] + "_meta.pickle", "rb") as handle:
        metadata = pickle.load(handle)
    return data, metadata


def returnlabelingdata(config):
    """ Returns a specific labeleing data set -- the user will be asked which one. """
    cfg = auxiliaryfunctions.read_config(config)
    videos = cfg["video_sets"].keys()
    video_names = [Path(i).stem for i in videos]
    folders = [Path(config).parent / "labeled-data" / Path(i) for i in video_names]
    for folder in folders:
        print("Do you want to get the data for folder:", folder, "?")
        askuser = input("yes/no")
        if (
            askuser == "y" or askuser == "yes" or askuser == "Ja" or askuser == "ha"
        ):  # multilanguage support :)
            fn = os.path.join(str(folder), "CollectedData_" + cfg["scorer"] + ".h5")
            Data = pd.read_hdf(fn)
            return Data


def convert2_maDLC(config, userfeedback=True, forceindividual=None):
    """
    Converts single animal annotation file into a multianimal annotation file,
    by introducing an individuals column with either the first individual
    in individuals list in config.yaml or whatever is passsed via "forceindividual".

    ----------
    config : string
        Full path of the config.yaml file as a string.

    userfeedback: bool, optional
            If this is set to false during automatic mode then frames for all videos are extracted. The user can set this to true, which will result in a dialog,
            where the user is asked for each video if (additional/any) frames from this video should be extracted. Use this, e.g. if you have already labeled
            some folders and want to extract data for new videos.

    forceindividual: None default
            If a string is given that is used in the individuals column.

    Examples
    --------
    Converts mulianimalbodyparts under the 'first individual' in individuals list in config.yaml
    and uniquebodyparts under 'single'
    >>> deeplabcut.convert2_maDLC('/socialrearing-task/config.yaml')

    --------
    Converts mulianimalbodyparts under the individual label mus17 and uniquebodyparts under 'single'
    >>> deeplabcut.convert2_maDLC('/socialrearing-task/config.yaml', forceindividual='mus17')
    """

    cfg = auxiliaryfunctions.read_config(config)
    videos = cfg["video_sets"].keys()
    video_names = [trainingsetmanipulation._robust_path_split(i)[1] for i in videos]
    folders = [Path(config).parent / "labeled-data" / Path(i) for i in video_names]

    individuals, uniquebodyparts, multianimalbodyparts = extractindividualsandbodyparts(
        cfg
    )

    if forceindividual is None:
        if len(individuals) == 0:
            print("At least one individual should exist...")
            folders = []
            forceindividual = ""
        else:
            forceindividual = individuals[0]  # note that single is added at then end!

        if forceindividual == "single":  # no specific individual ()
            if len(multianimalbodyparts) > 0:  # there should be an individual name...
                print(
                    "At least one individual should exist beyond 'single', as there are multianimalbodyparts..."
                )
                folders = []

    for folder in folders:
        if userfeedback == True:
            print("Do you want to convert the annotation file in folder:", folder, "?")
            askuser = input("yes/no")
        else:
            askuser = "yes"

        if (
            askuser == "y" or askuser == "yes" or askuser == "Ja" or askuser == "ha"
        ):  # multilanguage support :)

            fn = os.path.join(str(folder), "CollectedData_" + cfg["scorer"])
            Data = pd.read_hdf(fn + ".h5")
            imindex = Data.index

            print("This is a single animal data set, converting to multi...", folder)

            # -> adding (single,bpt) for uniquebodyparts
            for j, bpt in enumerate(uniquebodyparts):
                index = pd.MultiIndex.from_arrays(
                    np.array(
                        [2 * [cfg["scorer"]], 2 * ["single"], 2 * [bpt], ["x", "y"]]
                    ),
                    names=["scorer", "individuals", "bodyparts", "coords"],
                )

                if bpt in Data[cfg["scorer"]].keys():
                    frame = pd.DataFrame(
                        Data[cfg["scorer"]][bpt].values, columns=index, index=imindex
                    )
                else:
                    frame = pd.DataFrame(
                        np.ones((len(imindex), 2)) * np.nan,
                        columns=index,
                        index=imindex,
                    )

                if j == 0:
                    dataFrame = frame
                else:
                    dataFrame = pd.concat([dataFrame, frame], axis=1)

            if len(uniquebodyparts) == 0:
                dataFrame = None

            # -> adding (indivdual,bpt) for multianimalbodyparts
            for j, bpt in enumerate(multianimalbodyparts):
                index = pd.MultiIndex.from_arrays(
                    np.array(
                        [
                            2 * [cfg["scorer"]],
                            2 * [str(forceindividual)],
                            2 * [bpt],
                            ["x", "y"],
                        ]
                    ),
                    names=["scorer", "individuals", "bodyparts", "coords"],
                )

                if bpt in Data[cfg["scorer"]].keys():
                    frame = pd.DataFrame(
                        Data[cfg["scorer"]][bpt].values, columns=index, index=imindex
                    )
                else:
                    frame = pd.DataFrame(
                        np.ones((len(imindex), 2)) * np.nan,
                        columns=index,
                        index=imindex,
                    )

                if j == 0 and dataFrame is None:
                    dataFrame = frame
                else:
                    dataFrame = pd.concat([dataFrame, frame], axis=1)

            Data.to_hdf(
                fn + "singleanimal.h5", "df_with_missing", format="table", mode="w"
            )
            Data.to_csv(fn + "singleanimal.csv")

            dataFrame.to_hdf(fn + ".h5", "df_with_missing", format="table", mode="w")
            dataFrame.to_csv(fn + ".csv")


def convert_single2multiplelegacyAM(config, userfeedback=True, target=None):
    """ Convert multi animal to single animal code and vice versa. Note that by providing target='single'/'multi' this will be target! """
    cfg = auxiliaryfunctions.read_config(config)
    videos = cfg["video_sets"].keys()
    video_names = [Path(i).stem for i in videos]
    folders = [Path(config).parent / "labeled-data" / Path(i) for i in video_names]

    prefixes, uniquebodyparts, multianimalbodyparts = extractindividualsandbodyparts(
        cfg
    )
    for folder in folders:
        if userfeedback == True:
            print("Do you want to convert the annotation file in folder:", folder, "?")
            askuser = input("yes/no")
        else:
            askuser = "yes"

        if (
            askuser == "y" or askuser == "yes" or askuser == "Ja" or askuser == "ha"
        ):  # multilanguage support :)
            fn = os.path.join(str(folder), "CollectedData_" + cfg["scorer"])
            Data = pd.read_hdf(fn + ".h5")
            imindex = Data.index

            if "individuals" in Data.columns.names and (
                target == None or target == "single"
            ):
                print("This is a multianimal data set, converting to single...", folder)
                for prfxindex, prefix in enumerate(prefixes):
                    if prefix == "single":
                        for j, bpt in enumerate(uniquebodyparts):
                            index = pd.MultiIndex.from_product(
                                [[cfg["scorer"]], [bpt], ["x", "y"]],
                                names=["scorer", "bodyparts", "coords"],
                            )
                            frame = pd.DataFrame(
                                Data[cfg["scorer"]][prefix][bpt].values,
                                columns=index,
                                index=imindex,
                            )
                            if j == 0:
                                dataFrame = frame
                            else:
                                dataFrame = pd.concat([dataFrame, frame], axis=1)
                    else:
                        for j, bpt in enumerate(multianimalbodyparts):
                            index = pd.MultiIndex.from_product(
                                [[cfg["scorer"]], [prefix + bpt], ["x", "y"]],
                                names=["scorer", "bodyparts", "coords"],
                            )
                            frame = pd.DataFrame(
                                Data[cfg["scorer"]][prefix][bpt].values,
                                columns=index,
                                index=imindex,
                            )
                            if j == 0:
                                dataFrame = frame
                            else:
                                dataFrame = pd.concat([dataFrame, frame], axis=1)
                    if prfxindex == 0:
                        DataFrame = dataFrame
                    else:
                        DataFrame = pd.concat([DataFrame, dataFrame], axis=1)

                Data.to_hdf(
                    fn + "multianimal.h5", "df_with_missing", format="table", mode="w"
                )
                Data.to_csv(fn + "multianimal.csv")

                DataFrame.to_hdf(
                    fn + ".h5", "df_with_missing", format="table", mode="w"
                )
                DataFrame.to_csv(fn + ".csv")
            elif target == None or target == "multi":
                print(
                    "This is a single animal data set, converting to multi...", folder
                )
                for prfxindex, prefix in enumerate(prefixes):
                    if prefix == "single":
                        if cfg["uniquebodyparts"] != [None]:
                            for j, bpt in enumerate(uniquebodyparts):
                                index = pd.MultiIndex.from_arrays(
                                    np.array(
                                        [
                                            2 * [cfg["scorer"]],
                                            2 * [prefix],
                                            2 * [bpt],
                                            ["x", "y"],
                                        ]
                                    ),
                                    names=[
                                        "scorer",
                                        "individuals",
                                        "bodyparts",
                                        "coords",
                                    ],
                                )
                                if bpt in Data[cfg["scorer"]].keys():
                                    frame = pd.DataFrame(
                                        Data[cfg["scorer"]][bpt].values,
                                        columns=index,
                                        index=imindex,
                                    )
                                else:  # fill with nans...
                                    frame = pd.DataFrame(
                                        np.ones((len(imindex), 2)) * np.nan,
                                        columns=index,
                                        index=imindex,
                                    )

                                if j == 0:
                                    dataFrame = frame
                                else:
                                    dataFrame = pd.concat([dataFrame, frame], axis=1)
                        else:
                            dataFrame = None
                    else:
                        for j, bpt in enumerate(multianimalbodyparts):
                            index = pd.MultiIndex.from_arrays(
                                np.array(
                                    [
                                        2 * [cfg["scorer"]],
                                        2 * [prefix],
                                        2 * [bpt],
                                        ["x", "y"],
                                    ]
                                ),
                                names=["scorer", "individuals", "bodyparts", "coords"],
                            )
                            if prefix + "_" + bpt in Data[cfg["scorer"]].keys():
                                frame = pd.DataFrame(
                                    Data[cfg["scorer"]][prefix + "_" + bpt].values,
                                    columns=index,
                                    index=imindex,
                                )
                            else:
                                frame = pd.DataFrame(
                                    np.ones((len(imindex), 2)) * np.nan,
                                    columns=index,
                                    index=imindex,
                                )

                            if j == 0:
                                dataFrame = frame
                            else:
                                dataFrame = pd.concat([dataFrame, frame], axis=1)
                    if prfxindex == 0:
                        DataFrame = dataFrame
                    else:
                        DataFrame = pd.concat([DataFrame, dataFrame], axis=1)

                Data.to_hdf(
                    fn + "singleanimal.h5", "df_with_missing", format="table", mode="w"
                )
                Data.to_csv(fn + "singleanimal.csv")

                DataFrame.to_hdf(
                    fn + ".h5", "df_with_missing", format="table", mode="w"
                )
                DataFrame.to_csv(fn + ".csv")


def form_default_inferencecfg(cfg):
    # load defaults
    inferencecfg = auxiliaryfunctions.read_plainconfig(
        os.path.join(auxiliaryfunctions.get_deeplabcut_path(), "inference_cfg.yaml")
    )
    # set project specific parameters:
    inferencecfg["minimalnumberofconnections"] = (
        len(cfg["multianimalbodyparts"]) / 2
    )  # reasonable default
    inferencecfg["topktoretain"] = len(cfg["individuals"]) + 1 * (
        len(cfg["uniquebodyparts"]) > 0
    )  # reasonable default
    return inferencecfg


def check_inferencecfg_sanity(cfg, inferencecfg):
    template = form_default_inferencecfg(cfg)
    missing = [key for key in template if key not in inferencecfg]
    if missing:
        raise KeyError(f'Keys {", ".join(missing)} are missing in the inferencecfg.')


def read_inferencecfg(path_inference_config, cfg):
    """Load inferencecfg or initialize it."""
    try:
        inferencecfg = auxiliaryfunctions.read_plainconfig(str(path_inference_config))
    except FileNotFoundError:
        inferencecfg = form_default_inferencecfg(cfg)
        auxiliaryfunctions.write_plainconfig(
            str(path_inference_config), dict(inferencecfg)
        )
    return inferencecfg
