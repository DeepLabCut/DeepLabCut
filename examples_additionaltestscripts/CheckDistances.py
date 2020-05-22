import deeplabcut, pickle, os
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from tqdm import tqdm
import matplotlib.pyplot as plt

shuffle = 0
trainingsetindex = 0
from deeplabcut.utils import auxfun_multianimal

basepath = "/home/alex/Dropbox/InterestingCode/social_datasets/croppedNov18/CrackingParenting-Mostafizur-2019-08-08"
# basepath='/home/alex/Dropbox/InterestingCode/social_datasets/croppeddatasets/silversideschooling-Valentina-2019-07-14'


configfile = os.path.join(basepath, "config.yaml")
cfg = deeplabcut.auxiliaryfunctions.read_config(configfile)

deeplabcut.create_training_dataset(configfile)

(
    individuals,
    uniquebodyparts,
    multianimalbodyparts,
) = auxfun_multianimal.extractindividualsandbodyparts(cfg)

trainingsetfolder = deeplabcut.auxiliaryfunctions.GetTrainingSetFolder(cfg)

trainFraction = cfg["TrainingFraction"][trainingsetindex]
datafn, metadatafn = deeplabcut.auxiliaryfunctions.GetDataandMetaDataFilenames(
    trainingsetfolder, trainFraction, shuffle, cfg
)
# data, trainIndices, testIndices, trainFraction=auxiliaryfunctions.LoadMetadata(os.path.join(cfg["project_path"],metadatafn))
Data = pd.read_hdf(
    os.path.join(
        cfg["project_path"],
        str(trainingsetfolder),
        "CollectedData_" + cfg["scorer"] + ".h5",
    ),
    "df_with_missing",
)

Data = Data[cfg["scorer"]]


def distance(v, w):
    return np.sqrt(np.sum((v - w) ** 2))


skeleton = cfg["skeleton"]

imnames = Data.index

outfolder = (
    basepath  #'/home/alex/Dropbox/InterestingCode/social_datasets/Results-SDLCpaper'
)

## WHAT ABOUT GT costs?
numjoints = len(multianimalbodyparts) + len(uniquebodyparts)

Joints = multianimalbodyparts
Joints.extend(uniquebodyparts)

Joints = np.array(Joints)
Distances = np.zeros((numjoints, numjoints)) * np.nan
# Distances across bodyparts!
for p in skeleton:
    ds_within = []
    ds_across = []
    plt.close("all")
    for jj, imname in enumerate(Data.index):
        if p[0] in uniquebodyparts and p[1] in uniquebodyparts:
            ind = "single"
            source = np.array(Data[ind, p[0], "x"][jj], Data[ind, p[0], "y"][jj])
            target = np.array(Data[ind, p[1], "x"][jj], Data[ind, p[1], "y"][jj])
            ds_within.append(distance(source, target))
            if ind == ind2 and distance(source, target) > 100:
                print(imname, distance(source, target), p)
        else:
            for ind in individuals:
                for ind2 in individuals:
                    if ind != "single" and ind2 != "single":
                        source = np.array(
                            Data[ind, p[0], "x"][jj], Data[ind, p[0], "y"][jj]
                        )
                        target = np.array(
                            Data[ind2, p[1], "x"][jj], Data[ind2, p[1], "y"][jj]
                        )
                        if ind == ind2:
                            ds_within.append(distance(source, target))
                        else:
                            ds_across.append(distance(source, target))

                        if ind == ind2 and distance(source, target) > 66:
                            print(imname, distance(source, target), p)
    i, j = np.where(Joints == p[0])[0][0], np.where(Joints == p[1])[0][0]

    print(i, j)
    Distances[i, j] = np.nanmean(ds_within)
    plt.figure()
    # plt.vlines(150,0,3000)
    plt.hist(ds_across, 100, histtype="step")
    plt.hist(ds_within, 100, histtype="step")
    plt.savefig(os.path.join(outfolder, "connection" + str(p) + ".png"))

plt.figure()
plt.imshow(Distances)
plt.colorbar()
plt.savefig(os.path.join(outfolder, "distancematrix.png"))
