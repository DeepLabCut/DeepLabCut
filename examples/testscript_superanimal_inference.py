"""
Testscript human network

"""
import os, subprocess, deeplabcut
from pathlib import Path
import pandas as pd
import numpy as np

Task = "human_dancing"
YourName = "teamDLC"

basepath = os.path.dirname(os.path.abspath("testscript.py"))
videoname = "reachingvideo1"
video = [
    os.path.join(
        basepath, "Reaching-Mackenzie-2018-08-30", "videos", videoname + ".avi"
    )
]

# legacy mode:
"""
configfile, path_train_config=deeplabcut.create_pretrained_human_project(Task, YourName,video,
                                                                        videotype='avi', analyzevideo=True,
                                                                        createlabeledvideo=True, copy_videos=False) #must leave copy_videos=True
"""
# new way:
configfile, path_train_config = deeplabcut.create_pretrained_project(
    Task,
    YourName,
    video,
    model="full_human",
    videotype="avi",
    analyzevideo=True,
    createlabeledvideo=True,
    copy_videos=False,
)  # must leave copy_videos=True


lastvalue = 5
DLC_config = deeplabcut.auxiliaryfunctions.read_plainconfig(path_train_config)
pretrainedDeeperCutweights = DLC_config["init_weights"]

print("EXTRACTING FRAMES")
deeplabcut.extract_frames(configfile, mode="automatic", userfeedback=False)

print("CREATING-SOME LABELS FOR THE FRAMES")
cfg = deeplabcut.auxiliaryfunctions.read_config(configfile)
frames = os.listdir(os.path.join(cfg["project_path"], "labeled-data", videoname))
# As this next step is manual, we update the labels by putting them on the diagonal (fixed for all frames)
for index, bodypart in enumerate(cfg["bodyparts"]):
    columnindex = pd.MultiIndex.from_product(
        [[cfg["scorer"]], [bodypart], ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    frame = pd.DataFrame(
        100 + np.ones((len(frames), 2)) * 50 * index,
        columns=columnindex,
        index=[os.path.join("labeled-data", videoname, fn) for fn in frames],
    )
    if index == 0:
        dataFrame = frame
    else:
        dataFrame = pd.concat([dataFrame, frame], axis=1)

dataFrame.to_csv(
    os.path.join(
        cfg["project_path"],
        "labeled-data",
        videoname,
        "CollectedData_" + cfg["scorer"] + ".csv",
    )
)
dataFrame.to_hdf(
    os.path.join(
        cfg["project_path"],
        "labeled-data",
        videoname,
        "CollectedData_" + cfg["scorer"] + ".h5",
    ),
    "df_with_missing",
    format="table",
    mode="w",
)


scale_list = [200, 300, 400]
print ('start ---')
print (configfile,
       video)
print ('end --- ')
deeplabcut.video_inference_superanimal(
    configfile,
    video,
    videotype='avi',
    init_weights = pretrainedDeeperCutweights,
    scale_list = scale_list,
)
