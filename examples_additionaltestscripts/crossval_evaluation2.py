import deeplabcut
import os


projectpath = "/media/alex/dropboxdisk/Dropbox/InterestingCode/social_datasets/croppedNov18/MultiMouse-Daniel-2019-12-16"

projectpath = "/media/alex/dropboxdisk/Dropbox/InterestingCode/social_datasets/MultiMouse-Daniel-2019-12-16"
modelprefix = "simplebaseline"

config = os.path.join(projectpath, "config.yaml")

trainingsetindex = 0
shuffle = 0
# deeplabcut.evaluate_multianimal_crossvalidate(config,[0], trainingsetindex, modelprefix)

# that sets a good inference_cfg.yaml!#


##############################################
# let's assume we already predicted bpts!
###################################################

# deeplabcut.analyze_videos(config,[videopath],videotype='.mp4',shuffle=shuffle,trainingsetindex=trainingsetindex, destfolder=destfolder,modelprefix = modelprefix)

videopath = os.path.join(projectpath, "videos")
destfolder = None  # os.path.join(projectpath,modelprefix)
videotype = ".mp4"

##############################################
# Now convert the data:
###################################################

edgewisecondition = True
deeplabcut.convert_detections2tracklets(
    config,
    [videopath],
    videotype=".mp4",
    shuffle=shuffle,
    trainingsetindex=trainingsetindex,
    destfolder=destfolder,
    modelprefix=modelprefix,
    edgewisecondition=edgewisecondition,
)
# deeplabcut.create_labeled_video(config,[videopath],videotype='.mp4',shuffle=shuffle,trainingsetindex=trainingsetindex, destfolder=destfolder,modelprefix=modelprefix)
