import os, sys
#os.environ['DLClight']='True' ## THIS NEEDS To be turned off for refine tracklets!
import deeplabcut
#print("CREATING PROJECT") #NEW!
#path_config_file=deeplabcut.create_new_project(task,scorer,video,copy_videos=True,multianimal=True)

basefolder='/media/alex/dropboxdisk/Dropbox/InterestingCode/social_datasets/testprojects/CrackingParenting-Mostafizur-2019-08-08'
path_config_file=os.path.join(basefolder,'config.yaml')
videopath=os.path.join(basefolder,'videos')
videotype='.mp4'
shuffle=0
trainingsetindex=0

print("Plot labels...") #NEW:

deeplabcut.check_labels(path_config_file,draw_skeleton=True,visualizeindividuals=True)
deeplabcut.check_labels(path_config_file,draw_skeleton=True,visualizeindividuals=False)

print("Crop images...") #RECOMMENDED:
#deeplabcut.cropimagesandlabels(path_config_file,userfeedback=False)
#then we uncommented the large-scale full frame data > it is not used for training!

print("Plot labels...") #NEW:
print("Creating multianimal training set...")
deeplabcut.create_multianimaltraining_dataset(path_config_file,Shuffles=[shuffle])

saveiters=5000
displayiters=500

print("Creating multianimal training set...")

deeplabcut.train_network(path_config_file, shuffle=shuffle,trainingsetindex=trainingsetindex,
    saveiters=saveiters,displayiters=displayiters,max_snapshots_to_keep=2,maxiters=20000)

trainposeconfigfile,testposeconfigfile,snapshotfolder=deeplabcut.return_train_network_path(path_config_file,shuffle=shuffle)

edits = {'partaffinityfield_predict': True,
         'dataset_type': 'multi-animal-imgaug',
         'nmsradius': 5.,
         'minconfidence': .05}
cfg_dlc = deeplabcut.auxiliaryfunctions.edit_config(testposeconfigfile, edits)

#print("Evaluating network for shuffle ", shuffle) #not fully implemented in 2.2b1
#deeplabcut.evaluate_network(path_config_file,Shuffles=[shuffle])



################## Analyze video #NEW:
trainposeconfigfile,testposeconfigfile,snapshotfolder=deeplabcut.return_train_network_path(path_config_file,shuffle=shuffle)
edits = {'partaffinityfield_predict': True,
         'dataset_type': 'multi-animal-imgaug',
         'nmsradius': 5.,
         'minconfidence': .01}
cfg_dlc = deeplabcut.auxiliaryfunctions.edit_config(testposeconfigfile, edits)

print("Starting inference for", shuffle)
deeplabcut.analyze_videos(path_config_file,[videopath],shuffle=shuffle,videotype=videotype)


deeplabcut.convert_detections2tracklets(path_config_file,[videopath],shuffle=shuffle,videotype=videotype)

#this also works for the tracklets!
deeplabcut.create_labeled_video(path_config_file,[videopath],shuffle=shuffle,videotype=videotype)
