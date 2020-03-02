import os, sys
os.environ['DLClight']='True'
import deeplabcut

path_config_file='/media/alex/dropboxdisk/Dropbox/InterestingCode/social_datasets/sideviews/silversideschooling-valentina-2019-04-19/config.yaml'
videopath='/media/alex/dropboxdisk/Dropbox/InterestingCode/social_datasets/sideviews/silversideschooling-valentina-2019-04-19/videos'

print("Plot labels...")
#deeplabcut.check_labels(path_config_file)
#deeplabcut.check_labels(path_config_file,draw_skeleton=True,visualizeindividuals=True)
#deeplabcut.check_labels(path_config_file,draw_skeleton=True,visualizeindividuals=False)



print("Crop images...")
#deeplabcut.cropimagesandlabels(path_config_file,userfeedback=False)
#then we uncommented the large-scale full frame data > it is not used for training!


#cfg=deeplabcut.utils.auxiliaryfunctions.read_config(path_config_file)
#deeplabcut.utils.auxfun_multianimal.getpafgraph(cfg)

shuffle=0
trainingsetindex=0

#deeplabcut.comparevideolistsanddatafolders(path_config_file) #does not exist in light mode!
#print("Creating multianimal training set...")
#deeplabcut.create_multianimaltraining_dataset(path_config_file,Shuffles=[shuffle])

'''
import shutil
projectpath=Path(path_config_file).parents[0]
modelprefix='prefixtest'
print("Copying datasets")
try:
    shutil.copytree(os.path.join(projectpath,'dlc-models'),os.path.join(projectpath,modelprefix,'dlc-models'))
except FileExistsError:
    print("Folder exists already...")
'''
trainposeconfigfile,testposeconfigfile,snapshotfolder=deeplabcut.return_train_network_path(path_config_file,shuffle=shuffle)
'''
cfg_dlc=deeplabcut.auxiliaryfunctions.read_plainconfig(trainposeconfigfile)
cfg_dlc['bank3']=128
cfg_dlc['bank5']=128
cfg_dlc['smfactor']=4
cfg_dlc['stride']=4
cfg_dlc['augmentationprobability']=.6
cfg_dlc['weigh_only_present_joints']=False
cfg_dlc['cropratio']=.8
cfg_dlc['cropfactor']=.2
cfg_dlc['rotation']=180 #can also be an integer def. -10,10 if true.
cfg_dlc['covering']=True
cfg_dlc['motion_blur'] = True #[["k", 7],["angle", [-90, 90]]]
deeplabcut.auxiliaryfunctions.write_plainconfig(trainposeconfigfile,cfg_dlc)

saveiters=5000
displayiters=100

print("Creating multianimal training set...")
#deeplabcut.train_network(path_config_file, shuffle=shuffle,trainingsetindex=trainingsetindex,saveiters=saveiters,displayiters=displayiters)
deeplabcut.train_network(path_config_file, shuffle=shuffle,trainingsetindex=trainingsetindex,
    saveiters=saveiters,displayiters=displayiters,max_snapshots_to_keep=2,maxiters=20000)

'''
################## Analyze video
nmspath = 'deeplabcut/pose_estimation_tensorflow/lib/nms_cython'
sys.path.append(os.path.join('/usr/local/lib/python3.6/dist-packages',nmspath))


cfg_dlc=deeplabcut.auxiliaryfunctions.read_plainconfig(testposeconfigfile)
cfg_dlc['partaffinityfield_predict']=True
cfg_dlc['dataset_type']='multi-animal-imgaug'
cfg_dlc['nmsradius']=5.
cfg_dlc['minconfidence']=.1
cfg_dlc['bank3']=128
cfg_dlc['bank5']=128
cfg_dlc['smfactor']=4
cfg_dlc['stride']=4
deeplabcut.auxiliaryfunctions.write_plainconfig(testposeconfigfile,cfg_dlc)

print("Evaluating network for shuffle ", shuffle)
deeplabcut.evaluate_network(path_config_file,Shuffles=[shuffle])

print("Starting inference for", shuffle)
#deeplabcut.analyze_videos(path_config_file,[videopath],shuffle=shuffle,videotype='.avi')


'''

deeplabcut.convert_detections2tracklets(path_config_file,[videopath],videotype='.mov')

deeplabcut.create_labeled_video(path_config_file,[videopath],videotype='.mov')
'''
