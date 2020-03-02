import os, sys
os.environ['DLClight']='True'
import deeplabcut

task='MontBlanc'
scorer='Daniel'

video=['/home/alex/Dropbox/InterestingCode/social_datasets/croppedNov18/montblanc.mov']

#print("CREATING PROJECT")
#path_config_file=deeplabcut.create_new_project(task,scorer,video,copy_videos=True,multianimal=True)

#project already created!

#path_config_file='/home/alex/Hacking/DLCreleases/DLCdev/examples-multianimal/MontBlanc-Daniel-2019-12-16/config.yaml'

path_config_file='/media/alex/dropboxdisk/Dropbox/Collaborations/Cancer/DLCdev/examples-multianimal/MontBlanc-Daniel-2019-12-16/config.yaml'
videopath='/media/alex/dropboxdisk/Dropbox/Collaborations/Cancer/DLCdev/examples-multianimal/MontBlanc-Daniel-2019-12-16/videos'

shuffle=0
trainingsetindex=0



'''
print("Plot labels...")
#deeplabcut.check_labels(path_config_file)
#deeplabcut.check_labels(path_config_file,draw_skeleton=True,visualizeindividuals=True)
#deeplabcut.check_labels(path_config_file,draw_skeleton=True,visualizeindividuals=False)

print("Crop images...")
#deeplabcut.cropimagesandlabels(path_config_file,userfeedback=False)
#then we uncommented the large-scale full frame data > it is not used for training!


#cfg=deeplabcut.utils.auxiliaryfunctions.read_config(path_config_file)
#deeplabcut.utils.auxfun_multianimal.getpafgraph(cfg)

print("Creating multianimal training set...")
deeplabcut.create_multianimaltraining_dataset(path_config_file,Shuffles=[shuffle])

saveiters=5000
displayiters=500

print("Creating multianimal training set...")
#deeplabcut.train_network(path_config_file, shuffle=shuffle,trainingsetindex=trainingsetindex,saveiters=saveiters,displayiters=displayiters)
deeplabcut.train_network(path_config_file, shuffle=shuffle,trainingsetindex=trainingsetindex,
    saveiters=saveiters,displayiters=displayiters,max_snapshots_to_keep=2,maxiters=20000)

################## Analyze video
nmspath = 'deeplabcut/pose_estimation_tensorflow/lib/nms_cython'
sys.path.append(os.path.join('/usr/local/lib/python3.6/dist-packages',nmspath))

trainposeconfigfile,testposeconfigfile,snapshotfolder=deeplabcut.return_train_network_path(path_config_file,shuffle=shuffle)
cfg_dlc=deeplabcut.auxiliaryfunctions.read_plainconfig(testposeconfigfile)
cfg_dlc['partaffinityfield_predict']=True
cfg_dlc['dataset_type']='multi-animal-imgaug'
cfg_dlc['nmsradius']=5.
cfg_dlc['minconfidence']=.01
deeplabcut.auxiliaryfunctions.write_plainconfig(testposeconfigfile,cfg_dlc)

print("Starting inference for", shuffle)
deeplabcut.analyze_videos(path_config_file,[videopath],shuffle=shuffle,videotype='.mov')
'''

#deeplabcut.convert_detections2tracklets(path_config_file,[videopath],shuffle=shuffle,videotype='.mov')

deeplabcut.create_labeled_video(path_config_file,[videopath],shuffle=shuffle,videotype='.mov')
