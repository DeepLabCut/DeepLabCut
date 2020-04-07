import os, sys
#os.environ['DLClight']='True'
import deeplabcut

nmspath = 'deeplabcut/pose_estimation_tensorflow/lib/nms_cython'
sys.path.append(os.path.join('/usr/local/lib/python3.6/dist-packages',nmspath))

task='MontBlanc'
scorer='Daniel'


#print("CREATING PROJECT")
#path_config_file=deeplabcut.create_new_project(task,scorer,video,copy_videos=True,multianimal=True)

#project already created!

#basefolder='/media/alex/dropboxdisk/Dropbox/InterestingCode/social_datasets/sideviews/silversideschooling-valentina-2019-04-19'
#videotype='avi'

basefolder='/media/alex/dropboxdisk/Dropbox/Collaborations/Spider/merged/2020_03_05_3cm_angled_camera-1-kkim-2020-03-05'
videotype='.avi'

path_config_file=os.path.join(basefolder,'config.yaml')
videopath=os.path.join(basefolder,'videos')

shuffle=1
trainingsetindex=0


#deeplabcut.convertannotationdata_fromwindows2unixstyle(path_config_file)
#deeplabcut.conversioncodemulti2single(path_config_file)

print("Plot labels...")
#deeplabcut.check_labels(path_config_file)
#deeplabcut.check_labels(path_config_file,draw_skeleton=True,visualizeindividuals=True)
#deeplabcut.check_labels(path_config_file,draw_skeleton=True,visualizeindividuals=False)

print("Crop images...")
#deeplabcut.cropimagesandlabels(path_config_file,userfeedback=False,numcrops=5)
#then we uncommented the large-scale full frame data > it is not used for training!

#if 'Plat' not in bp: #FOR
#    animalincrop=True

#cfg=deeplabcut.utils.auxiliaryfunctions.read_config(path_config_file)
#deeplabcut.utils.auxfun_multianimal.getpafgraph(cfg)

#print("Creating multianimal training set...")
deeplabcut.create_multianimaltraining_dataset(path_config_file,Shuffles=[shuffle])

saveiters=5000
displayiters=500
trainposeconfigfile,testposeconfigfile,snapshotfolder=deeplabcut.return_train_network_path(path_config_file,shuffle=shuffle)

print("Creating multianimal training set...")
#deeplabcut.train_network(path_config_file, shuffle=shuffle,trainingsetindex=trainingsetindex,saveiters=saveiters,displayiters=displayiters)
deeplabcut.train_network(path_config_file, shuffle=shuffle,trainingsetindex=trainingsetindex,
    saveiters=saveiters,displayiters=displayiters,max_snapshots_to_keep=2,maxiters=200000)


edits = {'partaffinityfield_predict': True,
         'dataset_type': 'multi-animal-imgaug',
         'nmsradius': 5.,
         'minconfidence': .05}
cfg_dlc = deeplabcut.auxiliaryfunctions.edit_config(testposeconfigfile, edits)



################## Analyze video

cfg_dlc = deeplabcut.auxiliaryfunctions.edit_config(testposeconfigfile, {'partaffinityfield_predict': False,
                                                                         'dataset_type': 'imgaug'})

#print("Evaluating network for shuffle ", shuffle)
#deeplabcut.evaluate_network(path_config_file,Shuffles=[shuffle])


print("Starting inference for", shuffle)

deeplabcut.analyze_videos(path_config_file,[videopath],shuffle=shuffle,videotype=videotype)

deeplabcut.plot_trajectories(path_config_file,[videopath],shuffle=shuffle,videotype=videotype)
#deeplabcut.convert_detections2tracklets(path_config_file,[videopath],shuffle=shuffle,videotype=videotype)
deeplabcut.create_labeled_video(path_config_file,[videopath],shuffle=shuffle,videotype=videotype)
