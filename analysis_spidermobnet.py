import os, sys
os.environ['DLClight']='True'
import deeplabcut

nmspath = 'deeplabcut/pose_estimation_tensorflow/lib/nms_cython'
sys.path.append(os.path.join('/usr/local/lib/python3.6/dist-packages',nmspath))

task='MontBlanc'
scorer='Daniel'

basefolder='/media/alex/dropboxdisk/Dropbox/Collaborations/Spider/mobnet/2020_03_05_3cm_angled_camera-1-kkim-2020-03-05'
videotype='.avi'


### videos in merged data!
path_config_file=os.path.join(basefolder,'config.yaml')
videopath=os.path.join('/media/alex/dropboxdisk/Dropbox/Collaborations/Spider/merged/2020_03_05_3cm_angled_camera-1-kkim-2020-03-05','videos')

shuffle=0
trainingsetindex=0



#>> to convert to single!

#deeplabcut.conversioncodemulti2single(path_config_file)

print("Plot labels...")
#deeplabcut.check_labels(path_config_file)
#deeplabcut.check_labels(path_config_file,draw_skeleton=True,visualizeindividuals=True)
#deeplabcut.check_labels(path_config_file,draw_skeleton=True,visualizeindividuals=False)

print("Crop images...")
#deeplabcut.cropimagesandlabels(path_config_file,userfeedback=False,numcrops=5)

#deeplabcut.convertannotationdata_fromwindows2unixstyle(path_config_file)#
#deeplabcut.cropimagesandlabels(path_config_file,userfeedback=False,numcrops=15)
#then we uncommented the large-scale full frame data > it is not used for training!

#if 'Plat' not in bp: #FOR
#    animalincrop=True

#cfg=deeplabcut.utils.auxiliaryfunctions.read_config(path_config_file)
#deeplabcut.utils.auxfun_multianimal.getpafgraph(cfg)

shuffle=2

print("Creating multianimal training set...")
deeplabcut.create_training_dataset(path_config_file,Shuffles=[shuffle],augmenter_type='imgaug',net_type='mobilenet_v2_0.5')

saveiters=5000
displayiters=100

trainposeconfigfile,testposeconfigfile,snapshotfolder=deeplabcut.return_train_network_path(path_config_file,shuffle=shuffle)
edits = {'batch_size': 16,
         'optimizer': 'adam',
         'grayscale': True,
         'rotation': 20,
         'covering': True,
         'hist_eq': True,
         'fliplr': False,
         'motion_blur': True,
         'multi_step': [[1e-4, 5000], [5*1e-5, 7500], [1e-5, 12000], [1e-6,50000]],
         'weigh_only_present_joints': False}
cfg_dlc= deeplabcut.auxiliaryfunctions.edit_config(testposeconfigfile, edits)

print("Creating multianimal training set...")
#deeplabcut.train_network(path_config_file, shuffle=shuffle,trainingsetindex=trainingsetindex,saveiters=saveiters,displayiters=displayiters)
deeplabcut.train_network(path_config_file, shuffle=shuffle,trainingsetindex=trainingsetindex,
    saveiters=saveiters,displayiters=displayiters,max_snapshots_to_keep=2,maxiters=50000)

#deeplabcut.evaluate_network(path_config_file, Shuffles=[shuffle],trainingsetindex=trainingsetindex, plotting=True)


print("Starting inference for", shuffle)

deeplabcut.analyze_videos(path_config_file,[videopath],shuffle=shuffle,videotype=videotype,save_as_csv=True)
#deeplabcut.plot_trajectories(path_config_file,[videopath],shuffle=shuffle,videotype=videotype)
#deeplabcut.convert_detections2tracklets(path_config_file,[videopath],shuffle=shuffle,videotype=videotype)
#deeplabcut.create_labeled_video(path_config_file,[videopath],shuffle=shuffle,videotype=videotype,save_frames=False)
