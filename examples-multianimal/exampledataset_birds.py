#multimouse

import deeplabcut

task='MontBlanc'
scorer='Daniel'
video=['/home/alex/Dropbox/InterestingCode/social_datasets/croppedNov18/montblanc.mov']

#print("CREATING PROJECT")
#path_config_file=deeplabcut.create_new_project(task,scorer,video,copy_videos=True,multianimal=True)

#project already created!

#path_config_file='/home/alex/Hacking/DLCreleases/DLCdev/examples-multianimal/MontBlanc-Daniel-2019-12-16/config.yaml'

path_config_file='/media/alex/dropboxdisk/Dropbox/Collaborations/Cancer/DLCdev/examples-multianimal/MontBlanc-Daniel-2019-12-16/config.yaml'

print("Plot labels...")
#deeplabcut.check_labels(path_config_file)
deeplabcut.check_labels(path_config_file,draw_skeleton=True,visualizeindividuals=True)
deeplabcut.check_labels(path_config_file,draw_skeleton=True,visualizeindividuals=False)

print("Crop images...")
#deeplabcut.cropimagesandlabels(path_config_file,userfeedback=False)
#then we uncommented the large-scale full frame data > it is not used for training!

print("Creating multianimal training set...")
deeplabcut.create_multianimaltraining_dataset(path_config_file)

#cfg=deeplabcut.utils.auxiliaryfunctions.read_config(path_config_file)
#deeplabcut.utils.auxfun_multianimal.getpafgraph(cfg)

shuffle=1
trainingsetindex=0
Maxiter=11
saveiters=20
displayiters=1

print("Creating multianimal training set...")
deeplabcut.train_network(path_config_file, shuffle=shuffle,trainingsetindex=trainingsetindex,maxiters=Maxiter,saveiters=saveiters,displayiters=1)
