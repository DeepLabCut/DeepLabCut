#multimouse

import deeplabcut

task='MontBlanc'
scorer='Daniel'
video=['/home/alex/Dropbox/InterestingCode/social_datasets/croppedNov18/montblanc.mov']

#print("CREATING PROJECT")
#path_config_file=deeplabcut.create_new_project(task,scorer,video,copy_videos=True,multianimal=True)

#project already created!

#path_config_file='/home/alex/Hacking/DLCreleases/DLCdev/examples-multianimal/MontBlanc-Daniel-2019-12-16/config.yaml'

path_config_file='/home/alex/Hacking/DLCdev/examples-multianimal/MontBlanc-Daniel-2019-12-16/config.yaml'

#print("Plot labels...")
#deeplabcut.check_labels(path_config_file)


#deeplabcut.cropimagesandlabels(config,userfeedback=False)

deeplabcut.label_frames(path_config_file)
