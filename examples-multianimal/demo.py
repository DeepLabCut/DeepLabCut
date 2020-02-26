#multimouse

import deeplabcut

task='Birds'
scorer='Alex'
video=['/media/alex/dropboxdisk/Dropbox/Collaborations/Cancer/DLCdev/examples-multianimal/MontBlanc-Daniel-2019-12-16/videos/montblanc.mov']

#print("CREATING PROJECT")
path_config_file=deeplabcut.create_new_project(task,scorer,video,copy_videos=True,multianimal=True)

cfg=deeplabcut.auxiliaryfunctions.read_config(path_config_file)
cfg['numframes2pick']=7
cfg['individuals']=['bird1','bird2','bird3','birdx','single']

cfg['uniquebodyparts']=['montblanc']
cfg['multianimalbodyparts']=['wing','tip']

deeplabcut.auxiliaryfunctions.write_config(path_config_file,cfg)

deeplabcut.extract_frames(path_config_file,mode='automatic',algo='uniform',userfeedback=False)

#print("Plot labels...")
#deeplabcut.check_labels(path_config_file)
#deeplabcut.label_frames(path_config_file,multianimal=True)

#config='/home/alex/Hacking/DLCdev/examples-multianimal/MontBlanc-Daniel-2019-12-16'
