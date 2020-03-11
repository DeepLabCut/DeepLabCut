#testing conversions.py

import os, sys
os.environ['DLClight']='True'
import deeplabcut


basefolder='/media/alex/dropboxdisk/Dropbox/Collaborations/Cancer/DLCdev/examples-multianimal/MontBlanc-Daniel-2019-12-16'
#basefolder='/home/alex/Dropbox/Collaborations/Cancer/DLCdev/examples-multianimal/MontBlanc-Daniel-2019-12-16'
#basefolder='/media/alex/dropboxdisk/Dropbox/Collaborations/Cancer/DLCdev/examples-multianimal/MontBlanc-Daniel-2019-12-16'
videotype='mov'


video=[os.path.join(basefolder,'videos/montblanc.mov')]
path_config_file=os.path.join(basefolder,'config.yaml')
videopath=os.path.join(basefolder,'videos')

#deeplabcut.convertannotationdata_fromwindows2unixstyle(path_config_file)

deeplabcut.convertannotationdata_fromwindows2unixstyle(path_config_file,win2linux=False) #linux >> windows!
