import deeplabcut
import os


projectpath='/media/alex/dropboxdisk/Dropbox/InterestingCode/social_datasets/croppedNov18/MultiMouse-Daniel-2019-12-16'

projectpath = '/media/alex/dropboxdisk/Dropbox/InterestingCode/social_datasets/MultiMouse-Daniel-2019-12-16'
modelprefix = 'simplebaseline'

configfile=os.path.join(projectpath,'config.yaml')

trainingsetindex=0
shuffle=0
deeplabcut.evaluate_multianimal_crossvalidate(configfile,[0], trainingsetindex, modelprefix)
