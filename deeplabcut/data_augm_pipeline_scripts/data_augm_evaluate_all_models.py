import os
from evaluate_all_shuffles import evaluate_all_shuffles

config_path = "/media/data/stinkbugs-DLC-2022-07-15-SMALL/config.yaml"
dirs_in_project = os.listdir(str(os.path.dirname(config_path)))
modelprefixes = []

for directory in dirs_in_project:
    if directory.startswith("data_augm_"):
        modelprefixes.append(directory)
modelprefixes.sort()

for modelprefix in modelprefixes:
    evaluate_all_shuffles(config_path, # config.yaml, common to all models
                        trainingsetindex=0,
                        gputouse=3,
                        modelprefix=modelprefix,
                        train_iteration=1)