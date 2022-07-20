import os
import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import read_config
import re

def evaluate_all_shuffles(config_path, # config.yaml, common to all models
                        trainingsetindex=0,
                        gputouse=0,
                        modelprefix="",
                        train_iteration=0):

    ##########################################################
    ### Get config as dict and associated paths
    cfg = read_config(config_path)
    project_path = cfg["project_path"] # or: os.path.dirname(config_path) #dlc_models_path = os.path.join(project_path, "dlc-models")
    training_datasets_path = os.path.join(project_path, "training-datasets")

    #Get shuffles
    iteration_folder = os.path.join(training_datasets_path, 'iteration-' + str(train_iteration))
    dataset_top_folder = os.path.join(iteration_folder, os.listdir(iteration_folder)[0])
    files_in_dataset_top_folder = os.listdir(dataset_top_folder)
    shuffle_numbers = []
    for file in files_in_dataset_top_folder:
        if file.endswith(".mat"):
            shuffleNum = int(re.findall('[0-9]+',file)[-1])
            shuffle_numbers.append(shuffleNum)
    shuffle_numbers.sort()
    
    for sh in shuffle_numbers:
        deeplabcut.evaluate_network(config_path, # config.yaml, common to all models
                                Shuffles=[sh],
                                trainingsetindex=trainingsetindex,
                                gputouse=gputouse,
                                modelprefix=modelprefix,
                               )