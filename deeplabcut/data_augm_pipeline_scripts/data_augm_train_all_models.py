"""
Script for training all shuffles of a number of models in a specific GPU
We pass as command line inputs
- config.yaml path [required]
- string prefix identifying the subdirectories of the models we want to train, in the parent folder of the config.yaml file (e.g.: data_augm_)
- gpu to use [required]
- indices of the models to train from the subset that start with the input prefix, in alphabetical order [required]
For a detailed description of inputs run 'python data_augm_train_all_models.py --help'

Example usage:
1- To train the first three models in the sorted list of subdirs that start with data_augm_*, in gpu=3, run: 
    python data_augm_train_all_models.py <path_to_config.yaml> 'data_augm_' 3 -f=0 -l=2
2- To train the first two models in the sorted list of subdirs that start with data_augm_*, in gpu=0, with specific initialising weights, and train_iteration =0:
    python data_augm_train_all_models.py /Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/config.yaml 
    'data_augm_' 0 -f=0 -l=1 
    -w /Users/user/Desktop/CaseStudyScripts/initial_weights.yaml 
    --train_iteration=0

python data_augm_train_all_models.py /Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/config.yaml 'data_augm_' 0 -f 0 -l 3 -w /Users/user/Desktop/CaseStudyScripts/initial_weights.yaml

Contributors: Jonas, Sofia
"""

import os
import sys
import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
import re 
import argparse
import yaml
import pdb

###########################################
def train_all_shuffles(config_path, # config.yaml, common to all models
                        trainingsetindex=0,
                        max_snapshots_to_keep=10,
                        displayiters=1000,
                        maxiters=500000,
                        saveiters=100000,
                        gputouse=0,
                        modelprefix="",
                        train_iteration=0,
                        dict_init_weights_per_modelprefix_and_shuffle={},
                        dict_optimizer={}):
    """
    Train all shuffles for a given model

    """
    
    ##########################################################
    ### Get config as dict and associated paths
    cfg = read_config(config_path)
    project_path = cfg["project_path"] # or: os.path.dirname(config_path) #dlc_models_path = os.path.join(project_path, "dlc-models")
    training_datasets_path = os.path.join(project_path, "training-datasets")


    ##############################################################
    ### Get list of shuffles for this model
    iteration_folder = os.path.join(training_datasets_path, 'iteration-' + str(train_iteration))
    dataset_top_folder = os.path.join(iteration_folder, os.listdir(iteration_folder)[0])
    files_in_dataset_top_folder = os.listdir(dataset_top_folder)
    shuffle_numbers = []
    for file in files_in_dataset_top_folder:
        if file.endswith(".mat"):
            shuffleNum = int(re.findall('[0-9]+',file)[-1])
            shuffle_numbers.append(shuffleNum)
    shuffle_numbers.sort()

    ##########################################################
    ### Train every shuffle for this model
    for sh in shuffle_numbers:
        ## If specific initial weights are provided: edit pose_cfg for this shuffle
        if bool(dict_init_weights_per_modelprefix_and_shuffle): # empty dict will evaluate to false
            try:
                # if there is a snapshot defined for this modelprefix and shuffle in the dict, take it
                snapshot_path = dict_init_weights_per_modelprefix_and_shuffle[modelprefix][sh]

                # get path to train config for this shuffle
                one_train_pose_config_file_path,\
                _,_ = deeplabcut.return_train_network_path(config_path,
                                                            shuffle=sh,
                                                            trainingsetindex=trainingsetindex, 
                                                            modelprefix=modelprefix)
                # edit config
                edit_config(str(one_train_pose_config_file_path), 
                            {'init_weights': snapshot_path})

                # print
                print('Initialising weights for model {} - shuffle {}, with snapshot at {}'.format(modelprefix,sh,snapshot_path))
            except KeyError:
                pass

        ## Change optimizer, batch size and learning rate if a dict is passed
        if bool(dict_optimizer):
            pdb.set_trace()
            edit_config(str(one_train_pose_config_file_path), 
                            {'optimizer': dict_optimizer['optimizer'], #'adam',
                            'batch_size': dict_optimizer['batch_size'], #16,
                            'multi_step': dict_optimizer['multi_step']}) # learning rate schedule for adam: [[1e-4, 7500], [5 * 1e-5, 12000], [1e-5, 200000]]

        ## Train this shuffle
        deeplabcut.train_network(config_path, # config.yaml, common to all models
                                 shuffle=sh,
                                 trainingsetindex=trainingsetindex,
                                 max_snapshots_to_keep=max_snapshots_to_keep,
                                 displayiters=displayiters,
                                 maxiters=maxiters,
                                 saveiters=saveiters,
                                 gputouse=gputouse,
                                 allow_growth=True,
                                 modelprefix=modelprefix)


#############################################
if __name__ == "__main__":

    ##############################################################
    # ## Get command line input parameters
    # if an optional argument isn’t specified, it gets the None value (and None fails the truth test in an if statement)
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument("config_path", 
                        type=str,
                        help="path to config.yaml file [required]")
    parser.add_argument("subdir_prefix_str", 
                        type=str,
                        help="prefix common to all subdirectories to train [required]")
    parser.add_argument("gpu_to_use", 
                        type=int,
                        help="id of gpu to use (as given by nvidia-smi) [required]")
    # optional
    parser.add_argument("-f", "--first_model_index", 
                        type=int,
                        help="index of the first model to train, in a sorted list of the subset of subdirectories. \
                        If none provided, all models in matching subdirectories are trained [optional]")
    parser.add_argument("-l", "--last_model_index", 
                        type=int,
                        help="index of the last model to train, in a sorted list of the subset of subdirectories. \
                        If none provided, all models in matching subdirectories are trained [optional]")
    parser.add_argument("-w", "--initial_weights_yaml_file", 
                        type=str,
                        default='',
                        help="path to file that shows for each model and shuffle the paths to the snapshots to show as initial weights [optional]")
    # dict_optimizer
    parser.add_argument("-o", "--optimizer_yaml_file", 
                        type=str,
                        default='',
                        help="path to file that sets the optimizer parameters. If none is passed, Adam parameters are used \
                              (optimizer=Adam, batch_size=16, multi_step= [[1e-4, 7500], [5 * 1e-5, 12000], [1e-5, 200000]])\
                              [optional]")

    # Other  training params [otpional]
    parser.add_argument("--training_set_index", 
                        type=int,
                        default=0,
                        help="Integer specifying which TrainingsetFraction to use. Note that TrainingFraction is a list in config.yaml.[optional]")
    parser.add_argument("--max_snapshots", 
                        type=int,
                        default=10,
                        help="maximum number of snapshots to keep [optional]")
    parser.add_argument("--display_iters", 
                        type=int,
                        default=10000,
                        help="display average loss every N iterations [optional]")
    parser.add_argument("--max_iters", 
                        type=int,
                        default=300000,
                        help="maximum number of training iterations [optional]")
    parser.add_argument("--save_iters", 
                        type=int,
                        default=50000,
                        help="save snapshots every N iterations [optional]")
    parser.add_argument("--train_iteration", 
                        type=int,
                        default=0, # default is 0, but in stinkbug is 1. can this be extracted?
                        help="iteration number in terms of frames extraction and retraining workflow [optional]")
    args = parser.parse_args()

    ##############################################################
    ### Extract required input params
    config_path = args.config_path #str(sys.argv[1]) #"/media/data/stinkbugs-DLC-2022-07-15-SMALL/config.yaml"
    subdir_prefix_str = args.subdir_prefix_str #str(sys.argv[2]) # "data_augm_"
    gpu_to_use = args.gpu_to_use #int(sys.argv[5])

    ### Extract optional input params
    TRAINING_SET_INDEX = args.training_set_index # default;
    MAX_SNAPSHOTS = args.max_snapshots
    DISPLAY_ITERS = args.display_iters # display loss every N iters; one iter processes one batch
    MAX_ITERS = args.max_iters
    SAVE_ITERS = args.save_iters # save snapshots every n iters
    TRAIN_ITERATION = args.train_iteration # iteration in terms of frames extraction; default is 0, but in stinkbug is 1. can this be extracted?


    ### Get dict with initial weights per model and shuffle [optional]
    # if yaml file passed, read dict
    if bool(args.initial_weights_yaml_file): 
        with open(args.initial_weights_yaml_file,'r') as yaml_file:
            dict_ini_weights_per_model_and_shuffle = yaml.safe_load(yaml_file)
    else:
        dict_ini_weights_per_model_and_shuffle = {} # if no yaml file passed, initialise as an empty dict

    ### Get dict with optimizer parameters. If none provided, Adam is used [optional]
    # if yaml file passed, read dict
    pdb.set_trace()
    if bool(args.optimizer_yaml_file): 
        with open(args.optimizer_yaml_file,'r') as yaml_file:
            dict_optimizer = yaml.safe_load(yaml_file)
    else:
        dict_optimizer = {'optimizer':'adam',
                          'batch_size': 16,
                          'multi_step': [[1e-4, 7500], [5 * 1e-5, 12000], [1e-5, 200000]]} # if no yaml file passed, initialise as an empty dict


    #######################################################################################
    ## Compute list of subdirectories that start with 'subdir_prefix_str'
    list_all_dirs_in_project = os.listdir(str(os.path.dirname(config_path)))
    list_models_subdir = []
    for directory in list_all_dirs_in_project:
        if directory.startswith(subdir_prefix_str):
            list_models_subdir.append(directory)
    list_models_subdir.sort() # sorts in place


    ## Select range of subdirs to train (optional input args)
    if args.first_model_index:
        first_model_index = args.first_model_index 
    else:
        first_model_index = 0
    if args.last_model_index: 
        last_model_index = args.last_model_index 
    else:
        last_model_index = len(list_models_subdir)-1

    #######################################################################
    ## Set 'allow growth' before training (allow growth bug)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    ###################################################################
    ## Train models in list 
    for modelprefix in list_models_subdir[first_model_index:last_model_index+1]:

        # train all shuffles for each model
        train_all_shuffles(config_path, # config.yaml, common to all models
                            trainingsetindex=TRAINING_SET_INDEX,
                            max_snapshots_to_keep=MAX_SNAPSHOTS,
                            displayiters=DISPLAY_ITERS,
                            maxiters=MAX_ITERS,
                            saveiters=SAVE_ITERS,
                            gputouse=gpu_to_use,
                            modelprefix=modelprefix,
                            train_iteration=TRAIN_ITERATION,
                            dict_init_weights_per_modelprefix_and_shuffle=dict_ini_weights_per_model_and_shuffle,
                            dict_optimizer=dict_optimizer)
