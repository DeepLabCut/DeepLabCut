"""
This script generates a set of train config files (pose_config.yaml files) for a data augm study
- We consider 11 different data augmentation methods
- We train 12 models with different data augmentation settings. 
- The first model defines the baseline data augmentation settings.
- The next 11 models use the same data augmentation settings as the baseline, except for one data augmentation method
- This script also copies the test config file from the 'parent' project to all the sub-projects for each model

Contributors: Sofia, Jonas, Sabrina
"""

import os, shutil, sys
import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
from deeplabcut.generate_training_dataset.trainingsetmanipulation import create_training_dataset
import re
import argparse
import pdb

def create_parameters_dict():
    ##################################################################
    ### Define parameters for each data augmentation method --maybe read a separate file?
    # ATT! Parameters must be defined for True or False cases.
    # Not defining a set of parameters will result in applying the parameters from the pose_config.yaml template

    ## Initialise baseline dict with params per data augm type
    parameters_dict = dict() # define a class instead of a dict?

    ### General
    parameters_dict['general'] = {'dataset_type': 'imgaug', # OJO! not all the following will be available?
                                    'batch_size': 1, # 128
                                    'apply_prob': 0.5,
                                    'pre_resize': []} # Specify [width, height] if pre-resizing is desired

    ### Crop----is this applied if we select imgaug? I think so....
    parameters_dict['crop'] = {False: {'crop_by': 0.0,
                                    'cropratio': 0.0},
                            True: {'crop_by': 0.15,
                                    'cropratio': 0.4}}#---------- these are only used if height, width passed to pose_imgaug
    # from template:
    # parameters_dict['crop'] = {'crop_size':[400, 400],  # width, height,
    #                   'max_shift': 0.4,
    #                   'crop_sampling': 'hybrid',
    #                   'cropratio': 0.4}---------- crop ratio is used too

    ### Rotation
    parameters_dict['rotation'] = {False:{'rotation': 0,
                                        'rotratio': 0},
                                    True:{'rotation': 25,
                                        'rotratio': 0.4}}

    ### Scale
    parameters_dict['scale'] = {False:{'scale_jitter_lo': 1.0,
                                    'scale_jitter_up': 1.0},
                                True:{'scale_jitter_lo': 0.5,
                                    'scale_jitter_up': 1.25}}


    ### Motion blur
    # ATT motion_blur is not expected as a dictionary
    parameters_dict['motion_blur'] = {False: {'motion_blur': False}, # motion_blur_params should not be defined if False, but check if ok
                                    True: {'motion_blur': True,
                                            'motion_blur_params':{"k": 7, "angle": (-90, 90)}}}

    ### Contrast
    # ATT for Contrast a dict should be defined in the yaml file!
    # also: log, linear, sigmoid, gamma params...include those too? [I think if they are not defined in the template we are good, they wont be set]
    parameters_dict['contrast'] = {False: {'contrast': {'clahe': False,
                                                        'histeq': False}}, # ratios should not be defined if False, but check if ok
                                    True:{'contrast': {'clahe': True,
                                                        'claheratio': 0.1,
                                                        'histeq': True,
                                                        'histeqratio': 0.1}}}


    ### Convolution
    # ATT for Convolution a dict should be defined in the yaml file!
    parameters_dict['convolution'] = {False: {'convolution': {'sharpen': False,  # ratios should not be defined if False, but check if ok
                                                            'edge': False,
                                                            'emboss': False}}, # this needs to be fixed in pose_cfg.yaml template?
                                    True: {'convolution':{'sharpen': True,
                                                            'sharpenratio': 0.3, #---- in template: 0.3, in pose_imgaug default is 0.1
                                                            'edge': True,
                                                            'edgeratio': 0.1, #--------
                                                            'emboss': True,
                                                            'embossratio': 0.1}}}
    ### Mirror
    parameters_dict['mirror'] = {False: {'mirror': False},
                                True: {'mirror': True}}

    ### Grayscale
    parameters_dict['grayscale'] = {False: {'grayscale': False},
                                    True: {'grayscale': True}}

    ### Covering
    parameters_dict["covering"] = {False: {'covering': False},
                                True: {'covering': True}}

    ### Elastic transform
    parameters_dict["elastic_transform"] = {False: {'elastic_transform': False},
                                            True: {'elastic_transform': True}}

    ### Gaussian noise
    parameters_dict['gaussian_noise'] = {False: {'gaussian_noise': False},
                                        True: {'gaussian_noise': True}}

    return parameters_dict                                    


#############################################
if __name__ == "__main__":

    ##########################################################
    ### Set config path of project with labelled data
    # (we assume create_training_dataset has already been run)
    config_path = sys.argv[1] #'/media/data/stinkbugs-DLC-2022-07-15/config.yaml' # '/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/config.yaml'

    # each model subfolder is named with the format: <modelprefix_pre>_<id>_<str_id>
    modelprefix_pre = sys.argv[2] #"data_augm"

    # Other params
    TRAINING_SET_INDEX=0 # default;
    TRAIN_ITERATION=1 # iteration in terms of frames extraction; default is 0. can this be extracted?

    ##########################################################
    ### Get config as dict and associated paths
    cfg = read_config(config_path)
    project_path = cfg["project_path"] # or: os.path.dirname(config_path) #dlc_models_path = os.path.join(project_path, "dlc-models")
    training_datasets_path = os.path.join(project_path, "training-datasets")

    # Get shuffles
    iteration_folder = os.path.join(training_datasets_path, 'iteration-' + str(TRAIN_ITERATION))
    dataset_top_folder = os.path.join(iteration_folder, os.listdir(iteration_folder)[0])
    files_in_dataset_top_folder = os.listdir(dataset_top_folder)
    list_shuffle_numbers = []
    for file in files_in_dataset_top_folder:
        if file.endswith(".mat"):
            shuffleNum = int(re.findall('[0-9]+',file)[-1])
            list_shuffle_numbers.append(shuffleNum)
    list_shuffle_numbers.sort()

    # Get train and test pose config file paths from base project, for each shuffle
    list_base_train_pose_config_file_paths = []
    list_base_test_pose_config_file_paths = []
    for shuffle_number in list_shuffle_numbers:
        base_train_pose_config_file_path_TEMP,\
        base_test_pose_config_file_path_TEMP,\
        _ = deeplabcut.return_train_network_path(config_path,
                                                shuffle=shuffle_number,
                                                trainingsetindex=0)  # base_train_pose_config_file
        list_base_train_pose_config_file_paths.append(base_train_pose_config_file_path_TEMP)
        list_base_test_pose_config_file_paths.append(base_test_pose_config_file_path_TEMP)

    ###############################################################
    ## Create params dict
    parameters_dict = create_parameters_dict()

    ############################################################################
    ## Define baseline
    baseline = {'crop':             True, #----check
                'rotation':         True,
                'scale':            True,
                'mirror':           False,
                'contrast':         True,
                'motion_blur':      True,
                'convolution':      False,
                'grayscale':        False,
                'covering':         True,
                'elastic_transform': True,
                'gaussian_noise':   False}

    #################################################
    ## Create list of strings identifying each model
    list_of_data_augm_models_strs = ['baseline']
    for ky in baseline.keys() :
        list_of_data_augm_models_strs.append(ky) #'wo_' + ky)


    #########################################
    ## Loop to train each model
    for i, daug_str in enumerate(list_of_data_augm_models_strs):
        ###########################################################
        # Create subdirs for this augmentation method
        model_prefix = '_'.join([modelprefix_pre, "{0:0=2d}".format(i), daug_str]) # modelprefix_pre = aug_
        aug_project_path = os.path.join(project_path, model_prefix)
        aug_dlc_models = os.path.join(aug_project_path, "dlc-models", )
        aug_training_datasets = os.path.join(aug_project_path, "training-datasets")
        # create subdir for this model
        try:
            os.mkdir(aug_project_path)
        except OSError as error:
            print(error)
            print("Skipping this one as it already exists")
            continue
        # copy tree 'training-datasets' of dlc project under subdir for the current model---copies training_dataset subdir
        shutil.copytree(training_datasets_path, aug_training_datasets)

        ###########################################################
        # Copy base train pose config file to the directory of this augmentation method
        list_train_pose_config_path_per_shuffle = []
        list_test_pose_config_path_per_shuffle = []
        for j, sh in enumerate(list_shuffle_numbers):
            one_train_pose_config_file_path,\
            one_test_pose_config_file_path,\
            _ = deeplabcut.return_train_network_path(config_path,
                                                    shuffle=sh,
                                                    trainingsetindex=TRAINING_SET_INDEX, # default
                                                    modelprefix=model_prefix)

            # copy test and train config from base project to this subdir
            os.makedirs(str(os.path.dirname(one_train_pose_config_file_path))) # create parentdir 'train'
            os.makedirs(str(os.path.dirname(one_test_pose_config_file_path))) # create parentdir 'test'

            # copy base train config file
            shutil.copyfile(list_base_train_pose_config_file_paths[j],
                            one_train_pose_config_file_path) 

            # copy base test config file
            shutil.copyfile(list_base_test_pose_config_file_paths[j],
                            one_test_pose_config_file_path) 

            # add to list
            list_train_pose_config_path_per_shuffle.append(one_train_pose_config_file_path) 
            list_test_pose_config_path_per_shuffle.append(one_test_pose_config_file_path)

        #####################################################
        # Create dict with the data augm params for this model
        # initialise dict with gral params
        edits_dict = dict()
        edits_dict.update(parameters_dict['general'])
        for ky in baseline.keys():
            if daug_str == ky:
                # Get params that correspond to the opposite state of the method daug_str in the baseline
                d_temp = parameters_dict[ky][not baseline[ky]]
                # add to edits dict
                edits_dict.update(d_temp)
            else:
                # Get params that correspond to the same state as the baseline
                d_temp = parameters_dict[ky][baseline[ky]]
                # add to edits dict
                edits_dict.update(d_temp)

        # print
        print('-----------------------------------')
        if daug_str == 'baseline':
            print('Data augmentation model {}: {}'.format(i, daug_str))
        else:
            print('Data augmentation model {}: "{}" opposite to baseline'.format(i, daug_str))
        [print('{}: {}'.format(k,v)) for k,v in edits_dict.items()]
        print('-----------------------------------')

        ##################################################
        # Edit config for this data augmentation setting
        for j, sh in enumerate(list_shuffle_numbers):
            edit_config(str(list_train_pose_config_path_per_shuffle[j]), edits_dict)