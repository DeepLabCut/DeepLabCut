"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import os.path
import matplotlib as mpl
if os.environ.get('DLClight', default=False) == 'True':
    mpl.use('AGG') #anti-grain geometry engine #https://matplotlib.org/faq/usage_faq.html
    pass
else:
    mpl.use('TkAgg')

import matplotlib.pyplot as plt
import yaml
from deeplabcut import DEBUG
from deeplabcut.utils import auxiliaryfunctions, auxfun_models, auxfun_multianimal
from deeplabcut.generate_training_dataset import trainingsetmanipulation

def renamebodyparts(config,pairs):
    """
    Rename bodyparts for list of pairs, e.g. [['snout','nose']] rename snout to nose!

    TODO: FIX THIS!
    """
    cfg = auxiliaryfunctions.read_config(config)
    videos = cfg['video_sets'].keys()
    video_names = [Path(i).stem for i in videos]
    folders = [Path(config).parent / 'labeled-data' /Path(i) for i in video_names]

    for folder in folders:
        fn=os.path.join(str(folder),'CollectedData_' + cfg['scorer'] + '.h5')
        df = pd.read_hdf(fn, 'df_with_missing')
        bpts=list(df.columns.get_level_values(1)) #[::2])
        print(bpts,'pre')
        # now swap:
        for p in pairs:
            print('Changing',p)
            bpts=[p[1] if bpt==p[0] else bpt for bpt in bpts]

        d = dict(zip(df.columns.levels[1], bpts))
        print(bpts,"swapped?")
        #df.columns.set_levels(bpts,level=1,inplace=True)
        df.rename(columns=d,level=1)
        #DC.to_hdf(fn, key='df_with_missing', mode='w')
        df.to_csv(os.path.join(str(folder),'CollectedData_'+ cfg['scorer']+"TEST.csv"))

def create_multianimaltraining_dataset(config,num_shuffles=1,Shuffles=None,windows2linux=False, net_type=None, numdigits=2):
    """
    Creates a training dataset for multi-animal datasets. Labels from all the extracted frames are merged into a single .h5 file.\n
    Only the videos included in the config file are used to create this dataset.\n
    [OPTIONAL] Use the function 'add_new_video' at any stage of the project to add more videos to the project.

    Imporant differences to standard:
     - stores coordinates with numdigits as many digits
     - creates
    Parameter
    ----------
    config : string
        Full path of the config.yaml file as a string.

    num_shuffles : int, optional
        Number of shuffles of training dataset to create, i.e. [1,2,3] for num_shuffles=3. Default is set to 1.

    Shuffles: list of shuffles.
        Alternatively the user can also give a list of shuffles (integers!).

    windows2linux: bool.
        The annotation files contain path formated according to your operating system. If you label on windows
        but train & evaluate on a unix system (e.g. ubunt, colab, Mac) set this variable to True to convert the paths.

    net_type: string
        Type of networks. Currently resnet_50, resnet_101, and resnet_152 are supported (not the MobileNets!)

    numdigits: int, optional


    Example
    --------
    >>> deeplabcut.create_multianimaltraining_dataset('/analysis/project/reaching-task/config.yaml',num_shuffles=1)

    Windows:
    >>> deeplabcut.create_multianimaltraining_dataset(r'C:\\Users\\Ulf\\looming-task\\config.yaml',Shuffles=[3,17,5])
    --------
    """
    from skimage import io

    # Loading metadata from config file:
    cfg = auxiliaryfunctions.read_config(config)
    scorer = cfg['scorer']
    project_path = cfg['project_path']
    # Create path for training sets & store data there
    trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(cfg) #Path concatenatn OS platform independent
    auxiliaryfunctions.attempttomakefolder(Path(os.path.join(project_path,str(trainingsetfolder))),recursive=True)

    Data = trainingsetmanipulation.merge_annotateddatasets(cfg,project_path,Path(os.path.join(project_path,trainingsetfolder)),windows2linux)
    Data = Data[scorer] #extract labeled data
    #actualbpts=set(Data.columns.get_level_values(0))

    #loading & linking pretrained models
    # CURRENTLY ONLY ResNet supported!
    if net_type is None: #loading & linking pretrained models
        net_type =cfg.get('default_net_type', 'resnet_50')
    else:
        if 'resnet' in net_type: # or 'mobilenet' in net_type:
            pass
        else:
            raise ValueError('Currently only resnet is supported.')

    #multianimal case:
    dataset_type='multi-animal-imgaug'
    partaffinityfield_graph=auxfun_multianimal.getpafgraph(cfg)
    #ATTENTION: order has to be multibodyparts, then uniquebodyparts (for indexing)
    print("Utilizing the following graph:", partaffinityfield_graph)
    num_limbs=len(partaffinityfield_graph)
    partaffinityfield_predict=True

    import deeplabcut
    parent_path = Path(os.path.dirname(deeplabcut.__file__))
    defaultconfigfile = str(parent_path / 'pose_cfg.yaml')
    model_path,num_shuffles=auxfun_models.Check4weights(net_type,parent_path,num_shuffles)

    if Shuffles==None:
        Shuffles=range(1,num_shuffles+1,1)
    else:
        Shuffles=[i for i in Shuffles if isinstance(i,int)]

    individuals,uniquebodyparts,multianimalbodyparts=auxfun_multianimal.extractindividualsandbodyparts(cfg)

    TrainingFraction = cfg['TrainingFraction']
    for shuffle in Shuffles: # Creating shuffles starting from 1
        for trainFraction in TrainingFraction:
            trainIndexes, testIndexes = trainingsetmanipulation.SplitTrials(range(len(Data.index)), trainFraction)

            ####################################################
            # Generating data structure with labeled information & frame metadata (for deep cut)
            ####################################################

            # Make training file!
            data = []
            print("Creating training data for ", shuffle, trainFraction)
            print("This can take some time...")
            for jj in tqdm(trainIndexes):
                jointsannotated=False
                H = {}
                # load image to get dimensions:
                filename = Data.index[jj]
                im = io.imread(os.path.join(cfg['project_path'],filename))
                H['image'] = filename

                try:
                    H['size'] = np.array(
                        [np.shape(im)[2],
                         np.shape(im)[0],
                         np.shape(im)[1]])
                except:
                    # print "Grayscale!"
                    H['size'] = np.array([1, np.shape(im)[0], np.shape(im)[1]])

                Joints={}
                for prfxindex,prefix in enumerate(individuals):
                    joints=np.zeros((len(uniquebodyparts)+len(multianimalbodyparts),3))*np.nan
                    if prefix!='single': #first ones are multianimalparts!
                        indexjoints=0
                        for bpindex,bodypart in enumerate(multianimalbodyparts):
                                socialbdpt=bodypart #prefix+bodypart #build names!
                                #if socialbdpt in actualbpts:
                                try:
                                    x,y=Data[prefix][socialbdpt]['x'][jj],Data[prefix][socialbdpt]['y'][jj]
                                    if 0<x and x<np.shape(im)[1] and 0<y and y<np.shape(im)[0]: #are labels in image?
                                            joints[indexjoints,0]=int(bpindex)
                                            joints[indexjoints,1]=round(x,numdigits)
                                            joints[indexjoints,2]=round(y,numdigits)
                                            indexjoints+=1
                                except:
                                    pass
                    else:
                        indexjoints=len(multianimalbodyparts)
                        for bpindex,bodypart in enumerate(uniquebodyparts):
                                socialbdpt=bodypart #prefix+bodypart #build names!
                                #if socialbdpt in actualbpts:
                                try:
                                    x,y=Data[prefix][socialbdpt]['x'][jj],Data[prefix][socialbdpt]['y'][jj]
                                    if 0<x and x<np.shape(im)[1] and 0<y and y<np.shape(im)[0]: #are labels in image?
                                            joints[indexjoints,0]=len(multianimalbodyparts)+int(bpindex)
                                            joints[indexjoints,1]=round(x,2)
                                            joints[indexjoints,2]=round(y,2)
                                            indexjoints+=1
                                except:
                                    pass
                    joints = joints[np.where(
                        np.prod(np.isfinite(joints),
                                1))[0], :]  # drop NaN, i.e. lines for missing body parts

                    #print("TEST:", filename,joints)
                    assert (np.prod(np.array(joints[:, 2]) < np.shape(im)[0])
                            )  # y coordinate within image?
                    assert (np.prod(np.array(joints[:, 1]) < np.shape(im)[1])
                            )  # x coordinate within image?

                    if np.size(joints)>0: #exclude images without labels
                        jointsannotated=True

                    Joints[prfxindex]=joints #np.array(joints, dtype=int)

                H['joints'] = Joints
                if jointsannotated: #exclude images without labels
                        data.append(H)

            if len(trainIndexes)>0:
                datafilename,metadatafilename=auxiliaryfunctions.GetDataandMetaDataFilenames(trainingsetfolder,trainFraction,shuffle,cfg)
                ################################################################################
                # Saving metadata and data file (Pickle file)
                ################################################################################
                auxiliaryfunctions.SaveMetadata(os.path.join(project_path,metadatafilename),data, trainIndexes, testIndexes, trainFraction)

                datafilename=datafilename.split('.mat')[0]+'.pickle'
                import pickle
                with open(os.path.join(project_path,datafilename), 'wb') as f:
                    # Pickle the 'labeled-data' dictionary using the highest protocol available.
                    pickle.dump(data, f,pickle.HIGHEST_PROTOCOL)

                ################################################################################
                # Creating file structure for training &
                # Test files as well as pose_yaml files (containing training and testing information)
                #################################################################################

                modelfoldername=auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,cfg)
                auxiliaryfunctions.attempttomakefolder(Path(config).parents[0] / modelfoldername,recursive=True)
                auxiliaryfunctions.attempttomakefolder(str(Path(config).parents[0] / modelfoldername)+ '/'+ '/train')
                auxiliaryfunctions.attempttomakefolder(str(Path(config).parents[0] / modelfoldername)+ '/'+ '/test')

                path_train_config = str(os.path.join(cfg['project_path'],Path(modelfoldername),'train','pose_cfg.yaml'))
                path_test_config = str(os.path.join(cfg['project_path'],Path(modelfoldername),'test','pose_cfg.yaml'))
                #str(cfg['proj_path']+'/'+Path(modelfoldername) / 'test'  /  'pose_cfg.yaml')

                jointnames=[str(bpt) for bpt in multianimalbodyparts]
                jointnames.extend([str(bpt) for bpt in uniquebodyparts])
                items2change = {
                    "dataset": datafilename,
                    "metadataset": metadatafilename,
                    "num_joints": len(multianimalbodyparts)+len(uniquebodyparts), #cfg["uniquebodyparts"]),
                    "all_joints": [[i] for i in range(len(multianimalbodyparts)+len(uniquebodyparts))], #cfg["uniquebodyparts"]))],
                    "all_joints_names": jointnames,
                    "init_weights": model_path,
                    "project_path": str(cfg['project_path']),
                    "net_type": net_type,
                    "pairwise_loss_weight": .1,
                    "pafwidth": 20,
                    "partaffinityfield_graph": partaffinityfield_graph,
                    "partaffinityfield_predict": partaffinityfield_predict,
                    "weigh_only_present_joints": False,
                    "num_limbs": len(partaffinityfield_graph),
                    "dataset_type": dataset_type,
                    "optimzer": "adam",
                    "batch_size": 16,
                    "multi_step": [[1e-4, 7500], [5*1e-5, 12000], [1e-5, 200000]],
                    "save_iters": 10000,
                    "display_iters": 500
                }
                defaultconfigfile = str(Path(deeplabcut.__file__).parents[0] / 'pose_cfg.yaml')

                trainingdata = trainingsetmanipulation.MakeTrain_pose_yaml(items2change,path_train_config,defaultconfigfile)
                keys2save = [
                    "dataset", "num_joints", "all_joints", "all_joints_names",
                    "net_type", 'init_weights', 'global_scale', 'location_refinement',
                    'locref_stdev', 'dataset_type',
                    'partaffinityfield_predict', 'pairwise_predict','partaffinityfield_graph',
                    'num_limbs', 'dataset_type'
                ]

                trainingsetmanipulation.MakeTest_pose_yaml(trainingdata, keys2save,path_test_config)
                print("The training dataset is successfully created. Use the function 'train_network' to start training. Happy training!")
            else:
                pass
