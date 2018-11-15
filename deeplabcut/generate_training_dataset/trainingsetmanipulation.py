"""
DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import os.path
import matplotlib as mpl
if os.environ.get('DLClight', default=False) == 'True':
    mpl.use('AGG') #anti-grain geometry engine #https://matplotlib.org/faq/usage_faq.html
    pass
else:
    mpl.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import io
import yaml
from deeplabcut import DEBUG
from deeplabcut.utils import auxiliaryfunctions

#matplotlib.use('Agg')

def comparelists(config):
    """
    Auxiliary function, compares data sets in labeled-data & listed under video_sets. Try to make sure that they are the same!
    """
    cfg = auxiliaryfunctions.read_config(config)
    videos = cfg['video_sets'].keys()
    video_names = [Path(i).stem for i in videos]
    
    alldatafolders = [fn for fn in os.listdir(Path(config).parent / 'labeled-data') if '_labeled' not in fn]
    
    print("Config file contains:", len(video_names))
    print("Labeled-data contains:", len(alldatafolders))
    
    for vn in video_names:
        if vn in alldatafolders:
            pass
        else:
            print(vn, " is missing as a folder!")
    
    for vn in alldatafolders:
        if vn in video_names:
            pass
        else:
            print(vn, " is missing in config file!")

def dropduplicates(config):
    """
    Drop duplicates (of images) in annotation files. 
    """
    cfg = auxiliaryfunctions.read_config(config)
    videos = cfg['video_sets'].keys()
    video_names = [Path(i).stem for i in videos]
    folders = [Path(config).parent / 'labeled-data' /Path(i) for i in video_names]

    for folder in folders:
        try:
            fn=os.path.join(str(folder),'CollectedData_' + cfg['scorer'] + '.h5')
            DC = pd.read_hdf(fn, 'df_with_missing')
            numimages=len(DC.index)
            DC = DC[~DC.index.duplicated(keep='first')]
            if len(DC.index)<numimages:
                print("Dropped",numimages-len(DC.index))
                DC.to_hdf(fn, key='df_with_missing', mode='w')
                DC.to_csv(os.path.join(str(folder),'CollectedData_'+ cfg['scorer']+".csv"))
                
        except FileNotFoundError:
            print("Attention:", folder, "does not appear to have labeled data!")


def label_frames(config,scale =.9):
    """
    Manually label/annotate the extracted frames. Update the list of body parts you want to localize in the config.yaml file first

    Parameter
    ----------
    config : string
        Full path of the config.yaml file as a string.

    Example
    --------
    >>> deeplabcut.label_frames('/analysis/project/reaching-task/config.yaml')
    --------

    """
    startpath = os.getcwd()
    wd = Path(config).resolve().parents[0]
    os.chdir(str(wd))
    
    from deeplabcut.generate_training_dataset import labeling_toolbox
    
    labeling_toolbox.show(config) #,scale)
    os.chdir(startpath)

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def check_labels(config):
    """
    Double check if the labels were at correct locations and stored in a proper file format.\n
    This creates a new subdirectory for each video under the 'labeled-data' and all the frames are plotted with the labels.\n
    Make sure that these labels are fine.

    Parameter
    ----------
    config : string
        Full path of the config.yaml file as a string.

    Example
    --------
    for labeling the frames
    >>> deeplabcut.check_labels('/analysis/project/reaching-task/config.yaml')
    --------
    """
    cfg = auxiliaryfunctions.read_config(config)
    videos = cfg['video_sets'].keys()
    video_names = [Path(i).stem for i in videos]

   #plotting parameters:
    Labels = ['+','.','x']  # order of labels for different scorers. Currently using only the first (as by convention the human labeler is displayed as +).
    cc = 0 # label index / here only 0, for human labeler
    scale = 1
    Colorscheme = get_cmap(len( cfg['bodyparts']),cfg['colormap'])
    
    #folders = [Path(config).parent / 'labeled-data' /Path(i) for i in video_names]
    folders = [os.path.join(cfg['project_path'],'labeled-data',str(Path(i))) for i in video_names]
    print("Creating images with labels by %s." %cfg['scorer'])
    for folder in folders:
        try:
            DataCombined = pd.read_hdf(os.path.join(str(folder),'CollectedData_' + cfg['scorer'] + '.h5'), 'df_with_missing')
            MakeLabeledPlots(folder,DataCombined,cfg,Labels,Colorscheme,cc,scale)
        except FileNotFoundError:
            print("Attention:", folder, "does not appear to have labeled data!")

    print("If all the labels are ok, then use the function 'create_training_dataset' to create the training dataset!")

def MakeLabeledPlots(folder,DataCombined,cfg,Labels,Colorscheme,cc,scale):
    tmpfolder = str(folder) + '_labeled'
    auxiliaryfunctions.attempttomakefolder(tmpfolder)
    for index, imagename in enumerate(DataCombined.index.values):
        image = io.imread(os.path.join(cfg['project_path'],imagename))
        plt.axis('off')

        if np.ndim(image)==2:
            h, w = np.shape(image)
        else:
            h, w, nc = np.shape(image)

        plt.figure(
            frameon=False, figsize=(w * 1. / 100 * scale, h * 1. / 100 * scale))
        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        plt.imshow(image, 'gray')
        if index==0:
            print("They are stored in the following folder: %s." %tmpfolder) #folder)

        for c, bp in enumerate(cfg['bodyparts']):
            plt.plot(
                DataCombined[cfg['scorer']][bp]['x'].values[index],
                DataCombined[cfg['scorer']][bp]['y'].values[index],
                Labels[cc],
                color=Colorscheme(c),
                alpha=cfg['alphavalue'],
                ms=cfg['dotsize'])

        plt.xlim(0, w)
        plt.ylim(0, h)
        plt.axis('off')
        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.gca().invert_yaxis()

        plt.savefig(str(Path(tmpfolder)/imagename.split(os.sep)[-1])) #create file name
        plt.close("all")

def SplitTrials(trialindex, trainFraction=0.8):
    ''' Split a trial index into train and test sets. Also checks that the trainFraction is a two digit number between 0 an 1. The reason
    is that the folders contain the trainfraction as int(100*trainFraction). '''
    if trainFraction>1 or trainFraction<0:
        print("The training fraction should be a two digit number between 0 and 1; i.e. 0.95. Please change accordingly.")
        return ([],[])
    
    if abs(trainFraction-round(trainFraction,2))>0:
        print("The training fraction should be a two digit number between 0 and 1; i.e. 0.95. Please change accordingly.")
        return ([],[])
    else:
        trainsetsize = int(len(trialindex) * round(trainFraction,2))
        shuffle = np.random.permutation(trialindex)
        testIndexes = shuffle[trainsetsize:]
        trainIndexes = shuffle[:trainsetsize]
        return (trainIndexes, testIndexes)

def boxitintoacell(joints):
    ''' Auxiliary function for creating matfile.'''
    outer = np.array([[None]], dtype=object)
    outer[0, 0] = np.array(joints, dtype='int64')
    return outer

def MakeTrain_pose_yaml(itemstochange,saveasconfigfile,defaultconfigfile):
    raw = open(defaultconfigfile).read()
    docs = []
    for raw_doc in raw.split('\n---'):
        try:
            docs.append(yaml.load(raw_doc))
        except SyntaxError:
            docs.append(raw_doc)

    for key in itemstochange.keys():
        docs[0][key] = itemstochange[key]

    with open(saveasconfigfile, "w") as f:
        yaml.dump(docs[0], f)
    return docs[0]

def MakeTest_pose_yaml(dictionary, keys2save, saveasfile):
    dict_test = {}
    for key in keys2save:
        dict_test[key] = dictionary[key]

    dict_test['scoremap_dir'] = 'test'
    with open(saveasfile, "w") as f:
        yaml.dump(dict_test, f)

def merge_annotateddatasets(cfg,project_path,trainingsetfolder_full):
    """
    Merges all the h5 files for all labeled-datasets (from individual videos)
    """
    AnnotationData=None
    data_path = Path(os.path.join(project_path , 'labeled-data'))
    videos = cfg['video_sets'].keys()
    video_names = [Path(i).stem for i in videos]
    for i in video_names:
        try:
            data = pd.read_hdf((str(data_path / Path(i))+'/CollectedData_'+cfg['scorer']+'.h5'),'df_with_missing')
            if AnnotationData is None:
                AnnotationData=data
            else:
                AnnotationData=pd.concat([AnnotationData, data])
            
        except FileNotFoundError:
            print((str(data_path / Path(i))+'/CollectedData_'+cfg['scorer']+'.h5'), " not found (perhaps not annotated)")
    
    AnnotationData.to_hdf((str(trainingsetfolder_full)+'/'+'/CollectedData_'+cfg['scorer']+'.h5'), key='df_with_missing', mode='w')
    AnnotationData.to_csv(str(trainingsetfolder_full)+'/'+'/CollectedData_'+cfg['scorer']+'.csv') #human readable.
    return(AnnotationData)


def create_training_dataset(config,num_shuffles=1,Shuffles=None):
    """
    Creates a training dataset. Labels from all the extracted frames are merged into a single .h5 file.\n
    Only the videos included in the config file are used to create this dataset.\n
    [OPTIONAL]Use the function 'add_new_video' at any stage of the project to add more videos to the project.

    Parameter
    ----------
    config : string
        Full path of the config.yaml file as a string.

    num_shuffles : int, optional
        Number of shuffles of training dataset to create, i.e. [1,2,3] for num_shuffles=3. Default is set to 1.

    Shuffles: list of shuffles.
        Alternatively the user can also give a list of shuffles (integers!).
        
    Example
    --------
    >>> deeplabcut.create_training_dataset('/analysis/project/reaching-task/config.yaml',num_shuffles=1)
    Windows:
    >>> deeplabcut.create_training_dataset('C:\\Users\\Ulf\\looming-task\\config.yaml',Shuffles=[3,17,5])
    --------
    """
    from skimage import io
    import scipy.io as sio 
    import deeplabcut
    import subprocess

    # Loading metadata from config file:
    cfg = auxiliaryfunctions.read_config(config)
    scorer = cfg['scorer']
    project_path = cfg['project_path']
    # Create path for training sets & store data there 
    trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(cfg) #Path concatenation OS platform independent
    auxiliaryfunctions.attempttomakefolder(Path(os.path.join(project_path,str(trainingsetfolder))),recursive=True)
    Data = merge_annotateddatasets(cfg,project_path,Path(os.path.join(project_path,trainingsetfolder)))
    Data = Data[scorer] #extract labeled data

    #set model type. we will allow more in the future.
    if cfg['resnet']==50:
        net_type ='resnet_'+str(cfg['resnet'])
        resnet_path = str(Path(deeplabcut.__file__).parents[0] / 'pose_estimation_tensorflow/models/pretrained/resnet_v1_50.ckpt')
    elif cfg['resnet']==101:
        net_type ='resnet_'+str(cfg['resnet'])
        resnet_path = str(Path(deeplabcut.__file__).parents[0] / 'Pose_Estimation_Tensorflow/models/pretrained/resnet_v1_101.ckpt')
    else:
        print("Currently only ResNet 50 or 101 supported, please change 'resnet' entry in config.yaml!")
        num_shuffles=-1 #thus the loop below is empty...

    if not Path(resnet_path).is_file():
        """
        Downloads the ImageNet pretrained weights for ResNet. 
        """
        start = os.getcwd()
        os.chdir(str(Path(resnet_path).parents[0]))
        print("Downloading the pretrained model (ResNets)....")
        subprocess.call("download.sh", shell=True)
        os.chdir(start)
        
    if Shuffles==None:
        Shuffles=range(1,num_shuffles+1,1)
    else:
        Shuffles=[i for i in Shuffles if isinstance(i,int)]
    
    bodyparts = cfg['bodyparts']
    TrainingFraction = cfg['TrainingFraction']
    for shuffle in Shuffles: # Creating shuffles starting from 1
        for trainFraction in TrainingFraction:
            trainIndexes, testIndexes = SplitTrials(range(len(Data.index)), trainFraction)

            ####################################################
            # Generating data structure with labeled information & frame metadata (for deep cut)
            ####################################################

            # Make matlab train file!
            data = []
            for jj in trainIndexes:
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

                indexjoints=0
                joints=np.zeros((len(bodyparts),3))*np.nan
                for bpindex,bodypart in enumerate(bodyparts):
                    if Data[bodypart]['x'][jj]<np.shape(im)[1] and Data[bodypart]['y'][jj]<np.shape(im)[0]: #are labels in image?
                        	joints[indexjoints,0]=int(bpindex)
                        	joints[indexjoints,1]=Data[bodypart]['x'][jj]
                        	joints[indexjoints,2]=Data[bodypart]['y'][jj]
                        	indexjoints+=1

                joints = joints[np.where(
                    np.prod(np.isfinite(joints),
                            1))[0], :]  # drop NaN, i.e. lines for missing body parts

                assert (np.prod(np.array(joints[:, 2]) < np.shape(im)[0])
                        )  # y coordinate within image?
                assert (np.prod(np.array(joints[:, 1]) < np.shape(im)[1])
                        )  # x coordinate within image?

                H['joints'] = np.array(joints, dtype=int)
                if np.size(joints)>0: #exclude images without labels
                        data.append(H)

            if len(trainIndexes)>0:
                datafilename,metadatafilename=auxiliaryfunctions.GetDataandMetaDataFilenames(trainingsetfolder,trainFraction,shuffle,cfg)
                ################################################################################
                # Saving metadata (Pickle file)
                ################################################################################
                auxiliaryfunctions.SaveMetadata(os.path.join(project_path,metadatafilename),data, trainIndexes, testIndexes, trainFraction)
                ################################################################################
                # Saving data file (convert to training file for deeper cut (*.mat))
                ################################################################################
    
                DTYPE = [('image', 'O'), ('size', 'O'), ('joints', 'O')]
                MatlabData = np.array(
                    [(np.array([data[item]['image']], dtype='U'),
                      np.array([data[item]['size']]),
                      boxitintoacell(data[item]['joints']))
                     for item in range(len(data))],
                    dtype=DTYPE)
                    
                sio.savemat(os.path.join(project_path,datafilename), {'dataset': MatlabData})
                
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
                
                items2change = {
                    "dataset": datafilename,
                    "metadataset": metadatafilename,
                    "num_joints": len(bodyparts),
                    "all_joints": [[i] for i in range(len(bodyparts))],
                    "all_joints_names": bodyparts,
                    "init_weights": resnet_path,
                    "project_path": cfg['project_path'],
                    "net_type": net_type
                }
    
                defaultconfigfile = str(Path(deeplabcut.__file__).parents[0] / 'pose_cfg.yaml')
                
                trainingdata = MakeTrain_pose_yaml(items2change,path_train_config,defaultconfigfile)
                keys2save = [
                    "dataset", "num_joints", "all_joints", "all_joints_names",
                    "net_type", 'init_weights', 'global_scale', 'location_refinement',
                    'locref_stdev'
                ]
                MakeTest_pose_yaml(trainingdata, keys2save,path_test_config)
                print("The training dataset is successfully created. Use the function 'train_network' to start training. Happy training!")
            else:
                pass
