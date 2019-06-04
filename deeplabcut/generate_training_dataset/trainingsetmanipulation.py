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

import platform
if os.environ.get('DLClight', default=False) == 'True':
    mpl.use('AGG') #anti-grain geometry engine #https://matplotlib.org/faq/usage_faq.html
elif platform.system() == 'Darwin':
    mpl.use('WxAgg') #TkAgg
else:
    mpl.use('TkAgg')
import matplotlib.pyplot as plt



#if os.environ.get('DLClight', default=False) == 'True':
#    mpl.use('AGG') #anti-grain geometry engine #https://matplotlib.org/faq/usage_faq.html
#    pass
#else:
#    mpl.use('TkAgg')
#import matplotlib.pyplot as plt
from skimage import io

import yaml
from deeplabcut import DEBUG
from deeplabcut.utils import auxiliaryfunctions, conversioncode

#matplotlib.use('Agg')

def comparevideolistsanddatafolders(config):
    """
    Auxiliary function that compares the folders in labeled-data and the ones listed under video_sets (in the config file). 
    
    Parameter
    ----------
    config : string	
        String containing the full path of the config file in the project.
        
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


def adddatasetstovideolistandviceversa(config,prefix,width,height,suffix='.mp4'):
    """
    First run comparevideolistsanddatafolders(config) to compare the folders in labeled-data and the ones listed under video_sets (in the config file). 
    If you detect differences this function can be used to maker sure each folder has a video entry & vice versa.
    
    It corrects this problem in the following way:
    
    If a folder in labeled-data does not contain a video entry in the config file then the prefix path will be added in front of the name of the labeled-data folder and combined
    with the suffix variable as an ending. Width and height will be added as cropping variables as passed on. TODO: This should be written from the actual images!
    
    If a video entry in the config file does not contain a folder in labeled-data, then the entry is removed.
    
    Handle with care!
    
    Parameter
    ----------
    config : string	
        String containing the full path of the config file in the project.
        
    """
    cfg = auxiliaryfunctions.read_config(config)
    videos = cfg['video_sets'].keys()
    video_names = [Path(i).stem for i in videos]

    alldatafolders = [fn for fn in os.listdir(Path(config).parent / 'labeled-data') if '_labeled' not in fn]

    print("Config file contains:", len(video_names))
    print("Labeled-data contains:", len(alldatafolders))

    toberemoved=[]
    for vn in video_names:
        if vn in alldatafolders:
            pass
        else:
            print(vn, " is missing as a labeled folder >> removing key!")
            for fullvideo in cfg['video_sets'].keys():
                if vn in fullvideo:
                    toberemoved.append(fullvideo)

    for vid in toberemoved:
        del cfg['video_sets'][vid]

    #Load updated lists:
    videos = cfg['video_sets'].keys()
    video_names = [Path(i).stem for i in videos]

    for vn in alldatafolders:
        if vn in video_names:
            pass
        else:
            print(vn, " is missing in config file >> adding it!")
            #cfg['video_sets'][vn]
            cfg['video_sets'].update({os.path.join(prefix,vn+suffix) : {'crop': ', '.join(map(str, [0, width, 0, height]))}})

    auxiliaryfunctions.write_config(config,cfg)


def dropduplicatesinannotatinfiles(config):
    """
    
    Drop duplicate entries (of images) in annotation files (this should no longer happen, but might be useful).
    
    Parameter
    ----------
    config : string	
        String containing the full path of the config file in the project.
        
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

def dropannotationfileentriesduetodeletedimages(config):
    """
    Drop entries for all deleted images in annotation files, i.e. for folders of the type: /labeled-data/*folder*/CollectedData_*scorer*.h5
    Will be carried out iteratively for all *folders* in labeled-data.
    
    Parameter
    ----------
    config : string	
        String containing the full path of the config file in the project.
        
    """
    cfg = auxiliaryfunctions.read_config(config)
    videos = cfg['video_sets'].keys()
    video_names = [Path(i).stem for i in videos]
    folders = [Path(config).parent / 'labeled-data' /Path(i) for i in video_names]

    for folder in folders:
        fn=os.path.join(str(folder),'CollectedData_' + cfg['scorer'] + '.h5')
        DC = pd.read_hdf(fn, 'df_with_missing')
        dropped=False
        for imagename in DC.index:
            if os.path.isfile(os.path.join(cfg['project_path'],imagename)):
                pass
            else:
                print("Dropping...", imagename)
                DC = DC.drop(imagename)
                dropped=True
        if dropped==True:
            DC.to_hdf(fn, key='df_with_missing', mode='w')
            DC.to_csv(os.path.join(str(folder),'CollectedData_'+ cfg['scorer']+".csv"))


def label_frames(config):
    """
    Manually label/annotate the extracted frames. Update the list of body parts you want to localize in the config.yaml file first.

    Parameter
    ----------
    config : string	
        String containing the full path of the config file in the project.

    Example
    --------
    >>> deeplabcut.label_frames('/analysis/project/reaching-task/config.yaml')
    --------

    """
    startpath = os.getcwd()
    wd = Path(config).resolve().parents[0]
    os.chdir(str(wd))

    from deeplabcut.generate_training_dataset import labeling_toolbox

    # labeling_toolbox.show(config,Screens,scale_w,scale_h, winHack, img_scale)
    labeling_toolbox.show(config)
    os.chdir(startpath)

def get_cmap(n, name='jet'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def check_labels(config,Labels = ['+','.','x'],scale = 1):
    """
    Double check if the labels were at correct locations and stored in a proper file format.\n
    This creates a new subdirectory for each video under the 'labeled-data' and all the frames are plotted with the labels.\n
    Make sure that these labels are fine.

    Parameter
    ----------
    config : string
        Full path of the config.yaml file as a string.
        
    Labels: List of at least 3 matplotlib markers. The first one will be used to indicate the human ground truth location (Default: +)

    scale : float, default =1
        Change the relative size of the output images. 
    
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
    cc = 0 # label index / here only 0, for human labeler
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
            docs.append(yaml.load(raw_doc,Loader=yaml.SafeLoader))
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

def merge_annotateddatasets(cfg,project_path,trainingsetfolder_full,windows2linux):
    """
    Merges all the h5 files for all labeled-datasets (from individual videos).
    This is a bit of a mess because of cross platform compatablity. 
    
    Within platform comp. is straightforward. But if someone labels on windows and wants to train on a unix cluster or colab...
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

    if AnnotationData is None:
        print("Annotation data was not found by splitting video paths (from config['video_sets']). An alternative route is taken...")
        AnnotationData=conversioncode.merge_windowsannotationdataONlinuxsystem(cfg)
    if AnnotationData is None:
        print("No data was found!")
        windowspath=False
    else:
        windowspath=len((AnnotationData.index[0]).split('\\'))>1 #true if the first element is in windows path format
    
    # Let's check if the code is *not* run on windows (Source: #https://stackoverflow.com/questions/1325581/how-do-i-check-if-im-running-on-windows-in-python)
    # but the paths are in windows format...
    if os.name != 'nt' and windowspath and not windows2linux: 
        print("It appears that the images were labeled on a Windows system, but you are currently trying to create a training set on a Unix system. \n In this case the paths should be converted. Do you want to proceed with the conversion?")
        askuser = input("yes/no")
    else:
        askuser='no'
        
    filename=str(str(trainingsetfolder_full)+'/'+'/CollectedData_'+cfg['scorer'])
    if windows2linux or askuser=='yes' or askuser=='y' or askuser=='Ja': #convert windows path in pandas array \\ to unix / !
        AnnotationData=conversioncode.convertpaths_to_unixstyle(AnnotationData,filename,cfg)
        print("Annotation data converted to unix format...")
    else: #store as is
        AnnotationData.to_hdf(filename+'.h5', key='df_with_missing', mode='w')
        AnnotationData.to_csv(filename+'.csv') #human readable.
        
    return AnnotationData 

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

def mergeandsplit(config,trainindex=0,uniform=True,windows2linux=False):
    """
    This function allows additional control over "create_training_dataset". 
    
    Merge annotated data sets (from different folders) and split data in a specific way, returns the split variables (train/test indices). 
    Importantly, this allows one to freeze a split. 
    
    One can also either create a uniform split (uniform = True; thereby indexing TrainingFraction in config file) or leave-one-folder out split 
    by passing the index of the corrensponding video from the config.yaml file as variable trainindex.
    
    Parameter
    ----------
    config : string
        Full path of the config.yaml file as a string.

    trainindex: int, optional
        Either (in case uniform = True) indexes which element of TrainingFraction in the config file should be used (note it is a list!).
        Alternatively (uniform = False) indexes which folder is dropped, i.e. the first if trainindex=0, the second if trainindex =1, etc.

    uniform: bool, optional
        Perform uniform split (disregarding folder structure in labeled data), or (if False) leave one folder out.

    windows2linux: bool.
        The annotation files contain path formated according to your operating system. If you label on windows 
        but train & evaluate on a unix system (e.g. ubunt, colab, Mac) set this variable to True to convert the paths. 
    
    Examples
    --------
    To create a leave-one-folder-out model:
    >>> trainIndexes, testIndexes=deeplabcut.mergeandsplit(config,trainindex=0,uniform=False)
    returns the indices for the first video folder (as defined in config file) as testIndexes and all others as trainIndexes.
    You can then create the training set by calling (e.g. defining it as Shuffle 3):
    >>> deeplabcut.create_training_dataset(config,Shuffles=[3],trainIndexes=trainIndexes,testIndexes=testIndexes)
    
    To freeze a (uniform) split:
    >>> trainIndices, testIndices=deeplabcut.mergeandsplit(config,trainindex=0,uniform=True)
    You can then create two model instances that have the identical trainingset. Thereby you can assess the role of various parameters on the performance of DLC.
    
    >>> deeplabcut.create_training_dataset(config,Shuffles=[0],trainIndices=trainIndices,testIndices=testIndices)
    >>> deeplabcut.create_training_dataset(config,Shuffles=[1],trainIndices=trainIndices,testIndices=testIndices)
    --------
    
    """
    
    # Loading metadata from config file:
    cfg = auxiliaryfunctions.read_config(config)
    scorer = cfg['scorer']
    project_path = cfg['project_path']
    # Create path for training sets & store data there
    trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(cfg) #Path concatenation OS platform independent
    auxiliaryfunctions.attempttomakefolder(Path(os.path.join(project_path,str(trainingsetfolder))),recursive=True)
    fn=os.path.join(project_path,trainingsetfolder,'CollectedData_'+cfg['scorer'])
    
    try:
        Data= pd.read_hdf(fn+'.h5', 'df_with_missing')
    except FileNotFoundError:
        Data = merge_annotateddatasets(cfg,project_path,Path(os.path.join(project_path,trainingsetfolder)),windows2linux=windows2linux)
    
    Data = Data[scorer] #extract labeled data
    
    if uniform==True:
        TrainingFraction = cfg['TrainingFraction']
        trainFraction=TrainingFraction[trainindex]
        trainIndexes, testIndexes = SplitTrials(range(len(Data.index)), trainFraction)
    else: #leave one folder out split
        videos = cfg['video_sets'].keys()
        test_video_name = [Path(i).stem for i in videos][trainindex]
        print("Excluding the following folder (from training):", test_video_name)
        trainIndexes, testIndexes=[],[]
        for index,name in enumerate(Data.index):
            #print(index,name.split(os.sep)[1])
            if test_video_name==name.split(os.sep)[1]: #this is the video name
                #print(name,test_video_name)
                testIndexes.append(index)
            else:
                trainIndexes.append(index)
                
    return trainIndexes, testIndexes


def create_training_dataset(config,num_shuffles=1,Shuffles=None,windows2linux=False,trainIndices=None,testIndices=None):
    """
    Creates a training dataset. Labels from all the extracted frames are merged into a single .h5 file.\n
    Only the videos included in the config file are used to create this dataset.\n
    
    [OPTIONAL] Use the function 'add_new_video' at any stage of the project to add more videos to the project.

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
    
    trainIndices and testIndices: list of indices for traininng and testing. Use mergeandsplit(config,trainindex=0,uniform=True,windows2linux=False) to create them
    See help for deeplabcut.mergeandsplit?
    
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
    
    Data = merge_annotateddatasets(cfg,project_path,Path(os.path.join(project_path,trainingsetfolder)),windows2linux)
    Data = Data[scorer] #extract labeled data

    #set model type. we will allow more in the future.
    if cfg['resnet']==50:
        net_type ='resnet_'+str(cfg['resnet'])
        resnet_path = str(Path(deeplabcut.__file__).parents[0] / 'pose_estimation_tensorflow/models/pretrained/resnet_v1_50.ckpt')
    elif cfg['resnet']==101:
        net_type ='resnet_'+str(cfg['resnet'])
        resnet_path = str(Path(deeplabcut.__file__).parents[0] / 'pose_estimation_tensorflow/models/pretrained/resnet_v1_101.ckpt')
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
            #trainIndexes, testIndexes = SplitTrials(range(len(Data.index)), trainFraction)
            if trainIndices is None and testIndices is None:
                trainIndexes, testIndexes = SplitTrials(range(len(Data.index)), trainFraction)
            else: # set to passed values...
                trainIndexes=trainIndices
                testIndexes=testIndices
                print("You passed a split with the following fraction:", len(trainIndexes)*1./(len(testIndexes)+len(trainIndexes))*100)
            
            ####################################################
            # Generating data structure with labeled information & frame metadata (for deep cut)
            ####################################################

            # Make training file!
            data = []
            for jj in trainIndexes:
                H = {}
                # load image to get dimensions:
                filename = Data.index[jj]
                im = io.imread(os.path.join(cfg['project_path'],filename))
                H['image'] = filename

                if np.ndim(im)==3:
                    H['size'] = np.array(
                        [np.shape(im)[2],
                         np.shape(im)[0],
                         np.shape(im)[1]])
                else:
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
                    "all_joints_names": [str(bpt) for bpt in bodyparts],
                    "init_weights": resnet_path,
                    "project_path": str(cfg['project_path']),
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
