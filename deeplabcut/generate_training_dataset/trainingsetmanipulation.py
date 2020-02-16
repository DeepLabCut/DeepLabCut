"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

from pathlib import Path
import os,sys
import numpy as np
import pandas as pd
import os.path
import matplotlib as mpl
import logging
import platform
from functools import lru_cache


if os.environ.get('DLClight', default=False) == 'True':
    mpl.use('AGG') #anti-grain geometry engine #https://matplotlib.org/faq/usage_faq.html
elif platform.system() == 'Darwin':
    mpl.use('WxAgg') #TkAgg
else:
    mpl.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import io

import yaml
from deeplabcut import DEBUG
from deeplabcut.utils import auxiliaryfunctions, conversioncode, auxfun_models
from deeplabcut.pose_estimation_tensorflow import training

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

def dropimagesduetolackofannotation(config):
    """
    Drop images from corresponding folder for not annotated images: /labeled-data/*folder*/CollectedData_*scorer*.h5
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
        annotatedimages=[fn.split(os.sep)[-1] for fn in DC.index]
        imagelist=[fns for fns in os.listdir(str(folder)) if '.png' in fns]
        print("Annotated images: ", len(annotatedimages)," In folder:", len(imagelist))
        for imagename in imagelist:
            if imagename in annotatedimages:
                pass
            else:
                fullpath=os.path.join(cfg['project_path'],'labeled-data',folder,imagename)
                if os.path.isfile(fullpath):
                    print("Deleting", fullpath)
                    os.remove(fullpath)

        annotatedimages=[fn.split(os.sep)[-1] for fn in DC.index]
        imagelist=[fns for fns in os.listdir(str(folder)) if '.png' in fns]
        print("PROCESSED:", folder, " now # of annotated images: ", len(annotatedimages)," in folder:", len(imagelist))


def label_frames(config,multiple=False):
    """
    Manually label/annotate the extracted frames. Update the list of body parts you want to localize in the config.yaml file first.

    Parameter
    ----------
    config : string
        String containing the full path of the config file in the project.

    multiple: bool, optional
        If this is set to True, a user can label multiple individuals.
        The default is ``False``; if provided it must be either ``True`` or ``False``.


    Example
    --------
    To label multiple individuals
    >>> deeplabcut.label_frames('/analysis/project/reaching-task/config.yaml',multiple=True)
    --------

    """
    startpath = os.getcwd()
    wd = Path(config).resolve().parents[0]
    os.chdir(str(wd))

    if multiple==False:
        from deeplabcut.generate_training_dataset import labeling_toolbox

        # labeling_toolbox.show(config,Screens,scale_w,scale_h, winHack, img_scale)
        labeling_toolbox.show(config)
    else:
        from deeplabcut.generate_training_dataset import multiple_individuals_labeling_toolbox
        multiple_individuals_labeling_toolbox.show(config)

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

        #plt.savefig(str(Path(tmpfolder)/imagename.split(os.sep)[-1]))
        plt.savefig(os.path.join(tmpfolder,str(Path(imagename).name))) #create file name also on Windows for Unix projects (and vice versa)
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
    >>> trainIndexes, testIndexes=deeplabcut.mergeandsplit(config,trainindex=0,uniform=True)
    You can then create two model instances that have the identical trainingset. Thereby you can assess the role of various parameters on the performance of DLC.

    >>> deeplabcut.create_training_dataset(config,Shuffles=[0],trainIndexes=trainIndexes,testIndexes=testIndexes)
    >>> deeplabcut.create_training_dataset(config,Shuffles=[1],trainIndexes=trainIndexes,testIndexes=testIndexes)
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

@lru_cache(maxsize=None)
def _read_image_shape_fast(path):
    return io.imread(path).shape

def format_training_data(df, train_inds, nbodyparts, project_path):
    train_data = []
    matlab_data = []

    def to_matlab_cell(array):
        outer = np.array([[None]], dtype=object)
        outer[0, 0] = array.astype('int64')
        return outer

    for i in train_inds:
        data = dict()
        filename = df.index[i]
        data['image'] = filename
        img_shape = _read_image_shape_fast(os.path.join(project_path, filename))
        try:
            data['size'] = img_shape[2], img_shape[0], img_shape[1]
        except IndexError:
            data['size'] = 1, img_shape[0], img_shape[1]
        temp = df.iloc[i].values.reshape(-1, 2)
        joints = np.c_[range(nbodyparts), temp]
        joints = joints[~np.isnan(joints).any(axis=1)].astype(int)
        # Check that points lie within the image
        inside = np.logical_and(np.logical_and(joints[:, 1] < img_shape[1], joints[:, 1] > 0),
                                np.logical_and(joints[:, 2] < img_shape[0], joints[:, 2] > 0))
        if not all(inside):
            joints = joints[inside]
        if joints.size:  # Exclude images without labels
            data['joints'] = joints
            train_data.append(data)
            matlab_data.append((np.array([data['image']], dtype='U'),
                                np.array([data['size']]),
                                to_matlab_cell(data['joints'])))
    matlab_data = np.asarray(matlab_data, dtype=[('image', 'O'), ('size', 'O'), ('joints', 'O')])
    return train_data, matlab_data

def create_training_dataset(config,num_shuffles=1,Shuffles=None,windows2linux=False,userfeedback=False,
        trainIndexes=None,testIndexes=None,
        net_type=None,augmenter_type=None):
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

    userfeedback: bool, optional
        If this is set to false, then all requested train/test splits are created (no matter if they already exist). If you
        want to assure that previous splits etc. are not overwritten, then set this to True and you will be asked for each split.

    trainIndexes: list of lists, optional (default=None)
        List of one or multiple lists containing train indexes.
        A list containing two lists of training indexes will produce two splits.

    testIndexes: list of lists, optional (default=None)
        List of test indexes.

    net_type: string
        Type of networks. Currently resnet_50, resnet_101, resnet_152, mobilenet_v2_1.0,mobilenet_v2_0.75, mobilenet_v2_0.5, and mobilenet_v2_0.35 are supported.

    augmenter_type: string
        Type of augmenter. Currently default, imgaug, tensorpack, and deterministic are supported.

    Example
    --------
    >>> deeplabcut.create_training_dataset('/analysis/project/reaching-task/config.yaml',num_shuffles=1)
    Windows:
    >>> deeplabcut.create_training_dataset('C:\\Users\\Ulf\\looming-task\\config.yaml',Shuffles=[3,17,5])
    --------
    """
    import scipy.io as sio

    # Loading metadata from config file:
    cfg = auxiliaryfunctions.read_config(config)
    scorer = cfg['scorer']
    project_path = cfg['project_path']
    # Create path for training sets & store data there
    trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(cfg) #Path concatenation OS platform independent
    auxiliaryfunctions.attempttomakefolder(Path(os.path.join(project_path,str(trainingsetfolder))),recursive=True)

    Data = merge_annotateddatasets(cfg,project_path,Path(os.path.join(project_path,trainingsetfolder)),windows2linux)
    Data = Data[scorer] #extract labeled data

    #loading & linking pretrained models
    if net_type is None: #loading & linking pretrained models
        net_type =cfg.get('default_net_type', 'resnet_50')
    else:
        if 'resnet' in net_type or 'mobilenet' in net_type:
            pass
        else:
            raise ValueError('Invalid network type:', net_type)

    if augmenter_type is None:
        augmenter_type=cfg.get('default_augmenter', 'default')
    else:
        if augmenter_type in ['default','imgaug','tensorpack','deterministic']:
            pass
        else:
            raise ValueError('Invalid augmenter type:', augmenter_type)

    import deeplabcut
    parent_path = Path(os.path.dirname(deeplabcut.__file__))
    defaultconfigfile = str(parent_path / 'pose_cfg.yaml')
    model_path,num_shuffles=auxfun_models.Check4weights(net_type,parent_path,num_shuffles) #if the model does not exist >> throws error!

    if Shuffles is None:
        Shuffles = range(1, num_shuffles + 1)
    else:
        Shuffles = [i for i in Shuffles if isinstance(i, int)]

    #print(trainIndexes,testIndexes, Shuffles, augmenter_type,net_type)
    if trainIndexes is None and testIndexes is None:
        splits = [(trainFraction, shuffle, SplitTrials(range(len(Data.index)), trainFraction))
                  for trainFraction in cfg['TrainingFraction'] for shuffle in Shuffles]
    else:
        if len(trainIndexes) != len(testIndexes) != len(Shuffles):
            raise ValueError('Number of Shuffles and train and test indexes should be equal.')
        splits = []
        for shuffle, (train_inds, test_inds) in enumerate(zip(trainIndexes, testIndexes)):
            trainFraction = round(len(train_inds) * 1./ (len(train_inds) + len(test_inds)), 2)
            print(f"You passed a split with the following fraction: {int(100 * trainFraction)}%")
            splits.append((trainFraction, Shuffles[shuffle], (train_inds, test_inds)))

    bodyparts = cfg['bodyparts']
    nbodyparts = len(bodyparts)
    for trainFraction, shuffle, (trainIndexes, testIndexes) in splits:
        if len(trainIndexes)>0:
            if userfeedback:
                trainposeconfigfile, _, _ = training.return_train_network_path(config, shuffle=shuffle, trainFraction=trainFraction)
                if trainposeconfigfile.is_file():
                    askuser=input ("The model folder is already present. If you continue, it will overwrite the existing model (split). Do you want to continue?(yes/no): ")
                    if askuser=='no'or askuser=='No' or askuser=='N' or askuser=='No':
                        raise Exception("Use the Shuffles argument as a list to specify a different shuffle index. Check out the help for more details.")

            ####################################################
            # Generating data structure with labeled information & frame metadata (for deep cut)
            ####################################################
            # Make training file!
            datafilename, metadatafilename = auxiliaryfunctions.GetDataandMetaDataFilenames(trainingsetfolder,
                                                                                            trainFraction, shuffle, cfg)

            ################################################################################
            # Saving data file (convert to training file for deeper cut (*.mat))
            ################################################################################
            data, MatlabData = format_training_data(Data, trainIndexes, nbodyparts, project_path)
            sio.savemat(os.path.join(project_path,datafilename), {'dataset': MatlabData})

            ################################################################################
            # Saving metadata (Pickle file)
            ################################################################################
            auxiliaryfunctions.SaveMetadata(os.path.join(project_path,metadatafilename),data, trainIndexes, testIndexes, trainFraction)

            ################################################################################
            # Creating file structure for training &
            # Test files as well as pose_yaml files (containing training and testing information)
            #################################################################################
            modelfoldername=auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,cfg)
            auxiliaryfunctions.attempttomakefolder(Path(config).parents[0] / modelfoldername,recursive=True)
            auxiliaryfunctions.attempttomakefolder(str(Path(config).parents[0] / modelfoldername)+ '/train')
            auxiliaryfunctions.attempttomakefolder(str(Path(config).parents[0] / modelfoldername)+ '/test')

            path_train_config = str(os.path.join(cfg['project_path'],Path(modelfoldername),'train','pose_cfg.yaml'))
            path_test_config = str(os.path.join(cfg['project_path'],Path(modelfoldername),'test','pose_cfg.yaml'))
            #str(cfg['proj_path']+'/'+Path(modelfoldername) / 'test'  /  'pose_cfg.yaml')

            items2change = {
                "dataset": datafilename,
                "metadataset": metadatafilename,
                "num_joints": len(bodyparts),
                "all_joints": [[i] for i in range(len(bodyparts))],
                "all_joints_names": [str(bpt) for bpt in bodyparts],
                "init_weights": model_path,
                "project_path": str(cfg['project_path']),
                "net_type": net_type,
                "dataset_type": augmenter_type,
            }
            trainingdata = MakeTrain_pose_yaml(items2change,path_train_config,defaultconfigfile)
            keys2save = [
                "dataset", "num_joints", "all_joints", "all_joints_names",
                "net_type", 'init_weights', 'global_scale', 'location_refinement',
                'locref_stdev'
            ]
            MakeTest_pose_yaml(trainingdata, keys2save,path_test_config)
            print("The training dataset is successfully created. Use the function 'train_network' to start training. Happy training!")
    return splits


def get_largestshuffle_index(config):
    ''' Returns the largest shuffle for all dlc-models in the current iteration.'''
    cfg = auxiliaryfunctions.read_config(config)
    project_path = cfg['project_path']
    iterate = 'iteration-'+str(cfg['iteration'])
    dlc_model_path =  os.path.join(project_path,'dlc-models',iterate)
    if os.path.isdir(dlc_model_path):
        models = os.listdir(dlc_model_path)
        # sort the models directories
        models.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        # get the shuffle index
        max_shuffle_index = int(models[-1].split('shuffle')[-1])
    else:
        max_shuffle_index = 0
    return(max_shuffle_index)

def create_training_model_comparison(config,trainindex=0,num_shuffles=1,net_types=['resnet_50'],augmenter_types=['default'],userfeedback=False,windows2linux=False):
    """
    Creates a training dataset with different networks and augmentation types (dataset_loader) so that the shuffles
    have same training and testing indices.

    Therefore, this function is useful for benchmarking the performance of different network and augmentation types on the same training/testdata.\n

    Parameter
    ----------
    config : string
        Full path of the config.yaml file as a string.

    trainindex: int, optional
        Either (in case uniform = True) indexes which element of TrainingFraction in the config file should be used (note it is a list!).
        Alternatively (uniform = False) indexes which folder is dropped, i.e. the first if trainindex=0, the second if trainindex =1, etc.

    num_shuffles : int, optional
        Number of shuffles of training dataset to create, i.e. [1,2,3] for num_shuffles=3. Default is set to 1.

    net_types: list
        Type of networks. Currently resnet_50, resnet_101, resnet_152, mobilenet_v2_1.0,mobilenet_v2_0.75, mobilenet_v2_0.5, and mobilenet_v2_0.35 are supported.

    augmenter_types: list
        Type of augmenters. Currently "default", "imgaug", "tensorpack", and "deterministic" are supported.

    userfeedback: bool, optional
        If this is set to false, then all requested train/test splits are created (no matter if they already exist). If you
        want to assure that previous splits etc. are not overwritten, then set this to True and you will be asked for each split.

    windows2linux: bool.
        The annotation files contain path formated according to your operating system. If you label on windows
        but train & evaluate on a unix system (e.g. ubunt, colab, Mac) set this variable to True to convert the paths.

    Example
    --------
    >>> deeplabcut.create_training_model_comparison('/analysis/project/reaching-task/config.yaml',num_shuffles=1,net_types=['resnet_50','resnet_152'],augmenter_types=['tensorpack','deterministic'])

    Windows:
    >>> deeplabcut.create_training_model_comparison('C:\\Users\\Ulf\\looming-task\\config.yaml',num_shuffles=1,net_types=['resnet_50','resnet_152'],augmenter_types=['tensorpack','deterministic'])

    --------
    """
    # read cfg file
    cfg = auxiliaryfunctions.read_config(config)

    # create log file
    log_file_name = os.path.join(cfg['project_path'],'training_model_comparison.log')
    logger = logging.getLogger('training_model_comparison')
    if not logger.handlers:
        logger = logging.getLogger('training_model_comparison')
        hdlr = logging.FileHandler(log_file_name)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)
    else:
        pass

    largestshuffleindex = get_largestshuffle_index(config)

    for shuffle in range(num_shuffles):
        trainIndexes, testIndexes=mergeandsplit(config,trainindex=trainindex,uniform=True)
        for idx_net,net in enumerate(net_types):
            for idx_aug,aug in enumerate(augmenter_types):
                get_max_shuffle_idx=(largestshuffleindex+idx_aug+idx_net*len(augmenter_types)+shuffle*len(augmenter_types)*len(net_types))
                log_info = str("Shuffle index:" + str(get_max_shuffle_idx) + ", net_type:"+net +", augmenter_type:"+aug + ", trainsetindex:" +str(trainindex))
                create_training_dataset(config,Shuffles=[get_max_shuffle_idx],net_type=net,trainIndexes=[trainIndexes],testIndexes=[testIndexes],augmenter_type=aug,userfeedback=userfeedback,windows2linux=windows2linux)
                logger.info(log_info)
