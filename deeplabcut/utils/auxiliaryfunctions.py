"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

"""

import os, pickle, yaml
import pandas as pd
from pathlib import Path
import numpy as np

import ruamel.yaml

def create_config_template():
    """
    Creates a template for config.yaml file. This specific order is preserved while saving as yaml file.
    """
    import ruamel.yaml
    yaml_str = """\
# Project definitions (do not edit)
    Task:
    scorer:
    date:
    \n
# Project path (change when moving around)
    project_path:
    \n
# Annotation data set configuration (and individual video cropping parameters)
    video_sets:
    bodyparts:
    start:
    stop:
    numframes2pick:
    \n
# Plotting configuration
    pcutoff:
    dotsize:
    alphavalue:
    colormap:
    \n
# Training,Evaluation and Analysis configuration
    TrainingFraction:
    iteration:
    resnet:
    snapshotindex:
    batch_size:
    \n
# Cropping Parameters (for analysis and outlier frame detection)
    cropping:
#if cropping is true for analysis, then set the values here:
    x1:
    x2:
    y1:
    y2:
    \n
# Refinement configuration (parameters from annotation dataset configuration also relevant in this stage)
    corner2move2:
    move2corner:
    """
    ruamelFile = ruamel.yaml.YAML()
    cfg_file = ruamelFile.load(yaml_str)
    return(cfg_file,ruamelFile)
    
def read_config(configname):
    """
    Reads structured config file

    """
    ruamelFile = ruamel.yaml.YAML()
    path = Path(configname)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                cfg = ruamelFile.load(f)
        except Exception as err:
            if err.args[2] == "could not determine a constructor for the tag '!!python/tuple'":
                with open(path, 'r') as ymlfile:
                  cfg = yaml.load(ymlfile,Loader=yaml.SafeLoader)
                  write_config(configname,cfg)
    else:
        raise FileNotFoundError ("Config file is not found. Please make sure that the file exists and/or there are no unnecessary spaces in the path of the config file!")
    return(cfg)

def write_config(configname,cfg):
    """
    Write structured config file.
    """
    with open(configname, 'w') as cf:
        ruamelFile = ruamel.yaml.YAML()
        cfg_file,ruamelFile = create_config_template()
        for key in cfg.keys():
            cfg_file[key]=cfg[key]
            
        ruamelFile.dump(cfg_file, cf)

def read_plainconfig(filename = "pose_cfg.yaml"):
    ''' read unstructured yaml'''
    with open(filename, 'r') as f:
        yaml_cfg = yaml.load(f,Loader=yaml.SafeLoader)
    return yaml_cfg

def write_plainconfig(configname,cfg):
    with open(str(configname), 'w') as ymlfile:
                yaml.dump(cfg, ymlfile,default_flow_style=False)

def attempttomakefolder(foldername,recursive=False):
    ''' Attempts to create a folder with specified name. Does nothing if it already exists. '''

    try:
        os.path.isdir(foldername)
    except TypeError: #https://www.python.org/dev/peps/pep-0519/
        foldername=os.fspath(foldername) #https://github.com/AlexEMG/DeepLabCut/issues/105 (windows)

    if os.path.isdir(foldername):
        print(foldername, " already exists!")
    else:
        if recursive:
            os.makedirs(foldername)
        else:
            os.mkdir(foldername)

def Getlistofvideos(videos,videotype):
    from random import sample
    #checks if input is a directory
    if [os.path.isdir(i) for i in videos] == [True]:#os.path.isdir(video)==True:
        """
        Analyzes all the videos in the directory.
        """
        
        print("Analyzing all the videos in the directory")
        videofolder= videos[0]
        os.chdir(videofolder)
        videolist=[fn for fn in os.listdir(os.curdir) if (videotype in fn) and ('labeled.mp4' not in fn)] #exclude labeled-videos!
        Videos = sample(videolist,len(videolist)) # this is useful so multiple nets can be used to analzye simultanously
    else:
        if isinstance(videos,str):
            if os.path.isfile(videos): # #or just one direct path!
                Videos=[v for v in videos if os.path.isfile(v) and ('labeled.mp4' not in v)]
            else:
                Videos=[]
        else:
            Videos=[v for v in videos if os.path.isfile(v) and ('labeled.mp4' not in v)]
    return Videos

def SaveData(PredicteData, metadata, dataname, pdindex, imagenames,save_as_csv):
    ''' Save predicted data as h5 file and metadata as pickle file; created by predict_videos.py '''
    DataMachine = pd.DataFrame(PredicteData, columns=pdindex, index=imagenames)
    DataMachine.to_hdf(dataname, 'df_with_missing', format='table', mode='w')
    if save_as_csv:
        print("Saving csv poses!")
        DataMachine.to_csv(dataname.split('.h5')[0]+'.csv')
    with open(dataname.split('.h5')[0] + 'includingmetadata.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)

def LoadVideoMetadata(dataname):
    ''' Load meta data from analyzed video, created by predict_videos.py '''
    with open(dataname.split('.h5')[0] + 'includingmetadata.pickle', 'rb') as f: #same as in SaveData!
        metadata= pickle.load(f)
        return metadata

def SaveMetadata(metadatafilename, data, trainIndexes, testIndexes, trainFraction):
        with open(metadatafilename, 'wb') as f:
            # Pickle the 'labeled-data' dictionary using the highest protocol available.
            pickle.dump([data, trainIndexes, testIndexes, trainFraction], f,pickle.HIGHEST_PROTOCOL)

def LoadMetadata(metadatafile):
    with open(metadatafile, 'rb') as f:
        [trainingdata_details, trainIndexes, testIndexes,testFraction_data]= pickle.load(f)
        return trainingdata_details, trainIndexes, testIndexes, testFraction_data


def get_immediate_subdirectories(a_dir):
    ''' Get list of immediate subdirectories '''
    return [name for name in os.listdir(a_dir)
        if os.path.isdir(os.path.join(a_dir, name))]

def listfilesofaparticulartypeinfolder(a_dir,afiletype):
    ''' List files of a particular type in a folder a_dir '''
    return [
        name for name in os.listdir(a_dir)
        if afiletype in name]

def GetVideoList(filename,videopath,videtype):
    ''' Get list of videos in a path (if filetype == all), otherwise just a specific file.'''
    videos=listfilesofaparticulartypeinfolder(videopath,videtype)
    if filename=='all':
        return videos
    else:
        if filename in videos:
            videos=[filename]
        else:
            videos=[]
            print("Video not found!", filename)
    return videos

## Various functions to get filenames, foldernames etc. based on configuration parameters.
def GetTrainingSetFolder(cfg):
    ''' Training Set folder for config file based on parameters '''
    Task = cfg['Task']
    date = cfg['date']
    iterate = 'iteration-'+str(cfg['iteration'])
    return Path(os.path.join('training-datasets',iterate,'UnaugmentedDataSet_' + Task + date))

def GetModelFolder(trainFraction,shuffle,cfg):
    Task = cfg['Task']
    date = cfg['date']
    iterate = 'iteration-'+str(cfg['iteration'])
    return Path('dlc-models/'+ iterate+'/'+Task + date + '-trainset' + str(int(trainFraction * 100)) + 'shuffle' + str(shuffle))

def GetEvaluationFolder(trainFraction,shuffle,cfg):
    Task = cfg['Task']
    date = cfg['date']
    iterate = 'iteration-'+str(cfg['iteration'])
    return Path('evaluation-results/'+ iterate+'/'+Task + date + '-trainset' + str(int(trainFraction * 100)) + 'shuffle' + str(shuffle))

def GetDataandMetaDataFilenames(trainingsetfolder,trainFraction,shuffle,cfg):
    # Filename for metadata and data relative to project path for corresponding parameters
    metadatafn=os.path.join(str(trainingsetfolder) , 'Documentation_data-' + cfg["Task"] + "_" + str(int(trainFraction * 100)) + "shuffle" + str(shuffle) + '.pickle')
    datafn=os.path.join(str(trainingsetfolder) ,cfg["Task"] + "_" + cfg["scorer"] + str(int(100 * trainFraction)) + "shuffle" + str(shuffle)+ '.mat')
    return datafn,metadatafn


def IntersectionofBodyPartsandOnesGivenbyUser(cfg,comparisonbodyparts):
    ''' Returns all body parts when comparisonbodyparts=='all', otherwise all bpts that are in the intersection of comparisonbodyparts and the actual bodyparts '''
    allpbts = cfg['bodyparts']
    if comparisonbodyparts=="all":
        return allpbts
    else: #take only items in list that are actually bodyparts...
        cpbpts=[]
        for bp in comparisonbodyparts:
            if bp in allpbts:
                cpbpts.append(bp)
        return cpbpts

def GetScorerName(cfg,shuffle,trainFraction,trainingsiterations='unknown'):
    ''' Extract the scorer/network name for a particular shuffle, training fraction, etc. '''
    Task = cfg['Task']
    date = cfg['date']
    if trainingsiterations=='unknown':
        snapshotindex=cfg['snapshotindex']
        if cfg['snapshotindex'] == 'all':
            print("Changing snapshotindext to the last one -- plotting, videomaking, etc. should not be performed for all indices. For more selectivity enter the ordinal number of the snapshot you want (ie. 4 for the fifth) in the config file.")
            snapshotindex = -1
        else:
            snapshotindex=cfg['snapshotindex']

        modelfolder=os.path.join(cfg["project_path"],str(GetModelFolder(trainFraction,shuffle,cfg)),'train')
        Snapshots = np.array([fn.split('.')[0]for fn in os.listdir(modelfolder) if "index" in fn])
        increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
        Snapshots = Snapshots[increasing_indices]
        #dlc_cfg = read_config(os.path.join(modelfolder,'pose_cfg.yaml'))
        #dlc_cfg['init_weights'] = os.path.join(modelfolder , 'train', Snapshots[snapshotindex])
        SNP=Snapshots[snapshotindex]
        trainingsiterations = (SNP.split(os.sep)[-1]).split('-')[-1]

    scorer = 'DeepCut' + "_resnet" + str(cfg['resnet']) + "_" + Task + str(date) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations)
    return scorer
