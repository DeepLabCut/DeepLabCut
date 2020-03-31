"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""


import os
import argparse

# Dependencies for anaysis
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def evaluate_multianimal_full(config, Shuffles=[1], trainingsetindex=0,
    plotting=None, show_errors=True, comparisonbodyparts="all",
    gputouse=None, modelprefix='', c_engine=False):
    """
    WIP multi animal project.
    """
    import os
    from skimage import io
    import skimage.color
    from skimage.util import img_as_ubyte

    from deeplabcut.pose_estimation_tensorflow.nnet import predict
    from deeplabcut.pose_estimation_tensorflow.nnet import predict_multianimal as predictma
    from deeplabcut.utils import auxiliaryfunctions, visualization, auxfun_multianimal
    from deeplabcut.pose_estimation_tensorflow.config import load_config
    from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import data_to_input

    import tensorflow as tf

    if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
        del os.environ['TF_CUDNN_USE_AUTOTUNE'] #was potentially set during training

    tf.reset_default_graph()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #
    if gputouse is not None: #gpu selectinon
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)

    start_path=os.getcwd()

    ##################################################
    # Load data...
    ##################################################
    cfg = auxiliaryfunctions.read_config(config)
    if trainingsetindex=='all':
        TrainingFractions=cfg["TrainingFraction"]
    else:
        TrainingFractions=[cfg["TrainingFraction"][trainingsetindex]]

    # Loading human annotatated data
    trainingsetfolder=auxiliaryfunctions.GetTrainingSetFolder(cfg)
    Data=pd.read_hdf(os.path.join(cfg["project_path"],str(trainingsetfolder),'CollectedData_' + cfg["scorer"] + '.h5'),'df_with_missing')
    # Get list of body parts to evaluate network for
    comparisonbodyparts=auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(cfg,comparisonbodyparts)
    colors = visualization.get_cmap(len(comparisonbodyparts), name=cfg['colormap'])
    # Make folder for evaluation
    auxiliaryfunctions.attempttomakefolder(str(cfg["project_path"]+"/evaluation-results/"))
    for shuffle in Shuffles:
        for trainFraction in TrainingFractions:
            ##################################################
            # Load and setup CNN part detector
            ##################################################
            datafn,metadatafn=auxiliaryfunctions.GetDataandMetaDataFilenames(trainingsetfolder,trainFraction,shuffle,cfg)
            modelfolder=os.path.join(cfg["project_path"],str(auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,cfg,modelprefix=modelprefix)))
            path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'

            # Load meta data
            data, trainIndices, testIndices, trainFraction=auxiliaryfunctions.LoadMetadata(os.path.join(cfg["project_path"],metadatafn))

            try:
                dlc_cfg = load_config(str(path_test_config))
            except FileNotFoundError:
                raise FileNotFoundError("It seems the model for shuffle %s and trainFraction %s does not exist."%(shuffle,trainFraction))

            #TODO: IMPLEMENT for different batch sizes?
            dlc_cfg['batch_size']=1 #due to differently sized images!!!

            #Create folder structure to store results.
            evaluationfolder=os.path.join(cfg["project_path"],str(auxiliaryfunctions.GetEvaluationFolder(trainFraction,shuffle,cfg,modelprefix=modelprefix)))
            auxiliaryfunctions.attempttomakefolder(evaluationfolder,recursive=True)
            #path_train_config = modelfolder / 'train' / 'pose_cfg.yaml'

            # Check which snapshots are available and sort them by # iterations
            Snapshots = np.array([fn.split('.')[0]for fn in os.listdir(os.path.join(str(modelfolder), 'train'))if "index" in fn])
            if len(Snapshots)==0:
              print("Snapshots not found! It seems the dataset for shuffle %s and trainFraction %s is not trained.\nPlease train it before evaluating.\nUse the function 'train_network' to do so."%(shuffle,trainFraction))
            else:
                increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
                Snapshots = Snapshots[increasing_indices]

                if cfg["snapshotindex"] == -1:
                    snapindices = [-1]
                elif cfg["snapshotindex"] == "all":
                    snapindices = range(len(Snapshots))
                elif cfg["snapshotindex"]<len(Snapshots):
                    snapindices=[cfg["snapshotindex"]]
                else:
                    print("Invalid choice, only -1 (last), any integer up to last, or all (as string)!")

                individuals,uniquebodyparts,multianimalbodyparts=auxfun_multianimal.extractindividualsandbodyparts(cfg)

                final_result=[]
                ##################################################
                # Compute predictions over images
                ##################################################
                for snapindex in snapindices:
                    dlc_cfg['init_weights'] = os.path.join(str(modelfolder),'train',Snapshots[snapindex]) #setting weights to corresponding snapshot.
                    trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1] #read how many training siterations that corresponds to.

                    #name for deeplabcut net (based on its parameters)
                    DLCscorer,DLCscorerlegacy = auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction,trainingsiterations,modelprefix=modelprefix)
                    print("Running ", DLCscorer, " with # of trainingiterations:", trainingsiterations)
                    notanalyzed,resultsfilename,DLCscorer=auxiliaryfunctions.CheckifNotEvaluated(str(evaluationfolder),DLCscorer,DLCscorerlegacy,Snapshots[snapindex])

                    if os.path.isfile(resultsfilename.split('.h5')[0]+'_full.pickle'):
                            print("Model already evaluated.", resultsfilename)
                    else:
                        if plotting:
                            foldername = os.path.join(str(evaluationfolder),
                                                      'LabeledImages_' + DLCscorer + '_' + Snapshots[snapindex])
                            auxiliaryfunctions.attempttomakefolder(foldername)

                        # Specifying state of model (snapshot / training state)
                        sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)

                        PredicteData ={}
                        print("Analyzing data...")
                        for imageindex, imagename in tqdm(enumerate(Data.index)):
                            image_path = os.path.join(cfg['project_path'], imagename)
                            image = io.imread(image_path)
                            frame = img_as_ubyte(skimage.color.gray2rgb(image))

                            GT=Data.iloc[imageindex]

                            groundtruthcoordinates=[]
                            groundtruthidentity=[]
                            for bptindex, bpt in enumerate(dlc_cfg["all_joints_names"]):
                                coords=np.zeros([len(individuals),2])*np.nan
                                identity=[]
                                for prfxindex,prefix in enumerate(individuals):
                                    if bpt in uniquebodyparts and prefix=='single':
                                        coords[prfxindex,:]=np.array([GT[cfg["scorer"]][prefix][bpt]['x'],GT[cfg["scorer"]][prefix][bpt]['y']])
                                        identity.append(prefix)
                                    elif bpt in multianimalbodyparts and prefix!='single':
                                        coords[prfxindex,:]=np.array([GT[cfg["scorer"]][prefix][bpt]['x'],GT[cfg["scorer"]][prefix][bpt]['y']])
                                        identity.append(prefix)
                                    else:
                                        identity.append('nix')

                                groundtruthcoordinates.append(coords[np.isfinite(coords[:,0]),:])
                                groundtruthidentity.append(np.array(identity)[np.isfinite(coords[:,0])])

                            PredicteData[imagename]={}
                            PredicteData[imagename]['index']=imageindex
                            pred = predictma.get_detectionswithcostsandGT(frame,  groundtruthcoordinates, dlc_cfg, sess, inputs, outputs, outall=False,nms_radius=dlc_cfg.nmsradius,det_min_score=dlc_cfg.minconfidence, c_engine=c_engine)
                            PredicteData[imagename]['prediction'] = pred
                            PredicteData[imagename]['groundtruth']=[groundtruthidentity, groundtruthcoordinates,GT]

                            if plotting:
                                coords_pred = pred['coordinates'][0]
                                probs_pred = pred['confidence']
                                fig = visualization.make_multianimal_labeled_image(frame, groundtruthcoordinates, coords_pred, probs_pred, colors,
                                                                                   cfg['dotsize'], cfg['alphavalue'], cfg['pcutoff'])
                                visualization.save_labeled_frame(fig, image_path, foldername, imageindex in trainIndices)

                        sess.close() #closes the current tf session
                        PredicteData['metadata']={
                            'nms radius': dlc_cfg.nmsradius,
                            'minimal confidence': dlc_cfg.minconfidence,
                            'PAFgraph': dlc_cfg.partaffinityfield_graph,
                            "all_joints": [[i] for i in range(len(dlc_cfg.all_joints))],
                            "all_joints_names": [dlc_cfg.all_joints_names[i] for i in range(len(dlc_cfg.all_joints))],
                            "stride": dlc_cfg.get('stride',8)
                            }
                        print("Done and results stored for snapshot: ", Snapshots[snapindex])

                        dictionary = {
                            "Scorer": DLCscorer,
                            "DLC-model-config file": dlc_cfg,
                            "trainIndices": trainIndices,
                            "testIndices": testIndices,
                            "trainFraction": trainFraction
                        }
                        metadata = {'data': dictionary}
                        auxfun_multianimal.SaveFullMultiAnimalData(PredicteData, metadata, resultsfilename)

                        tf.reset_default_graph()

    #returning to intial folder
    os.chdir(str(start_path))

def evaluate_multianimal_crossvalidate(config,Shuffles=[1], trainingsetindex=0,
    plotting = None, show_errors = True, comparisonbodyparts="all", modelprefix=''):
    """
    TODO: integrate code from sDLC
    """
    import os
    from skimage import io
    import skimage.color
    from skimage.util import img_as_ubyte

    from deeplabcut.utils import auxiliaryfunctions, visualization, auxfun_multianimal
    from deeplabcut.pose_estimation_tensorflow.config import load_config
    start_path=os.getcwd()

    ##################################################
    # Load data...
    ##################################################
    cfg = auxiliaryfunctions.read_config(config)
    if trainingsetindex=='all':
        TrainingFractions=cfg["TrainingFraction"]
    else:
        TrainingFractions=[cfg["TrainingFraction"][trainingsetindex]]

    # Loading human annotatated data
    trainingsetfolder=auxiliaryfunctions.GetTrainingSetFolder(cfg)
    Data=pd.read_hdf(os.path.join(cfg["project_path"],str(trainingsetfolder),'CollectedData_' + cfg["scorer"] + '.h5'),'df_with_missing')
    # Get list of body parts to evaluate network for
    comparisonbodyparts=auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(cfg,comparisonbodyparts)
    # Make folder for evaluation
    auxiliaryfunctions.attempttomakefolder(str(cfg["project_path"]+"/evaluation-results/"))
    for shuffle in Shuffles:
        for trainFraction in TrainingFractions:
            ##################################################
            # Load and setup CNN part detector
            ##################################################
            datafn,metadatafn=auxiliaryfunctions.GetDataandMetaDataFilenames(trainingsetfolder,trainFraction,shuffle,cfg)
            modelfolder=os.path.join(cfg["project_path"],str(auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,cfg,modelprefix=modelprefix)))
            path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'

            # Load meta data
            data, trainIndices, testIndices, trainFraction=auxiliaryfunctions.LoadMetadata(os.path.join(cfg["project_path"],metadatafn))

            try:
                dlc_cfg = load_config(str(path_test_config))
            except FileNotFoundError:
                raise FileNotFoundError("It seems the model for shuffle %s and trainFraction %s does not exist."%(shuffle,trainFraction))

            #TODO: IMPLEMENT for different batch sizes?
            dlc_cfg['batch_size']=1 #due to differently sized images!!!

            #Create folder structure to store results.
            evaluationfolder=os.path.join(cfg["project_path"],str(auxiliaryfunctions.GetEvaluationFolder(trainFraction,shuffle,cfg,modelprefix=modelprefix)))
            auxiliaryfunctions.attempttomakefolder(evaluationfolder,recursive=True)
            #path_train_config = modelfolder / 'train' / 'pose_cfg.yaml'

            # Check which snapshots are available and sort them by # iterations
            Snapshots = np.array([fn.split('.')[0]for fn in os.listdir(os.path.join(str(modelfolder), 'train'))if "index" in fn])
            if len(Snapshots)==0:
              print("Snapshots not found! It seems the dataset for shuffle %s and trainFraction %s is not trained.\nPlease train it before evaluating.\nUse the function 'train_network' to do so."%(shuffle,trainFraction))
            else:
                increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
                Snapshots = Snapshots[increasing_indices]

                if cfg["snapshotindex"] == -1:
                    snapindices = [-1]
                elif cfg["snapshotindex"] == "all":
                    snapindices = range(len(Snapshots))
                elif cfg["snapshotindex"]<len(Snapshots):
                    snapindices=[cfg["snapshotindex"]]
                else:
                    print("Invalid choice, only -1 (last), any integer up to last, or all (as string)!")

                individuals,uniquebodyparts,multianimalbodyparts=auxfun_multianimal.extractindividualsandbodyparts(cfg)

                final_result=[]
                ##################################################
                # Compute predictions over images
                ##################################################
                for snapindex in snapindices:
                    dlc_cfg['init_weights'] = os.path.join(str(modelfolder),'train',Snapshots[snapindex]) #setting weights to corresponding snapshot.
                    trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1] #read how many training siterations that corresponds to.

                    #name for deeplabcut net (based on its parameters)
                    DLCscorer,DLCscorerlegacy = auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction,trainingsiterations,modelprefix=modelprefix)
                    print("Running ", DLCscorer, " with # of trainingiterations:", trainingsiterations)
                    notanalyzed,resultsfilename,DLCscorer=auxiliaryfunctions.CheckifNotEvaluated(str(evaluationfolder),DLCscorer,DLCscorerlegacy,Snapshots[snapindex])

                    if os.path.isfile(resultsfilename.split('.h5')[0]+'_full.pickle'):
                            print("Model already evaluated.", resultsfilename)

                    else:
                        Numimages = len(Data.index)

                        for imageindex, imagename in tqdm(enumerate(Data.index)):
                            image = io.imread(os.path.join(cfg['project_path'],imagename))
                            frame = img_as_ubyte(skimage.color.gray2rgb(image))

                            GT=Data.iloc[imageindex]
                            PredicteData[imagename]={}
                            PredicteData[imagename]['index']=imageindex
                            PredicteData[imagename]['prediction']=predictma.get_detectionswithcostsandGT(frame,  groundtruthcoordinates, dlc_cfg, sess, inputs, outputs, outall=False,nms_radius=dlc_cfg.nmsradius,det_min_score=dlc_cfg.minconfidence)
                            PredicteData[imagename]['groundtruth']=[groundtruthidentity, groundtruthcoordinates,GT]
                        PredicteData['metadata']={
                            'nms radius': dlc_cfg.nmsradius,
                            'minimal confidence': dlc_cfg.minconfidence,
                            'PAFgraph': dlc_cfg.partaffinityfield_graph,
                            "all_joints": [[i] for i in range(len(dlc_cfg.all_joints))],
                            "all_joints_names": [dlc_cfg.all_joints_names[i] for i in range(len(dlc_cfg.all_joints))],
                            "stride": dlc_cfg.get('stride',8)
                            }
                        print("Done and results stored for snapshot: ", Snapshots[snapindex])

                        dictionary = {
                            "Scorer": DLCscorer,
                            "DLC-model-config file": dlc_cfg,
                            "trainIndices": trainIndices,
                            "testIndices": testIndices,
                            "trainFraction": trainFraction
                        }
                        metadata = {'data': dictionary}
                        auxfun_multianimal.SaveFullMultiAnimalData(PredicteData, metadata, resultsfilename)

                        tf.reset_default_graph()

    #returning to intial folder
    os.chdir(str(start_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    cli_args = parser.parse_args()
