"""
DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu

"""

import os
import argparse

# Dependencies for anaysis
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def pairwisedistances(DataCombined,scorer1,scorer2,pcutoff=-1,bodyparts=None):
    ''' Calculates the pairwise Euclidean distance metric over body parts vs. images'''
    mask=DataCombined[scorer2].xs('likelihood',level=1,axis=1)>=pcutoff
    if bodyparts==None:
            Pointwisesquareddistance=(DataCombined[scorer1]-DataCombined[scorer2])**2
            RMSE=np.sqrt(Pointwisesquareddistance.xs('x',level=1,axis=1)+Pointwisesquareddistance.xs('y',level=1,axis=1)) #Euclidean distance (proportional to RMSE)
            return RMSE,RMSE[mask]
    else:
            Pointwisesquareddistance=(DataCombined[scorer1][bodyparts]-DataCombined[scorer2][bodyparts])**2
            RMSE=np.sqrt(Pointwisesquareddistance.xs('x',level=1,axis=1)+Pointwisesquareddistance.xs('y',level=1,axis=1)) #Euclidean distance (proportional to RMSE)
            return RMSE,RMSE[mask]

def evaluate_network(config,Shuffles=[1],plotting = None,show_errors = True,comparisonbodyparts="all",gputouse=None):
    """
    Evaluates the network based on the saved models at different stages of the training network.\n
    The evaluation results are stored in the .h5 and .csv file under the subdirectory 'evaluation_results'.
    Change the snapshotindex parameter in the config file to 'all' in order to evaluate all the saved models.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    Shuffles: list, optional
        List of integers specifying the shuffle indices of the training dataset. The default is [1]

    plotting: bool, optional
        Plots the predictions on the train and test images. The default is ``False``; if provided it must be either ``True`` or ``False``

    show_errors: bool, optional
        Display train and test errors. The default is `True``

    comparisonbodyparts: list of bodyparts, Default is "all".
        The average error will be computed for those body parts only (Has to be a subset of the body parts).

    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU put None.
    See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries
    
    Examples
    --------
    If you do not want to plot
    >>> deeplabcut.evaluate_network('/analysis/project/reaching-task/config.yaml', shuffle=[1])
    --------

    If you want to plot
    >>> deeplabcut.evaluate_network('/analysis/project/reaching-task/config.yaml',shuffle=[1],True)
    """
    import os
    from skimage import io
    import skimage.color

    from deeplabcut.pose_estimation_tensorflow.nnet import predict as ptf_predict
    from deeplabcut.pose_estimation_tensorflow.config import load_config
    from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import data_to_input
    from deeplabcut.utils import auxiliaryfunctions, visualization
    import tensorflow as tf
    
    if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
        del os.environ['TF_CUDNN_USE_AUTOTUNE'] #was potentially set during training
    

    tf.reset_default_graph()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 
#    tf.logging.set_verbosity(tf.logging.WARN)

    start_path=os.getcwd()
    # Read file path for pose_config file. >> pass it on
    cfg = auxiliaryfunctions.read_config(config)
    if gputouse is not None: #gpu selectinon
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)
            
    # Loading human annotatated data
    trainingsetfolder=auxiliaryfunctions.GetTrainingSetFolder(cfg)
    Data=pd.read_hdf(os.path.join(cfg["project_path"],str(trainingsetfolder),'CollectedData_' + cfg["scorer"] + '.h5'),'df_with_missing')
    # Get list of body parts to evaluate network for
    comparisonbodyparts=auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(cfg,comparisonbodyparts)
    # Make folder for evaluation
    auxiliaryfunctions.attempttomakefolder(str(cfg["project_path"]+"/evaluation-results/"))
    for shuffle in Shuffles:
        for trainFraction in cfg["TrainingFraction"]:
            ##################################################
            # Load and setup CNN part detector
            ##################################################
            datafn,metadatafn=auxiliaryfunctions.GetDataandMetaDataFilenames(trainingsetfolder,trainFraction,shuffle,cfg)
            modelfolder=os.path.join(cfg["project_path"],str(auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,cfg)))
            path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'
            # Load meta data
            data, trainIndices, testIndices, trainFraction=auxiliaryfunctions.LoadMetadata(os.path.join(cfg["project_path"],metadatafn))

            try:
                dlc_cfg = load_config(str(path_test_config))
            except FileNotFoundError:
                raise FileNotFoundError("It seems the model for shuffle %s and trainFraction %s does not exist."%(shuffle,trainFraction))
            
            #change batch size, if it was edited during analysis!
            dlc_cfg['batch_size']=1 #in case this was edited for analysis.
            #Create folder structure to store results.
            evaluationfolder=os.path.join(cfg["project_path"],str(auxiliaryfunctions.GetEvaluationFolder(trainFraction,shuffle,cfg)))
            auxiliaryfunctions.attempttomakefolder(evaluationfolder,recursive=True)
            #path_train_config = modelfolder / 'train' / 'pose_cfg.yaml'
            
            # Check which snapshots are available and sort them by # iterations
            Snapshots = np.array([fn.split('.')[0]for fn in os.listdir(os.path.join(str(modelfolder), 'train'))if "index" in fn])
            try: #check if any where found?
              Snapshots[0]
            except IndexError:
              raise FileNotFoundError("Snapshots not found! It seems the dataset for shuffle %s and trainFraction %s is not trained.\nPlease train it before evaluating.\nUse the function 'train_network' to do so."%(shuffle,trainFraction))

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

            final_result=[]
            ##################################################
            # Compute predictions over images
            ##################################################
            for snapindex in snapindices:
                dlc_cfg['init_weights'] = os.path.join(str(modelfolder),'train',Snapshots[snapindex]) #setting weights to corresponding snapshot.
                trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1] #read how many training siterations that corresponds to.
                
                #name for deeplabcut net (based on its parameters)
                DLCscorer = auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction,trainingsiterations)
                print("Running ", DLCscorer, " with # of trainingiterations:", trainingsiterations)
                resultsfilename=os.path.join(str(evaluationfolder),DLCscorer + '-' + Snapshots[snapindex]+  '.h5')
                try:
                    DataMachine = pd.read_hdf(resultsfilename,'df_with_missing')
                    print("This net has already been evaluated!")
                except FileNotFoundError:
                    # Specifying state of model (snapshot / training state)
                    sess, inputs, outputs = ptf_predict.setup_pose_prediction(dlc_cfg)

                    Numimages = len(Data.index)
                    PredicteData = np.zeros((Numimages,3 * len(dlc_cfg['all_joints_names'])))
                    print("Analyzing data...")
                    for imageindex, imagename in tqdm(enumerate(Data.index)):
                        image = io.imread(os.path.join(cfg['project_path'],imagename),mode='RGB')
                        image = skimage.color.gray2rgb(image)
                        image_batch = data_to_input(image)
                        
                        # Compute prediction with the CNN
                        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
                        scmap, locref = ptf_predict.extract_cnn_output(outputs_np, dlc_cfg)

                        # Extract maximum scoring location from the heatmap, assume 1 person
                        pose = ptf_predict.argmax_pose_predict(scmap, locref, dlc_cfg.stride)
                        PredicteData[imageindex, :] = pose.flatten()  # NOTE: thereby     cfg_test['all_joints_names'] should be same order as bodyparts!

                    sess.close() #closes the current tf session

                    index = pd.MultiIndex.from_product(
                        [[DLCscorer], dlc_cfg['all_joints_names'], ['x', 'y', 'likelihood']],
                        names=['scorer', 'bodyparts', 'coords'])

                    # Saving results
                    DataMachine = pd.DataFrame(PredicteData, columns=index, index=Data.index.values)
                    DataMachine.to_hdf(resultsfilename,'df_with_missing',format='table',mode='w')

                    print("Done and results stored for snapshot: ", Snapshots[snapindex])
                    DataCombined = pd.concat([Data.T, DataMachine.T], axis=0).T
                    RMSE,RMSEpcutoff = pairwisedistances(DataCombined, cfg["scorer"], DLCscorer,cfg["pcutoff"],comparisonbodyparts)
                    testerror = np.nanmean(RMSE.iloc[testIndices].values.flatten())
                    trainerror = np.nanmean(RMSE.iloc[trainIndices].values.flatten())
                    testerrorpcutoff = np.nanmean(RMSEpcutoff.iloc[testIndices].values.flatten())
                    trainerrorpcutoff = np.nanmean(RMSEpcutoff.iloc[trainIndices].values.flatten())
                    results = [trainingsiterations,int(100 * trainFraction),shuffle,np.round(trainerror,2),np.round(testerror,2),cfg["pcutoff"],np.round(trainerrorpcutoff,2), np.round(testerrorpcutoff,2)]
                    final_result.append(results)

                    if show_errors == True:
                            print("Results for",trainingsiterations," training iterations:", int(100 * trainFraction), shuffle, "train error:",np.round(trainerror,2), "pixels. Test error:", np.round(testerror,2)," pixels.")
                            print("With pcutoff of", cfg["pcutoff"]," train error:",np.round(trainerrorpcutoff,2), "pixels. Test error:", np.round(testerrorpcutoff,2), "pixels")
                            print("Thereby, the errors are given by the average distances between the labels by DLC and the scorer.")


                    if plotting == True:
                        print("Plotting...")
                        colors = visualization.get_cmap(len(comparisonbodyparts),name=cfg['colormap'])

                        foldername=os.path.join(str(evaluationfolder),'LabeledImages_' + DLCscorer + '_' + Snapshots[snapindex])
                        auxiliaryfunctions.attempttomakefolder(foldername)
                        NumFrames=np.size(DataCombined.index)
                        for ind in np.arange(NumFrames):
                            visualization.PlottingandSaveLabeledFrame(DataCombined,ind,trainIndices,cfg,colors,comparisonbodyparts,DLCscorer,foldername)
                            
                    tf.reset_default_graph()
                    #print(final_result)
            make_results_file(final_result,evaluationfolder,DLCscorer)
            print("The network is evaluated and the results are stored in the subdirectory 'evaluation_results'.")
            print("If it generalizes well, choose the best model for prediction and update the config file with the appropriate index for the 'snapshotindex'.\nUse the function 'analyze_video' to make predictions on new videos.")
            print("Otherwise consider retraining the network (see DeepLabCut workflow Fig 2)")
    
    #returning to intial folder
    os.chdir(str(start_path))
    
def make_results_file(final_result,evaluationfolder,DLCscorer):
    """
    Makes result file in .h5 and csv format and saves under evaluation_results directory
    """
    col_names = ["Training iterations:","%Training dataset","Shuffle number"," Train error(px)"," Test error(px)","p-cutoff used","Train error with p-cutoff","Test error with p-cutoff"]
    df = pd.DataFrame(final_result, columns = col_names)
    df.to_hdf(os.path.join(str(evaluationfolder),DLCscorer + '-results' + '.h5'),'df_with_missing',format='table',mode='w')
    df.to_csv(os.path.join(str(evaluationfolder),DLCscorer + '-results' + '.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    cli_args = parser.parse_args()
