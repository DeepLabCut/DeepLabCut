"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import logging, os

def visualizemaps(config,shuffle, trainingsetindex=0, comparisonbodyparts="all",
                gputouse=None, rescale=False, plotting=True, modelprefix='', Indices=None):
    from deeplabcut.utils.auxfun_videos import imread, imresize
    from deeplabcut.pose_estimation_tensorflow.nnet import predict
    from deeplabcut.pose_estimation_tensorflow.nnet import predict_multianimal as predictma
    from deeplabcut.pose_estimation_tensorflow.config import load_config
    from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import data_to_input
    from deeplabcut.utils import auxiliaryfunctions
    from tqdm import tqdm
    import tensorflow as tf
    vers = (tf.__version__).split('.')
    if int(vers[0])==1 and int(vers[1])>12:
        TF=tf.compat.v1
    else:
        TF=tf

    import pandas as pd
    from pathlib import Path
    import platform
    import numpy as np
    import platform
    import matplotlib as mpl
    if os.environ.get('DLClight', default=False) == 'True':
        mpl.use('AGG') #anti-grain geometry engine #https://matplotlib.org/faq/usage_faq.html
        pass
    elif platform.system() == 'Darwin':
        mpl.use('WXAgg')
    else:
        mpl.use('TkAgg') #TkAgg
    import matplotlib.pyplot as plt

    TF.reset_default_graph()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #
#    tf.logging.set_verbosity(tf.logging.WARN)

    start_path=os.getcwd()
    # Read file path for pose_config file. >> pass it on
    cfg = auxiliaryfunctions.read_config(config)

    if gputouse is not None: #gpu selectinon
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)
    if trainingsetindex=='all':
        TrainingFractions=cfg["TrainingFraction"]
    else:
        if trainingsetindex<len(cfg["TrainingFraction"]) and trainingsetindex>=0:
            TrainingFractions=[cfg["TrainingFraction"][int(trainingsetindex)]]
        else:
            raise Exception('Please check the trainingsetindex! ', trainingsetindex, ' should be an integer from 0 .. ', int(len(cfg["TrainingFraction"])-1))

    # Loading human annotatated data
    trainingsetfolder=auxiliaryfunctions.GetTrainingSetFolder(cfg)
    Data=pd.read_hdf(os.path.join(cfg["project_path"],str(trainingsetfolder),'CollectedData_' + cfg["scorer"] + '.h5'),'df_with_missing')

    # Get list of body parts to evaluate network for
    comparisonbodyparts=auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(cfg,comparisonbodyparts)
    # Make folder for evaluation
    auxiliaryfunctions.attempttomakefolder(str(cfg["project_path"]+"/evaluation-results/"))
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

            #change batch size, if it was edited during analysis!
            dlc_cfg['batch_size']=1 #in case this was edited for analysis.

            #Create folder structure to store results.
            evaluationfolder=os.path.join(cfg["project_path"],str(auxiliaryfunctions.GetEvaluationFolder(trainFraction,shuffle,cfg,modelprefix=modelprefix)))
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

            ########################### RESCALING (to global scale)
            if rescale==True:
                scale=dlc_cfg['global_scale']
                Data=pd.read_hdf(os.path.join(cfg["project_path"],str(trainingsetfolder),'CollectedData_' + cfg["scorer"] + '.h5'),'df_with_missing')*scale
            else:
                scale=1

            pagraph=dlc_cfg['partaffinityfield_graph']
            bptnames=[dlc_cfg['all_joints_names'][i] for i in range(len(dlc_cfg['all_joints']))]

            for snapindex in snapindices:
                dlc_cfg['init_weights'] = os.path.join(str(modelfolder),'train',Snapshots[snapindex]) #setting weights to corresponding snapshot.
                trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1] #read how many training siterations that corresponds to.

                # Name for deeplabcut net (based on its parameters)
                DLCscorer,DLCscorerlegacy = auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction,trainingsiterations)
                notanalyzed,resultsfilename,DLCscorer=auxiliaryfunctions.CheckifNotEvaluated(str(evaluationfolder),DLCscorer,DLCscorerlegacy,Snapshots[snapindex])
                print("Extracting maps for ", DLCscorer, " with # of trainingiterations:", trainingsiterations)
                if notanalyzed:
                    # Specifying state of model (snapshot / training state)
                    sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)
                    Numimages = len(Data.index)
                    PredicteData = np.zeros((Numimages,3 * len(dlc_cfg['all_joints_names'])))
                    print("Analyzing data...")
                    if Indices is None:
                        Indices=enumerate(Data.index)
                    else:
                        Ind = [Data.index[j] for j in Indices]
                        Indices=enumerate(Ind)

                    DATA={}
                    for imageindex, imagename in tqdm(Indices):
                        image = imread(os.path.join(cfg['project_path'],imagename),mode='RGB')
                        if scale!=1:
                            image = imresize(image, scale)

                        image_batch = data_to_input(image)
                        # Compute prediction with the CNN
                        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})

                        if cfg.get('multianimalproject',False):
                            scmap, locref, paf= predictma.extract_cnn_output(outputs_np, dlc_cfg)
                        else:
                            scmap, locref = predict.extract_cnn_output(outputs_np, dlc_cfg)
                            paf = None

                        if imageindex in testIndices:
                            trainingfram=False
                        else:
                            trainingfram=True
                        DATA[imageindex]=[image,scmap,locref,paf,bptnames,pagraph,imagename,trainingfram]
                    return DATA
'''

                        # Extract maximum scoring location from the heatmap, assume 1 person
                        pose = predict.argmax_pose_predict(scmap, locref, dlc_cfg.stride,dlc_cfg).flatten()
                        PredicteData[imageindex, :] = pose  # NOTE: thereby     cfg_test['all_joints_names'] should be same order as bodyparts!
                        print(np.shape(pose))
                        x,y=pose[0::3][:5],pose[1::3][:5]

                        DATA[imagename]={}
                        DATA[imagename]['image']=image
                        DATA[imagename]['scmap']=scmap[:,:,-2:]

                        if plotting==True:
                            plt.figure(figsize=(7,7))
                            numpanels=9
                            for jj in range(numpanels):
                                plt.subplot(3,3,jj+1)
                                plt.axis('off')

                                if jj==0:
                                    plt.imshow(image)
                                    plt.plot(x,y,'.')
                                elif jj<(numpanels-2):
                                    plt.imshow(scmap[:,:,jj])
                                elif jj==(numpanels-2):
                                    plt.imshow(np.log(scmap[:,:,-2]))
                                else:
                                    plt.imshow(np.log(scmap[:,:,-1]))
                                #scmap_part = imresize(scmap_part, 8.0, interp='nearest')

                            plt.savefig(os.path.join(evaluationfolder,str(imageindex)+Path(imagename).stem+'.png'))
                    sess.close() #closes the current tf session
                    print(np.min(scmap[:,:,-2:]),np.max(scmap[:,:,-2:]),np.mean(scmap[:,:,-2:]))
                    auxiliaryfunctions.write_pickle(os.path.join(evaluationfolder,'maps.pickle'),DATA)
    os.chdir(str(start_path))
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to yaml configuration file.')
    cli_args = parser.parse_args()

    display_dataset(Path(cli_args.config).resolve())
