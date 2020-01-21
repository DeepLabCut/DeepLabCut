#multimouse

import deeplabcut

task='MontBlanc'
scorer='Daniel'
video=['/home/alex/Dropbox/InterestingCode/social_datasets/croppedNov18/montblanc.mov']

#print("CREATING PROJECT")
#path_config_file=deeplabcut.create_new_project(task,scorer,video,copy_videos=True,multianimal=True)

#project already created!

#path_config_file='/home/alex/Hacking/DLCreleases/DLCdev/examples-multianimal/MontBlanc-Daniel-2019-12-16/config.yaml'

path_config_file='/home/alex/Hacking/DLCreleases/DLCdev/examples-multianimal/MontBlanc-Daniel-2019-12-16/config.yaml'

print("Plot labels...")
#deeplabcut.check_labels(path_config_file)
#deeplabcut.check_labels(path_config_file,draw_skeleton=True,visualizeindividuals=True)
#deeplabcut.check_labels(path_config_file,draw_skeleton=True,visualizeindividuals=False)

print("Crop images...")
#deeplabcut.cropimagesandlabels(path_config_file,userfeedback=False)
#then we uncommented the large-scale full frame data > it is not used for training!

print("Creating multianimal training set...")
deeplabcut.create_multianimaltraining_dataset(path_config_file)


'''

PUT INTO config file

    trainposeconfigfile,testposeconfigfile,snapshotfolder=deeplabcut.train_network_path(config,shuffle=shuffle,modelprefix=modelprefix,trainingsetindex=trainingsetindex)
        cfg_dlc=deeplabcut.auxiliaryfunctions.read_plainconfig(trainposeconfigfile)
        pts=cfg_dlc['all_joints'] #steps3?
        #PAF_graph=[[0,1],[2,3],[2,4],[2,5],[5,6],[6,7],[7,8],[8,9],[10,11],[11,12],[12,13]]

        #ADD MORE TAIL end point!
        PAF_graph=[[2,1],[2,0],[2,3],[2,12],[12,13],[13,14],[12,4],[4,10],[12,6],[6,11],[14,7],[7,9],[14,5],[5,8],[2,13],[2,14]]

        if 'fullyconnected' in modelsuffix:
            PAF_graph=[]
            for p1 in pts:
                for p2 in pts:
                    if p2[0]>p1[0]:
                        PAF_graph.append([p1[0],p2[0]])

        names=cfg_dlc['all_joints_names']
        for limbid,p in enumerate(PAF_graph):
            print(limbid,p[0],names[p[0]],names[p[1]])


        #now set it!
        num_limbs=len(PAF_graph)

        cfg_dlc['project_path']=str(projectpath)
        cfg_dlc['num_limbs']=int(num_limbs)
        cfg_dlc['partaffinityfield_graph']=PAF_graph
        cfg_dlc['augmentationprobability']=.6
        #cfg_dlc['save_iters']=5
        #cfg_dlc['display_iters']=1


        cfg_dlc['weigh_only_present_joints']=False
        cfg_dlc['pairwise_huber_loss']= True

        cfg_dlc['dataset_type']='multi-animal-imgaug'
        if '152' in modelsuffix:
            cfg_dlc['net_type']='resnet_152'
            cfg_dlc['intermediate_supervision']=True
            cfg_dlc['init_weights']='/usr/local/lib/python3.6/dist-packages/deeplabcut/pose_estimation_tensorflow/models/pretrained/resnet_v1_152.ckpt'
        elif '101' in modelsuffix:
            cfg_dlc['net_type']='resnet_101'
            cfg_dlc['intermediate_supervision']=True
            cfg_dlc['init_weights']='/usr/local/lib/python3.6/dist-packages/deeplabcut/pose_estimation_tensorflow/models/pretrained/resnet_v1_101.ckpt'
        else:
            cfg_dlc['intermediate_supervision']=False
            cfg_dlc['init_weights']='/usr/local/lib/python3.6/dist-packages/deeplabcut/pose_estimation_tensorflow/models/pretrained/resnet_v1_50.ckpt'


        if 'idchannel' in modelsuffix:
            ### TODO: Think about what to do if single exists
            cfg_dlc['num_idchannel']=len(cfg['individuals'])


        cfg_dlc['optimizer']='adam'
        cfg_dlc['batch_size']=16
        cfg_dlc['fliplr']=False
        cfg_dlc['hist_eq'] = False

        cfg_dlc['cropratio']=.6
        cfg_dlc['cropfactor']=.3

        cfg_dlc['rotation']=True #can also be an integer def. -10,10 if true.
        cfg_dlc['covering']=True
        cfg_dlc['motion_blur'] = [["k", 7],["angle", [-90, 90]]]

        cfg_dlc['elastic_transform']=False

        cfg_dlc['scmap_type']='plateau' #gaussian'
        cfg_dlc['pafwidth']=20

        cfg_dlc['save_iters']=si
        cfg_dlc['display_iters']=dp

        cfg_dlc['pairwise_loss_weight']= 0.2
        cfg_dlc['max_input_size']=1500
        cfg_dlc['scale_jitter_lo']= 0.5
        cfg_dlc['scale_jitter_up']=1.2
        cfg_dlc['global_scale']=.8
        cfg_dlc['partaffinityfield_predict']=True


        if cfg_dlc['optimizer'] == "adam":
            cfg_dlc['multi_step']=[[1e-4, 7500], [5*1e-5, 12000], [1e-5, 50000]]
        else:
            cfg_dlc['multi_step']=[[0.005, 10000], [0.01, 100000], [0.02, 430000]]
        #cfg_dlc['init_weights']='' >> set a priori

        deeplabcut.auxiliaryfunctions.write_plainconfig(trainposeconfigfile,cfg_dlc)

        cfg_dlc=deeplabcut.auxiliaryfunctions.read_plainconfig(testposeconfigfile)
        cfg_dlc['project_path']=str(projectpath)
        cfg_dlc['num_limbs']=int(num_limbs)
        cfg_dlc['partaffinityfield_predict']='True'
        cfg_dlc['dataset_type']='multi-animal-imgaug'
        cfg_dlc['partaffinityfield_graph']=PAF_graph #required for inference
        if 'idchannel' in modelsuffix:
            ### TODO: Think about what to do if single exists
            cfg_dlc['num_idchannel']=len(cfg['individuals'])
        deeplabcut.auxiliaryfunctions.write_plainconfig(testposeconfigfile,cfg_dlc)
'''
