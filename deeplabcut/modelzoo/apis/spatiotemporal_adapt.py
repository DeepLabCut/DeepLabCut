import deeplabcut
import os
import sys
from pathlib import Path
import shutil
import glob

class SpatiotemporalAdaptation:
    '''
    Minimal example how to perform 
    Drafting what a full pipeline will look like in the future
    '''

    def __init__(self,
                 video_path,
                 init_weights,
                 supermodel_name,
                 ref_proj_root = "",                 
                 scale_list = [],
                 videotype = 'mp4',
                 adapt_iterations = 1000,
                 modelfolder = "",
                 customized_pose_config = ''
                 ):        
        
        assert supermodel_name in ['superquadruped', 'supertopview']
        # TODO, make it accept a list. But I haven't tested multi video adaptation
        self.video_path = video_path
        # init weights is required for any model zoo project
        self.init_weights = init_weights
        # names need to be updated when there is a new model
        self.supermodel_name = supermodel_name
        # ref_proj_root if there is an existing dlc project
        self.ref_proj_root = ref_proj_root
        # scale list is required for spatial pyramid
        self.scale_list = scale_list
        # need video type for video inference
        self.videotype = videotype
        vname = str(Path(self.video_path).stem)
        self.adapt_modelprefix = vname + '_video_adaptation'
        # for how long video adaptation is trained
        self.adapt_iterations = adapt_iterations
        # if there is no project, use this as a light weight project
        self.modelfolder = modelfolder

        if modelfolder !="":
            os.makedirs(modelfolder, exist_ok = True)

            dlc_root_path = os.sep.join(deeplabcut.__file__.split(os.sep)[:-1])

            name_dict = {'supertopview': 'supertopview.yaml',
                         'superquadruped':'superquadruped.yaml'}

            self.customized_pose_config = os.path.join(
                dlc_root_path,
                'pose_estimation_tensorflow',
                'superanimal_configs',
                name_dict[self.supermodel_name]
            )

        if customized_pose_config != "":
        # if it's an old modelzoo model, this is also required
            self.customized_pose_config = customized_pose_config 

        
    def before_adapt_inference(self):

        # save frames have to be on 
        deeplabcut.video_inference_superanimal([self.video_path],
                                               self.supermodel_name,
                                               videotype = self.videotype,
                                               save_frames = True,
                                               scale_list = self.scale_list,
                                               init_weights = self.init_weights,
                                               customized_test_config = self.customized_pose_config)


        deeplabcut.create_labeled_video('',
                                        [self.video_path],
                                        videotype = self.videotype,
                                        filtered = False,
                                        init_weights = self.init_weights,
                                        draw_skeleton = True,
                                        superanimal_name = self.supermodel_name)
        
        
    def train_from_existing_project(self,pseudo_label_path):
        # just a naming convention for the adaptation model
        vname = str(Path(self.video_path).stem)
        
        modelprefix = vname + '_video_adaptation'
        config_path = os.path.join(self.ref_proj_root, 'config.yaml')
        
        if os.path.exists(os.path.join(self.ref_proj_root, 'template-dlc-models')):
            template_folder = 'template-dlc-models'
        else:
            template_folder = 'dlc-models'
                
        shutil.copytree(os.path.join(self.ref_proj_root,
                                     template_folder),
                        os.path.join(
                            self.ref_proj_root,
                            modelprefix,
                            'dlc-models'
                        ),
                        dirs_exist_ok = True
                        )

        deeplabcut.train_network(config_path,
                                 saveiters = 1000,
                                 init_weights = self.init_weights,
                                 load_pseudo_label = pseudo_label_path,
                                 pseudo_threshold = 0.3,
                                 max_snapshots_to_keep = 100,
                                 modelprefix = modelprefix,
                                 maxiters = self.adapt_iterations)        

    def train_without_project(self, pseudo_label_path):

        from deeplabcut.pose_estimation_tensorflow.core.train_multianimal import train

        print ('self.customized_pose_config', self.customized_pose_config)
        
        train(self.customized_pose_config,
              500, # displayiters
              1000, # saveiters,
              self.adapt_iterations, # maxiters
              modelfolder = self.modelfolder,
              init_weights = self.init_weights,
              load_pseudo_label = pseudo_label_path,
              pseudo_threshold = 0.3)
        
        
    def adaptation_training(self):
        '''
        There should be two choices, either taking a config, with is then assuming there is a DLC project.
        Or we make up a fake one, then we use a light way convention to do adaptation
        '''
        
        # looking for the pseudo label path
        DLCscorer = 'DLC_' + Path(self.init_weights).stem
        vname = str(Path(self.video_path).stem)
        video_root = Path(self.video_path).parent
        
        _, pseudo_label_path, _,  _ = deeplabcut.auxiliaryfunctions.load_analyzed_data(
            video_root, vname, DLCscorer, False, '')    

        if self.ref_proj_root == "":
            self.train_without_project(pseudo_label_path)
        else:
            self.train_from_existing_project(pseudo_label_path)
            
            
    def after_adapt_inference(self):


        if self.ref_proj_root !="":
            ref_proj_config_path = os.path.join(self.ref_proj_root,
                                                'config.yaml')


            deeplabcut.auxiliaryfunctions.read_config(ref_proj_config_path)

            # because the adaptation is not really attached to the dataset, we can't se aux function to get shuffle or something. In this case the model is just saved to the modelprefix so we just use glob to grap it

            pattern = os.path.join(self.ref_proj_root, self.adapt_modelprefix, f'*/*/*/train/snapshot-{self.adapt_iterations}.index')

        else:

            # make it empty
                        
            pattern = os.path.join(self.modelfolder,  f'*/snapshot-{self.adapt_iterations}.index')

            ref_proj_config_path = ''
                                   
        files = glob.glob(pattern)
        
        assert len(files) > 0

        adapt_weights = files[0].replace('.index','')

        # spatial pyramid is not for adapted model
                
        deeplabcut.video_inference_superanimal(
            [self.video_path],
            self.supermodel_name,
            videotype = self.videotype,
            init_weights = adapt_weights,
            customized_test_config = self.customized_pose_config)
                
        deeplabcut.create_labeled_video(ref_proj_config_path,
                                        [self.video_path],
                                        videotype = self.videotype,
                                        filtered = False,
                                        init_weights = adapt_weights,
                                        draw_skeleton = True,
                                        superanimal_name = self.supermodel_name)
        
