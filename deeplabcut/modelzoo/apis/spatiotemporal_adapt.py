import deeplabcut
import glob
import os
from deeplabcut.modelzoo.utils import parse_available_supermodels
from pathlib import Path


class SpatiotemporalAdaptation:
    def __init__(
        self,
        video_path,
        supermodel_name,
        scale_list=[],
        videotype="mp4",
        adapt_iterations=1000,
        modelfolder="",
        customized_pose_config="",
        pcutoff=0.1,
        pseudo_threshold=0.1,
    ):

        """
        This class supports video adaptation to a super model.

        Parameters
        ----------
        video_path: string
           The string to the path of the video
        supermodel_name: string
           Currently we support supertopview(LabMice) and superquadruped (quadruped side-view animals)
        scale_list: list
           A list of different resolutions for the spatial pyramid
        videotype: string
           Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed. The default is ``.avi``
        adapt_iterations: int
           Number of iterations for adaptation training. Empirically 1000 is sufficient. Training longer can cause worse performance depending whether there is occlusion in the video
        modelfolder: string, optional
           Because the API does not need a dlc project, the checkpoint and logs go to this temporary model folder, and otherwise model is saved to the current work place
        customized_pose_config: string, optional
           For future support of non modelzoo model
        pcutoff: float, optional
           below the confidence of pcutoff the keypoints won't be shown in the video
        pseudo_threshold: float, optional
           predictions that are under this threshold won't be used for video adaptation. Setting it higher reduces the false positive but might cause removal of true positive

        Examples
        --------

        from  deeplabcut.modelzoo.apis import SpatiotemporalAdaptation
        video_path = '/mnt/md0/shaokai/openfield_video/m3v1mp4.mp4'
        superanimal_name = 'superanimal_mouse_topview'
        videotype = 'mp4'
        >>> adapter = SpatiotemporalAdaptation(video_path,
                                       superanimal_name,
                                       modelfolder = "temp_topview",
                                       videotype = videotype)

        adapter.before_adapt_inference()
        adapter.adaptation_training()
        adapter.after_adapt_inference()    
        

        """
        supermodels = parse_available_supermodels()
        if supermodel_name not in supermodels:
            raise ValueError(f"`supermodel_name` should be one of: {', '.join(supermodels)}.")

        self.video_path = video_path
        self.supermodel_name = supermodel_name
        self.scale_list = scale_list
        self.videotype = videotype
        vname = str(Path(self.video_path).stem)
        self.adapt_modelprefix = vname + "_video_adaptation"
        self.adapt_iterations = adapt_iterations
        self.modelfolder = modelfolder
        self.customized_pose_config = customized_pose_config
        self.pcutoff = pcutoff
        self.pseudo_threshold = pseudo_threshold

        if modelfolder != "":
            os.makedirs(modelfolder, exist_ok=True)

            dlc_root_path = os.sep.join(deeplabcut.__file__.split(os.sep)[:-1])

            self.customized_pose_config = os.path.join(
                dlc_root_path,
                "pose_estimation_tensorflow",
                "superanimal_configs",
                supermodels[self.supermodel_name],
            )

        if customized_pose_config != "":
        # if it's an old modelzoo model, this is also required
            self.customized_pose_config = customized_pose_config 

        
    def before_adapt_inference(self):

        # save frames have to be on 
        deeplabcut.video_inference_superanimal([self.video_path],
                                               self.supermodel_name,
                                               videotype = self.videotype,
                                               scale_list = self.scale_list,
                                               init_weights = self.init_weights,
                                               customized_test_config = self.customized_pose_config)


        deeplabcut.create_labeled_video('',
                                        [self.video_path],
                                        videotype = self.videotype,
                                        filtered = False,
                                        init_weights = self.init_weights,
                                        draw_skeleton = True,
                                        superanimal_name = self.supermodel_name,
                                        pcutoff = self.pcutoff)
            

        deeplabcut.create_labeled_video(
            "",
            [self.video_path],
            videotype=self.videotype,
            filtered=False,
            init_weights=self.init_weights,
            draw_skeleton=True,
            superanimal_name=self.supermodel_name,
            pcutoff=self.pcutoff,
            **kwargs
        )

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
              video_path = self.video_path,
              pseudo_threshold = 0.3)        
        
    def adaptation_training(self):
        """
        There should be two choices, either taking a config, with is then assuming there is a DLC project.
        Or we make up a fake one, then we use a light way convention to do adaptation
        """

        # looking for the pseudo label path
        DLCscorer = "DLC_" + Path(self.init_weights).stem
        vname = str(Path(self.video_path).stem)
        video_root = Path(self.video_path).parent

        _, pseudo_label_path, _, _ = deeplabcut.auxiliaryfunctions.load_analyzed_data(
            video_root, vname, DLCscorer, False, ""
        )

        self.train_without_project(pseudo_label_path)

    def after_adapt_inference(self, **kwargs):

        pattern = os.path.join(
            self.modelfolder,  f"snapshot-{self.adapt_iterations}.index"
        )
        ref_proj_config_path = ""

        files = glob.glob(pattern)

        if not len(files):
            raise ValueError("Weights were not found.")

        adapt_weights = files[0].replace(".index", "")

        # spatial pyramid is not for adapted model

        deeplabcut.video_inference_superanimal(
            [self.video_path],
            self.supermodel_name,
            videotype=self.videotype,
            init_weights=adapt_weights,
            customized_test_config=self.customized_pose_config,
        )

        deeplabcut.create_labeled_video(
            ref_proj_config_path,
            [self.video_path],
            videotype=self.videotype,
            filtered=False,
            init_weights=adapt_weights,
            draw_skeleton=True,
            superanimal_name=self.supermodel_name,
            pcutoff=self.pcutoff,
            **kwargs
        )
