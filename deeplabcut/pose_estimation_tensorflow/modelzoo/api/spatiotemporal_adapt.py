#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import glob
import os
import io
from pathlib import Path
import yaml
from deeplabcut.pose_estimation_tensorflow.modelzoo.api.superanimal_inference import (
    video_inference,
)
from deeplabcut.utils.auxiliaryfunctions import (
    get_deeplabcut_path,
    load_analyzed_data,
    read_config,
)
from deeplabcut.utils.make_labeled_video import create_labeled_video
from deeplabcut.utils.plotting import _plot_trajectories


class SpatiotemporalAdaptation:
    def __init__(
        self,
        video_path,
        supermodel_name,
        scale_list=None,
        videotype="mp4",
        adapt_iterations=1000,
        modelfolder="",
        customized_pose_config="",
        init_weights="",
    ):
        """
        This class supports video adaptation to a super model.

        Parameters
        ----------
        video_path: string
           The string to the path of the video
        init_weights: string
           The path to a superanimal model's checkpoint
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

        Examples
        --------

        from  deeplabcut.modelzoo.apis import SpatiotemporalAdaptation
        video_path = '/mnt/md0/shaokai/openfield_video/m3v1mp4.mp4'
        superanimal_name = 'superanimal_topviewmouse'
        videotype = 'mp4'
        >>> adapter = SpatiotemporalAdaptation(video_path,
                                       superanimal_name,
                                       modelfolder = "temp_topview",
                                       videotype = videotype)

        adapter.before_adapt_inference()
        adapter.adaptation_training()
        adapter.after_adapt_inference()


        """
        if scale_list is None:
            scale_list = []

        self.video_path = video_path
        self.supermodel_name = supermodel_name
        self.scale_list = scale_list
        self.videotype = videotype
        vname = str(Path(self.video_path).stem)
        self.adapt_modelprefix = vname + "_video_adaptation"
        self.adapt_iterations = adapt_iterations
        self.modelfolder = modelfolder
        self.init_weights = init_weights

        project_name = "_".join(supermodel_name.split("_")[:-1])
        model_name = supermodel_name.split("_")[-1]
        self.project_name = project_name
        self.model_name = model_name

        if not customized_pose_config:
            dlc_root_path = get_deeplabcut_path()

            project_config = read_config(
                os.path.join(
                    dlc_root_path, "modelzoo", "project_configs", f"{project_name}.yaml"
                )
            )

            model_config = read_config(
                os.path.join(
                    dlc_root_path, "modelzoo", "model_configs", f"{model_name}.yaml"
                )
            )

            joints = [i for i in range(len(project_config["bodyparts"]))]
            num_joints = len(joints)
            model_config["all_joints"] = joints
            model_config["all_joints_names"] = project_config["bodyparts"]
            model_config["num_joints"] = num_joints
            model_config["num_limbs"] = int((num_joints * (num_joints - 1)) // 2)
            self.customized_pose_config = {**project_config, **model_config}
        else:
            self.customized_pose_config = customized_pose_config

    def before_adapt_inference(self, make_video=False, **kwargs):
        if self.init_weights != "":
            print("using customized weights", self.init_weights)
            _, datafiles = video_inference(
                [self.video_path],
                self.project_name,
                self.model_name,
                videotype=self.videotype,
                scale_list=self.scale_list,
                init_weights=self.init_weights,
                customized_test_config=self.customized_pose_config,
            )
        else:
            self.init_weights, datafiles = video_inference(
                [self.video_path],
                self.project_name,
                self.model_name,
                videotype=self.videotype,
                scale_list=self.scale_list,
                customized_test_config=self.customized_pose_config,
            )
        if kwargs.pop("plot_trajectories", True):
            if len(datafiles) == 0:
                print("No data files found for plotting trajectory")
            else:
                _plot_trajectories(datafiles[0])

        if make_video:
            create_labeled_video(
                "",
                [self.video_path],
                videotype=self.videotype,
                filtered=False,
                init_weights=self.init_weights,
                draw_skeleton=True,
                superanimal_name=self.supermodel_name,
                **kwargs,
            )

    def train_without_project(self, pseudo_label_path, **kwargs):
        from deeplabcut.pose_estimation_tensorflow.core.train_multianimal import train

        displayiters = kwargs.pop("displayiters", 500)
        saveiters = kwargs.pop("saveiters", 1000)
        self.adapt_iterations = kwargs.pop("adapt_iterations", self.adapt_iterations)

        train(
            self.customized_pose_config,
            displayiters=displayiters,
            saveiters=saveiters,
            maxiters=self.adapt_iterations,
            modelfolder=self.modelfolder,
            init_weights=self.init_weights,
            pseudo_labels=pseudo_label_path,
            video_path=self.video_path,
            superanimal=self.supermodel_name,
            **kwargs,
        )

    def adaptation_training(self, displayiters=500, saveiters=1000, **kwargs):
        """
        There should be two choices, either taking a config, with is then assuming there is a DLC project.
        Or we make up a fake one, then we use a light way convention to do adaptation
        """

        # looking for the pseudo label path
        DLCscorer = "DLC_" + Path(self.init_weights).stem
        vname = str(Path(self.video_path).stem)
        video_root = Path(self.video_path).parent

        _, pseudo_label_path, _, _ = load_analyzed_data(
            video_root, vname, DLCscorer, False, ""
        )
        if self.modelfolder != "":
            os.makedirs(self.modelfolder, exist_ok=True)

        self.adapt_iterations = kwargs.get("adapt_iterations", self.adapt_iterations)


        self.train_without_project(
            pseudo_label_path,
            displayiters=displayiters,
            saveiters=saveiters,
            **kwargs,
        )

    def after_adapt_inference(self, **kwargs):
        pattern = os.path.join(
            self.modelfolder, f"snapshot-{self.adapt_iterations}.index"
        )
        ref_proj_config_path = ""

        files = glob.glob(pattern)

        if not len(files):
            raise ValueError("Weights were not found.")

        adapt_weights = files[0].replace(".index", "")

        # spatial pyramid is not for adapted model

        scale_list = kwargs.pop("scale_list", [])

        # spatial pyramid can still be useful for reducing jittering and quantization error

        _, datafiles = video_inference(
            [self.video_path],
            self.project_name,
            self.model_name,
            videotype=self.videotype,
            init_weights=adapt_weights,
            scale_list=scale_list,
            customized_test_config=self.customized_pose_config,
        )

        if kwargs.pop("plot_trajectories", True):
            _plot_trajectories(datafiles[0])

        create_labeled_video(
            ref_proj_config_path,
            [self.video_path],
            videotype=self.videotype,
            filtered=False,
            init_weights=adapt_weights,
            draw_skeleton=True,
            superanimal_name=self.supermodel_name,
            **kwargs,
        )
