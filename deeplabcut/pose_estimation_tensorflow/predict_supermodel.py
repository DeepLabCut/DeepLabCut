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
from pathlib import Path
from deeplabcut.modelzoo.api import SpatiotemporalAdaptation


def video_inference_superanimal(
    videos,
    superanimal_name,
    scale_list=[],
    videotype=".mp4",
    video_adapt=False,
    plot_trajectories=True,
    pcutoff=0.1,
):
    """
    Makes prediction based on a super animal model. Note right now we only support single animal video inference

    The index of the trained network is specified by parameters in the config file (in particular the variable 'snapshotindex')

    Output: The labels are stored as MultiIndex Pandas Array, which contains the name of the network, body part name, (x, y) label position \n
            in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient Hierarchical Data Format (HDF) \n
            in the same directory, where the video is stored.

    Parameters
    ----------
    videos: list
        A list of strings containing the full paths to videos for analysis or a path to the directory, where all the videos with same extension are stored.

    superanimal_name: str
        The name of the superanimal model. We currently only support supertopview and superquadruped
    scale_list: list
        A list of int containing the target height of the multi scale test time augmentation. By default it uses the original size. Users are advised to try a wide range of scale list when the super model does not give reasonable results

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed. The default is ``.avi``

    video_adapt: bool, optional
        Set True if you want to apply video adaptation to make the resulted video less jittering and better. However, adaptation training takes more time than usual video inference

    plot_trajectories: bool, optional (default=True)
        By default, plot the trajectories of various body parts across the video.

    pcutoff: float, optional
        Keypoints confidence that are under pcutoff will not be shown in the resulted video

    Given a list of scales for spatial pyramid, i.e. [600, 700]

    scale_list = range(600,800,100)

    superanimal_name = 'superanimal_topviewmouse'
    videotype = 'mp4'
    scale_list = [200, 300, 400]
    deeplabcut.video_inference_superanimal(
         video,
         superanimal_name,
         videotype = '.avi',
         scale_list = scale_list,
    )
    >>>


    """
    from deeplabcut.utils.auxiliaryfunctions import get_deeplabcut_path

    for video in videos:
        vname = Path(video).stem
        dlcparent_path = get_deeplabcut_path()
        modelfolder = (
            Path(dlcparent_path)
            / "pose_estimation_tensorflow"
            / "models"
            / "pretrained"
            / (superanimal_name + "_" + vname + "_weights")
        )
        adapter = SpatiotemporalAdaptation(
            video,
            superanimal_name,
            modelfolder=modelfolder,
            videotype=videotype,
            scale_list=scale_list,
        )

        if not video_adapt:
            adapter.before_adapt_inference(make_video=True, pcutoff=pcutoff)
        else:
            adapter.before_adapt_inference(make_video=False)
            adapter.adaptation_training()
            adapter.after_adapt_inference(
                pcutoff=pcutoff,
                plot_trajectories=plot_trajectories,
            )
