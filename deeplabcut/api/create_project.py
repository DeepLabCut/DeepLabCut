#
# DeepLabCut Toolbox (deeplabcut.org)
# (c) A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Public API for DeepLabCut project creation.

Currently covers modelzoo / pretrained project creation via
``create_pretrained_project``. Regular project creation
(``create_new_project``) will be added in a follow-up.
"""

from __future__ import annotations

from collections.abc import Sequence

from deeplabcut.api._tf_routing import with_tensorflow_fallback
from deeplabcut.core.deprecation import renamed_parameter
from deeplabcut.core.engine import Engine
from deeplabcut.create_project.modelzoo import create_pretrained_project as _create_pretrained_project_impl


@with_tensorflow_fallback(
    when=lambda *a, **kw: kw.get("engine") == Engine.TF,
    tensorflow_module="deeplabcut.tensorflow_compat.create_project",
    tensorflow_name="_tf_create_pretrained_project",
)
@renamed_parameter(old="videotype", new="video_extensions", since="3.0.0")
def create_pretrained_project(
    project: str,
    experimenter: str,
    videos: list[str],
    model: str | None = None,
    working_directory: str | None = None,
    copy_videos: bool = False,
    video_extensions: str | Sequence[str] | None = None,
    analyzevideo: bool = True,
    filtered: bool = True,
    createlabeledvideo: bool = True,
    trainFraction: float | None = None,
    engine: Engine = Engine.PYTORCH,
    multi_animal: bool = False,
    individuals: list[str] | None = None,
    net_name: str | None = None,
    detector_name: str | None = None,
):
    r"""Creates a new project directory, sub-directories and a basic configuration file.
    Change its parameters to your projects need.

    The project will also be initialized with a pre-trained model from the DeepLabCut model zoo!

    http://modelzoo.deeplabcut.org

    Args:
        project (string): String containing the name of the project.
        experimenter (string): String containing the name of the experimenter.
        model (string | None, optional): The model / dataset to use as basis for the
            project. If None, the default model / dataset for the selected engine will
            be used. Defaults to None.
        videos (list[string]): A list of string containing the full paths of the videos
            to include in the project.
        working_directory (string, optional): The directory where the project will be
            created. If None - the current working directory will be used. Defaults to
            None.
        copy_videos (bool, optional): If this is set to True, the videos are copied to
            the ``videos`` directory. If it is False, symlink of the videos are copied
            to the project/videos directory.
            Note: on Windows, True is necessary when not running in Administrator mode.
            The same applies whenever symlinks are disabled or unsupported.
            Defaults to False.
        analyzevideo (bool, optional): If true, then the video is analyzed and a labeled
            video is created. If false, then only the project will be created and the
            weights downloaded.
        filtered (bool, optional): Indicates if filtered pose data output should be
            plotted rather than frame-by-frame predictions. Filtered version can be
            calculated with deeplabcut.filterpredictions(). Defaults to True.
        createlabeledvideo (bool, optional): Specifies if a labeled video needs to be
            created. Defaults to True.
        trainFraction (float | None, optional): Fraction that will be used in
            dlc-model/trainingset folder name. If None - default value (0.95) from new
            projects will be used. Defaults to None.
        engine (Engine, optional): Engine on which the pretrained weights are based.
            Defaults to Engine.PYTORCH.
        multi_animal (bool, optional): Specifies if the project is single or
            multi-animal. Implemented only for Pytorch-based models. Defaults to False.
        individuals (list[str] | None, optional): Only if multianimal is True. Defines
            the names of the individuals. Defaults to None.
        net_name (str | None, optional): Valid only if using Pytorch engine. Name of the
            pose model on which the superanimal dataset has been trained on. If None -
            "hrnet_w32" will be used as default. Defaults to None.
        detector_name (str | None, optional): Valid only if using Pytorch engine. Name
            of the detector model on which the superanimal dataset has been trained on.
            If None - "fasterrcnn_resnet50_fpn_v2" will be used as default. Defaults to
            None.

    Examples:
        Linux/MacOs loading full_human model and analyzing video /homosapiens1.avi:

            deeplabcut.create_pretrained_project(
                "humanstrokestudy", "Linus", ["/data/videos/homosapiens1.avi"], copy_videos=False
            )

        Loading full_cat model and analyzing video "felixfeliscatus3.avi":

            deeplabcut.create_pretrained_project(
                "humanstrokestudy", "Linus", ["/data/videos/felixfeliscatus3.avi"], model="full_cat", engine=Engine.TF
            )

        Windows:

            deeplabcut.create_pretrained_project(
                "humanstrokestudy",
                "Bill",
                [r"C:\yourusername\rig-95\Videos\reachingvideo1.avi"],
                r"C:\yourusername\analysis\project",
                copy_videos=True,
            )

        On Windows, paths should be formatted as ``r`"C:\"`` or ``"C:\\"`` (i.e. a double backslash).
    """
    return _create_pretrained_project_impl(
        project=project,
        experimenter=experimenter,
        videos=videos,
        model=model,
        working_directory=working_directory,
        copy_videos=copy_videos,
        video_extensions=video_extensions,
        analyzevideo=analyzevideo,
        filtered=filtered,
        createlabeledvideo=createlabeledvideo,
        trainFraction=trainFraction,
        multi_animal=multi_animal,
        individuals=individuals,
        net_name=net_name,
        detector_name=detector_name,
    )
