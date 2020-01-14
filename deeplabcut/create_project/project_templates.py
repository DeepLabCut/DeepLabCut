"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

def create_config_template(multianimal=False):
    """
    Creates a template for config.yaml file. This specific order is preserved while saving as yaml file.
    """
    import ruamel.yaml
    if multianimal:
        yaml_str = """\
    # Project definitions (do not edit)
        Task:
        scorer:
        date:
        multianimalproject:
        \n
    # Project path (change when moving around)
        project_path:
        \n
    # Annotation data set configuration (and individual video cropping parameters)
        video_sets:
        individuals:
        uniquebodyparts:
        multianimalbodyparts:
        skeleton:
        bodyparts:
        start:
        stop:
        numframes2pick:
        \n
    # Plotting configuration
        skeleton_color:
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
    else:
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
        skeleton:
        skeleton_color:
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

def create_config_template_3d():
    """
    Creates a template for config.yaml file for 3d project. This specific order is preserved while saving as yaml file.
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
# Plotting configuration
    skeleton: # Note that the pairs must be defined, as you want them linked!
    skeleton_color:
    pcutoff:
    colormap:
    dotsize:
    alphaValue:
    markerType:
    markerColor:
    \n
# Number of cameras, camera names, path of the config files, shuffle index and trainingsetindex used to analyze videos:
    num_cameras:
    camera_names:
    scorername_3d: # Enter the scorer name for the 3D output
    """
    ruamelFile_3d = ruamel.yaml.YAML()
    cfg_file_3d = ruamelFile_3d.load(yaml_str)
    return(cfg_file_3d,ruamelFile_3d)
