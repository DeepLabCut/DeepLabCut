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
"""
Test script for super animal adaptation
"""
import deeplabcut
import os

print (deeplabcut.__file__)
if __name__ == "__main__":


    superanimal_name = "superanimal_topviewmouse"
    config_path = os.path.join(os.getcwd(), "golden_mouse-shaokai-2022-10-19", "config.yaml")
    
    deeplabcut.create_training_dataset(config_path, net_type = "",
                                       superanimal_name = superanimal_name)

    
    deeplabcut.train_network(config_path,
                             maxiters=10,
                             superanimal_name = superanimal_name)

    deeplabcut.analyze_videos(
        config_path, [os.path.join(os.getcwd(), "golden_mouse-shaokai-2022-10-19", "videos", "golden_mouse.mp4")],
    )  # ,videotype='.mp4')
    
    deeplabcut.create_labeled_video(
        config_path,
        [os.path.join(os.getcwd(), "golden_mouse-shaokai-2022-10-19", "videos", "golden_mouse.mp4")],
        "mp4",
        save_frames=False,
    )    
    
