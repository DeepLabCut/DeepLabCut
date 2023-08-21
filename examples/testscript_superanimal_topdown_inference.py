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
Testscript for super animal inference

"""
import deeplabcut
import deeplabcut.superanimal_pytorch
import os


if __name__ == "__main__":

    relative_root = '/mnt/md0/shaokai/integration/DeepLabCut/deeplabcut/superanimal_pytorch/third_party/' 
    
    video_path = os.path.join(relative_root, 'Sample15.mp4')
    det_config_path = os.path.join(relative_root, 'det_config_SA_quadruped.py')
    det_checkpoint = '/mnt/md0/shaokai/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_quadruped/epoch_80.pth'

    #pose_config = os.path.join(relative_root, 'hrnet_pose_config_SA_quadruped.py')
    pose_config = os.path.join(relative_root, 'tokenpose_pose_config_SA_quadruped.py')
    #pose_checkpoint = os.path.join(relative_root, 'work_dirs/hrnet_w32_quadruped_256x256_splitD/latest.pth')
    pose_checkpoint = os.path.join(relative_root, 'work_dirs/Tokenpose_L_quadruped_256x256_data_split_D/latest.pth')
        
    deeplabcut.superanimal_pytorch.topdown_video_inference(
        video_path,
        det_config_path,
        det_checkpoint,
        pose_config, 
        pose_checkpoint,
        video_adapt = False,
        device = 'cpu',
        kpt_threshold = 0.6,
        videotype = '.mp4',
        apply_median_filter = True)
                               
    
    
