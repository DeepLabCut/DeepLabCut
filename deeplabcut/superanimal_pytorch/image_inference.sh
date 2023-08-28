det_model_name='superquadruped_faster_rcnn'
pose_model_name='superquadruped_hrnetw32'
det_checkpoint='/mnt/md0/shaokai/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_quadruped/epoch_80.pth'
pose_checkpoint='/mnt/md0/shaokai/integration/DeepLabCut/deeplabcut/superanimal_pytorch/third_party/work_dirs/hrnet_w32_quadruped_256x256_splitD/latest.pth'
image_path='/mnt/md0/shaokai/many_dogs.jpeg'

python topdown_superanimal_img_inference.py $det_model_name $pose_model_name $det_checkpoint $pose_checkpoint $image_path
