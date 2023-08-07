
det_config=/mnt/md0/shaokai/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_quadruped.py
det_checkpoint=/mnt/md0/shaokai/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_quadruped/epoch_80.pth
pose_config=hrnet_pose_config_SA_quadruped.py
pose_checkpoint=''
videoname=black_dog

python demo/top_down_video_demo_with_mmdet.py  $det_config \
       $det_checkpoint \
       $pose_config \
       $pose_checkpoint \
       --video-path ${video_name}.mp4 \
       --out-video-root video_pred_${videoname}/${videoname}.mp4.json \
       --kpt-thr 0.0 &&

python videopseudo2annotation.py --video_result_path video_pred_${videoname}/${videoname}.mp4.json \
       --video_path ${videoname}.mp4 &&
    
python tools/train.py  hrnet_pose_config_SA_quadruped.py --cfg-options data.train.ann_file=annotation_${videoname}/annotations/train.json data.train.img_prefix=annotation_${videoname}/images/ data.val.img_prefix=annotation_${videoname}/images/  data.val.ann_file=annotation_${videoname}/annotations/test.json total_epochs=4  lr_config.warmup_iters=1 optimizer.lr=5e-5 load_from=work_dirs/hrnet_w32_quadruped_256x256/latest.pth   --work-dir ${videoname}_adapted &&

python demo/top_down_video_demo_with_mmdet.py  $det_config \
       $det_checkpoint \
       $pose_config \
       ${videoname}_adapted/latest.pth \
       --video-path ${videoname}.mp4 \
       --out-video-root ${videoname}_adapted_video_predict \
       --kpt-median-filter \
       --kpt-thr 0.6
