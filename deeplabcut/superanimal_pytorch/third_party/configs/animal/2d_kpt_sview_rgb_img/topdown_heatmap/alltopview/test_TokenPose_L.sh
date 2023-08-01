pretrain=T/best_AP_epoch_186.pth

#test_json=data/all_topview_70-D-O/annotations/3CSI_test.json
#test_json=data/all_topview_70-D-O/annotations/ChanLab_test.json
#test_json=data/all_topview_70-D-O/annotations/FST_test.json
#test_json=data/all_topview_70-D-O/annotations/MackenzieMausHaus_test.json
#test_json=data/all_topview_70-D-O/annotations/TwoWhiteMice_GoldenLab_test.json
#test_json=data/all_topview_70-D-O/annotations/BM_test.json
#test_json=data/all_topview_70-D-O/annotations/EPM_test.json
#test_json=data/all_topview_70-D-O/annotations/LDB_test.json
test_json=data/all_topview_70-D-O/annotations/OFT_test.json

python tools/test.py configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/alltopview/hrnet_w32_alltopview_256x256.py $pretrain  --cfg-options data.test.data_cfg.use_gt_bbox=True   data.test.ann_file=$test_json

