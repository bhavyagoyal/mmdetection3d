#!/usr/bin/env bash


######### SUN RGB-D ##############

### PPC Model Testing
#EXPERIMENT=sunrgbd/ppc/
#CHECKPOINT=checkpoints/sunrgbd/ppc.pth
#
#SBR=("clean" "5_50" "5_100" "1_50" "1_100")
#for i in "${!SBR[@]}"
#do
#DATAPATH=points_ppc/${SBR[$i]}/
#CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/votenet/votenet_8xb16_sunrgbd-3d.py ${CHECKPOINT} --cfg-options \
#	work_dir=mapresults/${EXPERIMENT} \
#	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	test_dataloader.dataset.pipeline.0.load_dim=8 \
#	test_dataloader.dataset.pipeline.1.transforms.2.num_points=50000 \
#	test_dataloader.dataset.pipeline.1.transforms.2.firstk_sampling=True \
#	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,4]" \
#	model.data_preprocessor.max_ball_neighbors=64 \
#	model.data_preprocessor.ball_radius=0.2 \
#	model.data_preprocessor.neighbor_score=0.003 \
#	model.data_preprocessor.filter_index=4 \
#	model.data_preprocessor.post=True \
#	model.data_preprocessor.same_sizes=True \
#
#done



### Baselines
### Matched Filtering
#EXPERIMENT=sunrgbd/matchedfiltering/
#CHECKPOINT=checkpoints/sunrgbd/matchedfiltering.pth
##EXPERIMENT=off-the-shelf-20000/
##CHECKPOINT=checkpoints/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth
#
#SBR=("clean" "5_50" "5_100" "1_50" "1_100")
#for i in "${!SBR[@]}"
#do
#DATAPATH=points_ppc/${SBR[$i]}/
#CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/votenet/votenet_8xb16_sunrgbd-3d.py ${CHECKPOINT} --cfg-options \
#	work_dir=mapresults/${EXPERIMENT} \
#	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	test_dataloader.dataset.pipeline.0.load_dim=8 \
#	test_dataloader.dataset.pipeline.1.transforms.2.num_points=2048 \
#	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2]" \
#
#done



#### Matched Filtering + Thresholding
#EXPERIMENT=sunrgbd/thresholding/
#CHECKPOINT=checkpoints/sunrgbd/thresholding.pth
#
#SBR=("clean" "5_50" "5_100" "1_50" "1_100")
#for i in "${!SBR[@]}"
#do
#DATAPATH=points_ppc/${SBR[$i]}/
#CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/votenet/votenet_8xb16_sunrgbd-3d.py ${CHECKPOINT} --cfg-options \
#	work_dir=mapresults/${EXPERIMENT} \
#	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	test_dataloader.dataset.pipeline.0.load_dim=8 \
#	test_dataloader.dataset.pipeline.1.transforms.2.num_points=50000 \
#	test_dataloader.dataset.pipeline.1.transforms.2.thresh_sampling=1.1 \
#	test_dataloader.dataset.pipeline.1.transforms.2.thresh_index=4 \
#	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,3]" \
#
#done


######## Kitti ##############

### PPC Model Testing 
#EXPERIMENT=kitti/ppc/
#CHECKPOINT=checkpoints/kitti/ppc.pth
#
#SBR=("clean" "5_100" "5_250" "5_500" "5_1000")
#for i in "${!SBR[@]}"
#do
#DATAPATH=points_ppc/${SBR[$i]}/
##DATAPATH=training/points2048_r025_dist10/0.3/argmax-filtering-sbr/${SBR[$i]}/
#CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py ${CHECKPOINT} --cfg-options \
#	work_dir=mapresults/${EXPERIMENT} \
#	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	test_dataloader.dataset.pipeline.0.load_dim=6 \
#	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,5,4,3]" \
#	model.data_preprocessor.in_channels=4 \
#	model.data_preprocessor.neighbor_score=0.1 \
#	model.data_preprocessor.filter_index=4 \
#	model.data_preprocessor.ad_neighbor_score=True \
#	model.data_preprocessor.max_ball_neighbors=32 \
#	model.data_preprocessor.ball_radius=0.8 \
#
#
##	model.data_preprocessor.post=True \
#
#done


#### Baselines
#### Matched Filtering
#EXPERIMENT=kitti/matchedfiltering/
#CHECKPOINT=checkpoints/kitti/matchedfiltering.pth
##EXPERIMENT=work_dir_py/kitti/pvrcnn/3class/joint_5fluxunder2048/0.3/baseline/
##EXPERIMENT=work_dir_py/kitti/pvrcnn/3class/off-the-shelf/
##CHECKPOINT=${EXPERIMENT}/epoch_10.pth
##CHECKPOINT=checkpoints/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth
#
#SBR=("clean" "5_100" "5_250" "5_500" "5_1000")
#for i in "${!SBR[@]}"
#do
#DATAPATH=points_ppc/${SBR[$i]}/
##DATAPATH=training/points2048_r025_dist10/0.3/argmax-filtering-sbr/${SBR[$i]}/
#CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py ${CHECKPOINT} --cfg-options \
#	work_dir=mapresults/${EXPERIMENT} \
#	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	test_dataloader.dataset.pipeline.0.load_dim=6 \
#	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,5]" \
#
#done


### Thresholding
#EXPERIMENT=kitti/thresholding/
#CHECKPOINT=checkpoints/kitti/thresholding.pth
#
#SBR=("clean" "5_100" "5_250" "5_500" "5_1000")
#for i in "${!SBR[@]}"
#do
#DATAPATH=points_ppc/${SBR[$i]}/
#CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class_thresholding.py ${CHECKPOINT} --cfg-options \
#	work_dir=mapresults/${EXPERIMENT} \
#	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	test_dataloader.dataset.pipeline.0.load_dim=6 \
#	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,5,4,3]" \
#	test_dataloader.dataset.pipeline.1.thresh_index=5 \
#	test_dataloader.dataset.pipeline.1.threshall_sampling=1.0 \
#	test_dataloader.dataset.pipeline.1.ad_threshall_sampling=True \
#	model.data_preprocessor.in_channels=4 \
#
#
#done


########## SUN RGB-D Fusion models ##############
#
#### PPC Model Testing
#EXPERIMENT=work_dir_py/imvotenet/0.3/jointDP/first50000_spupdated0003_post
#CHECKPOINT=${EXPERIMENT}/epoch_12.pth
#
#SBR=("clean" "5_50" "5_100" "1_50" "1_100")
#for i in "${!SBR[@]}"
#do
#DATAPATH=points_min2/0.3/argmax-filtering-sbr/${SBR[$i]}/
#CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/imvotenet/imvotenet_stage2_8xb16_sunrgbd-3d.py ${CHECKPOINT} --cfg-options \
#	work_dir=mapresults/${EXPERIMENT} \
#	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	test_dataloader.dataset.data_prefix.img='sunrgbd_trainval/image' \
#	test_dataloader.dataset.pipeline.0.load_dim=8 \
#	test_dataloader.dataset.pipeline.3.num_points=50000 \
#	test_dataloader.dataset.pipeline.3.firstk_sampling=True \
#	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,3,4]" \
#	model.data_preprocessor.max_ball_neighbors=64 \
#	model.data_preprocessor.ball_radius=0.2 \
#	model.data_preprocessor.neighbor_score=0.003 \
#	model.data_preprocessor.filter_index=5 \
#	model.data_preprocessor.post=True \
#	model.data_preprocessor.same_sizes=True \
#
#
#done



### Baselines
###
#### Matched Filtering
#EXPERIMENT=work_dir_py/imvotenet/0.3/jointDP/points2048
#CHECKPOINT=${EXPERIMENT}/epoch_12.pth
#
#SBR=("clean" "5_50" "5_100" "1_50" "1_100")
#for i in "${!SBR[@]}"
#do
#DATAPATH=points_min2/0.3/argmax-filtering-sbr/${SBR[$i]}/
#CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/imvotenet/imvotenet_stage2_8xb16_sunrgbd-3d.py ${CHECKPOINT} --cfg-options \
#	work_dir=mapresults/${EXPERIMENT} \
#	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	test_dataloader.dataset.data_prefix.img='sunrgbd_trainval/image' \
#	test_dataloader.dataset.pipeline.0.load_dim=8 \
#	test_dataloader.dataset.pipeline.3.num_points=2048 \
#	test_dataloader.dataset.pipeline.3.firstk_sampling=True \
#	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2]" \
#
#
#done



##### Matched Filtering + Thresholding
#
#EXPERIMENT=work_dir_py/imvotenet/0.3/jointDP/thresh50000_11
#CHECKPOINT=${EXPERIMENT}/epoch_12.pth
#
#SBR=("clean" "5_50" "5_100" "1_50" "1_100")
#for i in "${!SBR[@]}"
#do
#DATAPATH=points_min2/0.3/argmax-filtering-sbr/${SBR[$i]}/
#CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/imvotenet/imvotenet_stage2_8xb16_sunrgbd-3d.py ${CHECKPOINT} --cfg-options \
#	work_dir=mapresults/${EXPERIMENT} \
#	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	test_dataloader.dataset.data_prefix.img='sunrgbd_trainval/image' \
#	test_dataloader.dataset.pipeline.0.load_dim=8 \
#	test_dataloader.dataset.pipeline.3.num_points=50000 \
#	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,3,4]" \
#	test_dataloader.dataset.pipeline.3.thresh_sampling=1.1 \
#	test_dataloader.dataset.pipeline.3.thresh_index=4 \
#
#
#done

