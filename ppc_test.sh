#!/usr/bin/env bash
#SBATCH --partition=research
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:gtx1080:1
#SBATCH --mem=24G
#SBATCH --time=24:0:0
###SBATCH --nodelist=euler21
#SBATCH --exclude=euler05,euler07
#SBATCH -o slurm.%j.%N.out # STDOUT
#SBATCH -e slurm.%j.%N.err # STDERR
#SBATCH --job-name=testmm
###SBATCH --no-requeue

#module load anaconda/mini/23.3.1
module load conda/miniforge/23.1.0
module load nvidia/cuda/11.8.0
bootstrap_conda
conda activate openmmlab2
export CUDA_HOME=/opt/apps/cuda/x86_64/11.8.0/default
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1


######### SUN RGB-D ##############

## PPC Model Testing
EXPERIMENT=work_dir_py/sbr/0.3/joint/first50000_spupdatedX/first50000_spupdated0003/
EXPERIMENT=work_dir_py/sbr/0.048/joint/first50000_spupdated0001/
CHECKPOINT=${EXPERIMENT}/epoch_12.pth

SBR=("clean" "5_50" "5_100" "1_50" "1_100")
for i in "${!SBR[@]}"
do
#DATAPATH=points_min2/0.3/argmax-filtering-sbr/${SBR[$i]}/
DATAPATH=points_min2/0.048/gaussfilter-argmax-filtering-sbr/${SBR[$i]}/
CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/votenet/votenet_8xb16_sunrgbd-3d.py ${CHECKPOINT} --cfg-options \
	work_dir=mapresults/${EXPERIMENT} \
	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
	test_dataloader.dataset.pipeline.0.load_dim=8 \
	test_dataloader.dataset.pipeline.1.transforms.2.num_points=50000 \
	test_dataloader.dataset.pipeline.1.transforms.2.firstk_sampling=True \
	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,4]" \
	model.neighbor_score=0.001 \
	model.filter_index=4 \


#	model.post_sort=4 \
#	model.updated_fps=0.005 \

done



### Baselines
##
### Matched Filtering
##EXPERIMENT=off-the-shelf-20000/
##CHECKPOINT=checkpoints/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth
#EXPERIMENT=work_dir_py/sbr/0.3/joint/point2048/
#CHECKPOINT=${EXPERIMENT}/epoch_12.pth
#
#SBR=("clean" "5_50" "5_100" "1_50" "1_100")
#for i in "${!SBR[@]}"
#do
#DATAPATH=points_min2/0.3/argmax-filtering-sbr/${SBR[$i]}/
#CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/votenet/votenet_8xb16_sunrgbd-3d.py ${CHECKPOINT} --cfg-options \
#	work_dir=mapresults/${EXPERIMENT} \
#	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	test_dataloader.dataset.pipeline.0.load_dim=8 \
#	test_dataloader.dataset.pipeline.1.transforms.2.num_points=2048 \
#	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2]" \
#
#done



#### Matched Filtering + Thresholding
#EXPERIMENT=work_dir_py/sbr/0.3/joint/thresh/thresh50000_11/
#CHECKPOINT=${EXPERIMENT}/epoch_12.pth
#
#SBR=("clean" "5_50" "5_100" "1_50" "1_100")
#for i in "${!SBR[@]}"
#do
#DATAPATH=points_min2/0.3/argmax-filtering-sbr/${SBR[$i]}/
#CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/votenet/votenet_8xb16_sunrgbd-3d.py ${CHECKPOINT} --cfg-options \
#	work_dir=mapresults/${EXPERIMENT} \
#	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	test_dataloader.dataset.pipeline.0.load_dim=8 \
#	test_dataloader.dataset.pipeline.1.transforms.2.num_points=50000 \
#	test_dataloader.dataset.pipeline.1.transforms.2.thresh_sampling=1.1 \
#	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,3]" \
#
#done


######### Kitti ##############
#
### PPC Model Testing 
#EXPERIMENT=work_dir_py/kitti/pvrcnn/3class/joint_1flux/0.3/npupdated00001/
#CHECKPOINT=${EXPERIMENT}/epoch_10.pth
##CHECKPOINT=checkpoints/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth
#
#SBR=("clean" "1_10" "1_20" "1_50" "1_100")
#for i in "${!SBR[@]}"
#do
#DATAPATH=training/points8192_r025_dist10/0.3/argmax-filtering-sbr/${SBR[$i]}/
#CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py ${CHECKPOINT} --cfg-options \
#	work_dir=testing/${EXPERIMENT} \
#	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	test_dataloader.dataset.pipeline.0.load_dim=6 \
#	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,5,4,3]" \
#	model.data_preprocessor.in_channels=4 \
#	model.data_preprocessor.neighbor_score=0.0001 \
#	model.data_preprocessor.filter_index=4 \
#
#
##	model.data_preprocessor.post=True \
#
#done


#### Baselines
#### Matched Filtering
#EXPERIMENT=work_dir_py/kitti/pvrcnn/3class/joint_1flux/0.3/baseline/
#CHECKPOINT=${EXPERIMENT}/epoch_10.pth
#
#SBR=("clean" "5_50" "5_100" "1_50" "1_100")
#SBR=("clean" "1_10" "1_20" "1_50" "1_100")
#for i in "${!SBR[@]}"
#do
#DATAPATH=training/points8192_r025_dist10/0.3/argmax-filtering-sbr/${SBR[$i]}/
#CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py ${CHECKPOINT} --cfg-options \
#	work_dir=mapresults/${EXPERIMENT} \
#	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	test_dataloader.dataset.pipeline.0.load_dim=6 \
#	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,5]" \
#
#
#done


#### Thresholding
#EXPERIMENT=work_dir_py/kitti/pvrcnn/3class/joint/0.3/thresh035_repeat/
#CHECKPOINT=${EXPERIMENT}/epoch_10.pth
#
#SBR=("clean" "5_50" "5_100" "1_50" "1_100")
#for i in "${!SBR[@]}"
#do
#DATAPATH=training/points8192_r025_dist10/0.3/argmax-filtering-sbr/${SBR[$i]}/
#CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py ${CHECKPOINT} --cfg-options \
#	work_dir=mapresults/${EXPERIMENT} \
#	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	test_dataloader.dataset.pipeline.0.load_dim=6 \
#	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,5,4,3]" \
#	test_dataloader.dataset.pipeline.1.thresh_index=5 \
#	test_dataloader.dataset.pipeline.1.threshall_sampling=0.35 \
#	model.data_preprocessor.in_channels=4 \
#
#
#done


