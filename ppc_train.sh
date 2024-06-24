#!/usr/bin/env bash
#SBATCH --partition=research
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=36G
#SBATCH --time=48:0:0
###SBATCH --nodelist=euler28
#SBATCH --exclude=euler05,euler01
#SBATCH -o slurm.%j.%N.out # STDOUT
#SBATCH -e slurm.%j.%N.err # STDERR
#SBATCH --job-name=mmdt3d
###SBATCH --no-requeue


# Environment Setup

#conda 23.3.1, cuda11.8
#conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
#install mmcv without mim, using pip instead
#pip install "mmcv>=2.0.1" -f  https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
#pip install mmdet==3.1.0
#git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
## "-b dev-1.x" means checkout to the `dev-1.x` branch.
#cd mmdetection3d
#pip install -v -e .


# Load Environment
#module load anaconda/mini/23.3.1
module load conda/miniforge/23.1.0
module load nvidia/cuda/11.8.0
bootstrap_conda
conda activate openmmlab
export CUDA_HOME=/opt/apps/cuda/x86_64/11.8.0/default

GPUS=2
PORTUSED=$(( $RANDOM + 10000 ))


# PPC Model Training
EXPERIMENT=work_dir_py/sbr/0.3/joint/first50000_spupdated0004_post_newfps001
DATAPATH=points_min2/0.3/argmax-filtering-sbr/
PORT=${PORTUSED} ./tools/dist_train.sh configs/votenet/votenet_8xb16_sunrgbd-3d.py ${GPUS} --auto-scale-lr --resume --cfg-options \
	train_dataloader.dataset.dataset.data_prefix.pts=${DATAPATH} \
	val_dataloader.dataset.data_prefix.pts=${DATAPATH} \
	train_dataloader.batch_size=8 \
	work_dir=${EXPERIMENT} \
	default_hooks.checkpoint.interval=1 \
	train_dataloader.dataset.dataset.pipeline.4.num_points=50000 \
	val_dataloader.dataset.pipeline.1.transforms.2.num_points=50000 \
	train_dataloader.dataset.dataset.pipeline.0.load_dim=8 \
	val_dataloader.dataset.pipeline.0.load_dim=8 \
	train_dataloader.dataset.dataset.pipeline.0.use_dim="[0,1,2,4,5,6,7]" \
	val_dataloader.dataset.pipeline.0.use_dim="[0,1,2,4,5,6,7]" \
	train_dataloader.dataset.dataset.ann_file='sunrgbd_infos_train_1_100_1_50_5_100_5_50_clean.pkl' \
	val_dataloader.dataset.ann_file='sunrgbd_infos_val_1_100_1_50_5_100_5_50_clean.pkl' \
	param_scheduler.0.end=12 \
	param_scheduler.0.milestones=[8,10] \
	train_cfg.max_epochs=12 \
	train_dataloader.dataset.dataset.pipeline.4.firstk_sampling=True \
	val_dataloader.dataset.pipeline.1.transforms.2.firstk_sampling=True \
	model.neighbor_score=0.004 \
	model.filter_index=4 \
	model.post_sort=4 \
	model.updated_fps=0.01 \


	#train_dataloader.dataset.dataset.pipeline.0.unit_probabilities=3 \
	#val_dataloader.dataset.pipeline.0.unit_probabilities=3 \


## Baselines 
### Matched Filtering
#EXPERIMENT=work_dir_py/sbr/0.3/joint/point2048
#DATAPATH=points_min2/0.3/argmax-filtering-sbr/
#PORT=${PORTUSED} ./tools/dist_train.sh configs/votenet/votenet_8xb16_sunrgbd-3d.py ${GPUS} --auto-scale-lr --resume --cfg-options \
#	train_dataloader.dataset.dataset.data_prefix.pts=${DATAPATH} \
#	val_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	train_dataloader.batch_size=8 \
#	work_dir=${EXPERIMENT} \
#	default_hooks.checkpoint.interval=1 \
#	train_dataloader.dataset.dataset.pipeline.4.num_points=2048 \
#	val_dataloader.dataset.pipeline.1.transforms.2.num_points=2048 \
#	train_dataloader.dataset.dataset.pipeline.0.load_dim=8 \
#	val_dataloader.dataset.pipeline.0.load_dim=8 \
#	train_dataloader.dataset.dataset.pipeline.0.use_dim="[0,1,2]" \
#	val_dataloader.dataset.pipeline.0.use_dim="[0,1,2]" \
#	train_dataloader.dataset.dataset.ann_file='sunrgbd_infos_train_1_100_1_50_5_100_5_50_clean.pkl' \
#	val_dataloader.dataset.ann_file='sunrgbd_infos_val_1_100_1_50_5_100_5_50_clean.pkl' \
#	param_scheduler.0.end=12 \
#	param_scheduler.0.milestones=[8,10] \
#	train_cfg.max_epochs=12 \



#### Matched Filtering + Thresholding
#EXPERIMENT=work_dir_py/sbr/0.3/joint/thresh50000_10
#DATAPATH=points_min2/0.3/argmax-filtering-sbr/
#PORT=${PORTUSED} ./tools/dist_train.sh configs/votenet/votenet_8xb16_sunrgbd-3d.py ${GPUS} --auto-scale-lr --resume --cfg-options \
#	train_dataloader.dataset.dataset.data_prefix.pts=${DATAPATH} \
#	val_dataloader.dataset.data_prefix.pts=${DATAPATH} \
#	train_dataloader.batch_size=8 \
#	work_dir=${EXPERIMENT} \
#	default_hooks.checkpoint.interval=1 \
#	train_dataloader.dataset.dataset.pipeline.4.num_points=50000 \
#	val_dataloader.dataset.pipeline.1.transforms.2.num_points=50000 \
#	train_dataloader.dataset.dataset.pipeline.0.load_dim=8 \
#	val_dataloader.dataset.pipeline.0.load_dim=8 \
#	train_dataloader.dataset.dataset.pipeline.0.use_dim="[0,1,2,3]" \
#	val_dataloader.dataset.pipeline.0.use_dim="[0,1,2,3]" \
#	train_dataloader.dataset.dataset.ann_file='sunrgbd_infos_train_1_100_1_50_5_100_5_50_clean.pkl' \
#	val_dataloader.dataset.ann_file='sunrgbd_infos_val_1_100_1_50_5_100_5_50_clean.pkl' \
#	param_scheduler.0.end=12 \
#	param_scheduler.0.milestones=[8,10] \
#	train_cfg.max_epochs=12 \
#	val_dataloader.dataset.pipeline.1.transforms.2.thresh_sampling=1.0 \
#	train_dataloader.dataset.dataset.pipeline.4.thresh_sampling=1.0 \
#
