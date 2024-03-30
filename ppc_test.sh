#!/usr/bin/env bash
#SBATCH --partition=research
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:0:0
###SBATCH --nodelist=euler19
###SBATCH --exclude=euler05,euler24,euler25,euler26,euler27
#SBATCH -o slurm.%j.%N.out # STDOUT
#SBATCH -e slurm.%j.%N.err # STDERR
#SBATCH --job-name=testmmdt3d
###SBATCH --no-requeue

module load anaconda/mini/23.3.1
module load nvidia/cuda/11.8.0
bootstrap_conda
conda activate openmmlab


## PPC Model Testing
EXPERIMENT=work_dir_py/sbr/0.3/joint/first50000_spupdated0004_post_newfps001/
CHECKPOINT=${EXPERIMENT}/epoch_12.pth

SBR=("clean" "5_50" "5_100" "1_50" "1_100")
for i in "${!SBR[@]}"
do
DATAPATH=points_min2/0.3/argmax-filtering-sbr/${SBR[$i]}/
CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/votenet/votenet_8xb16_sunrgbd-3d.py ${CHECKPOINT} --cfg-options \
	work_dir=mapresults/${EXPERIMENT} \
	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
	test_dataloader.dataset.pipeline.0.load_dim=8 \
	test_dataloader.dataset.pipeline.1.transforms.2.num_points=50000 \
	test_dataloader.dataset.pipeline.1.transforms.2.firstk_sampling=True \
	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,4]" \
	model.neighbor_score=0.004 \
	model.filter_index=4 \
	model.post_sort=4 \
	model.updated_fps=0.01 \
done

## Baselines
#
## Matched Filtering
EXPERIMENT=work_dir_py/sbr/0.3/joint/point2048/
CHECKPOINT=${EXPERIMENT}/epoch_12.pth

SBR=("clean" "5_50" "5_100" "1_50" "1_100")
for i in "${!SBR[@]}"
do
DATAPATH=points_min2/0.3/argmax-filtering-sbr/${SBR[$i]}/
CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/votenet/votenet_8xb16_sunrgbd-3d.py ${CHECKPOINT} --cfg-options \
	work_dir=mapresults/${EXPERIMENT} \
	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
	test_dataloader.dataset.pipeline.0.load_dim=8 \
	test_dataloader.dataset.pipeline.1.transforms.2.num_points=2048 \
	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2]" \
done



### Matched Filtering + Thresholding
EXPERIMENT=work_dir_py/sbr/0.3/joint/thresh2000_10/
CHECKPOINT=${EXPERIMENT}/epoch_12.pth

SBR=("clean" "5_50" "5_100" "1_50" "1_100")
for i in "${!SBR[@]}"
do
DATAPATH=points_min2/0.3/argmax-filtering-sbr/${SBR[$i]}/
CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/votenet/votenet_8xb16_sunrgbd-3d.py ${CHECKPOINT} --cfg-options \
	work_dir=mapresults/${EXPERIMENT} \
	test_dataloader.dataset.data_prefix.pts=${DATAPATH} \
	test_dataloader.dataset.pipeline.0.load_dim=8 \
	test_dataloader.dataset.pipeline.1.transforms.2.num_points=20000 \
	test_dataloader.dataset.pipeline.1.transforms.2.thresh_sampling=1.0 \
	test_dataloader.dataset.pipeline.0.use_dim="[0,1,2,3]" \
done



