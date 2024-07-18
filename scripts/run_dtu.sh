export CUDA_VISIBLE_DEVICES=$1
dataset=$2 
workspace=$3


python train.py \
--source_path $dataset -m $workspace \
--eval  -r 4 --n_views 3 \
--iterations 10000 --position_lr_max_steps 10000 \
--densify_until_iter 10000 \
--densify_grad_threshold 0.0005 \
--gaussiansN 2 \
--coprune --coprune_threshold 10 \
--coreg --sample_pseudo_interval 1 --start_sample_pseudo 2000

bash ./scripts/copy_mask_dtu.sh

python render.py \
--source_path $dataset -m $workspace \
--render_depth

python metrics_dtu.py \
--source_path $dataset -m $workspace \