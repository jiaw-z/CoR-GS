export CUDA_VISIBLE_DEVICES=$1
dataset=$2 
workspace=$3


python train.py \
--source_path $dataset -m $workspace \
--eval  -r 4 --n_views 24 \
--random_background \
--iterations 30000 --position_lr_max_steps 30000 \
--densify_until_iter 15000 \
--densify_grad_threshold 0.0005 \
--gaussiansN 2 \
--coprune --coprune_threshold 20 \
--coreg --sample_pseudo_interval 1 --start_sample_pseudo 500 --end_sample_pseudo 30000


python render.py \
--source_path $dataset -m $workspace \
--render_depth

python metrics.py \
--source_path $dataset -m $workspace \