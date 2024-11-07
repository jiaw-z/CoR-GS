export CUDA_VISIBLE_DEVICES=$1
scan=$2
workspace=$3


python train.py \
--source_path data/DTU/Rectified/$scan -m $workspace \
--eval  -r 4 --n_views 3 \
--random_background \
--iterations 10000 --position_lr_max_steps 10000 \
--densify_until_iter 10000 \
--densify_grad_threshold 0.0005 \
--gaussiansN 2 \
--coprune --coprune_threshold 10 \
--coreg --sample_pseudo_interval 1 --start_sample_pseudo 2000

bash ./scripts/copy_mask_dtu.sh $workspace $scan

python render.py \
--source_path data/DTU/Rectified/$scan -m $workspace

python metrics_dtu.py -m $workspace