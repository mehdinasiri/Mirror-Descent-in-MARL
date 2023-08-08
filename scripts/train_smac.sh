#!/bin/sh
env="StarCraft2"
algo="hamdpo"
exp="mlp"
map="3s5z"
running_max=5
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}"
for number in `seq ${running_max}`;
do
    echo "the ${number}-th running:"
    python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --running_id ${number} --lr 5e-4 --gamma 0.95 --n_training_threads 16 --n_rollout_threads 20 --num_mini_batch 1 --episode_length 160 --num_env_steps 20000000 --mdpo_epoch 5 --stacked_frames 1 --use_value_active_masks --use_eval --add_center_xy --use_state_agent --share_policy
done
