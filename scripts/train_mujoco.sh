#!/bin/sh
env="mujoco"
scenario="Ant-v2"
agent_conf="2x4"
agent_obsk=2
algo="hamdpo"
exp="mlp"
running_max=5
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}"
for number in `seq ${running_max}`;
do
    echo "the ${number}-th running:"
    python train/train_mujoco.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --agent_conf ${agent_conf} --agent_obsk ${agent_obsk} --lr 5e-5 --std_x_coef 1 --std_y_coef 5e-1 --n_training_threads 16 --n_rollout_threads 8 --num_mini_batch 40 --episode_length 1000 --num_env_steps 10000000 --mdpo_epoch 10 --use_value_active_masks --use_eval --add_center_xy --use_state_agent --share_policy
done
