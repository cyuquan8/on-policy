#!/bin/sh
env="StarCraft2"
map="bane_vs_bane"
algo="gcm_gin_mappo"
seed_max=1
user="cyuquan8"
lr=0.0001
critic_lr=0.0001
ppo_epoch=15
clip_param=0.2
n_gnn_layers=2
rni_ratio=0.25
exp="lr_${lr}_critic_lr_${critic_lr}_ppo_epoch_${ppo_epoch}_clip_param_${clip_param}_n_gnn_layers_${n_gnn_layers}_rni_ratio_${rni_ratio}"

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python ../../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp}\
    --map_name ${map} --seed ${seed} --user_name ${user} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1\
    --episode_length 400 --num_env_steps 10000000 --lr ${lr} --critic_lr ${critic_lr} --ppo_epoch ${ppo_epoch}\
    --clip_param ${clip_param} --n_gnn_layers ${n_gnn_layers} --somu_n_layers 1 --somu_lstm_hidden_size 64\
    --somu_multi_att_n_heads 2 --scmu_n_layers 1 --scmu_lstm_hidden_size 64 --scmu_multi_att_n_heads 2 --fc_output_dims 64 --n_fc_layers 2\
    --n_gin_fc_layers 2 --k 1 --rni --rni_ratio ${rni_ratio} --use_value_active_masks --use_policy_active_masks --use_eval\
    --eval_episodes 32
done