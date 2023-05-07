#!/bin/sh
env="StarCraft2"
map="2m_vs_1z"
algo="gcmnet_dna_gatv2_mappo"
seed_max=1
user="cyuquan8"
n_training_threads=1
n_rollout_threads=8
num_mini_batch=1
episode_length=400
num_env_steps=10000000
lr=0.0001
critic_lr=0.0001
ppo_epoch=15
clip_param=0.2
gcmnet_n_gnn_layers=2
gcmnet_somu_n_layers=2
gcmnet_somu_lstm_hidden_size=64
gcmnet_somu_multi_att_n_heads=2
gcmnet_scmu_n_layers=2
gcmnet_scmu_lstm_hidden_size=64
gcmnet_scmu_multi_att_n_heads=2
gcmnet_fc_output_dims=64
gcmnet_n_fc_layers=2
gcmnet_n_gin_fc_layers=2
gcmnet_k=1
gcmnet_rni_ratio=0.25
eval_episodes=32
exp="lr_${lr}_critic_lr_${critic_lr}_ppo_epoch_${ppo_epoch}_clip_param_${clip_param}_gcmnet_n_gnn_layers_${gcmnet_n_gnn_layers}_gcmnet_rni_ratio_${gcmnet_rni_ratio}"

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python ../../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp}\
    --map_name ${map} --seed ${seed} --user_name ${user} --n_training_threads ${n_training_threads}\
    --n_rollout_threads ${n_rollout_threads} --num_mini_batch ${num_mini_batch} --episode_length ${episode_length}\
    --num_env_steps ${num_env_steps} --lr ${lr} --critic_lr ${critic_lr} --ppo_epoch ${ppo_epoch}\
    --clip_param ${clip_param} --gcmnet_n_gnn_layers ${gcmnet_n_gnn_layers} \
    --gcmnet_somu_n_layers ${gcmnet_somu_n_layers} --gcmnet_somu_lstm_hidden_size ${gcmnet_somu_lstm_hidden_size}\
    --gcmnet_somu_multi_att_n_heads ${gcmnet_somu_multi_att_n_heads} --gcmnet_scmu_n_layers ${gcmnet_scmu_n_layers}\
    --gcmnet_scmu_lstm_hidden_size ${gcmnet_scmu_lstm_hidden_size}\
    --gcmnet_scmu_multi_att_n_heads ${gcmnet_scmu_multi_att_n_heads} --gcmnet_fc_output_dims ${gcmnet_fc_output_dims}\
    --gcmnet_n_fc_layers ${gcmnet_n_fc_layers} --gcmnet_n_gin_fc_layers ${gcmnet_n_gin_fc_layers}\
    --gcmnet_k ${gcmnet_k} --gcmnet_rni --gcmnet_rni_ratio ${gcmnet_rni_ratio} --use_value_active_masks --use_eval\
    --eval_episodes ${eval_episodes}
done