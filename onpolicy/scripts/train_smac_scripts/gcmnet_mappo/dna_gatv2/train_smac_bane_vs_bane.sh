#!/bin/sh
env="StarCraft2"
map="bane_vs_bane"
algo="gcmnet_mappo"
seed_max=1
user="cyuquan8"
n_training_threads=1
n_rollout_threads=8
num_env_steps=10000000
episode_length=400
data_chunk_length=10
num_mini_batch=4
lr=0.0005
critic_lr=0.0005
ppo_epoch=4
clip_param=0.2
eval_episodes=32

gcmnet_gnn_architecture="dna_gatv2"
gcmnet_gnn_output_dims=64  
gcmnet_gnn_att_heads=8
gcmnet_gnn_dna_gatv2_multi_att_heads=1
gcmnet_cpa_model='f_additive'
gcmnet_n_gnn_layers=8
gcmnet_n_gnn_fc_layers=2
gcmnet_somu_n_layers=4
gcmnet_somu_lstm_hidden_size=64
gcmnet_somu_multi_att_n_heads=8
gcmnet_scmu_n_layers=4
gcmnet_scmu_lstm_hidden_size=64
gcmnet_scmu_multi_att_n_heads=8
gcmnet_fc_output_dims=64
gcmnet_n_fc_layers=2
gcmnet_k=1
gcmnet_rni_ratio=0.2
gcmnet_dynamics_fc_output_dims=64
gcmnet_dynamics_n_fc_layers=2
gcmnet_dynamics_loss_coef=0.01
gcmnet_dynamics_reward_coef=1

exp="gnn_arch_${gcmnet_gnn_architecture}_\
gnn_att_heads_${gcmnet_gnn_att_heads}_\
cpa_model_${gcmnet_cpa_model}_\
n_gnn_layers_${gcmnet_n_gnn_layers}_\
episode_length_${episode_length}_\
data_chunk_length_${data_chunk_length}_\
num_mini_batch_${num_mini_batch}_\
lr_${lr}_\
critic_lr_${critic_lr}_\
ppo_epoch_${ppo_epoch}_\
clip_param_${clip_param}_somu_actor_scmu_actor_somu_critic_scmu_critic_dyna_dyna_rew"

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python ../../../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp}\
    --map_name ${map} --seed ${seed} --user_name ${user} --n_training_threads ${n_training_threads}\
    --n_rollout_threads ${n_rollout_threads} --num_env_steps ${num_env_steps} --episode_length ${episode_length}\
    --data_chunk_length ${data_chunk_length} --num_mini_batch ${num_mini_batch} --lr ${lr} --critic_lr ${critic_lr}\
    --ppo_epoch ${ppo_epoch} --clip_param ${clip_param} --eval_episodes ${eval_episodes} --use_value_active_masks\
    --use_eval\
    --gcmnet_gnn_architecture ${gcmnet_gnn_architecture}\
    --gcmnet_gnn_output_dims ${gcmnet_gnn_output_dims}\
    --gcmnet_gnn_att_heads ${gcmnet_gnn_att_heads}\
    --gcmnet_gnn_dna_gatv2_multi_att_heads ${gcmnet_gnn_dna_gatv2_multi_att_heads}\
    --gcmnet_gnn_att_concat\
    --gcmnet_cpa_model ${gcmnet_cpa_model}\
    --gcmnet_n_gnn_layers ${gcmnet_n_gnn_layers}\
    --gcmnet_n_gnn_fc_layers ${gcmnet_n_gnn_fc_layers}\
    --gcmnet_somu_actor\
    --gcmnet_scmu_actor\
    --gcmnet_somu_critic\
    --gcmnet_scmu_critic\
    --gcmnet_somu_n_layers ${gcmnet_somu_n_layers}\
    --gcmnet_somu_lstm_hidden_size ${gcmnet_somu_lstm_hidden_size}\
    --gcmnet_somu_multi_att_n_heads ${gcmnet_somu_multi_att_n_heads}\
    --gcmnet_scmu_n_layers ${gcmnet_scmu_n_layers}\
    --gcmnet_scmu_lstm_hidden_size ${gcmnet_scmu_lstm_hidden_size}\
    --gcmnet_scmu_multi_att_n_heads ${gcmnet_scmu_multi_att_n_heads}\
    --gcmnet_fc_output_dims ${gcmnet_fc_output_dims}\
    --gcmnet_n_fc_layers ${gcmnet_n_fc_layers}\
    --gcmnet_k ${gcmnet_k}\
    --gcmnet_rni_ratio ${gcmnet_rni_ratio}\
    --gcmnet_dynamics\
    --gcmnet_dynamics_reward\
    --gcmnet_dynamics_fc_output_dims ${gcmnet_dynamics_fc_output_dims}\
    --gcmnet_dynamics_n_fc_layers ${gcmnet_dynamics_n_fc_layers}\
    --gcmnet_dynamics_loss_coef ${gcmnet_dynamics_loss_coef}\
    --gcmnet_dynamics_reward_coef ${gcmnet_dynamics_reward_coef}\
    # --gcmnet_gnn_att_concat\
    # --gcmnet_train_eps\
    # --gcmnet_somu_actor\
    # --gcmnet_scmu_actor\
    # --gcmnet_somu_critic\
    # --gcmnet_scmu_critic\
    # --gcmnet_knn\
    # --gcmnet_rni\
    # --gcmnet_dynamics\
    # --gcmnet_dynamics_reward\
done