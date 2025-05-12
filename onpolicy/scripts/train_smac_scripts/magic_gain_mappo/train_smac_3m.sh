#!/bin/sh
env="StarCraft2"
map="3m"
algo="magic_mappo"
seed_max=5
user='cyuquan8'
n_training_threads=1
n_rollout_threads=8
num_env_steps=10000000
episode_length=400
data_chunk_length=10
num_mini_batch=1
lr=0.0005
critic_lr=0.0005
ppo_epoch=15
clip_param=0.2
eval_interval=10
eval_episodes=32

magic_gat_encoder_out_size=128
magic_gat_encoder_num_heads=8
magic_gat_hidden_size=128
magic_gat_num_heads=1
magic_gat_num_heads_out=1
magic_self_loop_type1=2
magic_self_loop_type2=2
magic_comm_init="uniform"
magic_gat_architecture="gat"
magic_n_gnn_fc_layers=2
magic_gnn_norm="none"
hidden_size=128
recurrent_N=1

exp="magic_gat_architecture_${magic_gat_architecture}_\
magic_self_loop_type1_${magic_self_loop_type1}_\
magic_self_loop_type2_${magic_self_loop_type2}_\
hidden_size_${hidden_size}_\
recurrent_N_${recurrent_N}_\
ep_len_${episode_length}_\
data_chunk_len_${data_chunk_length}_\
num_mini_batch_${num_mini_batch}_\
ppo_epoch_${ppo_epoch}_\
clip_param_${clip_param}\
"

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python ../../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp}\
    --map_name ${map} --seed ${seed} --user_name ${user} --n_training_threads ${n_training_threads}\
    --n_rollout_threads ${n_rollout_threads} --num_env_steps ${num_env_steps} --episode_length ${episode_length}\
    --data_chunk_length ${data_chunk_length} --num_mini_batch ${num_mini_batch} --lr ${lr} --critic_lr ${critic_lr}\
    --ppo_epoch ${ppo_epoch} --clip_param ${clip_param} --eval_interval ${eval_interval}\
    --eval_episodes ${eval_episodes} --use_value_active_masks --use_eval\
    --magic_gat_encoder_out_size ${magic_gat_encoder_out_size}\
    --magic_gat_encoder_num_heads ${magic_gat_encoder_num_heads}\
    --magic_gat_hidden_size ${magic_gat_hidden_size}\
    --magic_gat_num_heads ${magic_gat_num_heads}\
    --magic_gat_num_heads_out ${magic_gat_num_heads_out}\
    --magic_self_loop_type1 ${magic_self_loop_type1}\
    --magic_self_loop_type2 ${magic_self_loop_type2}\
    --magic_comm_init ${magic_comm_init}\
    --magic_gat_architecture ${magic_gat_architecture}\
    --magic_n_gnn_fc_layers ${magic_n_gnn_fc_layers}\
    --magic_gnn_norm ${magic_gnn_norm}\
    --hidden_size ${hidden_size}\
    --recurrent_N ${recurrent_N}\
    --magic_message_encoder\
    --magic_message_decoder\
    --magic_use_gat_encoder\
    --magic_first_gat_normalize\
    --magic_second_gat_normalize\
    --magic_directed\
    # --magic_comm_mask_zero\
    # --magic_gat_encoder_normalize\
    # --magic_first_graph_complete\
    # --magic_second_graph_complete\
    # --magic_learn_second_graph\
    # --magic_gnn_train_eps\
done