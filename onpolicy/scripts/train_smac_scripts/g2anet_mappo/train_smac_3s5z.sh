#!/bin/sh
env="StarCraft2"
map="3s5z"
algo="g2anet_mappo"
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
ppo_epoch=5
clip_param=0.05
eval_interval=10
eval_episodes=32

g2anet_gumbel_softmax_tau=0.01
hidden_size=128
recurrent_N=1

exp="g2anet_gumbel_softmax_tau_${g2anet_gumbel_softmax_tau}_\
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
    --g2anet_gumbel_softmax_tau ${g2anet_gumbel_softmax_tau}\
    --hidden_size ${hidden_size}\
    --recurrent_N ${recurrent_N}\ 
done
