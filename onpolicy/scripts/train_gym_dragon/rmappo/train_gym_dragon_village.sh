#!/bin/sh
seed_max=1
env="gym_dragon"
region="village"
algo="rmappo"
user="cyuquan8"
n_training_threads=1
n_rollout_threads=16
num_mini_batch=1
episode_length=900
num_env_steps=10000000000
lr=0.0001
critic_lr=0.0001
ppo_epoch=15
clip_param=0.2
n_eval_rollout_threads=16

budget_weight_desert_perturbations=10 
budget_weight_desert_communications=10 
budget_weight_desert_communications=10 
budget_weight_forest_perturbations=10 
budget_weight_forest_communications=10 
budget_weight_forest_communications=10 
budget_weight_village_perturbations=10 
budget_weight_village_communications=10 
budget_weight_village_communications=10 

explore_reward_weight=0.05
inspect_reward_weight=0.05
defusal_reward_weight=0.25
beacon_reward_weight=0.1
proximity_reward_weight=0.1

exp="region_${region}_n_rollout_threads_${n_rollout_threads}_lr_${lr}_critic_lr_${critic_lr}_ppo_epoch_${ppo_epoch}_clip_param_${clip_param}"

echo "env is ${env}, region is ${region}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python ../../train/train_gym_dragon.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --region ${region} \
    --seed ${seed} --user_name ${user} --n_training_threads ${n_training_threads} --n_rollout_threads ${n_rollout_threads} \
    --num_mini_batch ${num_mini_batch} --episode_length ${episode_length} --num_env_steps ${num_env_steps} --lr ${lr} \
    --critic_lr ${critic_lr} --ppo_epoch ${ppo_epoch} --clip_param ${clip_param} --use_eval --n_eval_rollout_threads ${n_eval_rollout_threads} \
    --color_tools_only \
    --include_full_obs \
    --include_explore_reward --include_inspect_reward --include_defusal_reward \
    --budget_weight_desert_perturbations ${budget_weight_desert_perturbations} \
    --budget_weight_desert_communications ${budget_weight_desert_communications} \
    --budget_weight_desert_communications ${budget_weight_desert_communications} \
    --budget_weight_forest_perturbations ${budget_weight_forest_perturbations} \
    --budget_weight_forest_communications ${budget_weight_forest_communications} \
    --budget_weight_forest_communications ${budget_weight_forest_communications} \
    --budget_weight_village_perturbations ${budget_weight_village_perturbations} \
    --budget_weight_village_communications ${budget_weight_village_communications} \
    --budget_weight_village_communications ${budget_weight_village_communications} \
    --explore_reward_weight ${explore_reward_weight} \
    --inspect_reward_weight ${inspect_reward_weight} \
    --defusal_reward_weight ${defusal_reward_weight} \
    --beacon_reward_weight ${beacon_reward_weight} \
    --proximity_reward_weight ${proximity_reward_weight} \
    # --include_perturbations
    # --include_explore_reward --include_inspect_reward --include_defusal_reward --include_beacon_reward --include_proximity_reward
    # --include_memory_obs --include_edge_index_obs --include_all_agent_locations_obs --include_all_agent_nodes_obs --include_full_obs
    # --budget_weight_desert_perturbations --budget_weight_desert_communications --budget_weight_desert_communications
    # --budget_weight_forest_perturbations --budget_weight_forest_communications --budget_weight_forest_communications
    # --budget_weight_village_perturbations --budget_weight_village_communications --budget_weight_village_communications
    # --explore_reward_weight --inspect_reward_weight --defusal_reward_weight --beacon_reward_weight --proximity_reward_weight
done
