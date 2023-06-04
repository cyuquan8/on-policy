#!/bin/sh
seed_max=0
env="gym_dragon"
algo="gcmnet_mappo"
user="cyuquan8"
n_training_threads=1
n_rollout_threads=8
num_mini_batch=1
num_env_steps=10000000000
lr=0.0001
critic_lr=0.0001
ppo_epoch=15
clip_param=0.2
n_eval_rollout_threads=32

gcmnet_gnn_architecture="dna_gatv2"
gcmnet_n_gnn_layers=8
gcmnet_somu_n_layers=2
gcmnet_somu_lstm_hidden_size=128
gcmnet_somu_multi_att_n_heads=2
gcmnet_scmu_n_layers=2
gcmnet_scmu_lstm_hidden_size=128
gcmnet_scmu_multi_att_n_heads=2
gcmnet_fc_output_dims=512
gcmnet_n_fc_layers=3
gcmnet_n_gin_fc_layers=2
gcmnet_k=1
gcmnet_rni_ratio=0.25
gcmnet_cpa_model='f_additive'

episode_length=480
region="forest"
recon_phase_length=0
seconds_per_timestep=2.0

budget_weight_desert_perturbations=10 
budget_weight_desert_communications=10 
budget_weight_desert_bomb_additonal=10 
budget_weight_forest_perturbations=10 
budget_weight_forest_communications=10 
budget_weight_forest_bomb_additonal=10 
budget_weight_village_perturbations=10 
budget_weight_village_communications=10 
budget_weight_village_bomb_additonal=10 

explore_reward_weight=0.01
inspect_reward_weight=0.01
defusal_reward_weight=0.1
beacon_reward_weight=0.01
proximity_reward_weight=0.01

exp="region_${region}_lr_${lr}_critic_lr_${critic_lr}_ppo_epoch_${ppo_epoch}_clip_param_${clip_param}_gnn_arch_${gcmnet_gnn_architecture}_n_gnn_layers_${gcmnet_n_gnn_layers}_rni_ratio_${gcmnet_rni_ratio}_cpa_model_${gcmnet_cpa_model}"

echo "env is ${env}, region is ${region}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
if [ "$seed_max" -eq 0 ]; then
    echo "seed is ${seed_max} (seed == None for reset() for gym_dragon):"
    python ../../../train/train_gym_dragon.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp}\
    --region ${region} --seed ${seed_max} --user_name ${user} --n_training_threads ${n_training_threads}\
    --n_rollout_threads ${n_rollout_threads} --num_mini_batch ${num_mini_batch} --episode_length ${episode_length}\
    --num_env_steps ${num_env_steps} --lr ${lr} --critic_lr ${critic_lr} --ppo_epoch ${ppo_epoch}\
    --clip_param ${clip_param} --gcmnet_gnn_architecture ${gcmnet_gnn_architecture}\
    --gcmnet_n_gnn_layers ${gcmnet_n_gnn_layers} --gcmnet_somu_n_layers ${gcmnet_somu_n_layers}\
    --gcmnet_somu_lstm_hidden_size ${gcmnet_somu_lstm_hidden_size}\
    --gcmnet_somu_multi_att_n_heads ${gcmnet_somu_multi_att_n_heads}\
    --gcmnet_scmu_n_layers ${gcmnet_scmu_n_layers} --gcmnet_scmu_lstm_hidden_size ${gcmnet_scmu_lstm_hidden_size}\
    --gcmnet_scmu_multi_att_n_heads ${gcmnet_scmu_multi_att_n_heads}\
    --gcmnet_fc_output_dims ${gcmnet_fc_output_dims} --gcmnet_n_fc_layers ${gcmnet_n_fc_layers}\
    --gcmnet_n_gin_fc_layers ${gcmnet_n_gin_fc_layers} --gcmnet_k ${gcmnet_k} --gcmnet_rni\
    --gcmnet_rni_ratio ${gcmnet_rni_ratio} --gcmnet_cpa_model ${gcmnet_cpa_model}\
    --use_value_active_masks --use_eval --n_eval_rollout_threads ${n_eval_rollout_threads}\
    --recon_phase_length ${recon_phase_length}\
    --seconds_per_timestep ${seconds_per_timestep}\
    --color_tools_only\
    --include_fuse_bombs\
    --include_chained_bombs\
    --include_explore_reward\
    --include_inspect_reward\
    --include_defusal_reward\
    --include_beacon_reward\
    --include_proximity_reward\
    --explore_reward_weight ${explore_reward_weight}\
    --inspect_reward_weight ${inspect_reward_weight}\
    --defusal_reward_weight ${defusal_reward_weight}\
    --beacon_reward_weight ${beacon_reward_weight}\
    --proximity_reward_weight ${proximity_reward_weight}\
    --include_full_obs\
    --budget_weight_desert_perturbations ${budget_weight_desert_perturbations}\
    --budget_weight_desert_communications ${budget_weight_desert_communications}\
    --budget_weight_desert_bomb_additonal ${budget_weight_desert_bomb_additonal}\
    --budget_weight_forest_perturbations ${budget_weight_forest_perturbations}\
    --budget_weight_forest_communications ${budget_weight_forest_communications}\
    --budget_weight_forest_bomb_additonal ${budget_weight_forest_bomb_additonal}\
    --budget_weight_village_perturbations ${budget_weight_village_perturbations}\
    --budget_weight_village_communications ${budget_weight_village_communications}\
    --budget_weight_village_bomb_additonal ${budget_weight_village_bomb_additonal}\
    # --recon_phase_length ${recon_phase_length}\
    # --seconds_per_timestep ${seconds_per_timestep}\
    
    # --color_tools_only\
    # --include_fuse_bombs\
    # --include_fire_bombs\
    # --include_chained_bombs\
    
    # --include_explore_reward\
    # --include_inspect_reward\
    # --include_defusal_reward\
    # --include_beacon_reward\
    # --include_proximity_reward\

    # --explore_reward_weight ${explore_reward_weight}\
    # --inspect_reward_weight ${inspect_reward_weight}\
    # --defusal_reward_weight ${defusal_reward_weight}\
    # --beacon_reward_weight ${beacon_reward_weight}\
    # --proximity_reward_weight ${proximity_reward_weight}\

    # --include_memory_obs\
    # --include_edge_index_obs\
    # --include_all_agent_locations_obs\
    # --include_all_agent_nodes_obs\
    # --include_full_obs\
    
    # --budget_weight_desert_perturbations ${budget_weight_desert_perturbations}\
    # --budget_weight_desert_communications ${budget_weight_desert_communications}\
    # --budget_weight_desert_bomb_additonal ${budget_weight_desert_bomb_additonal}\
    # --budget_weight_forest_perturbations ${budget_weight_forest_perturbations}\
    # --budget_weight_forest_communications ${budget_weight_forest_communications}\
    # --budget_weight_forest_bomb_additonal ${budget_weight_forest_bomb_additonal}\
    # --budget_weight_village_perturbations ${budget_weight_village_perturbations}\
    # --budget_weight_village_communications ${budget_weight_village_communications}\
    # --budget_weight_village_bomb_additonal ${budget_weight_village_bomb_additonal}\
else
    for seed in `seq ${seed_max}`;
    do
        echo "seed is ${seed}:"
        python ../../../train/train_gym_dragon.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp}\
        --region ${region} --seed ${seed} --user_name ${user} --n_training_threads ${n_training_threads}\
        --n_rollout_threads ${n_rollout_threads} --num_mini_batch ${num_mini_batch} --episode_length ${episode_length}\
        --num_env_steps ${num_env_steps} --lr ${lr} --critic_lr ${critic_lr} --ppo_epoch ${ppo_epoch}\
        --clip_param ${clip_param} --gcmnet_gnn_architecture ${gcmnet_gnn_architecture}\
        --gcmnet_n_gnn_layers ${gcmnet_n_gnn_layers} --gcmnet_somu_n_layers ${gcmnet_somu_n_layers}\
        --gcmnet_somu_lstm_hidden_size ${gcmnet_somu_lstm_hidden_size}\
        --gcmnet_somu_multi_att_n_heads ${gcmnet_somu_multi_att_n_heads}\
        --gcmnet_scmu_n_layers ${gcmnet_scmu_n_layers} --gcmnet_scmu_lstm_hidden_size ${gcmnet_scmu_lstm_hidden_size}\
        --gcmnet_scmu_multi_att_n_heads ${gcmnet_scmu_multi_att_n_heads}\
        --gcmnet_fc_output_dims ${gcmnet_fc_output_dims} --gcmnet_n_fc_layers ${gcmnet_n_fc_layers}\
        --gcmnet_n_gin_fc_layers ${gcmnet_n_gin_fc_layers} --gcmnet_k ${gcmnet_k} --gcmnet_rni\
        --gcmnet_rni_ratio ${gcmnet_rni_ratio} --gcmnet_cpa_model ${gcmnet_cpa_model}\
        --use_value_active_masks --use_eval --n_eval_rollout_threads ${n_eval_rollout_threads}\
        --recon_phase_length ${recon_phase_length}\
        --seconds_per_timestep ${seconds_per_timestep}\
        --color_tools_only\
        --include_fuse_bombs\
        --include_chained_bombs\
        --include_explore_reward\
        --include_inspect_reward\
        --include_defusal_reward\
        --include_beacon_reward\
        --include_proximity_reward\
        --explore_reward_weight ${explore_reward_weight}\
        --inspect_reward_weight ${inspect_reward_weight}\
        --defusal_reward_weight ${defusal_reward_weight}\
        --beacon_reward_weight ${beacon_reward_weight}\
        --proximity_reward_weight ${proximity_reward_weight}\
        --include_full_obs\
        --budget_weight_desert_perturbations ${budget_weight_desert_perturbations}\
        --budget_weight_desert_communications ${budget_weight_desert_communications}\
        --budget_weight_desert_bomb_additonal ${budget_weight_desert_bomb_additonal}\
        --budget_weight_forest_perturbations ${budget_weight_forest_perturbations}\
        --budget_weight_forest_communications ${budget_weight_forest_communications}\
        --budget_weight_forest_bomb_additonal ${budget_weight_forest_bomb_additonal}\
        --budget_weight_village_perturbations ${budget_weight_village_perturbations}\
        --budget_weight_village_communications ${budget_weight_village_communications}\
        --budget_weight_village_bomb_additonal ${budget_weight_village_bomb_additonal}\
        # --recon_phase_length ${recon_phase_length}\
        # --seconds_per_timestep ${seconds_per_timestep}\
        
        # --color_tools_only\
        # --include_fuse_bombs\
        # --include_fire_bombs\
        # --include_chained_bombs\
        
        # --include_explore_reward\
        # --include_inspect_reward\
        # --include_defusal_reward\
        # --include_beacon_reward\
        # --include_proximity_reward\

        # --explore_reward_weight ${explore_reward_weight}\
        # --inspect_reward_weight ${inspect_reward_weight}\
        # --defusal_reward_weight ${defusal_reward_weight}\
        # --beacon_reward_weight ${beacon_reward_weight}\
        # --proximity_reward_weight ${proximity_reward_weight}\

        # --include_memory_obs\
        # --include_edge_index_obs\
        # --include_all_agent_locations_obs\
        # --include_all_agent_nodes_obs\
        # --include_full_obs\
        
        # --budget_weight_desert_perturbations ${budget_weight_desert_perturbations}\
        # --budget_weight_desert_communications ${budget_weight_desert_communications}\
        # --budget_weight_desert_bomb_additonal ${budget_weight_desert_bomb_additonal}\
        # --budget_weight_forest_perturbations ${budget_weight_forest_perturbations}\
        # --budget_weight_forest_communications ${budget_weight_forest_communications}\
        # --budget_weight_forest_bomb_additonal ${budget_weight_forest_bomb_additonal}\
        # --budget_weight_village_perturbations ${budget_weight_village_perturbations}\
        # --budget_weight_village_communications ${budget_weight_village_communications}\
        # --budget_weight_village_bomb_additonal ${budget_weight_village_bomb_additonal}\
    done
fi