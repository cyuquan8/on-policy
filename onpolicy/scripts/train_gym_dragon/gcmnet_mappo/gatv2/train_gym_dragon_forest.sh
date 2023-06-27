#!/bin/sh
seed_max=0
env="gym_dragon"
algo="gcmnet_mappo"
user="cyuquan8"
n_training_threads=1
n_rollout_threads=32
num_env_steps=10000000000
data_chunk_length=10
num_mini_batch=5
lr=0.0005
critic_lr=0.0005
ppo_epoch=2
clip_param=0.2
eval_episodes=32

gcmnet_gnn_architecture="gatv2"
gcmnet_gnn_output_dims=64  
gcmnet_gnn_att_heads=8
gcmnet_gnn_dna_gatv2_multi_att_heads=1
gcmnet_cpa_model='f_additive'
gcmnet_n_gnn_layers=4
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
clip_param_${clip_param}_\
region_${region}"

echo "env is ${env}, region is ${region}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
if [ "$seed_max" -eq 0 ]; then
    echo "seed is ${seed_max} (seed == None for reset() for gym_dragon):"
    python ../../../train/train_gym_dragon.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp}\
    --region ${region} --seed ${seed_max} --user_name ${user} --n_training_threads ${n_training_threads}\
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
    --gcmnet_somu_n_layers ${gcmnet_somu_n_layers}\
    --gcmnet_somu_lstm_hidden_size ${gcmnet_somu_lstm_hidden_size}\
    --gcmnet_somu_multi_att_n_heads ${gcmnet_somu_multi_att_n_heads}\
    --gcmnet_scmu_n_layers ${gcmnet_scmu_n_layers}\
    --gcmnet_scmu_lstm_hidden_size ${gcmnet_scmu_lstm_hidden_size}\
    --gcmnet_scmu_multi_att_n_heads ${gcmnet_scmu_multi_att_n_heads}\
    --gcmnet_fc_output_dims ${gcmnet_fc_output_dims}\
    --gcmnet_n_fc_layers ${gcmnet_n_fc_layers}\
    --gcmnet_k ${gcmnet_k}\
    --gcmnet_rni\
    --gcmnet_rni_ratio ${gcmnet_rni_ratio}\
    --gcmnet_dynamics\
    --gcmnet_dynamics_reward\
    --gcmnet_dynamics_fc_output_dims ${gcmnet_dynamics_fc_output_dims}\
    --gcmnet_dynamics_n_fc_layers ${gcmnet_dynamics_n_fc_layers}\
    --gcmnet_dynamics_loss_coef ${gcmnet_dynamics_loss_coef}\
    --gcmnet_dynamics_reward_coef ${gcmnet_dynamics_reward_coef}\
    --recon_phase_length ${recon_phase_length}\
    --seconds_per_timestep ${seconds_per_timestep}\
    --color_tools_only\
    --include_fuse_bombs\
    --include_chained_bombs\
    --include_explore_reward\
    --include_inspect_reward\
    --include_defusal_reward\
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
    # --gcmnet_gnn_att_concat\
    # --gcmnet_train_eps\
    # --gcmnet_knn\
    # --gcmnet_dynamics\
    # --gcmnet_dynamics_reward\

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
        --region ${region} --seed ${seed_max} --user_name ${user} --n_training_threads ${n_training_threads}\
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
        --gcmnet_somu_n_layers ${gcmnet_somu_n_layers}\
        --gcmnet_somu_lstm_hidden_size ${gcmnet_somu_lstm_hidden_size}\
        --gcmnet_somu_multi_att_n_heads ${gcmnet_somu_multi_att_n_heads}\
        --gcmnet_scmu_n_layers ${gcmnet_scmu_n_layers}\
        --gcmnet_scmu_lstm_hidden_size ${gcmnet_scmu_lstm_hidden_size}\
        --gcmnet_scmu_multi_att_n_heads ${gcmnet_scmu_multi_att_n_heads}\
        --gcmnet_fc_output_dims ${gcmnet_fc_output_dims}\
        --gcmnet_n_fc_layers ${gcmnet_n_fc_layers}\
        --gcmnet_k ${gcmnet_k}\
        --gcmnet_rni\
        --gcmnet_rni_ratio ${gcmnet_rni_ratio}\
        --gcmnet_dynamics\
        --gcmnet_dynamics_reward\
        --gcmnet_dynamics_fc_output_dims ${gcmnet_dynamics_fc_output_dims}\
        --gcmnet_dynamics_n_fc_layers ${gcmnet_dynamics_n_fc_layers}\
        --gcmnet_dynamics_loss_coef ${gcmnet_dynamics_loss_coef}\
        --gcmnet_dynamics_reward_coef ${gcmnet_dynamics_reward_coef}\
        --recon_phase_length ${recon_phase_length}\
        --seconds_per_timestep ${seconds_per_timestep}\
        --color_tools_only\
        --include_fuse_bombs\
        --include_chained_bombs\
        --include_explore_reward\
        --include_inspect_reward\
        --include_defusal_reward\
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
        # --gcmnet_gnn_att_concat\
        # --gcmnet_train_eps\
        # --gcmnet_knn\
        # --gcmnet_dynamics\
        # --gcmnet_dynamics_reward\

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