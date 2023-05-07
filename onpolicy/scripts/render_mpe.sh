#!/bin/sh
env="MPE"
scenario="simple_spread"
num_landmarks=3
num_agents=3
algo="rmappo"
exp="shared_r_actor_cen_r_critic_3_agents_3_targets"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 25 --render_episodes 5 \
    --model_dir "results/MPE/simple_spread/rmappo/shared_r_actor_cen_r_critic_3_agents_3_targets/wandb/latest-run/files" \
    # --share_policy --use_centralized_V --use_recurrent_policy
    # --share_policy --use_centralized_V --use_recurrent_policy
done
