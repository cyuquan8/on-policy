#!/bin/sh
env="MPE"
scenario="simple_spread" 
num_landmarks=3
num_agents=3
algo="rmappo" #"rmappo" "mappo" "ippo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 20000000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 5e-4 --critic_lr 5e-4 --use_eval\
    --wandb_name "MPE" --user_name "cyuquan8"\
    # --share_policy --use_centralized_V --use_recurrent_policy \
done
