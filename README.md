Code for 10-701 project on multi-agent reinforcement learning in the SMAC environment. Starting point for codebase was this repository: https://github.com/zoeyuchao/mappo

The relevant top level script is https://github.com/cyuquan8/on-policy/blob/main/onpolicy/scripts/train/train_smac.py. This script handles the training setup, loop, and logging for all SMAC tasks and all algorithms, including dgcn, which is what we implemented.

The DGCN model was implemented in this directory: https://github.com/cyuquan8/on-policy/tree/main/onpolicy/algorithms/dgcn_mappo. You can find the actor and critic models here: https://github.com/cyuquan8/on-policy/tree/main/onpolicy/algorithms/dgcn_mappo/algorithm.

To replicate our training, run the script with the following arguments: `--user_name {wandb username} --n_rollout_threads 32 --algorithm_name mappo_gnn --map_name {SMAC task name} --num_env_steps 10000000 --experiment_name {experiment name}`

We ran one experiment, 2m_vs_1z, with 6 rollout threads due to hardware constraints. We suspect that this change in hyperparameter hurt performance.

Results in our paper for baseline models were taken from the FCMNet paper: https://arxiv.org/pdf/2201.11994.pdf.
