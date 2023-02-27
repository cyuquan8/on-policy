#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.envs.gym_dragon.gym_dragon_env import GymDragonEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv, AvailableActionsDummyVecEnv, AvailableActionsSubprocVecEnv
from pathlib import Path

"""Train script for gym_dragon."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "gym_dragon":
                env = GymDragonEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return AvailableActionsDummyVecEnv([get_env_fn(0)])
    else:
        return AvailableActionsSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "gym_dragon":
                env = GymDragonEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return AvailableActionsDummyVecEnv([get_env_fn(0)])
    else:
        return AvailableActionsSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--region', type=str, default='all',
                        help="Which region of gym_dragon to run on")
    parser.add_argument("--include_perturbations", action='store_true', default=False)
    parser.add_argument('--num_agents', type=int, default=3, help="number of agents")
    parser.add_argument("--color_tools_only", action='store_true', default=False)
    
    # reward wrappers
    parser.add_argument("--include_explore_reward", action='store_true', default=False)
    parser.add_argument("--include_inspect_reward", action='store_true', default=False)
    parser.add_argument("--include_defusal_reward", action='store_true', default=False)
    parser.add_argument("--include_beacon_reward", action='store_true', default=False)
    parser.add_argument("--include_proximity_reward", action='store_true', default=False)

    # observations wrappers
    parser.add_argument("--include_memory_obs", action='store_true', default=False)
    parser.add_argument("--include_edge_index_obs", action='store_true', default=False)
    parser.add_argument("--include_all_agent_locations_obs", action='store_true', default=False)
    parser.add_argument("--include_all_agent_nodes_obs", action='store_true', default=False)
    parser.add_argument("--include_full_obs", action='store_true', default=False)

    # budget weights for each region

    # desert
    parser.add_argument('--budget_weight_desert_perturbations', type=float, default=0.2)
    parser.add_argument('--budget_weight_desert_communications', type=float, default=0.2)
    parser.add_argument('--budget_weight_desert_bomb_additonal', type=float, default=0.2)

    # forest
    parser.add_argument('--budget_weight_forest_perturbations', type=float, default=0.2)
    parser.add_argument('--budget_weight_forest_communications', type=float, default=0.2)
    parser.add_argument('--budget_weight_forest_bomb_additonal', type=float, default=0.2)

    # village
    parser.add_argument('--budget_weight_village_perturbations', type=float, default=0.2)
    parser.add_argument('--budget_weight_village_communications', type=float, default=0.2)
    parser.add_argument('--budget_weight_village_bomb_additonal', type=float, default=0.2)

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    elif all_args.algorithm_name == "gcm_dna_gatv2_mappo":
        print("u are choosing to use gcm_dna_gatv2_mappo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    elif all_args.algorithm_name == "gcm_gin_mappo":
        print("u are choosing to use gcm_gin_mappo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.region / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        if all_args.wandb_resume_run_id:
            run = wandb.init(id=all_args.wandb_resume_run_id,
                             config=all_args,
                             project=all_args.env_name,
                             entity=all_args.user_name,
                             notes=socket.gethostname(),
                             name=str(all_args.algorithm_name) + "_" +
                                  str(all_args.experiment_name) +
                                  "_seed" + str(all_args.seed),
                             group=all_args.region,
                             dir=str(run_dir),
                             job_type="training",
                             reinit=True,
                             resume="must")
        else:
            run = wandb.init(config=all_args,
                             project=all_args.env_name,
                             entity=all_args.user_name,
                             notes=socket.gethostname(),
                             name=str(all_args.algorithm_name) + "_" +
                                  str(all_args.experiment_name) +
                                  "_seed" + str(all_args.seed),
                             group=all_args.region,
                             dir=str(run_dir),
                             job_type="training",
                             reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.algorithm_name == "gcm_dna_gatv2_mappo" or all_args.algorithm_name == "gcm_gin_mappo":
        from onpolicy.runner.shared.gnn_gym_dragon_runner import GNNGymDragonRunner as Runner
    elif all_args.share_policy:
        from onpolicy.runner.shared.gym_dragon_runner import GymDragonRunner as Runner
    else:
        from onpolicy.runner.separated.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
