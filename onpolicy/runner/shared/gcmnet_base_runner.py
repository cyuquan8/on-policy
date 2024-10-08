import numpy as np
import os
import torch
import wandb

from onpolicy.utils.shared_gcmnet_buffer import SharedGCMNetReplayBuffer as SharedReplayBuffer
from tensorboardX import SummaryWriter

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class GCMNetRunner(object):
    """
    Base class for training GCMNet policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # generic parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render

        # gcmnet somu and scmu parameters
        assert ('gcmnet' in self.algorithm_name or 'gcnet' in self.algorithm_name), \
            f'algorithm name is {self.algorithm_name} and not gcmnet_mappo'
        self.somu_actor = self.all_args.gcmnet_somu_actor
        self.scmu_actor = self.all_args.gcmnet_scmu_actor
        self.somu_critic = self.all_args.gcmnet_somu_critic
        self.scmu_critic = self.all_args.gcmnet_scmu_critic
        self.somu_n_layers = self.all_args.gcmnet_somu_n_layers
        self.somu_lstm_hidden_size = self.all_args.gcmnet_somu_lstm_hidden_size
        self.scmu_n_layers = self.all_args.gcmnet_scmu_n_layers
        self.scmu_lstm_hidden_size = self.all_args.gcmnet_scmu_lstm_hidden_size

        # dynamics
        self.dynamics = self.all_args.gcmnet_dynamics
        self.dynamics_reward = self.all_args.gcmnet_dynamics_reward
        assert not (self.dynamics == False and self.dynamics_reward == True), "Can't have dynamics reward w/o dynamics"

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
	
        from onpolicy.algorithms.gcmnet_mappo.gcmnet_mappo import GCMNet_MAPPO as TrainAlgo
        from onpolicy.algorithms.gcmnet_mappo.algorithm.gcmnetMAPPOPolicy import GCMNet_MAPPOPolicy as Policy

        if self.all_args.env_name == 'gym_dragon':
            share_observation_space = self.envs.share_observation_space if self.use_centralized_V else self.envs.observation_space

            self.all_args.num_agents = self.num_agents

            # policy network
            self.policy = Policy(self.all_args,
                                 self.envs.observation_space,
                                 share_observation_space,
                                 self.envs.action_space,
                                 device=self.device)

            # algorithm
            self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)

            if self.model_dir is not None:
                self.restore()

            # buffer
            self.buffer = SharedReplayBuffer(self.all_args,
                                             self.num_agents,
                                             self.envs.observation_space,
                                             share_observation_space,
                                             self.envs.action_space)
        else:
            share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

            self.all_args.num_agents = self.num_agents

            # policy network
            self.policy = Policy(self.all_args,
                                 self.envs.observation_space[0],
                                 share_observation_space,
                                 self.envs.action_space[0],
                                 device=self.device)

            # algorithm
            self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)

            if self.model_dir is not None:
                self.restore()

            # buffer
            self.buffer = SharedReplayBuffer(self.all_args,
                                             self.num_agents,
                                             self.envs.observation_space[0],
                                             share_observation_space,
                                             self.envs.action_space[0])

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            cent_obs=self.buffer.share_obs[-1],
            masks=self.buffer.masks[-1],
            somu_hidden_states_critic=self.buffer.somu_hidden_states_critic[-1] \
                if self.buffer.somu_hidden_states_critic is not None else None,
            somu_cell_states_critic=self.buffer.somu_cell_states_critic[-1] \
                if self.buffer.somu_cell_states_critic is not None else None,
            scmu_hidden_states_critic=self.buffer.scmu_hidden_states_critic[-1] \
                if self.buffer.scmu_hidden_states_critic is not None else None,
            scmu_cell_states_critic=self.buffer.scmu_cell_states_critic[-1] \
                if self.buffer.scmu_cell_states_critic is not None else None
        )
        next_values = _t2n(next_values)
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")
        if self.trainer._use_valuenorm:
            policy_vnorm = self.trainer.value_normalizer
            torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
            if self.trainer._use_valuenorm:
                policy_vnorm_state_dict = torch.load(str(self.model_dir) + '/vnorm.pt')
                self.trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
