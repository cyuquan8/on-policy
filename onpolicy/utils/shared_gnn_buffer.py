import torch
import numpy as np
from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space

class SharedGNNReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.num_somu_lstm = args.num_somu_lstm
        self.num_scmu_lstm = args.num_scmu_lstm
        self.somu_lstm_hidden_size = args.somu_lstm_hidden_size
        self.scmu_lstm_hidden_size = args.scmu_lstm_hidden_size
        self.num_agents = num_agents
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape),
                                  dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)

        self.somu_hidden_states_actor = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.num_somu_lstm, self.somu_lstm_hidden_size),
            dtype=np.float32)
        self.somu_cell_states_actor = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.num_somu_lstm, self.somu_lstm_hidden_size),
            dtype=np.float32)
        self.scmu_hidden_states_actor = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.num_scmu_lstm, self.scmu_lstm_hidden_size),
            dtype=np.float32)
        self.scmu_cell_states_actor = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.num_scmu_lstm, self.scmu_lstm_hidden_size),
            dtype=np.float32)

        self.rnn_states_critic = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)

        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, act_space.n),
                                             dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(self, share_obs, obs, somu_hidden_states_actor, somu_cell_states_actor, scmu_hidden_states_actor, 
               scmu_cell_states_actor, rnn_states_critic, actions, action_log_probs, value_preds, rewards, masks, bad_masks=None, 
               active_masks=None, available_actions=None):
        """
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param somu_hidden_states_actor: (np.ndarray) hidden states for somu (LSTMCell) network.
        :param somu_cell_states_actor: (np.ndarray) cell states for somu (LSTMCell) network.
        :param scmu_hidden_states_actor: (np.ndarray) hidden states for scmu (LSTMCell) network.
        :param scmu_cell_states_actor: (np.ndarray) hidden states for scmu (LSTMCell) network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) action space for agents.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.somu_hidden_states_actor[self.step + 1] = somu_hidden_states_actor.copy()
        self.somu_cell_states_actor[self.step + 1] = somu_cell_states_actor.copy()
        self.scmu_hidden_states_actor[self.step + 1] = scmu_hidden_states_actor.copy()
        self.scmu_cell_states_actor[self.step + 1] = scmu_cell_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def chooseinsert(self, share_obs, obs, somu_hidden_states_actor, somu_cell_states_actor, scmu_hidden_states_actor, 
                     scmu_cell_states_actor, rnn_states_critic, actions, action_log_probs, value_preds, rewards, masks, bad_masks=None, 
                     active_masks=None, available_actions=None):
        """
        Insert data into the buffer. This insert function is used specifically for Hanabi, which is turn based.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param somu_hidden_states_actor: (np.ndarray) hidden states for somu (LSTMCell) network.
        :param somu_cell_states_actor: (np.ndarray) cell states for somu (LSTMCell) network.
        :param scmu_hidden_states_actor: (np.ndarray) hidden states for scmu (LSTMCell) network.
        :param scmu_cell_states_actor: (np.ndarray) hidden states for scmu (LSTMCell) network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) denotes indicate whether whether true terminal state or due to episode limit
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.somu_hidden_states_actor[self.step + 1] = somu_hidden_states_actor.copy()
        self.somu_cell_states_actor[self.step + 1] = somu_cell_states_actor.copy()
        self.scmu_hidden_states_actor[self.step + 1] = scmu_hidden_states_actor.copy()
        self.scmu_cell_states_actor[self.step + 1] = scmu_cell_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.somu_hidden_states_actor[self.step + 1] = self.somu_hidden_states_actor[-1].copy()
        self.somu_cell_states_actor[self.step + 1] = self.somu_cell_states_actor[-1].copy()
        self.scmu_hidden_states_actor[self.step + 1] = self.scmu_hidden_states_actor[-1].copy()
        self.scmu_cell_states_actor[self.step + 1] = self.scmu_cell_states_actor[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def chooseafter_update(self):
        """Copy last timestep data to first index. This method is used for Hanabi."""
        self.somu_hidden_states_actor[self.step + 1] = self.somu_hidden_states_actor[-1].copy()
        self.somu_cell_states_actor[self.step + 1] = self.somu_cell_states_actor[-1].copy()
        self.scmu_hidden_states_actor[self.step + 1] = self.scmu_hidden_states_actor[-1].copy()
        self.scmu_cell_states_actor[self.step + 1] = self.scmu_cell_states_actor[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(
                            self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP / GNN policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        # episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        # batch_size = n_rollout_threads * episode_length * num_agents

        # if mini_batch_size is None:
        #     assert batch_size >= num_mini_batch, (
        #         "PPO requires the number of processes ({}) "
        #         "* number of steps ({}) * number of agents ({}) = {} "
        #         "to be greater than or equal to the number of PPO mini batches ({})."
        #         "".format(n_rollout_threads, episode_length, num_agents,
        #                   n_rollout_threads * episode_length * num_agents,
        #                   num_mini_batch))
        #     mini_batch_size = batch_size // num_mini_batch

        # rand = torch.randperm(batch_size).numpy()
        # sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        # share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
        # obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        # somu_hidden_states_actor = self.somu_hidden_states_actor[:-1].reshape(-1, *self.somu_hidden_states_actor.shape[3:])
        # somu_cell_states_actor = self.somu_cell_states_actor[:-1].reshape(-1, *self.somu_cell_states_actor.shape[3:])
        # scmu_hidden_states_actor = self.scmu_hidden_states_actor[:-1].reshape(-1, *self.scmu_hidden_states_actor.shape[3:])
        # scmu_cell_states_actor = self.scmu_cell_states_actor[:-1].reshape(-1, *self.scmu_cell_states_actor.shape[3:])
        # rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[3:])
        # actions = self.actions.reshape(-1, self.actions.shape[-1])
        # if self.available_actions is not None:
        #     available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        # value_preds = self.value_preds[:-1].reshape(-1, 1)
        # returns = self.returns[:-1].reshape(-1, 1)
        # masks = self.masks[:-1].reshape(-1, 1)
        # active_masks = self.active_masks[:-1].reshape(-1, 1)
        # action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        # advantages = advantages.reshape(-1, 1)

        # for indices in sampler:
        #     # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
        #     share_obs_batch = share_obs[indices]
        #     obs_batch = obs[indices]
        #     somu_hidden_states_actor_batch = somu_hidden_states_actor[indices]
        #     somu_cell_states_actor_batch = somu_cell_states_actor[indices]
        #     scmu_hidden_states_actor_batch = scmu_hidden_states_actor[indices]
        #     scmu_cell_states_actor_batch = scmu_cell_states_actor[indices]
        #     rnn_states_critic_batch = rnn_states_critic[indices]
        #     actions_batch = actions[indices]
        #     if self.available_actions is not None:
        #         available_actions_batch = available_actions[indices]
        #     else:
        #         available_actions_batch = None
        #     value_preds_batch = value_preds[indices]
        #     return_batch = returns[indices]
        #     masks_batch = masks[indices]
        #     active_masks_batch = active_masks[indices]
        #     old_action_log_probs_batch = action_log_probs[indices]
        #     if advantages is None:
        #         adv_targ = None
        #     else:
        #         adv_targ = advantages[indices]

        #     yield share_obs_batch, obs_batch, somu_hidden_states_actor_batch, somu_cell_states_actor_batch, \
        #           scmu_hidden_states_actor_batch, scmu_cell_states_actor_batch, rnn_states_critic_batch, actions_batch, \
        #           value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        #           adv_targ, available_actions_batch

        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, num_agents,
                          n_rollout_threads * episode_length * num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        # rand = torch.randperm(batch_size // num_agents).numpy()
        rand = torch.arange(batch_size // num_agents).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, self.num_agents, *self.share_obs.shape[3:])
        obs = self.obs[:-1].reshape(-1, self.num_agents, *self.obs.shape[3:])
        somu_hidden_states_actor = self.somu_hidden_states_actor[:-1].reshape(-1, self.num_agents, *self.somu_hidden_states_actor.shape[3:])
        somu_cell_states_actor = self.somu_cell_states_actor[:-1].reshape(-1, self.num_agents, *self.somu_cell_states_actor.shape[3:])
        scmu_hidden_states_actor = self.scmu_hidden_states_actor[:-1].reshape(-1, self.num_agents, *self.scmu_hidden_states_actor.shape[3:])
        scmu_cell_states_actor = self.scmu_cell_states_actor[:-1].reshape(-1, self.num_agents, *self.scmu_cell_states_actor.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, self.num_agents, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, self.num_agents, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.num_agents, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, self.num_agents, 1)
        returns = self.returns[:-1].reshape(-1, self.num_agents, 1)
        masks = self.masks[:-1].reshape(-1, self.num_agents, 1)
        active_masks = self.active_masks[:-1].reshape(-1, self.num_agents, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.num_agents, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, self.num_agents, 1)

        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N,M,Dim]-->[index,M,Dim]-->[index*M,Dim]
            share_obs_batch = share_obs[indices].reshape(-1, *self.share_obs.shape[3:])
            obs_batch = obs[indices].reshape(-1, *self.obs.shape[3:])
            somu_hidden_states_actor_batch = somu_hidden_states_actor[indices].reshape(-1, *self.somu_hidden_states_actor.shape[3:])
            somu_cell_states_actor_batch = somu_cell_states_actor[indices].reshape(-1, *self.somu_cell_states_actor.shape[3:])
            scmu_hidden_states_actor_batch = scmu_hidden_states_actor[indices].reshape(-1, *self.scmu_hidden_states_actor.shape[3:])
            scmu_cell_states_actor_batch = scmu_cell_states_actor[indices].reshape(-1, *self.scmu_cell_states_actor.shape[3:])
            rnn_states_critic_batch = rnn_states_critic[indices].reshape(-1, *self.rnn_states_critic.shape[3:])
            actions_batch = actions[indices].reshape(-1, self.actions.shape[-1])
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices].reshape(-1, self.available_actions.shape[-1])
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices].reshape(-1, 1)
            return_batch = returns[indices].reshape(-1, 1)
            masks_batch = masks[indices].reshape(-1, 1)
            active_masks_batch = active_masks[indices].reshape(-1, 1)
            old_action_log_probs_batch = action_log_probs[indices].reshape(-1, self.action_log_probs.shape[-1])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, 1)

            yield share_obs_batch, obs_batch, somu_hidden_states_actor_batch, somu_cell_states_actor_batch, \
                  scmu_hidden_states_actor_batch, scmu_cell_states_actor_batch, rnn_states_critic_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch
