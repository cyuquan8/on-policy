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
        self.somu_n_layers = args.somu_n_layers
        self.scmu_num_layers = args.scmu_num_layers
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

        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, *share_obs_shape),
                                  dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)

        self.somu_hidden_states_actor = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.num_agents, self.somu_n_layers, self.somu_lstm_hidden_size),
            dtype=np.float32)
        self.somu_cell_states_actor = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.num_agents, self.somu_n_layers, self.somu_lstm_hidden_size),
            dtype=np.float32)
        self.scmu_hidden_states_actor = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.num_agents, self.scmu_n_layers, self.scmu_lstm_hidden_size),
            dtype=np.float32)
        self.scmu_cell_states_actor = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.num_agents, self.scmu_n_layers, self.scmu_lstm_hidden_size),
            dtype=np.float32)

        self.somu_hidden_states_critic = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.num_agents, self.somu_n_layers, self.somu_lstm_hidden_size),
            dtype=np.float32)
        self.somu_cell_states_critic = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.num_agents, self.somu_n_layers, self.somu_lstm_hidden_size),
            dtype=np.float32)
        self.scmu_hidden_states_critic = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.num_agents, self.scmu_n_layers, self.scmu_lstm_hidden_size),
            dtype=np.float32)
        self.scmu_cell_states_critic = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.num_agents, self.scmu_n_layers, self.scmu_lstm_hidden_size),
            dtype=np.float32)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)

        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, self.num_agents, act_space.n),
                                             dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, self.num_agents, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, self.num_agents, act_shape), dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(self, share_obs, obs, somu_hidden_states_actor, somu_cell_states_actor, scmu_hidden_states_actor, 
               scmu_cell_states_actor, somu_hidden_states_critic, somu_cell_states_critic, scmu_hidden_states_critic, 
               scmu_cell_states_critic, actions, action_log_probs, value_preds, rewards, masks, bad_masks=None, 
               active_masks=None, available_actions=None):
        """
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param somu_hidden_states_actor: (np.ndarray) hidden states for somu in actor network.
        :param somu_cell_states_actor: (np.ndarray) cell states for somu in actor network.
        :param scmu_hidden_states_actor: (np.ndarray) hidden states for scmu in actor network.
        :param scmu_cell_states_actor: (np.ndarray) hidden states for scmu in actor network.
        :param somu_hidden_states_critic: (np.ndarray) hidden states for somu in critic network.
        :param somu_cell_states_critic: (np.ndarray) cell states for somu in critic network.
        :param scmu_hidden_states_critic: (np.ndarray) hidden states for scmu in critic network.
        :param scmu_cell_states_critic: (np.ndarray) hidden states for scmu in critic network.
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
        self.somu_hidden_states_critic[self.step + 1] = somu_hidden_states_critic.copy()
        self.somu_cell_states_critic[self.step + 1] = somu_cell_states_critic.copy()
        self.scmu_hidden_states_critic[self.step + 1] = scmu_hidden_states_critic.copy()
        self.scmu_cell_states_critic[self.step + 1] = scmu_cell_states_critic.copy()
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
                     scmu_cell_states_actor, somu_hidden_states_critic, somu_cell_states_critic, scmu_hidden_states_critic, 
                     scmu_cell_states_critic, actions, action_log_probs, value_preds, rewards, masks, bad_masks=None, 
                     active_masks=None, available_actions=None):
        """
        Insert data into the buffer. This insert function is used specifically for Hanabi, which is turn based.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param somu_hidden_states_actor: (np.ndarray) hidden states for somu (LSTMCell) network.
        :param somu_cell_states_actor: (np.ndarray) cell states for somu (LSTMCell) network.
        :param scmu_hidden_states_actor: (np.ndarray) hidden states for scmu (LSTMCell) network.
        :param scmu_cell_states_actor: (np.ndarray) hidden states for scmu (LSTMCell) network.
        :param somu_hidden_states_critic: (np.ndarray) hidden states for somu in critic network.
        :param somu_cell_states_critic: (np.ndarray) cell states for somu in critic network.
        :param scmu_hidden_states_critic: (np.ndarray) hidden states for scmu in critic network.
        :param scmu_cell_states_critic: (np.ndarray) hidden states for scmu in critic network.
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
        self.somu_hidden_states_critic[self.step + 1] = somu_hidden_states_critic.copy()
        self.somu_cell_states_critic[self.step + 1] = somu_cell_states_critic.copy()
        self.scmu_hidden_states_critic[self.step + 1] = scmu_hidden_states_critic.copy()
        self.scmu_cell_states_critic[self.step + 1] = scmu_cell_states_critic.copy()
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
        self.somu_hidden_states_actor[0] = self.somu_hidden_states_actor[-1].copy()
        self.somu_cell_states_actor[0] = self.somu_cell_states_actor[-1].copy()
        self.scmu_hidden_states_actor[0] = self.scmu_hidden_states_actor[-1].copy()
        self.scmu_cell_states_actor[0] = self.scmu_cell_states_actor[-1].copy()
        self.somu_hidden_states_critic[0] = self.somu_hidden_states_critic[-1].copy()
        self.somu_cell_states_critic[0] = self.somu_cell_states_critic[-1].copy()
        self.scmu_hidden_states_critic[0] = self.scmu_hidden_states_critic[-1].copy()
        self.scmu_cell_states_critic[0] = self.scmu_cell_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def chooseafter_update(self):
        """Copy last timestep data to first index. This method is used for Hanabi."""
        self.somu_hidden_states_actor[0] = self.somu_hidden_states_actor[-1].copy()
        self.somu_cell_states_actor[0] = self.somu_cell_states_actor[-1].copy()
        self.scmu_hidden_states_actor[0] = self.scmu_hidden_states_actor[-1].copy()
        self.scmu_cell_states_actor[0] = self.scmu_cell_states_actor[-1].copy()
        self.somu_hidden_states_critic[0] = self.somu_hidden_states_critic[-1].copy()
        self.somu_cell_states_critic[0] = self.somu_cell_states_critic[-1].copy()
        self.scmu_hidden_states_critic[0] = self.scmu_hidden_states_critic[-1].copy()
        self.scmu_cell_states_critic[0] = self.scmu_cell_states_critic[-1].copy()
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

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        """
        Yield training data for chunked RNN / LSTM training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param data_chunk_length: (int) length of sequence chunks with which to train RNN / LSTM.
        """
        batch_size = self.n_rollout_threads * self.episode_length * self.num_agents
        data_chunks = batch_size // self.num_agents // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        # helper reshaping functions
        def _cast(x):
            return x.swapaxes(0, 1).reshape(-1, self.num_agents, *x.shape[3:])

        # [shape: (episode_length, n_rollout_threads, num_agents, *)] -->
        # [shape: (n_rollout_threads * episode_length, num_agents, *)] 
        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        # [shape: (episode_length + 1, n_rollout_threads, num_agents, *)] -->
        # [shape: (n_rollout_threads * episode_length, num_agents, *)] 
        share_obs = _cast(self.share_obs[:-1])
        obs = _cast(self.obs[:-1])
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        somu_hidden_states_actor = _cast(self.somu_hidden_states_actor[:-1])
        somu_cell_states_actor = _cast(self.somu_cell_states_actor[:-1])
        scmu_hidden_states_actor = _cast(self.scmu_hidden_states_actor[:-1])
        scmu_cell_states_actor = _cast(self.scmu_cell_states_actor[:-1])
        somu_hidden_states_critic = _cast(self.somu_hidden_states_critic[:-1])
        somu_cell_states_critic = _cast(self.somu_cell_states_critic[:-1])
        scmu_hidden_states_critic = _cast(self.scmu_hidden_states_critic[:-1])
        scmu_cell_states_critic = _cast(self.scmu_cell_states_critic[:-1])
        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            somu_hidden_states_actor_batch = []
            somu_cell_states_actor_batch = []
            scmu_hidden_states_actor_batch = []
            scmu_cell_states_actor_batch = [] 
            somu_hidden_states_critic_batch = [] 
            somu_cell_states_critic_batch = []
            scmu_hidden_states_critic_batch = []
            scmu_cell_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:
                ind = index * data_chunk_length
                # [shape: (data_chunk_length, num_agents, *)]
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                return_batch.append(returns[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind + data_chunk_length])
                adv_targ.append(advantages[ind:ind + data_chunk_length])
                # [shape: (1, num_agents, *)]
                somu_hidden_states_actor_batch.append(somu_hidden_states_actor[ind])
                somu_cell_states_actor_batch.append(somu_cell_states_actor[ind])
                scmu_hidden_states_actor_batch.append(scmu_hidden_states_actor[ind])
                scmu_cell_states_actor_batch.append(scmu_cell_states_actor[ind])
                somu_hidden_states_critic_batch.append(somu_hidden_states_critic[ind])
                somu_cell_states_critic_batch.append(somu_cell_states_critic[ind])
                scmu_hidden_states_critic_batch.append(scmu_hidden_states_critic[ind])
                scmu_cell_states_critic_batch.append(scmu_cell_states_critic[ind])

            # [shape: (mini_batch_size, data_chunk_length, num_agents, *)]           
            share_obs_batch = np.stack(share_obs_batch, axis=0)
            obs_batch = np.stack(obs_batch, axis=0)
            actions_batch = np.stack(actions_batch, axis=0)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=0)
            value_preds_batch = np.stack(value_preds_batch, axis=0)
            return_batch = np.stack(return_batch, axis=0)
            masks_batch = np.stack(masks_batch, axis=0)
            active_masks_batch = np.stack(active_masks_batch, axis=0)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=0)
            adv_targ = np.stack(adv_targ, axis=0)
            # [shape: (mini_batch_size, num_agents, *)]  
            somu_hidden_states_actor_batch = np.stack(somu_hidden_states_actor_batch)\
                                               .reshape(mini_batch_size, self.num_agents, 
                                                        *self.somu_hidden_states_actor.shape[3:])
            somu_cell_states_actor_batch = np.stack(somu_cell_states_actor_batch)\
                                             .reshape(mini_batch_size, self.num_agents, 
                                                      *self.somu_cell_states_actor.shape[3:])
            scmu_hidden_states_actor_batch = np.stack(scmu_hidden_states_actor_batch)\
                                               .reshape(mini_batch_size, self.num_agents, 
                                                        *self.scmu_hidden_states_actor.shape[3:])
            scmu_cell_states_actor_batch = np.stack(scmu_cell_states_actor_batch)\
                                             .reshape(mini_batch_size, self.num_agents, 
                                                      *self.scmu_cell_states_actor.shape[3:])
            somu_hidden_states_critic_batch = np.stack(somu_hidden_states_critic_batch)\
                                                .reshape(mini_batch_size, self.num_agents, 
                                                         *self.somu_hidden_states_critic.shape[3:])
            somu_cell_states_critic_batch = np.stack(somu_cell_states_critic_batch)\
                                              .reshape(mini_batch_size, self.num_agents, 
                                                       *self.somu_cell_states_critic.shape[3:])
            scmu_hidden_states_critic_batch = np.stack(scmu_hidden_states_critic_batch)\
                                                .reshape(mini_batch_size, self.num_agents, 
                                                         *self.scmu_hidden_states_critic.shape[3:])
            scmu_cell_states_critic_batch = np.stack(scmu_cell_states_critic_batch)\
                                              .reshape(mini_batch_size, self.num_agents, 
                                                       *self.scmu_cell_states_critic.shape[3:])

            yield share_obs_batch, obs_batch, somu_hidden_states_actor_batch, somu_cell_states_actor_batch, \
                  scmu_hidden_states_actor_batch, scmu_cell_states_actor_batch, somu_hidden_states_critic_batch, \
                  somu_cell_states_critic_batch, scmu_hidden_states_critic_batch, scmu_cell_states_critic_batch, \
                  actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, \
                  old_action_log_probs_batch, adv_targ, available_actions_batch