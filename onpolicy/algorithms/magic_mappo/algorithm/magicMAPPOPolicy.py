import torch
from onpolicy.algorithms.magic_mappo.algorithm.magic_actor_critic import MAGIC_Actor, MAGIC_Critic
from onpolicy.utils.util import update_linear_schedule


class MAGIC_MAPPOPolicy:
    """
    MAGIC MAPPO Policy class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = MAGIC_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = MAGIC_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, lstm_hidden_states_actor, lstm_cell_states_actor, 
                    lstm_hidden_states_critic, lstm_cell_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param lstm_hidden_states_actor: (np.ndarray) hidden states for lstm for actor network.
        :param lstm_cell_states_actor: (np.ndarray) cell states for lstm for actor network.
        :param lstm_hidden_states_critic: (np.ndarray) hidden states for lstm for critic network.
        :param lstm_cell_states_critic: (np.ndarray) cell states for lstm for critic network.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return lstm_hidden_states_actor: (torch.Tensor) updated hidden states for lstm for actor network.
        :return lstm_cell_states_actor: (torch.Tensor) updated cell states for lstm for actor network.
        :return lstm_hidden_states_critic: (torch.Tensor) updated hidden states for lstm for critic network.
        :return lstm_cell_states_critic: (torch.Tensor) updated cell states for lstm for critic network.
        """
        actions, action_log_probs, lstm_hidden_states_actor, lstm_cell_states_actor = \
            self.actor(obs,
                       lstm_hidden_states_actor,
                       lstm_cell_states_actor,
                       masks,
                       available_actions,
                       deterministic)

        values, lstm_hidden_states_critic, lstm_cell_states_critic = self.critic(cent_obs,
                                                                                 lstm_hidden_states_critic,
                                                                                 lstm_cell_states_critic, 
                                                                                 masks)
        return values, actions, action_log_probs, lstm_hidden_states_actor, lstm_cell_states_actor, \
               lstm_hidden_states_critic, lstm_cell_states_critic

    def get_values(self, cent_obs, lstm_hidden_states_critic, lstm_cell_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param lstm_hidden_states_critic: (np.ndarray) hidden states for lstm for critic network.
        :param lstm_cell_states_critic: (np.ndarray) cell states for lstm for critic network.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _, _ = self.critic(cent_obs, lstm_hidden_states_critic, lstm_cell_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, lstm_hidden_states_actor, lstm_cell_states_actor, 
                         lstm_hidden_states_critic, lstm_cell_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param lstm_hidden_states_actor: (np.ndarray) hidden states for lstm for actor network.
        :param lstm_cell_states_actor: (np.ndarray) cell states for lstm for actor network.
        :param lstm_hidden_states_critic: (np.ndarray) hidden states for lstm for critic network.
        :param lstm_cell_states_critic: (np.ndarray) cell states for lstm for critic network.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     lstm_hidden_states_actor,
                                                                     lstm_cell_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values = self.critic.evaluate_actions(cent_obs, lstm_hidden_states_critic, lstm_cell_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, lstm_hidden_states_actor, lstm_cell_states_actor, masks, available_actions=None, 
            deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param lstm_hidden_states_actor: (np.ndarray) hidden states for lstm for actor network.
        :param lstm_cell_states_actor: (np.ndarray) cell states for lstm for actor network.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, lstm_hidden_states_actor, lstm_cell_states_actor = \
            self.actor(obs, 
                       lstm_hidden_states_actor, 
                       lstm_cell_states_actor, 
                       masks, 
                       available_actions, 
                       deterministic)
        return actions, lstm_hidden_states_actor, lstm_cell_states_actor
