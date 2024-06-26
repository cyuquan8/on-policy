import torch

from onpolicy.algorithms.gcmnet_mappo.algorithm.gcmnet_actor_critic import GCMNetActor, GCMNetCritic
from onpolicy.utils.util import update_linear_schedule


class GCMNet_MAPPOPolicy:
    """
    GCMNet_MAPPO Policy class. Wraps actor and critic networks to compute actions and value function predictions.

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

        self.actor = GCMNetActor(args, self.obs_space, self.act_space, self.device)
        self.critic = GCMNetCritic(args, self.share_obs_space, self.device)

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

    def get_actions(self, cent_obs, obs, masks, available_actions=None, somu_hidden_states_actor=None, 
                    somu_cell_states_actor=None, scmu_hidden_states_actor=None, scmu_cell_states_actor=None, 
                    somu_hidden_states_critic=None, somu_cell_states_critic=None, scmu_hidden_states_critic=None, 
                    scmu_cell_states_critic=None, deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param masks: (np.ndarray) denotes points at which somu, scmu hidden states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param somu_hidden_states_actor: (np.ndarray) hidden states for somu network.
        :param somu_cell_states_actor: (np.ndarray) cell states for somu network.
        :param scmu_hidden_states_actor: (np.ndarray) hidden states for scmu network.
        :param scmu_cell_states_actor: (np.ndarray) hidden states for scmu network.
        :param somu_hidden_states_critic: (np.ndarray) hidden states for somu network.
        :param somu_cell_states_critic: (np.ndarray) cell states for somu network.
        :param scmu_hidden_states_critic: (np.ndarray) hidden states for scmu network.
        :param scmu_cell_states_critic: (np.ndarray) hidden states for scmu network.
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return somu_hidden_states_actor: (torch.Tensor) hidden states for somu network.
        :return somu_cell_states_actor: (torch.Tensor) cell states for somu network.
        :return scmu_hidden_states_actor: (torch.Tensor) hidden states for scmu network.
        :return scmu_cell_states_actor: (torch.Tensor) hidden states for scmu network.
        :return somu_hidden_states_critic: (torch.Tensor) hidden states for somu network.
        :return somu_cell_states_critic: (torch.Tensor) cell states for somu network.
        :return scmu_hidden_states_critic: (torch.Tensor) hidden states for scmu network.
        :return scmu_cell_states_critic: (torch.Tensor) hidden states for scmu network.
        :return obs_pred: (torch.Tensor) observation predictions from dynamics models if used else None.
        """
        actions, action_log_probs, somu_hidden_states_actor, somu_cell_states_actor, \
        scmu_hidden_states_actor, scmu_cell_states_actor, obs_pred = \
            self.actor(
                obs=obs,
                masks=masks, 
                available_actions=available_actions,
                somu_hidden_states_actor=somu_hidden_states_actor,
                somu_cell_states_actor=somu_cell_states_actor,
                scmu_hidden_states_actor=scmu_hidden_states_actor,
                scmu_cell_states_actor=scmu_cell_states_actor,
                deterministic=deterministic
            )
        values, somu_hidden_states_critic, somu_cell_states_critic, \
        scmu_hidden_states_critic, scmu_cell_states_critic = \
            self.critic(
                    cent_obs=cent_obs, 
                    masks=masks,
                    somu_hidden_states_critic=somu_hidden_states_critic,
                    somu_cell_states_critic=somu_cell_states_critic,
                    scmu_hidden_states_critic=scmu_hidden_states_critic,
                    scmu_cell_states_critic=scmu_cell_states_critic
                )
        return values, actions, action_log_probs, somu_hidden_states_actor, somu_cell_states_actor, \
               scmu_hidden_states_actor, scmu_cell_states_actor, somu_hidden_states_critic, \
               somu_cell_states_critic, scmu_hidden_states_critic, scmu_cell_states_critic, obs_pred

    def get_values(self, cent_obs, masks, somu_hidden_states_critic=None, somu_cell_states_critic=None, 
                   scmu_hidden_states_critic=None, scmu_cell_states_critic=None):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param masks: (np.ndarray) denotes points at which somu, scmu hidden states should be reset.
        :param somu_hidden_states_critic: (np.ndarray) hidden states for somu network.
        :param somu_cell_states_critic: (np.ndarray) cell states for somu network.
        :param scmu_hidden_states_critic: (np.ndarray) hidden states for scmu network.
        :param scmu_cell_states_critic: (np.ndarray) hidden states for scmu network.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _, _, _, _ = self.critic(cent_obs=cent_obs,
                                         masks=masks,
                                         somu_hidden_states_critic=somu_hidden_states_critic,
                                         somu_cell_states_critic=somu_cell_states_critic,
                                         scmu_hidden_states_critic=scmu_hidden_states_critic,
                                         scmu_cell_states_critic=scmu_cell_states_critic 
                                        )
        return values

    def evaluate_actions(self, cent_obs, obs, action, masks, available_actions=None, active_masks=None, 
                         somu_hidden_states_actor=None, somu_cell_states_actor=None, scmu_hidden_states_actor=None, 
                         scmu_cell_states_actor=None, somu_hidden_states_critic=None, somu_cell_states_critic=None, 
                         scmu_hidden_states_critic=None, scmu_cell_states_critic=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which somu, scmu hidden states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.
        :param somu_hidden_states_actor: (np.ndarray) hidden states for somu network.
        :param somu_cell_states_actor: (np.ndarray) cell states for somu network.
        :param scmu_hidden_states_actor: (np.ndarray) hidden states for scmu network.
        :param scmu_cell_states_actor: (np.ndarray) hidden states for scmu network.
        :param somu_hidden_states_critic: (np.ndarray) hidden states for somu network.
        :param somu_cell_states_critic: (np.ndarray) cell states for somu network.
        :param scmu_hidden_states_critic: (np.ndarray) hidden states for scmu network.
        :param scmu_cell_states_critic: (np.ndarray) hidden states for scmu network.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        :return obs_pred: (torch.Tensor) observation predictions from dynamics models if used else None.
        """
        action_log_probs, dist_entropy, obs_pred = \
            self.actor.evaluate_actions(obs=obs,
                                        action=action,
                                        masks=masks,
                                        available_actions=available_actions,
                                        active_masks=active_masks,
                                        somu_hidden_states_actor=somu_hidden_states_actor,
                                        somu_cell_states_actor=somu_cell_states_actor,
                                        scmu_hidden_states_actor=scmu_hidden_states_actor,
                                        scmu_cell_states_actor=scmu_cell_states_actor
                                       )
        values = self.critic.evaluate_actions(cent_obs=cent_obs,
                                              masks=masks,
                                              somu_hidden_states_critic=somu_hidden_states_critic,
                                              somu_cell_states_critic=somu_cell_states_critic,
                                              scmu_hidden_states_critic=scmu_hidden_states_critic,
                                              scmu_cell_states_critic=scmu_cell_states_critic 
                                             )
        return values, action_log_probs, dist_entropy, obs_pred

    def act(self, obs, masks, available_actions=None, somu_hidden_states_actor=None, somu_cell_states_actor=None, 
            scmu_hidden_states_actor=None, scmu_cell_states_actor=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param masks: (np.ndarray) denotes points at which somu, scmu hidden states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param somu_hidden_states_actor: (np.ndarray) hidden states for somu network.
        :param somu_cell_states_actor: (np.ndarray) cell states for somu network.
        :param scmu_hidden_states_actor: (np.ndarray) hidden states for scmu network.
        :param scmu_cell_states_actor: (np.ndarray) hidden states for scmu network.
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return actions: (torch.Tensor) actions to take.
        :return somu_hidden_states_actor: (torch.Tensor) hidden states for somu network.
        :return somu_cell_states_actor: (torch.Tensor) cell states for somu network.
        :return scmu_hidden_states_actor: (torch.Tensor) hidden states for scmu network.
        :return scmu_cell_states_actor: (torch.Tensor) hidden states for scmu network.
        """
        actions, _ , somu_hidden_states_actor, somu_cell_states_actor, \
        scmu_hidden_states_actor, scmu_cell_states_actor, _ = \
            self.actor(
                obs=obs,
                masks=masks, 
                available_actions=available_actions,
                somu_hidden_states_actor=somu_hidden_states_actor,
                somu_cell_states_actor=somu_cell_states_actor,
                scmu_hidden_states_actor=scmu_hidden_states_actor,
                scmu_cell_states_actor=scmu_cell_states_actor,
                deterministic=deterministic
            )
        return actions, somu_hidden_states_actor, somu_cell_states_actor, scmu_hidden_states_actor, \
               scmu_cell_states_actor