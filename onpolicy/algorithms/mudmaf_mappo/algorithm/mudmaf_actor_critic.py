import torch
import torch.nn as nn

from functools import partial
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.lstm import LSTMLayer
from onpolicy.algorithms.utils.nn import Conv2DAutoPadding, MLPBlock, NNLayers, VGGBlock
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.algorithms.utils.util import check, init
from onpolicy.utils.util import get_shape_from_obs_space

class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)
     
        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states

class MuDMAF_Actor(nn.Module):
    """
    MuDMAF actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.mudmaf_conv_output_dims = args.mudmaf_conv_output_dims
        self.mudmaf_n_vgg_conv_layers = args.mudmaf_n_vgg_conv_layers
        self.mudmaf_vgg_conv_kernel_size = args.mudmaf_vgg_conv_kernel_size
        self.mudmaf_vgg_maxpool_kernel_size = args.mudmaf_vgg_maxpool_kernel_size
        self.mudmaf_n_goal_fc_layers = args.mudmaf_n_goal_fc_layers
        self.mudmaf_n_post_concat_fc_layers = args.mudmaf_n_post_concat_fc_layers
        self.mudmaf_lstm_hidden_size = args.mudmaf_lstm_hidden_size
        self.mudmaf_lstm_n_layers = args.mudmaf_lstm_n_layers       

        obs_shape = get_shape_from_obs_space(obs_space)
        if isinstance(obs_shape, (list, tuple)):
            self.obs_dims = obs_shape[0]
        else:
            self.obs_dims = obs_shape
        
        # model architecture for MuDMAF_Actor

        # vgg convolution layers
        self.vgg_conv_layers = NNLayers(input_channels=self.obs_dims, 
                                        block=VGGBlock, 
                                        output_channels=[mudmaf_conv_output_dims \
                                                         for _ in range(mudmaf_n_vgg_conv_layers)], 
                                        activation_func = 'relu', 
                                        conv = partial(Conv2DAutoPadding, 
                                                       kernel_size=mudmaf_vgg_conv_kernel_size, 
                                                       bias=False), 
                                        dropout_p=0, 
                                        max_pool_kernel=self.mudmaf_vgg_maxpool_kernel_size)
        # goal fc layers 
        self.goal_fc_layers = NNLayers(input_channels=2, 
                                       block=MLPBlock, 
                                       output_channels=[self.mudmaf_lstm_hidden_size - \
                                                        self.vgg_conv_layers.get_flat_output_shape(self.obs_dims)
                                                        for _ in range(self.mudmaf_n_goal_fc_layers)], 
                                       activation_func='relu', 
                                       dropout_p=0, 
                                       weight_initialisation="orthogonal" if self._use_orthogonal else "default")
        # post concat fc layers
        self.post_concat_fc_layers = NNLayers(input_channels=self.mudmaf_lstm_hidden_size, 
                                              block=MLPBlock, 
                                              output_channels=[self.mudmaf_lstm_hidden_size \
                                                               for _ in range(self.mudmaf_n_post_concat_fc_layers)], 
                                              activation_func='relu', 
                                              dropout_p=0, 
                                              weight_initialisation="orthogonal" if self._use_orthogonal else "default")
        # lstm layer
        self.lstm_layer = LSTMLayer(input_size=self.mudmaf_lstm_hidden_size, 
                                    hidden_size=self.mudmaf_lstm_hidden_size, 
                                    num_layers=self.mudmaf_lstm_n_layers, 
                                    use_orthogonal=self._use_orthogonal)
        # shared final action layer for each agent
        self.act = SingleACTLayer(action_space=action_space, 
                                  inputs_dim=self.mudmaf_lstm_hidden_size, 
                                  use_orthogonal=self._use_orthogonal, 
                                  gain=self._gain)

        self.to(device) 

    def forward(self, obs, goals, lstm_hidden_states_actor, lstm_cell_states_actor, masks, available_actions=None, 
                deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param goals: (np.ndarray / torch.Tensor) goal inputs into network.
        :param lstm_hidden_states_actor: (np.ndarray / torch.Tensor) hidden states for lstm for actor network.
        :param lstm_cell_states_actor: (np.ndarray / torch.Tensor) cell states for lstm for actor network.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return lstm_hidden_states_actor: (torch.Tensor) updated hidden states for lstm for actor network.
        :return lstm_cell_states_actor: (torch.Tensor) updated cell states for lstm for actor network.
        """
        obs = check(obs).to(**self.tpdv)
        goals = check(goals).to(**self.tpdv)
        lstm_hidden_states_actor = check(lstm_hidden_states_actor).to(**self.tpdv)
        lstm_cell_states_actor = check(lstm_cell_states_actor).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        # obs --> vgg_conv_layers
        obs_features = self.vgg_conv_layers(obs)
        # goals --> goal_fc_layers
        goal_features = self.goal_fc_layers(goals)
        # flatten obs_features 
        obs_features_flatten = torch.flatten(obs_features, start_dim=1)
        # concatenate obs and goal features
        concat_features = torch.cat((obs_features_flatten, goal_features), dim=-1)
        # concat_features --> post_concat_fc_layers
        post_concat_features = self.post_concat_fc_layers(concat_features)
        # residual connection between concat_features and post_concat_features
        actor_features = concat_features + post_concat_features
        # actor_features --> lstm_layer
        actor_features, (lstm_hidden_states_actor, lstm_cell_states_actor) = \
            self.lstm_layer(actor_features, lstm_hidden_states_actor, lstm_cell_states_actor, masks)
        # lstm --> actions
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, lstm_hidden_states_actor, lstm_cell_states_actor

    def evaluate_actions(self, obs, goals, lstm_hidden_states_actor, lstm_cell_states_actor, action, masks, 
                         available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param goals: (torch.Tensor) goal inputs into network.
        :param lstm_hidden_states_actor: (torch.Tensor) hidden states for lstm for actor network.
        :param lstm_cell_states_actor: (torch.Tensor) cell states for lstm for actor network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        goals = check(goals).to(**self.tpdv)
        lstm_hidden_states_actor = check(lstm_hidden_states_actor).to(**self.tpdv)
        lstm_cell_states_actor = check(lstm_cell_states_actor).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        # obs --> vgg_conv_layers
        obs_features = self.vgg_conv_layers(obs)
        # goals --> goal_fc_layers
        goal_features = self.goal_fc_layers(goals)
        # flatten obs_features 
        obs_features_flatten = torch.flatten(obs_features, start_dim=1)
        # concatenate obs and goal features
        concat_features = torch.cat((obs_features_flatten, goal_features), dim=-1)
        # concat_features --> post_concat_fc_layers
        post_concat_features = self.post_concat_fc_layers(concat_features)
        # residual connection between concat_features and post_concat_features
        actor_features = concat_features + post_concat_features
        # actor_features --> lstm_layer
        actor_features, (lstm_hidden_states_actor, lstm_cell_states_actor) = \
            self.lstm_layer(actor_features, lstm_hidden_states_actor, lstm_cell_states_actor, masks)
        # evaluate actions to obtain log probabilities and distribution entropy
        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)
        return action_log_probs, dist_entropy

class MuDMAF_Critic(nn.Module):
    """
    MuDMAF critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
    
        self._use_orthogonal = args.use_orthogonal
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = args.num_agents
        self.use_centralized_V = arg.use_centralized_V
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.mudmaf_conv_output_dims = args.mudmaf_conv_output_dims
        self.mudmaf_n_vgg_conv_layers = args.mudmaf_n_vgg_conv_layers
        self.mudmaf_vgg_conv_kernel_size = args.mudmaf_vgg_conv_kernel_size
        self.mudmaf_vgg_maxpool_kernel_size = args.mudmaf_vgg_maxpool_kernel_size
        self.mudmaf_n_goal_fc_layers = args.mudmaf_n_goal_fc_layers
        self.mudmaf_n_post_concat_fc_layers = args.mudmaf_n_post_concat_fc_layers
        self.mudmaf_lstm_hidden_size = args.mudmaf_lstm_hidden_size
        self.mudmaf_lstm_n_layers = args.mudmaf_lstm_n_layers   

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        if isinstance(obs_shape, (list, tuple)):
            self.obs_dims = cent_obs_shape[0]
        else:
            self.obs_dims = cent_obs_shape

        # model architecture for MuDMAF_Critic

        # vgg convolution layers
        self.vgg_conv_layers = NNLayers(input_channels=self.obs_dims, 
                                        block=VGGBlock, 
                                        output_channels=[mudmaf_conv_output_dims \
                                                         for _ in range(mudmaf_n_vgg_conv_layers)], 
                                        activation_func = 'relu', 
                                        conv = partial(Conv2DAutoPadding, 
                                                       kernel_size=mudmaf_vgg_conv_kernel_size, 
                                                       bias=False), 
                                        dropout_p=0, 
                                        max_pool_kernel=self.mudmaf_vgg_maxpool_kernel_size)
        # goal fc layers 
        self.goal_fc_layers = NNLayers(input_channels=2 * self.num_agents if self.use_centralized_V else 2, 
                                       block=MLPBlock, 
                                       output_channels=[self.mudmaf_lstm_hidden_size - \
                                                        self.vgg_conv_layers.get_flat_output_shape(self.obs_dims)
                                                        for _ in range(self.mudmaf_n_goal_fc_layers)], 
                                       activation_func='relu', 
                                       dropout_p=0, 
                                       weight_initialisation="orthogonal" if self._use_orthogonal else "default")
        # post concat fc layers
        self.post_concat_fc_layers = NNLayers(input_channels=self.mudmaf_lstm_hidden_size, 
                                              block=MLPBlock, 
                                              output_channels=[self.mudmaf_lstm_hidden_size \
                                                               for _ in range(self.mudmaf_n_post_concat_fc_layers)], 
                                              activation_func='relu', 
                                              dropout_p=0, 
                                              weight_initialisation="orthogonal" if self._use_orthogonal else "default")
        # lstm layer
        self.lstm_layer = LSTMLayer(input_size=self.mudmaf_lstm_hidden_size, 
                                    hidden_size=self.mudmaf_lstm_hidden_size, 
                                    num_layers=self.mudmaf_lstm_n_layers, 
                                    use_orthogonal=self._use_orthogonal)
        # value function
        if self._use_popart:
            self.v_out = init_(PopArt(self.mudmaf_lstm_hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.mudmaf_lstm_hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, goals, lstm_hidden_states_critic, lstm_cell_states_critic, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param goals: (np.ndarray / torch.Tensor) goal inputs into network.
        :param lstm_hidden_states_critic: (np.ndarray / torch.Tensor) hidden states for lstm for critic network.
        :param lstm_cell_states_critic: (np.ndarray / torch.Tensor) cell states for lstm for critic network.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return lstm_hidden_states_critic: (torch.Tensor) updated hidden states for lstm for critic network.
        :return lstm_cell_states_critic: (torch.Tensor) updated cell states for lstm for critic network.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        goals = check(goals).to(**self.tpdv)
        lstm_hidden_states_critic = check(lstm_hidden_states_critic).to(**self.tpdv)
        lstm_cell_states_critic = check(lstm_cell_states_critic).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # obs --> vgg_conv_layers
        obs_features = self.vgg_conv_layers(obs)
        # goals --> goal_fc_layers
        goal_features = self.goal_fc_layers(goals)
        # flatten obs_features 
        obs_features_flatten = torch.flatten(obs_features, start_dim=1)
        # concatenate obs and goal features
        concat_features = torch.cat((obs_features_flatten, goal_features), dim=-1)
        # concat_features --> post_concat_fc_layers
        post_concat_features = self.post_concat_fc_layers(concat_features)
        # residual connection between concat_features and post_concat_features
        critic_features = concat_features + post_concat_features
        # actor_features --> lstm_layer
        critic_features, (lstm_hidden_states_critic, lstm_cell_states_critic) = \
            self.lstm_layer(actor_features, lstm_hidden_states_critic, lstm_cell_states_critic, masks)
        # lstm_layer --> value 
        values = self.v_out(critic_features)

        return values, lstm_hidden_states_critic, lstm_cell_states_critic