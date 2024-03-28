import torch
import torch.nn as nn

from onpolicy.algorithms.utils.nn import (
    MLPBlock, 
    NNLayers
)
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space


class CommNet_Actor(nn.Module):
    """
    CommNet Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(CommNet_Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.data_chunk_length = args.data_chunk_length
        self.num_agents = args.num_agents
        self.commnet_k = args.commnet_k

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        if len(obs_shape) == 3:
            raise NotImplementedError("CNN-based observations not implemented for CommNet")
        if isinstance(obs_shape, (list, tuple)):
            self.obs_dims = obs_shape[0]
        else:
            self.obs_dims = obs_shape
        self.act_dims = get_shape_from_act_space(action_space)

        self.obs_encoder = NNLayers(
            input_channels=self.obs_dims, 
            block=MLPBlock, 
            output_channels=[self.hidden_size],
            norm_type='none', 
            activation_func='relu', 
            dropout_p=0, 
            weight_initialisation="default"
        )
        self.obs_rnn = nn.GRU(
            input_size=self.hidden_size, 
            hidden_size=self.hidden_size, 
            num_layers=self._recurrent_N, 
            batch_first=True)
        self.comms_rnn = nn.GRU(
            input_size=self.hidden_size, 
            hidden_size=self.hidden_size, 
            num_layers=self._recurrent_N, 
            batch_first=True)
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
        # [shape: (batch_size * num_agents, obs_dims)]
        obs = check(obs).to(**self.tpdv)
        batch_size = obs.shape[0] // self.num_agents
        # [shape: (batch_size * num_agents, recurrent_N, hidden_size)]
        rnn_states = check(rnn_states).to(**self.tpdv)
        # [shape: (batch_size * num_agents, 1)]
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            # [shape: (batch_size * num_agents, act_dims)]
            available_actions = check(available_actions).to(**self.tpdv)

        # encode observation 
        # [shape: (batch_size * num_agents, obs_dims)] --> [shape: (batch_size * num_agents, 1, hidden_size)]
        h = self.obs_encoder(obs).unsqueeze(1)
        # zeros for communication state at start of communication [shape: (batch_size * num_agents, 1, hidden_size)]
        c = torch.zeros_like(h).to(**self.tpdv)
        # iterate over number of rounds of communication
        for k in range(self.commnet_k):
            if k == 0:
                # hidden state at start of communication, h [shape: (batch_size * num_agents, 1, hidden_size)]
                # rnn_states.transpose(0, 1).contiguous() [shape: (recurrent_N, batch_size * num_agents, hidden_size)]
                # masks.repeat(1, self._recurrent_N).transpose(0, 1).unsqueeze(-1).contiguous()
                # [shape: (recurrent_N, batch_size * num_agents, 1)] --> 
                # h [shape: (batch_size * num_agents, 1, hidden_size)]
                # rnn_states [shape: (recurrent_N, batch_size * num_agents, hidden_size)]
                h, rnn_states = self.obs_rnn(
                    h, 
                    rnn_states.transpose(0, 1).contiguous() * \
                    masks.repeat(1, self._recurrent_N).transpose(0, 1).unsqueeze(-1).contiguous()
                )
                # [shape: (recurrent_N, batch_size * num_agents, hidden_size)] --> 
                # [shape: (batch_size * num_agents, recurrent_N, hidden_size)]
                rnn_states = rnn_states.transpose(0, 1)
            else:
                # obtain hidden states of all agents for all agents
                # c [shape: (batch_size, 1, num_agents, hidden_size)] --> 
                # [shape: (batch_size, num_agents, num_agents, hidden_size)]
                c = h.reshape(batch_size, 1, self.num_agents, self.hidden_size).repeat(1, self.num_agents, 1, 1)
                # communication mask that precludes information from self 
                # [shape: (batch_size, num_agents, num_agents, 1)]
                m = (1 - torch.eye(self.num_agents)).unsqueeze(0).repeat(batch_size, 1, 1).unsqueeze(-1).to(**self.tpdv)
                # communication is average of neighbours hidden state
                # [shape: (batch_size, num_agents, num_agents, hidden_size)] --> 
                # [shape: (batch_size, num_agents, hidden_size)] --> [shape: (batch_size * num_agents, 1, hidden_size)]
                c = (c * m).mean(dim=2).reshape(batch_size * self.num_agents, 1, self.hidden_size)
            # c [shape: (batch_size * num_agents, 1, hidden_size)]
            # h.transpose(0, 1).contiguous() [shape: (1, batch_size * num_agents, hidden_size)] -->
            # h [shape: (1, batch_size * num_agents, hidden_size)]
            _, h = self.comms_rnn(c, h.transpose(0, 1).contiguous())
            # h [shape: (1, batch_size * num_agents, hidden_size)] --> 
            # [shape: (batch_size * num_agents, 1, hidden_size)]
            h = h.transpose(0, 1)

        # h [shape: (batch_size * num_agents, 1, hidden_size)] --> [shape: (batch_size * num_agents, hidden_size)] -->
        # actions, action_log_probs [shape: (batch_size * num_agents, act_dims)]
        actions, action_log_probs = self.act(h.squeeze(1), available_actions, deterministic)

        # [shape: (batch_size * num_agents, act_dims)], [shape: (batch_size * num_agents, recurrent_N, hidden_size)]
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
        # [shape: (mini_batch_size * num_agents * data_chunk_length, obs_dims)] --> 
        # [shape: (mini_batch_size * num_agents, data_chunk_length, obs_dims)]
        obs = check(obs).to(**self.tpdv)
        obs = obs.reshape(-1, self.data_chunk_length, self.obs_dims)
        mini_batch_size = obs.shape[0] // self.num_agents
        # [shape: (mini_batch_size * num_agents, recurrent_N, hidden_size)] --> 
        # [shape: (recurrent_N, mini_batch_size * num_agents, hidden_size)]
        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_states = rnn_states.transpose(0, 1).contiguous()
        # [shape: (mini_batch_size * num_agents * data_chunk_length, act_dims)] -->
        # [shape: (mini_batch_size * num_agents, data_chunk_length, act_dims)]
        action = check(action).to(**self.tpdv)
        action = action.reshape(mini_batch_size * self.num_agents, self.data_chunk_length, -1)
        # [shape: (mini_batch_size * num_agents * data_chunk_length, 1)] --> 
        # [shape: (mini_batch_size * num_agents, data_chunk_length, 1)]
        masks = check(masks).to(**self.tpdv)
        masks = masks.reshape(mini_batch_size * self.num_agents, self.data_chunk_length, -1)
        if available_actions is not None:
            # [shape: (mini_batch_size * num_agents * data_chunk_length, act_dims)] -->
            # [shape: (mini_batch_size * num_agents, data_chunk_length, act_dims)]
            available_actions = check(available_actions).to(**self.tpdv)
            available_actions = available_actions.reshape(mini_batch_size * self.num_agents, self.data_chunk_length, -1)
        if active_masks is not None:
            # [shape: (mini_batch_size * num_agents * data_chunk_length, 1)] -->
            # [shape: (mini_batch_size * num_agents, data_chunk_length, 1)]
            active_masks = check(active_masks).to(**self.tpdv)
            active_masks = active_masks.reshape(mini_batch_size * self.num_agents, self.data_chunk_length, -1)
        # store actions and actions_log_probs
        action_log_probs_list = []
        dist_entropy_list = []

        # iterate over data_chunk_length 
        for j in range(self.data_chunk_length):
            # encode observation 
            # [shape: (mini_batch_size * num_agents, 1, hidden_size)]
            h = self.obs_encoder(obs[:, j]).unsqueeze(1)
            # zeros for communication state at start of communication 
            # [shape: (mini_batch_size * num_agents, 1, hidden_size)]
            c = torch.zeros_like(h).to(**self.tpdv)
            # iterate over number of rounds of communication
            for k in range(self.commnet_k):
                if k == 0:
                    # hidden state at start of communication, h [shape: (mini_batch_size * num_agents, 1, hidden_size)]
                    # rnn_states [shape: (recurrent_N, mini_batch_size * num_agents, hidden_size)]
                    # masks[:, j].repeat(1, self._recurrent_N).transpose(0, 1).unsqueeze(-1).contiguous()
                    # [shape: (recurrent_N, mini_batch_size * num_agents, 1)] --> 
                    # h [shape: (mini_batch_size * num_agents, 1, hidden_size)]
                    h, rnn_states = self.obs_rnn(
                        h, 
                        rnn_states * \
                        masks[:, j].repeat(1, self._recurrent_N).transpose(0, 1).unsqueeze(-1).contiguous()
                    )
                else:
                    # obtain hidden states of all agents for all agents
                    # c [shape: (mini_batch_size, 1, num_agents, hidden_size)] --> 
                    # [shape: (mini_batch_size, num_agents, num_agents, hidden_size)]
                    c = h.reshape(mini_batch_size, 1, self.num_agents, self.hidden_size)\
                         .repeat(1, self.num_agents, 1, 1)
                    # communication mask that precludes information from self 
                    # [shape: (mini_batch_size, num_agents, num_agents, 1)]
                    m = (1 - torch.eye(self.num_agents)).unsqueeze(0)\
                                                        .repeat(mini_batch_size, 1, 1)\
                                                        .unsqueeze(-1)\
                                                        .to(**self.tpdv)
                    # communication is average of neighbours hidden state
                    # [shape: (mini_batch_size, num_agents, num_agents, hidden_size)] --> 
                    # [shape: (mini_batch_size, num_agents, hidden_size)] --> 
                    # [shape: (mini_batch_size * num_agents, 1, hidden_size)]
                    c = (c * m).mean(dim=2).reshape(mini_batch_size * self.num_agents, 1, self.hidden_size)
                # c [shape: (mini_batch_size * num_agents, 1, hidden_size)]
                # h.transpose(0, 1).contiguous() [shape: (1, mini_batch_size * num_agents, hidden_size)] -->
                # h [shape: (1, mini_batch_size * num_agents, hidden_size)]
                _, h = self.comms_rnn(c, h.transpose(0, 1).contiguous())
                # h [shape: (1, mini_batch_size * num_agents, hidden_size)] --> 
                # [shape: (mini_batch_size * num_agents, 1, hidden_size)]
                h = h.transpose(0, 1)

            # [shape: (mini_batch_size * num_agents, hidden_size)] -->
            # [shape: (mini_batch_size * num_agents, act_dims)], [shape: () == scalar] 
            action_log_probs, dist_entropy = self.act.evaluate_actions(h.squeeze(1),
                                                                       action[:, j], available_actions[:, j],
                                                                       active_masks=active_masks[:, j] if \
                                                                       self._use_policy_active_masks else None)
            # append action_log_probs and dist_entropy to respective lists
            action_log_probs_list.append(action_log_probs)
            dist_entropy_list.append(dist_entropy)
     
        # [shape: (mini_batch_size * num_agents * data_chunk_length, act_dims)]
        # [shape: () == scalar]
        return torch.stack(action_log_probs_list, dim=1)\
                    .reshape(mini_batch_size * self.num_agents * self.data_chunk_length, -1), \
               torch.stack(dist_entropy_list, dim=0).mean()

class CommNet_Critic(nn.Module):
    """
    CommNet Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(CommNet_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self.data_chunk_length = args.data_chunk_length
        self.num_agents = args.num_agents
        self.commnet_k = args.commnet_k

        self._use_orthogonal = args.use_orthogonal
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        if len(cent_obs_shape) == 3:
            raise NotImplementedError("CNN-based observations not implemented for CommNet")
        if isinstance(cent_obs_shape, (list, tuple)):
            self.obs_dims = cent_obs_shape[0]
        else:
            self.obs_dims = cent_obs_shape

        self.obs_encoder = NNLayers(
            input_channels=self.obs_dims, 
            block=MLPBlock, 
            output_channels=[self.hidden_size],
            norm_type='none', 
            activation_func='relu', 
            dropout_p=0, 
            weight_initialisation="default"
        )
        self.obs_rnn = nn.GRU(
            input_size=self.hidden_size, 
            hidden_size=self.hidden_size, 
            num_layers=self._recurrent_N, 
            batch_first=True)
        self.comms_rnn = nn.GRU(
            input_size=self.hidden_size, 
            hidden_size=self.hidden_size, 
            num_layers=self._recurrent_N, 
            batch_first=True)

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
        # [shape: (batch_size * num_agents, obs_dims)]
        cent_obs = check(cent_obs).to(**self.tpdv)
        batch_size = cent_obs.shape[0] // self.num_agents
        # [shape: (batch_size * num_agents, hidden_size)]
        rnn_states = check(rnn_states).to(**self.tpdv)
        # [shape: (batch_size * num_agents, 1)]
        masks = check(masks).to(**self.tpdv)

        # encode observation 
        # [shape: (batch_size * num_agents, obs_dims)] --> [shape: (batch_size * num_agents, 1, hidden_size)]
        h = self.obs_encoder(cent_obs).unsqueeze(1)
        # zeros for communication state at start of communication [shape: (batch_size * num_agents, 1, hidden_size)]
        c = torch.zeros_like(h).to(**self.tpdv)
        # iterate over number of rounds of communication
        for k in range(self.commnet_k):
            if k == 0:
                # hidden state at start of communication, h [shape: (batch_size * num_agents, 1, hidden_size)]
                # rnn_states.transpose(0, 1).contiguous() [shape: (recurrent_N, batch_size * num_agents, hidden_size)]
                # masks.repeat(1, self._recurrent_N).transpose(0, 1).unsqueeze(-1).contiguous()
                # [shape: (recurrent_N, batch_size * num_agents, 1)] --> 
                # h [shape: (batch_size * num_agents, 1, hidden_size)]
                # rnn_states [shape: (recurrent_N, batch_size * num_agents, hidden_size)]
                h, rnn_states = self.obs_rnn(
                    h, 
                    rnn_states.transpose(0, 1).contiguous() * \
                    masks.repeat(1, self._recurrent_N).transpose(0, 1).unsqueeze(-1).contiguous()
                )
                # [shape: (recurrent_N, batch_size * num_agents, hidden_size)] --> 
                # [shape: (batch_size * num_agents, recurrent_N, hidden_size)]
                rnn_states = rnn_states.transpose(0, 1)
            else:
                # obtain hidden states of all agents for all agents
                # c [shape: (batch_size, 1, num_agents, hidden_size)] --> 
                # [shape: (batch_size, num_agents, num_agents, hidden_size)]
                c = h.reshape(batch_size, 1, self.num_agents, self.hidden_size).repeat(1, self.num_agents, 1, 1)
                # communication mask that precludes information from self 
                # [shape: (batch_size, num_agents, num_agents, 1)]
                m = (1 - torch.eye(self.num_agents)).unsqueeze(0).repeat(batch_size, 1, 1).unsqueeze(-1).to(**self.tpdv)
                # communication is average of neighbours hidden state
                # [shape: (batch_size, num_agents, num_agents, hidden_size)] --> 
                # [shape: (batch_size, num_agents, hidden_size)] --> [shape: (batch_size * num_agents, 1, hidden_size)]
                c = (c * m).mean(dim=2).reshape(batch_size * self.num_agents, 1, self.hidden_size)
            # c [shape: (batch_size * num_agents, 1, hidden_size)]
            # h.transpose(0, 1).contiguous() [shape: (1, batch_size * num_agents, hidden_size)] -->
            # h [shape: (1, batch_size * num_agents, hidden_size)]
            _, h = self.comms_rnn(c, h.transpose(0, 1).contiguous())
            # h [shape: (1, batch_size * num_agents, hidden_size)] --> 
            # [shape: (batch_size * num_agents, 1, hidden_size)]
            h = h.transpose(0, 1)

        # h [shape: (batch_size * num_agents, 1, hidden_size)] --> [shape: (batch_size * num_agents, hidden_size)] -->
        # values [shape: (batch_size * num_agents, 1)]
        values = self.v_out(h.squeeze(1))

        # [shape: (batch_size * num_agents, 1)], [shape: (batch_size * num_agents, recurrent_N, hidden_size)]
        return values, rnn_states

    def evaluate_actions(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        """
        # [shape: (mini_batch_size * num_agents * data_chunk_length, obs_dims)] --> 
        # [shape: (mini_batch_size * num_agents, data_chunk_length, obs_dims)]
        cent_obs = check(cent_obs).to(**self.tpdv)
        cent_obs = cent_obs.reshape(-1, self.data_chunk_length, self.obs_dims)
        mini_batch_size = cent_obs.shape[0] // self.num_agents
        # [shape: (mini_batch_size * num_agents, recurrent_N, hidden_size)] --> 
        # [shape: (recurrent_N, mini_batch_size * num_agents, hidden_size)]
        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_states = rnn_states.transpose(0, 1).contiguous()
        # [shape: (mini_batch_size * num_agents * data_chunk_length, 1)] --> 
        # [shape: (mini_batch_size * num_agents, data_chunk_length, 1)]
        masks = check(masks).to(**self.tpdv)
        masks = masks.reshape(mini_batch_size * self.num_agents, self.data_chunk_length, -1)
        # list to store values
        values_list = []

        # iterate over data_chunk_length 
        for j in range(self.data_chunk_length):
            # encode observation 
            # [shape: (mini_batch_size * num_agents, 1, hidden_size)]
            h = self.obs_encoder(cent_obs[:, j]).unsqueeze(1)
            # zeros for communication state at start of communication 
            # [shape: (mini_batch_size * num_agents, 1, hidden_size)]
            c = torch.zeros_like(h).to(**self.tpdv)
            # iterate over number of rounds of communication
            for k in range(self.commnet_k):
                if k == 0:
                    # hidden state at start of communication, h [shape: (mini_batch_size * num_agents, 1, hidden_size)]
                    # rnn_states [shape: (recurrent_N, mini_batch_size * num_agents, hidden_size)]
                    # masks[:, j].repeat(1, self._recurrent_N).transpose(0, 1).unsqueeze(-1).contiguous()
                    # [shape: (recurrent_N, mini_batch_size * num_agents, 1)] --> 
                    # h [shape: (mini_batch_size * num_agents, 1, hidden_size)]
                    h, rnn_states = self.obs_rnn(
                        h, 
                        rnn_states * \
                        masks[:, j].repeat(1, self._recurrent_N).transpose(0, 1).unsqueeze(-1).contiguous()
                    )
                else:
                    # obtain hidden states of all agents for all agents
                    # c [shape: (mini_batch_size, 1, num_agents, hidden_size)] --> 
                    # [shape: (mini_batch_size, num_agents, num_agents, hidden_size)]
                    c = h.reshape(mini_batch_size, 1, self.num_agents, self.hidden_size)\
                         .repeat(1, self.num_agents, 1, 1)
                    # communication mask that precludes information from self 
                    # [shape: (mini_batch_size, num_agents, num_agents, 1)]
                    m = (1 - torch.eye(self.num_agents)).unsqueeze(0)\
                                                        .repeat(mini_batch_size, 1, 1)\
                                                        .unsqueeze(-1)\
                                                        .to(**self.tpdv)
                    # communication is average of neighbours hidden state
                    # [shape: (mini_batch_size, num_agents, num_agents, hidden_size)] --> 
                    # [shape: (mini_batch_size, num_agents, hidden_size)] --> 
                    # [shape: (mini_batch_size * num_agents, 1, hidden_size)]
                    c = (c * m).mean(dim=2).reshape(mini_batch_size * self.num_agents, 1, self.hidden_size)
                # c [shape: (mini_batch_size * num_agents, 1, hidden_size)]
                # h.transpose(0, 1).contiguous() [shape: (1, mini_batch_size * num_agents, hidden_size)] -->
                # h [shape: (1, mini_batch_size * num_agents, hidden_size)]
                _, h = self.comms_rnn(c, h.transpose(0, 1).contiguous())
                # h [shape: (1, mini_batch_size * num_agents, hidden_size)] --> 
                # [shape: (mini_batch_size * num_agents, 1, hidden_size)]
                h = h.transpose(0, 1)

            # h [shape: (mini_batch_size * num_agents, 1, hidden_size)] --> 
            # [shape: (mini_batch_size * num_agents, hidden_size)] --> values [shape: (mini_batch_size * num_agents, 1)]
            values = self.v_out(h.squeeze(1))
            values_list.append(values)
     
        # [shape: (mini_batch_size * num_agents * data_chunk_length, 1)]
        return torch.stack(values_list, dim=1).reshape(-1, 1)