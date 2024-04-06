import math
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


class G2ANet_Actor(nn.Module):
    """
    G2ANet Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(G2ANet_Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.data_chunk_length = args.data_chunk_length
        self.num_agents = args.num_agents
        self.gumbel_softmax_tau = args.g2anet_gumbel_softmax_tau

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        if len(obs_shape) == 3:
            raise NotImplementedError("CNN-based observations not implemented for G2ANet")
        if isinstance(obs_shape, (list, tuple)):
            self.obs_dims = obs_shape[0]
        else:
            self.obs_dims = obs_shape
        self.act_dims = get_shape_from_act_space(action_space)

        # encoder
        self.obs_encoder = NNLayers(
            input_channels=self.obs_dims, 
            block=MLPBlock, 
            output_channels=[self.hidden_size],
            norm_type='none', 
            activation_func='relu', 
            dropout_p=0, 
            weight_initialisation="default"
        )
        self.obs_lstm_encoder = torch.nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self._recurrent_N, 
            batch_first=True
        )
        # hard attention
        self.hard_bi_rnn = nn.GRU(
            input_size=self.hidden_size * 2, 
            hidden_size=self.hidden_size, 
            num_layers=self._recurrent_N, 
            batch_first=True,
            bidirectional=True
        )
        self.hard_encoder = nn.Linear(self.hidden_size * 2, 2)
        # soft attention
        self.q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # decoder
        self.act = ACTLayer(action_space, self.hidden_size * 2, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, lstm_hidden_states, lstm_cell_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param lstm_hidden_states: (np.ndarray / torch.Tensor) hidden states for LSTM.
        :param lstm_cell_states: (np.ndarray / torch.Tensor) cell states for LSTM.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return lstm_hidden_states: (torch.Tensor) updated LSTM hidden states.
        :return lstm_cell_states: (torch.Tensor) updated LSTM cell states.
        """
        # [shape: (batch_size * num_agents, obs_dims)]
        obs = check(obs).to(**self.tpdv)
        batch_size = obs.shape[0] // self.num_agents
        # [shape: (batch_size * num_agents, recurrent_N, hidden_size)]
        lstm_hidden_states = check(lstm_hidden_states).to(**self.tpdv)
        lstm_cell_states = check(lstm_cell_states).to(**self.tpdv)
        # [shape: (batch_size * num_agents, 1)]
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            # [shape: (batch_size * num_agents, act_dims)]
            available_actions = check(available_actions).to(**self.tpdv)

        # obs --> h [shape: (batch_size * num_agents, hidden_size)]
        h = self.obs_encoder(obs)
        # h.unsqueeze(1) [shape: (batch_size * num_agents, 1, hidden_size)]
        # lstm_hidden_states.transpose(0, 1).contiguous() / lstm_cell_states.transpose(0, 1).contiguous() 
        # [shape: (recurrent_N, batch_size * num_agents, hidden_size)]
        # masks.repeat(1, self._recurrent_N).transpose(0, 1).unsqueeze(-1).contiguous()
        # [shape: (recurrent_N, batch_size * num_agents, 1)] --> 
        # h [shape: (batch_size * num_agents, 1, hidden_size)]
        # lstm_hidden_states / lstm_cell_states [shape: (recurrent_N, batch_size * num_agents, hidden_size)]
        h, (lstm_hidden_states, lstm_cell_states) = \
            self.obs_lstm_encoder(
                h.unsqueeze(1), 
                (lstm_hidden_states.transpose(0, 1).contiguous() * \
                 masks.repeat(1, self._recurrent_N).transpose(0, 1).unsqueeze(-1).contiguous(), 
                 lstm_cell_states.transpose(0, 1).contiguous() * \
                 masks.repeat(1, self._recurrent_N).transpose(0, 1).unsqueeze(-1).contiguous()
                )
            )
        # [shape: (recurrent_N, batch_size * num_agents, hidden_size)] --> 
        # [shape: (batch_size * num_agents, recurrent_N, hidden_size)]
        lstm_hidden_states = lstm_hidden_states.transpose(0, 1)
        lstm_cell_states = lstm_cell_states.transpose(0, 1) 

        # hard attention mechanism to compute hard weight (whether there is (one-hot) communication between agents)

        # repeat hidden state for each agent i
        # h_hard_1 [shape: (batch_size * num_agents, 1, hidden_size)] -->
        # [shape: (batch_size, num_agents, num_agents, hidden_size)]
        h_hard_1 = h.reshape(batch_size, self.num_agents, 1, self.hidden_size).repeat(1, 1, self.num_agents, 1)
        # repeat hidden state for each agent j
        # h_hard_2 [shape: (batch_size * num_agents, 1, hidden_size)] -->
        # [shape: (batch_size, num_agents, num_agents, hidden_size)]
        h_hard_2 = h.reshape(batch_size, 1, self.num_agents, self.hidden_size).repeat(1, self.num_agents, 1, 1)
        # concatenate hidden state of agent i to each hidden state of agent j
        # [shape: (batch_size, num_agents, num_agents, hidden_size * 2)]
        h_hard = torch.cat((h_hard_1, h_hard_2), dim=-1) 
        # mask to remove instances of where i == j (concatenating the same hidden state)
        # [shape: (batch_size, num_agents, num_agents, hidden_size * 2)]
        h_hard_mask = (1 - torch.eye(self.num_agents)).unsqueeze(0)\
                                                      .unsqueeze(-1)\
                                                      .repeat(batch_size, 1, 1, self.hidden_size * 2)\
                                                      .to(**self.tpdv)
        # mask h_hard and reshape where sequence length is num_agents - 1 (all agents where i != j)
        # [shape: (batch_size * num_agents, num_agents - 1, hidden_size * 2)]
        h_hard = h_hard[h_hard_mask == 1].reshape(batch_size * self.num_agents, 
                                                  self.num_agents - 1, 
                                                  self.hidden_size * 2)
        # zeros for initial hidden states for bidirectional rnn 
        # [shape: (2 * recurrent_N, batch_size * num_agents, hidden_size)]
        h_hard_hidden = torch.zeros((2 * self._recurrent_N, batch_size * self.num_agents, self.hidden_size))\
                             .to(**self.tpdv)
        # h_hard, h_hard_hidden --> h_hard [shape: (batch_size * num_agents, num_agents - 1, hidden_size * 2)]
        h_hard, _ = self.hard_bi_rnn(h_hard, h_hard_hidden)
        # h_hard --> h_hard [shape: (batch_size * num_agents, num_agents - 1, 2)]
        h_hard = self.hard_encoder(h_hard)
        # h_hard --> hard_weights [shape: (batch_size * num_agents, num_agents - 1, 2)] (one-hot) -->
        # [shape: (batch_size * num_agents, num_agents - 1)] --> [shape: (batch_size, num_agents, num_agents - 1, 1)] 
        hard_weights = nn.functional.gumbel_softmax(h_hard, tau=self.gumbel_softmax_tau, hard=True)[:, :, 1]\
                                    .reshape(batch_size, self.num_agents, self.num_agents - 1, 1)

        # soft attention mechanism to compute soft weight (weighting feature vectors of communicating agents)

        # h --> q, k [shape: (batch_size * num_agents, 1, hidden_size)]
        q = self.q(h)
        k = self.k(h)
        # q [shape: (batch_size, num_agents, num_agents, hidden_size)] 
        q = q.reshape(batch_size, self.num_agents, 1, self.hidden_size)
        # repeat key hidden state for each agent j
        k = k.reshape(batch_size, 1, self.num_agents, self.hidden_size).repeat(1, self.num_agents, 1, 1)
        # mask to remove instances of where i == j in key hidden state
        # [shape: (batch_size, num_agents, num_agents, hidden_size)]
        mask = (1 - torch.eye(self.num_agents)).unsqueeze(0)\
                                               .unsqueeze(-1)\
                                               .repeat(batch_size, 1, 1, self.hidden_size)\
                                               .to(**self.tpdv) 
        # mask k and transpose [shape: (batch_size, num_agents, num_agents - 1, hidden_size)] --> 
        # [shape: (batch_size, num_agents, hidden_size, num_agents - 1)]
        k = k[mask == 1].reshape(batch_size, self.num_agents, self.num_agents - 1, self.hidden_size).transpose(2, 3)
        # dot product between query (agent i) and key for each agent j != i for each agent i
        # q, k --> soft_weights [shape: (batch_size, num_agents, 1, num_agents - 1)] --> 
        # [shape: (batch_size, num_agents, num_agents - 1)]
        soft_weights = torch.matmul(q, k).squeeze(2) 
        # scale soft_weights and softmax [shape: (batch_size, num_agents, num_agents - 1)] --> 
        # [shape: (batch_size, num_agents, num_agents - 1, 1)]
        soft_weights = soft_weights / math.sqrt(self.hidden_size)
        soft_weights = nn.functional.softmax(soft_weights, dim=-1).unsqueeze(-1)

        # hidden state for each agent j != i for each agent i
        # h_hard_2 [shape: (batch_size, num_agents, num_agents, hidden_size)] --> 
        # x [shape: (batch_size, num_agents, num_agents - 1, hidden_size)]
        x = h_hard_2[mask == 1].reshape(batch_size, self.num_agents, self.num_agents - 1, self.hidden_size)
        # apply hard and soft weights [shape: (batch_size, num_agents, num_agents - 1, hidden_size)]
        x *= hard_weights * soft_weights
        # summation over neighbours (each agent j != i for each agent i) 
        # x [shape: (batch_size, num_agents, hidden_size)] --> [shape: (batch_size * num_agents, hidden_size)]
        x = x.sum(dim=2).reshape(batch_size * self.num_agents, self.hidden_size)
        # [shape: (batch_size * num_agents, hidden_size * 2)] -->
        # actions, action_log_probs [shape: (batch_size * num_agents, act_dims)]
        actions, action_log_probs = self.act(torch.cat((h.squeeze(1), x), dim=-1), available_actions, deterministic)

        # [shape: (batch_size * num_agents, act_dims)], [shape: (batch_size * num_agents, recurrent_N, hidden_size)]
        return actions, action_log_probs, lstm_hidden_states, lstm_cell_states

    def evaluate_actions(self, obs, lstm_hidden_states, lstm_cell_states, action, masks, available_actions=None, 
                         active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param lstm_hidden_states: (torch.Tensor) hidden states for LSTM.
        :param lstm_cell_states: (torch.Tensor) cell states for LSTM.
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
        lstm_hidden_states = check(lstm_hidden_states).to(**self.tpdv)
        lstm_cell_states = check(lstm_cell_states).to(**self.tpdv)
        lstm_hidden_states = lstm_hidden_states.transpose(0, 1).contiguous()
        lstm_cell_states = lstm_cell_states.transpose(0, 1).contiguous()
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

        # obs --> obs_enc [shape: (mini_batch_size * num_agents, data_chunk_length, hidden_size)]
        obs_enc = self.obs_encoder(obs)

        # iterate over data_chunk_length 
        for j in range(self.data_chunk_length):
            # obs_enc[:, j].unsqueeze(1) [shape: (mini_batch_size * num_agents, 1, hidden_size)]
            # lstm_hidden_states / lstm_cell_states [shape: (recurrent_N, mini_batch_size * num_agents, hidden_size)]
            # masks[:, j].repeat(1, self._recurrent_N).transpose(0, 1).unsqueeze(-1).contiguous()
            # [shape: (recurrent_N, mini_batch_size * num_agents, 1)] --> 
            # h [shape: (mini_batch_size * num_agents, 1, hidden_size)]
            # lstm_hidden_states / lstm_cell_states [shape: (recurrent_N, mini_batch_size * num_agents, hidden_size)]
            h, (lstm_hidden_states, lstm_cell_states) = \
                self.obs_lstm_encoder(
                    obs_enc[:, j].unsqueeze(1), 
                    (lstm_hidden_states * \
                     masks[:, j].repeat(1, self._recurrent_N).transpose(0, 1).unsqueeze(-1).contiguous(), 
                     lstm_cell_states * \
                     masks[:, j].repeat(1, self._recurrent_N).transpose(0, 1).unsqueeze(-1).contiguous()
                    )
                )

            # hard attention mechanism to compute hard weight (whether there is (one-hot) communication between agents)

            # repeat hidden state for each agent i
            # h_hard_1 [shape: (mini_batch_size * num_agents, 1, hidden_size)] -->
            # [shape: (mini_batch_size, num_agents, num_agents, hidden_size)]
            h_hard_1 = h.reshape(mini_batch_size, self.num_agents, 1, self.hidden_size).repeat(1, 1, self.num_agents, 1)
            # repeat hidden state for each agent j
            # h_hard_2 [shape: (mini_batch_size * num_agents, 1, hidden_size)] -->
            # [shape: (mini_batch_size, num_agents, num_agents, hidden_size)]
            h_hard_2 = h.reshape(mini_batch_size, 1, self.num_agents, self.hidden_size).repeat(1, self.num_agents, 1, 1)
            # concatenate hidden state of agent i to each hidden state of agent j
            # [shape: (mini_batch_size, num_agents, num_agents, hidden_size * 2)]
            h_hard = torch.cat((h_hard_1, h_hard_2), dim=-1) 
            # mask to remove instances of where i == j (concatenating the same hidden state)
            # [shape: (mini_batch_size, num_agents, num_agents, hidden_size * 2)]
            h_hard_mask = (1 - torch.eye(self.num_agents)).unsqueeze(0)\
                                                          .unsqueeze(-1)\
                                                          .repeat(mini_batch_size, 1, 1, self.hidden_size * 2)\
                                                          .to(**self.tpdv)
            # mask h_hard and reshape where sequence length is num_agents - 1 (all agents where i != j)
            # [shape: (mini_batch_size * num_agents, num_agents - 1, hidden_size * 2)]
            h_hard = h_hard[h_hard_mask == 1].reshape(mini_batch_size * self.num_agents, 
                                                      self.num_agents - 1, 
                                                      self.hidden_size * 2)
            # zeros for initial hidden states for bidirectional rnn 
            # [shape: (2 * recurrent_N, mini_batch_size * num_agents, hidden_size)]
            h_hard_hidden = torch.zeros((2 * self._recurrent_N, mini_batch_size * self.num_agents, self.hidden_size))\
                                 .to(**self.tpdv)
            # h_hard, h_hard_hidden --> h_hard [shape: (mini_batch_size * num_agents, num_agents - 1, hidden_size * 2)]
            h_hard, _ = self.hard_bi_rnn(h_hard, h_hard_hidden)
            # h_hard --> h_hard [shape: (mini_batch_size * num_agents, num_agents - 1, 2)]
            h_hard = self.hard_encoder(h_hard)
            # h_hard --> hard_weights [shape: (mini_batch_size * num_agents, num_agents - 1, 2)] (one-hot) -->
            # [shape: (mini_batch_size * num_agents, num_agents - 1)] --> 
            # [shape: (mini_batch_size, num_agents, num_agents - 1, 1)] 
            hard_weights = nn.functional.gumbel_softmax(h_hard, tau=self.gumbel_softmax_tau, hard=True)[:, :, 1]\
                                        .reshape(mini_batch_size, self.num_agents, self.num_agents - 1, 1)

            # soft attention mechanism to compute soft weight (weighting feature vectors of communicating agents)

            # h --> q, k [shape: (mini_batch_size * num_agents, 1, hidden_size)]
            q = self.q(h)
            k = self.k(h)
            # q [shape: (mini_batch_size, num_agents, num_agents, hidden_size)] 
            q = q.reshape(mini_batch_size, self.num_agents, 1, self.hidden_size)
            # repeat key hidden state for each agent j
            k = k.reshape(mini_batch_size, 1, self.num_agents, self.hidden_size).repeat(1, self.num_agents, 1, 1)
            # mask to remove instances of where i == j in key hidden state
            # [shape: (mini_batch_size, num_agents, num_agents, hidden_size)]
            mask = (1 - torch.eye(self.num_agents)).unsqueeze(0)\
                                                   .unsqueeze(-1)\
                                                   .repeat(mini_batch_size, 1, 1, self.hidden_size)\
                                                   .to(**self.tpdv) 
            # mask k and transpose [shape: (mini_batch_size, num_agents, num_agents - 1, hidden_size)] --> 
            # [shape: (mini_batch_size, num_agents, hidden_size, num_agents - 1)]
            k = k[mask == 1].reshape(mini_batch_size, self.num_agents, self.num_agents - 1, self.hidden_size)\
                            .transpose(2, 3)
            # dot product between query (agent i) and key for each agent j != i for each agent i
            # q, k --> soft_weights [shape: (mini_batch_size, num_agents, 1, num_agents - 1)] --> 
            # [shape: (mini_batch_size, num_agents, num_agents - 1)]
            soft_weights = torch.matmul(q, k).squeeze(2) 
            # scale soft_weights and softmax [shape: (mini_batch_size, num_agents, num_agents - 1)] --> 
            # [shape: (mini_batch_size, num_agents, num_agents - 1, 1)]
            soft_weights = soft_weights / math.sqrt(self.hidden_size)
            soft_weights = nn.functional.softmax(soft_weights, dim=-1).unsqueeze(-1)

            # hidden state for each agent j != i for each agent i
            # h_hard_2 [shape: (mini_batch_size, num_agents, num_agents, hidden_size)] --> 
            # x [shape: (mini_batch_size, num_agents, num_agents - 1, hidden_size)]
            x = h_hard_2[mask == 1].reshape(mini_batch_size, self.num_agents, self.num_agents - 1, self.hidden_size)
            # apply hard and soft weights [shape: (mini_batch_size, num_agents, num_agents - 1, hidden_size)]
            x *= hard_weights * soft_weights
            # summation over neighbours (each agent j != i for each agent i) 
            # x [shape: (mini_batch_size, num_agents, hidden_size)] --> 
            # [shape: (mini_batch_size * num_agents, hidden_size)]
            x = x.sum(dim=2).reshape(mini_batch_size * self.num_agents, self.hidden_size)
            # [shape: (mini_batch_size * num_agents, hidden_size * 2)] -->
            # [shape: (mini_batch_size * num_agents, act_dims)], [shape: () == scalar] 
            action_log_probs, dist_entropy = self.act.evaluate_actions(torch.cat((h.squeeze(1), x), dim=-1),
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

class G2ANet_Critic(nn.Module):
    """
    G2ANet Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(G2ANet_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self.data_chunk_length = args.data_chunk_length
        self.num_agents = args.num_agents
        self.gumbel_softmax_tau = args.g2anet_gumbel_softmax_tau

        self._use_orthogonal = args.use_orthogonal
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        if len(cent_obs_shape) == 3:
            raise NotImplementedError("CNN-based observations not implemented for G2ANet")
        if isinstance(cent_obs_shape, (list, tuple)):
            self.obs_dims = cent_obs_shape[0]
        else:
            self.obs_dims = cent_obs_shape

        # encoder
        self.obs_encoder = NNLayers(
            input_channels=self.obs_dims, 
            block=MLPBlock, 
            output_channels=[self.hidden_size],
            norm_type='none', 
            activation_func='relu', 
            dropout_p=0, 
            weight_initialisation="default"
        )
        self.obs_lstm_encoder = torch.nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self._recurrent_N, 
            batch_first=True
        )
        # hard attention
        self.hard_bi_rnn = nn.GRU(
            input_size=self.hidden_size * 2, 
            hidden_size=self.hidden_size, 
            num_layers=self._recurrent_N, 
            batch_first=True,
            bidirectional=True
        )
        self.hard_encoder = nn.Linear(self.hidden_size * 2, 2)
        # soft attention
        self.q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # decoder
        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size * 2, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size * 2, 1))

        self.to(device)

    def forward(self, cent_obs, lstm_hidden_states, lstm_cell_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param lstm_hidden_states: (np.ndarray / torch.Tensor) hidden states for LSTM.
        :param lstm_cell_states: (np.ndarray / torch.Tensor) cell states for LSTM.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return lstm_hidden_states: (torch.Tensor) updated LSTM hidden states.
        :return lstm_cell_states: (torch.Tensor) updated LSTM cell states.
        """
        # [shape: (batch_size * num_agents, obs_dims)]
        cent_obs = check(cent_obs).to(**self.tpdv)
        batch_size = cent_obs.shape[0] // self.num_agents
        # [shape: (batch_size * num_agents, recurrent_N, hidden_size)]
        lstm_hidden_states = check(lstm_hidden_states).to(**self.tpdv)
        lstm_cell_states = check(lstm_cell_states).to(**self.tpdv)
        # [shape: (batch_size * num_agents, 1)]
        masks = check(masks).to(**self.tpdv)

        # obs --> h [shape: (batch_size * num_agents, hidden_size)]
        h = self.obs_encoder(cent_obs)
        # h.unsqueeze(1) [shape: (batch_size * num_agents, 1, hidden_size)]
        # lstm_hidden_states.transpose(0, 1).contiguous() / lstm_cell_states.transpose(0, 1).contiguous() 
        # [shape: (recurrent_N, batch_size * num_agents, hidden_size)]
        # masks.repeat(1, self._recurrent_N).transpose(0, 1).unsqueeze(-1).contiguous()
        # [shape: (recurrent_N, batch_size * num_agents, 1)] --> 
        # h [shape: (batch_size * num_agents, 1, hidden_size)]
        # lstm_hidden_states / lstm_cell_states [shape: (recurrent_N, batch_size * num_agents, hidden_size)]
        h, (lstm_hidden_states, lstm_cell_states) = \
            self.obs_lstm_encoder(
                h.unsqueeze(1), 
                (lstm_hidden_states.transpose(0, 1).contiguous() * \
                 masks.repeat(1, self._recurrent_N).transpose(0, 1).unsqueeze(-1).contiguous(), 
                 lstm_cell_states.transpose(0, 1).contiguous() * \
                 masks.repeat(1, self._recurrent_N).transpose(0, 1).unsqueeze(-1).contiguous()
                )
            )
        # [shape: (recurrent_N, batch_size * num_agents, hidden_size)] --> 
        # [shape: (batch_size * num_agents, recurrent_N, hidden_size)]
        lstm_hidden_states = lstm_hidden_states.transpose(0, 1)
        lstm_cell_states = lstm_cell_states.transpose(0, 1) 

        # hard attention mechanism to compute hard weight (whether there is (one-hot) communication between agents)

        # repeat hidden state for each agent i
        # h_hard_1 [shape: (batch_size * num_agents, 1, hidden_size)] -->
        # [shape: (batch_size, num_agents, num_agents, hidden_size)]
        h_hard_1 = h.reshape(batch_size, self.num_agents, 1, self.hidden_size).repeat(1, 1, self.num_agents, 1)
        # repeat hidden state for each agent j
        # h_hard_2 [shape: (batch_size * num_agents, 1, hidden_size)] -->
        # [shape: (batch_size, num_agents, num_agents, hidden_size)]
        h_hard_2 = h.reshape(batch_size, 1, self.num_agents, self.hidden_size).repeat(1, self.num_agents, 1, 1)
        # concatenate hidden state of agent i to each hidden state of agent j
        # [shape: (batch_size, num_agents, num_agents, hidden_size * 2)]
        h_hard = torch.cat((h_hard_1, h_hard_2), dim=-1) 
        # mask to remove instances of where i == j (concatenating the same hidden state)
        # [shape: (batch_size, num_agents, num_agents, hidden_size * 2)]
        h_hard_mask = (1 - torch.eye(self.num_agents)).unsqueeze(0)\
                                                      .unsqueeze(-1)\
                                                      .repeat(batch_size, 1, 1, self.hidden_size * 2)\
                                                      .to(**self.tpdv)
        # mask h_hard and reshape where sequence length is num_agents - 1 (all agents where i != j)
        # [shape: (batch_size * num_agents, num_agents - 1, hidden_size * 2)]
        h_hard = h_hard[h_hard_mask == 1].reshape(batch_size * self.num_agents, 
                                                  self.num_agents - 1, 
                                                  self.hidden_size * 2)
        # zeros for initial hidden states for bidirectional rnn 
        # [shape: (2 * recurrent_N, batch_size * num_agents, hidden_size)]
        h_hard_hidden = torch.zeros((2 * self._recurrent_N, batch_size * self.num_agents, self.hidden_size))\
                             .to(**self.tpdv)
        # h_hard, h_hard_hidden --> h_hard [shape: (batch_size * num_agents, num_agents - 1, hidden_size * 2)]
        h_hard, _ = self.hard_bi_rnn(h_hard, h_hard_hidden)
        # h_hard --> h_hard [shape: (batch_size * num_agents, num_agents - 1, 2)]
        h_hard = self.hard_encoder(h_hard)
        # h_hard --> hard_weights [shape: (batch_size * num_agents, num_agents - 1, 2)] (one-hot) -->
        # [shape: (batch_size * num_agents, num_agents - 1)] --> [shape: (batch_size, num_agents, num_agents - 1, 1)] 
        hard_weights = nn.functional.gumbel_softmax(h_hard, tau=self.gumbel_softmax_tau, hard=True)[:, :, 1]\
                                    .reshape(batch_size, self.num_agents, self.num_agents - 1, 1)

        # soft attention mechanism to compute soft weight (weighting feature vectors of communicating agents)

        # h --> q, k [shape: (batch_size * num_agents, 1, hidden_size)]
        q = self.q(h)
        k = self.k(h)
        # q [shape: (batch_size, num_agents, num_agents, hidden_size)] 
        q = q.reshape(batch_size, self.num_agents, 1, self.hidden_size)
        # repeat key hidden state for each agent j
        k = k.reshape(batch_size, 1, self.num_agents, self.hidden_size).repeat(1, self.num_agents, 1, 1)
        # mask to remove instances of where i == j in key hidden state
        # [shape: (batch_size, num_agents, num_agents, hidden_size)]
        mask = (1 - torch.eye(self.num_agents)).unsqueeze(0)\
                                               .unsqueeze(-1)\
                                               .repeat(batch_size, 1, 1, self.hidden_size)\
                                               .to(**self.tpdv) 
        # mask k and transpose [shape: (batch_size, num_agents, num_agents - 1, hidden_size)] --> 
        # [shape: (batch_size, num_agents, hidden_size, num_agents - 1)]
        k = k[mask == 1].reshape(batch_size, self.num_agents, self.num_agents - 1, self.hidden_size).transpose(2, 3)
        # dot product between query (agent i) and key for each agent j != i for each agent i
        # q, k --> soft_weights [shape: (batch_size, num_agents, 1, num_agents - 1)] --> 
        # [shape: (batch_size, num_agents, num_agents - 1)]
        soft_weights = torch.matmul(q, k).squeeze(2) 
        # scale soft_weights and softmax [shape: (batch_size, num_agents, num_agents - 1)] --> 
        # [shape: (batch_size, num_agents, num_agents - 1, 1)]
        soft_weights = soft_weights / math.sqrt(self.hidden_size)
        soft_weights = nn.functional.softmax(soft_weights, dim=-1).unsqueeze(-1)

        # hidden state for each agent j != i for each agent i
        # h_hard_2 [shape: (batch_size, num_agents, num_agents, hidden_size)] --> 
        # x [shape: (batch_size, num_agents, num_agents - 1, hidden_size)]
        x = h_hard_2[mask == 1].reshape(batch_size, self.num_agents, self.num_agents - 1, self.hidden_size)
        # apply hard and soft weights [shape: (batch_size, num_agents, num_agents - 1, hidden_size)]
        x *= hard_weights * soft_weights
        # summation over neighbours (each agent j != i for each agent i) 
        # x [shape: (batch_size, num_agents, hidden_size)] --> [shape: (batch_size * num_agents, hidden_size)]
        x = x.sum(dim=2).reshape(batch_size * self.num_agents, self.hidden_size)
        # [shape: (batch_size * num_agents, hidden_size * 2)] --> values [shape: (batch_size * num_agents, 1)]
        values = self.v_out(torch.cat((h.squeeze(1), x), dim=-1))

        # [shape: (batch_size * num_agents, 1)], [shape: (batch_size * num_agents, recurrent_N, hidden_size)]
        return values, lstm_hidden_states, lstm_cell_states

    def evaluate_actions(self, cent_obs, lstm_hidden_states, lstm_cell_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param lstm_hidden_states: (np.ndarray / torch.Tensor) hidden states for LSTM.
        :param lstm_cell_states: (np.ndarray / torch.Tensor) cell states for LSTM.
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
        lstm_hidden_states = check(lstm_hidden_states).to(**self.tpdv)
        lstm_cell_states = check(lstm_cell_states).to(**self.tpdv)
        lstm_hidden_states = lstm_hidden_states.transpose(0, 1).contiguous()
        lstm_cell_states = lstm_cell_states.transpose(0, 1).contiguous()
        # [shape: (mini_batch_size * num_agents * data_chunk_length, 1)] --> 
        # [shape: (mini_batch_size * num_agents, data_chunk_length, 1)]
        masks = check(masks).to(**self.tpdv)
        masks = masks.reshape(mini_batch_size * self.num_agents, self.data_chunk_length, -1)
        # list to store values
        values_list = []

        # obs --> obs_enc [shape: (mini_batch_size * num_agents, data_chunk_length, hidden_size)]
        obs_enc = self.obs_encoder(cent_obs)

        # iterate over data_chunk_length 
        for j in range(self.data_chunk_length):
            # obs_enc[:, j].unsqueeze(1) [shape: (mini_batch_size * num_agents, 1, hidden_size)]
            # lstm_hidden_states / lstm_cell_states [shape: (recurrent_N, mini_batch_size * num_agents, hidden_size)]
            # masks[:, j].repeat(1, self._recurrent_N).transpose(0, 1).unsqueeze(-1).contiguous()
            # [shape: (recurrent_N, mini_batch_size * num_agents, 1)] --> 
            # h [shape: (mini_batch_size * num_agents, 1, hidden_size)]
            # lstm_hidden_states / lstm_cell_states [shape: (recurrent_N, mini_batch_size * num_agents, hidden_size)]
            h, (lstm_hidden_states, lstm_cell_states) = \
                self.obs_lstm_encoder(
                    obs_enc[:, j].unsqueeze(1), 
                    (lstm_hidden_states * \
                     masks[:, j].repeat(1, self._recurrent_N).transpose(0, 1).unsqueeze(-1).contiguous(), 
                     lstm_cell_states * \
                     masks[:, j].repeat(1, self._recurrent_N).transpose(0, 1).unsqueeze(-1).contiguous()
                    )
                )

            # hard attention mechanism to compute hard weight (whether there is (one-hot) communication between agents)

            # repeat hidden state for each agent i
            # h_hard_1 [shape: (mini_batch_size * num_agents, 1, hidden_size)] -->
            # [shape: (mini_batch_size, num_agents, num_agents, hidden_size)]
            h_hard_1 = h.reshape(mini_batch_size, self.num_agents, 1, self.hidden_size).repeat(1, 1, self.num_agents, 1)
            # repeat hidden state for each agent j
            # h_hard_2 [shape: (mini_batch_size * num_agents, 1, hidden_size)] -->
            # [shape: (mini_batch_size, num_agents, num_agents, hidden_size)]
            h_hard_2 = h.reshape(mini_batch_size, 1, self.num_agents, self.hidden_size).repeat(1, self.num_agents, 1, 1)
            # concatenate hidden state of agent i to each hidden state of agent j
            # [shape: (mini_batch_size, num_agents, num_agents, hidden_size * 2)]
            h_hard = torch.cat((h_hard_1, h_hard_2), dim=-1) 
            # mask to remove instances of where i == j (concatenating the same hidden state)
            # [shape: (mini_batch_size, num_agents, num_agents, hidden_size * 2)]
            h_hard_mask = (1 - torch.eye(self.num_agents)).unsqueeze(0)\
                                                          .unsqueeze(-1)\
                                                          .repeat(mini_batch_size, 1, 1, self.hidden_size * 2)\
                                                          .to(**self.tpdv)
            # mask h_hard and reshape where sequence length is num_agents - 1 (all agents where i != j)
            # [shape: (mini_batch_size * num_agents, num_agents - 1, hidden_size * 2)]
            h_hard = h_hard[h_hard_mask == 1].reshape(mini_batch_size * self.num_agents, 
                                                      self.num_agents - 1, 
                                                      self.hidden_size * 2)
            # zeros for initial hidden states for bidirectional rnn 
            # [shape: (2 * recurrent_N, mini_batch_size * num_agents, hidden_size)]
            h_hard_hidden = torch.zeros((2 * self._recurrent_N, mini_batch_size * self.num_agents, self.hidden_size))\
                                 .to(**self.tpdv)
            # h_hard, h_hard_hidden --> h_hard [shape: (mini_batch_size * num_agents, num_agents - 1, hidden_size * 2)]
            h_hard, _ = self.hard_bi_rnn(h_hard, h_hard_hidden)
            # h_hard --> h_hard [shape: (mini_batch_size * num_agents, num_agents - 1, 2)]
            h_hard = self.hard_encoder(h_hard)
            # h_hard --> hard_weights [shape: (mini_batch_size * num_agents, num_agents - 1, 2)] (one-hot) -->
            # [shape: (mini_batch_size * num_agents, num_agents - 1)] --> 
            # [shape: (mini_batch_size, num_agents, num_agents - 1, 1)] 
            hard_weights = nn.functional.gumbel_softmax(h_hard, tau=self.gumbel_softmax_tau, hard=True)[:, :, 1]\
                                        .reshape(mini_batch_size, self.num_agents, self.num_agents - 1, 1)

            # soft attention mechanism to compute soft weight (weighting feature vectors of communicating agents)

            # h --> q, k [shape: (mini_batch_size * num_agents, 1, hidden_size)]
            q = self.q(h)
            k = self.k(h)
            # q [shape: (mini_batch_size, num_agents, num_agents, hidden_size)] 
            q = q.reshape(mini_batch_size, self.num_agents, 1, self.hidden_size)
            # repeat key hidden state for each agent j
            k = k.reshape(mini_batch_size, 1, self.num_agents, self.hidden_size).repeat(1, self.num_agents, 1, 1)
            # mask to remove instances of where i == j in key hidden state
            # [shape: (mini_batch_size, num_agents, num_agents, hidden_size)]
            mask = (1 - torch.eye(self.num_agents)).unsqueeze(0)\
                                                   .unsqueeze(-1)\
                                                   .repeat(mini_batch_size, 1, 1, self.hidden_size)\
                                                   .to(**self.tpdv) 
            # mask k and transpose [shape: (mini_batch_size, num_agents, num_agents - 1, hidden_size)] --> 
            # [shape: (mini_batch_size, num_agents, hidden_size, num_agents - 1)]
            k = k[mask == 1].reshape(mini_batch_size, self.num_agents, self.num_agents - 1, self.hidden_size)\
                            .transpose(2, 3)
            # dot product between query (agent i) and key for each agent j != i for each agent i
            # q, k --> soft_weights [shape: (mini_batch_size, num_agents, 1, num_agents - 1)] --> 
            # [shape: (mini_batch_size, num_agents, num_agents - 1)]
            soft_weights = torch.matmul(q, k).squeeze(2) 
            # scale soft_weights and softmax [shape: (mini_batch_size, num_agents, num_agents - 1)] --> 
            # [shape: (mini_batch_size, num_agents, num_agents - 1, 1)]
            soft_weights = soft_weights / math.sqrt(self.hidden_size)
            soft_weights = nn.functional.softmax(soft_weights, dim=-1).unsqueeze(-1)

            # hidden state for each agent j != i for each agent i
            # h_hard_2 [shape: (mini_batch_size, num_agents, num_agents, hidden_size)] --> 
            # x [shape: (mini_batch_size, num_agents, num_agents - 1, hidden_size)]
            x = h_hard_2[mask == 1].reshape(mini_batch_size, self.num_agents, self.num_agents - 1, self.hidden_size)
            # apply hard and soft weights [shape: (mini_batch_size, num_agents, num_agents - 1, hidden_size)]
            x *= hard_weights * soft_weights
            # summation over neighbours (each agent j != i for each agent i) 
            # x [shape: (mini_batch_size, num_agents, hidden_size)] --> 
            # [shape: (mini_batch_size * num_agents, hidden_size)]
            x = x.sum(dim=2).reshape(mini_batch_size * self.num_agents, self.hidden_size)
            # [shape: (mini_batch_size * num_agents, hidden_size * 2)] --> 
            # values [shape: (mini_batch_size * num_agents, 1)]
            values = self.v_out(torch.cat((h.squeeze(1), x), dim=-1))
            values_list.append(values)
     
        # [shape: (mini_batch_size * num_agents * data_chunk_length, 1)]
        return torch.stack(values_list, dim=1).reshape(-1, 1)