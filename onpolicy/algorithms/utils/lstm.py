import torch
import torch.nn as nn

"""LSTM modules."""

class LSTMLayer(nn.Module):
    
    def __init__(self, input_size, hidden_size, n_lstm_layers, use_orthogonal):
        super(LSTMLayer, self).__init__()
        self.n_lstm_layers = n_lstm_layers
        self._use_orthogonal = use_orthogonal

        # lstm. note that batch_first=False (default)
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=n_lstm_layers)
        # layer norm
        self.norm = nn.LayerNorm(hidden_size)

        # initialisation
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        
    def forward(self, x, hxs, cxs, masks):
        # no batch size
        if x.size(0) == hxs.size(0) and x.size(0) == cxs.size(0):
            x, (hxs, cxs) = \
                self.lstm(x.unsqueeze(0),
                          ((hxs * masks.repeat(1, self.n_lstm_layers).unsqueeze(-1)).transpose(0, 1).contiguous(),
                           (cxs * masks.repeat(1, self.n_lstm_layers).unsqueeze(-1)).transpose(0, 1).contiguous()))
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
            cxs = cxs.transpose(0, 1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)
            # unflatten
            x = x.view(T, N, x.size(1))
            # Same deal with masks
            masks = masks.view(T, N)
            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())
            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()
            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)
            cxs = cxs.transpose(0, 1)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                hxs_temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(self.n_lstm_layers, 1, 1)).contiguous()
                cxs_temp = (cxs * masks[start_idx].view(1, -1, 1).repeat(self.n_lstm_layers, 1, 1)).contiguous()
                lstm_scores, hxs = self.lstm(x[start_idx:end_idx], (hxs_temp, cxs_temp))
                outputs.append(lstm_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)
            cxs = cxs.transpose(0, 1)

        x = self.norm(x)
        return x, (hxs, cxs)