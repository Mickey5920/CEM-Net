import torch
import torch.nn as nn

import transformer.Constants as Constants


def reverse_valid_sequence(values, lengths):
    positions = torch.arange(values.size(1), device=values.device).unsqueeze(0).expand(values.size(0), -1)
    reversed_positions = torch.where(positions < lengths.unsqueeze(1), lengths.unsqueeze(1) - positions - 1, positions)
    return values.gather(1, reversed_positions.unsqueeze(-1).expand(-1, -1, values.size(-1)))


class ClinicalRETAINEncoder(nn.Module):
    """Event-level RETAIN adaptation with reverse-time alpha and beta attention."""

    def __init__(self, num_types, d_model=64, hidden_dim=64, dropout=0.1, time_scale_hours=24.0):
        super().__init__()
        self.d_model = int(d_model)
        self.time_scale_hours = float(time_scale_hours)
        self.event_emb = nn.Embedding(int(num_types) + 1, d_model, padding_idx=Constants.PAD)
        self.time_proj = nn.Sequential(nn.Linear(2, d_model), nn.Tanh())
        self.alpha_rnn = nn.GRU(d_model, hidden_dim, batch_first=True)
        self.beta_rnn = nn.GRU(d_model, hidden_dim, batch_first=True)
        self.alpha = nn.Linear(hidden_dim, 1)
        self.beta = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, event_type, event_time, time_gap=None):
        non_pad = event_type.ne(Constants.PAD)
        lengths = non_pad.sum(dim=1).long().clamp_min(1)
        if time_gap is None:
            time_gap = torch.zeros_like(event_time)
            time_gap[:, 1:] = torch.clamp(event_time[:, 1:] - event_time[:, :-1], min=0.0)
        time_features = torch.stack(
            [event_time / self.time_scale_hours, time_gap / self.time_scale_hours],
            dim=-1,
        )
        values = self.dropout(self.event_emb(event_type) + self.time_proj(time_features))
        reversed_values = reverse_valid_sequence(values, lengths)
        packed = nn.utils.rnn.pack_padded_sequence(
            reversed_values,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        alpha_packed, _ = self.alpha_rnn(packed)
        beta_packed, _ = self.beta_rnn(packed)
        alpha_hidden, _ = nn.utils.rnn.pad_packed_sequence(alpha_packed, batch_first=True, total_length=event_type.size(1))
        beta_hidden, _ = nn.utils.rnn.pad_packed_sequence(beta_packed, batch_first=True, total_length=event_type.size(1))
        alpha_hidden = reverse_valid_sequence(alpha_hidden, lengths)
        beta_hidden = reverse_valid_sequence(beta_hidden, lengths)
        alpha_logits = self.alpha(alpha_hidden).squeeze(-1).masked_fill(~non_pad, torch.finfo(values.dtype).min)
        alpha_weights = torch.softmax(alpha_logits, dim=1).unsqueeze(-1)
        beta_weights = torch.tanh(self.beta(beta_hidden))
        sequence_hidden = values * beta_weights * non_pad.unsqueeze(-1).float()
        pooled = (alpha_weights * sequence_hidden).sum(dim=1)
        return pooled, sequence_hidden
