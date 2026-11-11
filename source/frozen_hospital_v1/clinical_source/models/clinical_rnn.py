import torch
import torch.nn as nn

import transformer.Constants as Constants
from models.clinical_pooling import build_attention_pooling, pool_sequence


class ClinicalRNNEncoder(nn.Module):
    """Vanilla Elman RNN over event tokens and continuous-time features."""

    def __init__(self, num_types, d_model=64, hidden_dim=64, dropout=0.1, pooling='last', time_scale_hours=24.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.pooling = pooling
        self.time_scale_hours = float(time_scale_hours)
        self.attention = build_attention_pooling(pooling, hidden_dim)
        self.event_emb = nn.Embedding(int(num_types) + 1, d_model, padding_idx=Constants.PAD)
        self.time_proj = nn.Sequential(nn.Linear(2, d_model), nn.Tanh())
        self.rnn = nn.RNN(d_model, hidden_dim, nonlinearity='tanh', batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, event_type, event_time, time_gap=None):
        non_pad = event_type.ne(Constants.PAD)
        if time_gap is None:
            time_gap = torch.zeros_like(event_time)
            time_gap[:, 1:] = torch.clamp(event_time[:, 1:] - event_time[:, :-1], min=0.0)
        time_features = torch.stack(
            [event_time / self.time_scale_hours, time_gap / self.time_scale_hours],
            dim=-1,
        )
        x = self.dropout(self.event_emb(event_type) + self.time_proj(time_features))
        lengths = non_pad.sum(dim=1).long().clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        sequence_hidden, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=event_type.size(1),
        )
        sequence_hidden = sequence_hidden * non_pad.unsqueeze(-1).float()
        pooled = pool_sequence(sequence_hidden, non_pad, pooling=self.pooling, attention=self.attention)
        return pooled, sequence_hidden

