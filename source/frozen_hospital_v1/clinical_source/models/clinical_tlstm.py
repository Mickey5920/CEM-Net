import math

import torch
import torch.nn as nn

import transformer.Constants as Constants
from models.clinical_pooling import build_attention_pooling, pool_sequence


class TimeAwareLSTMCell(nn.Module):
    """T-LSTM cell following the short-term memory decay formulation."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.memory_decompose = nn.Linear(hidden_dim, hidden_dim)
        self.gates = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)

    def forward(self, x, elapsed_time, state):
        hidden, cell = state
        short_term = torch.tanh(self.memory_decompose(cell))
        decay = 1.0 / torch.log(elapsed_time.clamp_min(0.0).unsqueeze(-1) + math.e)
        adjusted_cell = cell - short_term + short_term * decay
        input_gate, forget_gate, output_gate, candidate = self.gates(torch.cat([x, hidden], dim=-1)).chunk(4, dim=-1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)
        cell = forget_gate * adjusted_cell + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell


class ClinicalTLSTMEncoder(nn.Module):
    """Event-level adaptation of T-LSTM using inter-event time gaps."""

    def __init__(self, num_types, d_model=64, hidden_dim=64, dropout=0.1, pooling='mean', time_scale_hours=24.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.pooling = pooling
        self.time_scale_hours = float(time_scale_hours)
        self.event_emb = nn.Embedding(int(num_types) + 1, d_model, padding_idx=Constants.PAD)
        self.cell = TimeAwareLSTMCell(d_model, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = build_attention_pooling(pooling, hidden_dim)

    def forward(self, event_type, event_time, time_gap=None):
        non_pad = event_type.ne(Constants.PAD)
        if time_gap is None:
            time_gap = torch.zeros_like(event_time)
            time_gap[:, 1:] = torch.clamp(event_time[:, 1:] - event_time[:, :-1], min=0.0)
        x = self.dropout(self.event_emb(event_type))
        hidden = x.new_zeros((x.size(0), self.hidden_dim))
        cell = x.new_zeros((x.size(0), self.hidden_dim))
        outputs = []
        for step in range(x.size(1)):
            next_hidden, next_cell = self.cell(x[:, step], time_gap[:, step] / self.time_scale_hours, (hidden, cell))
            valid = non_pad[:, step].unsqueeze(-1)
            hidden = torch.where(valid, next_hidden, hidden)
            cell = torch.where(valid, next_cell, cell)
            outputs.append(torch.where(valid, hidden, torch.zeros_like(hidden)))
        sequence_hidden = torch.stack(outputs, dim=1)
        pooled = pool_sequence(sequence_hidden, non_pad, pooling=self.pooling, attention=self.attention)
        return pooled, sequence_hidden
