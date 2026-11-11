import torch
import torch.nn as nn

import transformer.Constants as Constants
from models.clinical_pooling import build_attention_pooling, pool_sequence
from transformer.Models_mhp import Encoder_pure as MHPEncoderPure
from transformer.mambapy.mamba_mhp import MambaConfig as MHPConfig


class MHPEncoderAdapter(nn.Module):
    """Use the MHP event encoder as a clinical sequence encoder."""

    def __init__(
        self,
        num_types,
        d_model=64,
        d_inner=256,
        n_layers=2,
        n_head=4,
        d_k=16,
        d_v=16,
        dropout=0.1,
        pooling='last',
        model_type='Mamba_pure',
    ):
        super().__init__()
        self.num_types = int(num_types)
        self.d_model = int(d_model)
        self.pooling = pooling
        self.attention = build_attention_pooling(pooling, d_model)
        config = MHPConfig(d_model=d_model, n_layers=n_layers)
        self.encoder = MHPEncoderPure(
            config=config,
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            model_type=model_type,
        )

    def forward(self, event_type, event_time):
        non_pad_mask = event_type.ne(Constants.PAD).float().unsqueeze(-1)
        sequence_hidden = self.encoder(event_type, event_time, non_pad_mask)
        if isinstance(sequence_hidden, tuple):
            sequence_hidden = sequence_hidden[0]
        sequence_hidden = sequence_hidden * non_pad_mask
        pooled = self.pool(sequence_hidden, non_pad_mask)
        return pooled, sequence_hidden

    def pool(self, sequence_hidden, non_pad_mask):
        return pool_sequence(sequence_hidden, non_pad_mask, pooling=self.pooling, attention=self.attention)
