"""Native comparator architectures with fixed, audited numerical input handling."""
import torch
from torch import nn

def tabm_model(variant,bins,small=False):
    from tabm import TabM
    import rtdl_num_embeddings as rtdl
    embedding=rtdl.PiecewiseLinearEmbeddings(bins,d_embedding=16,activation=False,version='B')
    return TabM.make(n_num_features=len(bins),d_out=1,num_embeddings=embedding,
        k=32,d_block=16 if small else 256,n_blocks=3,dropout=.2,
        arch_type='tabm-mini' if variant=='TabM_mini_PLE' else 'tabm')

def make_model(variant,seed,data):
    from scripts.train_tracehr_unified import seed_all
    from scripts.cem_additional_common import preparation
    seed_all(seed)
    if variant in ('SeFT','GRU_D','mTAND_Enc'):
        from models.cem_sequence_baselines import MeasurementSeFT,GRUD,SparseMTAND
        data.variant=variant
        return MeasurementSeFT(data.input_dim) if variant=='SeFT' else GRUD(data.input_dim,data.num_types) if variant=='GRU_D' else SparseMTAND(data.input_dim)
    bins=torch.load(preparation()/'bins.pt',map_location='cpu',weights_only=False)
    return tabm_model(variant,bins)
