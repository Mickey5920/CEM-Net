"""Fixed-input source/time memory with summary-conditioned task queries.

This is a project architecture, not a renamed implementation of DGL/FAM/tPatch.
The record pool is permutation invariant with timestamps supplied explicitly.
"""
import math
import torch
from torch import nn


def segment_pool(tokens, logits, segments, n_segments):
    maximum = tokens.new_full((n_segments,), -torch.inf)
    maximum.scatter_reduce_(0, segments, logits, reduce='amax', include_self=True)
    weights = torch.exp(logits-maximum[segments])
    denominator = tokens.new_zeros(n_segments).index_add(0, segments, weights)
    numerator = tokens.new_zeros((n_segments,tokens.shape[-1])).index_add(0, segments, weights[:,None]*tokens)
    counts = tokens.new_zeros(n_segments).index_add(0, segments, torch.ones_like(weights))
    return numerator/denominator.clamp_min(1e-12)[:,None], counts


class SourceTimeMemory(nn.Module):
    def __init__(self, n_types, n_sources, bins=8, width=128, dropout=.1):
        super().__init__()
        self.n_sources, self.bins, self.width = n_sources, bins, width
        self.event = nn.Embedding(n_types+32,64)
        self.source = nn.Embedding(n_sources,64)
        self.continuous = nn.Linear(6,64)
        self.record = nn.Sequential(nn.GELU(),nn.Linear(64,64),nn.GELU())
        self.score = nn.Linear(64,1)
        self.block = nn.Linear(66,width)
        self.temporal = nn.Sequential(nn.Conv1d(width,width,3,padding=1),nn.GELU(),nn.Dropout(dropout),
                                      nn.Conv1d(width,width,3,padding=1),nn.GELU())
        self.block_norm = nn.LayerNorm(width)
        self.source_bias = nn.Parameter(torch.zeros(n_sources,n_sources))
        self.time_bias = nn.Parameter(torch.zeros(2*bins-1))
        source_ids = torch.arange(n_sources).repeat_interleave(bins)
        time_ids = torch.arange(bins).repeat(n_sources)
        self.register_buffer('source_ids',source_ids)
        self.register_buffer('relative_time',time_ids[:,None]-time_ids[None,:]+bins-1)
        self.interaction = nn.MultiheadAttention(width,4,dropout=dropout,batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Dropout(dropout),nn.Linear(width*2,width))
        self.norm2 = nn.LayerNorm(width)

    def forward(self, events, batch_size):
        typ, source, time, value, numeric, patient = [events[k] for k in ['type','source','time','value','numeric','patient']]
        bins = torch.clamp((time*self.bins).long(),0,self.bins-1)
        features = torch.stack([value,numeric,time,torch.sin(2*math.pi*time),torch.cos(2*math.pi*time),time*self.bins-bins],1)
        tokens = self.record(self.event(typ)+self.source(source)+self.continuous(features))
        segment = (patient*self.n_sources+source)*self.bins+bins
        pooled, counts = segment_pool(tokens,self.score(tokens).squeeze(-1),segment,batch_size*self.n_sources*self.bins)
        blocks = self.block(torch.cat([pooled,torch.log1p(counts)[:,None],(counts>0).float()[:,None]],1))
        blocks = blocks.reshape(batch_size*self.n_sources,self.bins,self.width)
        blocks = self.block_norm(blocks+self.temporal(blocks.transpose(1,2)).transpose(1,2))
        blocks = blocks.reshape(batch_size,self.n_sources*self.bins,self.width)
        observed = counts.reshape(batch_size,-1)>0
        blocks = blocks*observed[:,:,None]
        valid = observed.clone()
        empty = ~valid.any(1)
        valid[empty,0] = True  # Numerical placeholder only; all-empty memory remains zero.
        key_mask = blocks.new_zeros(valid.shape).masked_fill(~valid,-torch.inf)
        bias = self.source_bias[self.source_ids[:,None],self.source_ids[None,:]]+self.time_bias[self.relative_time]
        cross,_ = self.interaction(blocks,blocks,blocks,key_padding_mask=key_mask,attn_mask=bias,need_weights=False)
        blocks = self.norm1(blocks+cross)
        blocks = self.norm2(blocks+self.ffn(blocks))*observed[:,:,None]
        return blocks, valid, observed


class MultiView(nn.Module):
    def __init__(self,n_types,n_sources,mode='conditioned',bins=8,width=128,dropout=.1,a_dim=192,v_dim=1024):
        super().__init__()
        assert mode in ['temporal','static','concat','unconditioned','conditioned','no_a']
        self.mode,self.width = mode,width
        self.a_project = nn.Sequential(nn.LayerNorm(a_dim),nn.Linear(a_dim,width),nn.GELU())
        self.v_project = nn.Sequential(nn.LayerNorm(v_dim),nn.Linear(v_dim,width),nn.GELU())
        self.static_head = nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.Dropout(dropout),nn.Linear(width,2))
        if mode=='static':
            return
        self.temporal = SourceTimeMemory(n_types,n_sources,bins,width,dropout)
        self.aux_query = nn.Parameter(torch.randn(2,width)*.02)
        self.aux_attention = nn.MultiheadAttention(width,4,dropout=dropout,batch_first=True)
        self.aux_head = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,1))
        self.query = nn.Parameter(torch.randn(2,width)*.02)
        self.condition = nn.Sequential(nn.Linear(2*width,2*width),nn.GELU(),nn.Linear(2*width,2*width))
        self.read = nn.MultiheadAttention(width,4,dropout=dropout,batch_first=True)
        self.joint_head = nn.Sequential(nn.Linear(3*width,width),nn.GELU(),nn.Dropout(dropout),nn.Linear(width,1))

    def forward(self,h_a,h_v,events=None):
        n = len(h_a)
        a,v = self.a_project(h_a),self.v_project(h_v)
        if self.mode=='no_a':
            a = torch.zeros_like(a)
        context = torch.cat([a,v],1)
        if self.mode=='static':
            logits = self.static_head(context)
            return dict(logits=logits,aux=None)
        memory,valid,observed = self.temporal(events,n)
        aux,_ = self.aux_attention(self.aux_query[None].expand(n,-1,-1),memory,memory,key_padding_mask=~valid,need_weights=False)
        aux_logits = self.aux_head(aux).squeeze(-1)
        if self.mode=='temporal':
            return dict(logits=aux_logits,aux=None)
        if self.mode=='concat':
            # The shared scalar head still needs endpoint identity after global pooling.
            u = (memory.sum(1)/observed.sum(1).clamp_min(1)[:,None])[:,None]+self.query[None]
        else:
            q = self.query[None].expand(n,-1,-1)
            if self.mode in ['conditioned','no_a']:
                q = q+self.condition(context).reshape(n,2,self.width)
            u,_ = self.read(q,memory,memory,key_padding_mask=~valid,need_weights=False)
        logits = self.joint_head(torch.cat([context[:,None].expand(-1,2,-1),u],-1)).squeeze(-1)
        return dict(logits=logits,aux=aux_logits)

    def load_temporal(self, state):
        prefixes = ('temporal.','aux_query','aux_attention.','aux_head.')
        selected = {k:v for k,v in state.items() if k.startswith(prefixes)}
        result = self.load_state_dict(selected,strict=False)
        assert not result.unexpected_keys
        assert all(not k.startswith(prefixes) for k in result.missing_keys)
