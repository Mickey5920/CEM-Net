"""Bounded transforms of existing rich statistics and four-bin time pooling."""
import torch
from torch import nn

class DenseRepresentation(nn.Module):
    def __init__(self, original, variant):
        super().__init__()
        if variant not in ['N1','N2','T','N1T','N2T','C','R','S','C_no_decay','C_no_last']:raise ValueError(variant)
        self.base=original;self.variant=variant;self.num_types=original.num_types
        self.log_counts=variant.startswith('N');self.standardize=variant.startswith('N2')
        self.has_bins=variant.endswith('T')
        self.register_buffer('feature_mean',torch.zeros(original.input_dim))
        self.register_buffer('feature_scale',torch.ones(original.input_dim))
        self.register_buffer('fit_count',torch.tensor(0,dtype=torch.long))
        if self.standardize:self.base.net[0]=nn.Identity()
        if self.has_bins:
            self.bin_embeddings=nn.Parameter(torch.empty(self.num_types,16))
            nn.init.normal_(self.bin_embeddings,std=.02)
            self.bin_projection=nn.Linear(4*16,original.net[-1].normalized_shape[0],bias=False)
            nn.init.zeros_(self.bin_projection.weight)

    def unscaled_features(self,types,times,sources):
        f=self.base.build_static_features(types,times,sources)
        if self.log_counts:
            n=self.num_types
            f=torch.cat([torch.log1p(f[:,:n]),f[:,n:3*n],torch.log1p(f[:,3*n:4*n]),f[:,4*n:]],dim=1)
        elif self.variant in ['C','R','S','C_no_decay','C_no_last']:
            n=self.num_types
            counts,decay=f[:,:n],f[:,3*n:4*n]
            if self.variant.startswith('C'):counts=torch.log1p(counts)
            elif self.variant=='R':decay=torch.log1p(decay)
            else:
                # Algebraically sqrt(1+x)-1, without cancellation near zero.
                counts=counts/(torch.sqrt(1+counts)+1)
                decay=decay/(torch.sqrt(1+decay)+1)
            f=torch.cat([counts,f[:,n:3*n],decay,f[:,4*n:]],dim=1)
            if self.variant=='C_no_decay':
                f=torch.cat([f[:,:3*n],torch.zeros_like(f[:,3*n:4*n]),f[:,4*n:]],1)
            elif self.variant=='C_no_last':
                f=torch.cat([f[:,:2*n],torch.zeros_like(f[:,2*n:3*n]),f[:,3*n:]],1)
        return f

    @torch.no_grad()
    def fit_normalization(self,batches):
        if not self.standardize:raise ValueError('Only N2 fits train statistics')
        count=0;mean=torch.zeros_like(self.feature_mean,dtype=torch.float64);m2=mean.clone()
        for types,times,sources in batches:
            f=self.unscaled_features(types,times,sources).double()
            assert torch.isfinite(f).all()
            n=f.shape[0];avg=f.mean(0);delta=avg-mean
            m2+=((f-avg)**2).sum(0)+delta.square()*count*n/(count+n)
            mean+=delta*n/(count+n);count+=n
        if count<2:raise ValueError('Need at least two training samples')
        scale=(m2/count).sqrt();scale=torch.where(scale>1e-6,scale,torch.ones_like(scale))
        # Presence indicators stay binary; all statistics come from train only.
        mean[self.num_types:2*self.num_types]=0;scale[self.num_types:2*self.num_types]=1
        self.feature_mean.copy_(mean);self.feature_scale.copy_(scale);self.fit_count.fill_(count)

    def bin_fractions(self,types,times):
        mask=types.ne(0)
        if torch.any(mask & ((times<0)|(times>self.base.time_scale_hours+1e-3))):
            raise ValueError('Observed event outside prediction window')
        bins=(times/self.base.time_scale_hours*4).floor().long().clamp(0,3)
        index=bins*self.num_types+(types-1).clamp(0,self.num_types-1)
        counts=times.new_zeros((types.size(0),4*self.num_types))
        counts.scatter_add_(1,index,mask.to(times.dtype))
        counts=counts.reshape(types.size(0),4,self.num_types)
        return counts/counts.sum(1,keepdim=True).clamp_min(1)

    def forward(self,event_type,event_time,source_features=None):
        f=self.unscaled_features(event_type,event_time,source_features)
        if self.standardize:
            if self.fit_count.item()<2:raise ValueError('Train-only normalization has not been fitted')
            f=(f-self.feature_mean)/self.feature_scale
        pooled=self.base.net(f)
        if self.has_bins:
            fractions=self.bin_fractions(event_type,event_time)
            denom=fractions.sum(1).gt(0).sum(1).clamp_min(1).sqrt().reshape(-1,1,1)
            compressed=(fractions@self.bin_embeddings)/denom
            pooled=pooled+self.bin_projection(compressed.flatten(1))
        hidden=pooled.unsqueeze(1).expand(-1,event_type.size(1),-1)*event_type.ne(0).unsqueeze(-1)
        return pooled,hidden
