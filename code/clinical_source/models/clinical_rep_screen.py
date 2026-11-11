"""Count PLE, raw-count bypass, and official compact TabM screening models."""
from collections import Counter
import math
import torch
from torch import nn

class RepresentationScreen(nn.Module):
    def __init__(self, single, variant, tasks):
        super().__init__();self.stats=single.encoder.active;self.head=single.head
        self.variant=variant;self.n=self.stats.num_types;self.members=None
        self.register_buffer('fit_count',torch.tensor(0,dtype=torch.long))
        if variant=='L':
            with torch.random.fork_rng(devices=[]):self.bypass=nn.Linear(self.n,tasks,bias=False)
            nn.init.zeros_(self.bypass.weight)
        elif variant=='P':
            self.register_buffer('low',torch.zeros(self.n,8));self.register_buffer('high',torch.ones(self.n,8))
            self.register_buffer('valid',torch.zeros(self.n,8,dtype=torch.bool))
            self.register_buffer('last',torch.zeros(self.n,8,dtype=torch.bool))
            width=self.stats.input_dim+7*self.n
            self.stats.net[0]=nn.LayerNorm(width)
            self.stats.net[1]=nn.Linear(width,self.stats.hidden_dim)
        elif variant=='TabM':
            from tabm import TabM
            width=self.stats.input_dim
            self.stats.net=nn.Identity();self.head=nn.Identity()
            self.register_buffer('mean',torch.zeros(width));self.register_buffer('scale',torch.ones(width))
            self.tabm=TabM.make(n_num_features=width,d_out=tasks,k=8,d_block=192,n_blocks=3,dropout=.15)
        else:raise ValueError(variant)

    @torch.no_grad()
    def fit_bins(self,samples):
        assert self.variant=='P'
        hist=[Counter() for _ in range(self.n)]
        for sample in samples:
            for index,count in Counter(sample['type_event']).items():
                assert 0<=int(index)<self.n
                hist[int(index)][count]+=1
        for i,h in enumerate(hist):
            if not h:edges=[0.,1.]
            else:
                pairs=sorted(h.items());total=sum(h.values());edges=[0.]
                for q in range(1,9):
                    threshold=math.ceil(total*q/8);cum=0
                    for value,freq in pairs:
                        cum+=freq
                        if cum>=threshold:
                            if float(value)>edges[-1]:edges.append(float(value))
                            break
            k=len(edges)-1
            self.low[i,:k]=torch.tensor(edges[:-1]);self.high[i,:k]=torch.tensor(edges[1:])
            self.valid[i,:k]=True;self.last[i,k-1]=True
        self.fit_count.fill_(len(samples))

    def encode_counts(self,c):
        assert self.fit_count.item()>0
        z=(c.unsqueeze(-1)-self.low)/(self.high-self.low).clamp_min(1e-8)
        # Last active interval extrapolates above train max; zero remains zero.
        return torch.where(self.last,z.clamp_min(0),z.clamp(0,1))*self.valid

    def c_features(self,f):return torch.cat([torch.log1p(f[:,:self.n]),f[:,self.n:]],1)

    @torch.no_grad()
    def fit_tabm(self,batches):
        assert self.variant=='TabM'
        count=0;mean=torch.zeros_like(self.mean,dtype=torch.float64);m2=mean.clone()
        for x,t,s in batches:
            f=self.c_features(self.stats.build_static_features(x,t,s)).double()
            n=len(f);avg=f.mean(0);delta=avg-mean
            m2+=((f-avg)**2).sum(0)+delta.square()*count*n/(count+n)
            mean+=delta*n/(count+n);count+=n
        assert count>1
        scale=(m2/count).sqrt();scale=torch.where(scale>1e-6,scale,torch.ones_like(scale))
        mean[self.n:2*self.n]=0;scale[self.n:2*self.n]=1
        self.mean.copy_(mean);self.scale.copy_(scale);self.fit_count.fill_(count)

    def forward(self,event_type,event_time,time_gap=None,source_features=None):
        f=self.stats.build_static_features(event_type,event_time,source_features)
        if self.variant=='P':
            f=torch.cat([self.encode_counts(f[:,:self.n]).flatten(1),f[:,self.n:]],1)
            pooled=self.stats.net(f);logits=self.head(pooled)
        elif self.variant=='L':
            pooled=self.stats.net(self.c_features(f));logits=self.head(pooled)+self.bypass(f[:,:self.n])
        else:
            assert self.fit_count.item()>1
            values=self.tabm((self.c_features(f)-self.mean)/self.scale)
            self.members=values if self.training else None
            pooled=values.mean(1)
            logits=torch.logit(values.sigmoid().mean(1).clamp(1e-7,1-1e-7))
        hidden=pooled.unsqueeze(1).expand(-1,event_type.size(1),-1)*event_type.ne(0).unsqueeze(-1)
        return logits,hidden,pooled
