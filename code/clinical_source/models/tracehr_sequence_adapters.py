"""Native KEDGN/MTM encoders and audited sparse-grid MTM pooling."""
import ast,importlib.util,math,sys,types
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange,repeat,reduce

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def sparse_mtm_downsample(self,x,x_mask,idx_b,idx_t,idx_c,imp):
    """Algebraically equivalent to expand->pool->shrink on observed groups.

    The original expands 24h seconds into 86,401 mostly empty time positions.
    This computes exactly the same per-grid-bin max/mean without those zeros.
    No time quantization or input event truncation is introduced.
    """
    bsz,_,channels,dim=x.shape;xs=[];masks=[];times=[]
    for b in range(bsz):
        valid=idx_t[b,:,0]>=0
        t=idx_t[b,valid,0]//self.ratio
        groups,inverse=torch.unique_consecutive(t,return_inverse=True)
        if not len(groups):
            groups=t.new_tensor([0]);inverse=t.new_empty(0)
        v=x[b,valid];mask=x_mask[b,valid];index=inverse[:,None,None].expand(-1,channels,dim)
        sums=x.new_zeros((len(groups),channels,dim));counts=x.new_zeros((len(groups),channels))
        sums.scatter_add_(0,index,v.masked_fill(mask[...,None],0))
        counts.scatter_add_(0,inverse[:,None].expand(-1,channels),(~mask).to(x.dtype))
        maximum=x.new_full((len(groups),channels,dim),-torch.inf)
        maximum.scatter_reduce_(0,index,v.masked_fill(mask[...,None],-torch.inf),reduce='amax',include_self=True)
        maximum=maximum.nan_to_num(0,0,0)
        avg=sums/(counts[...,None]+1e-7)
        if self.mode=='concat':out=self.lin(torch.cat([maximum,avg],-1))
        elif self.mode=='max':out=maximum
        elif self.mode=='avg':out=avg
        else:raise ValueError(self.mode)
        xs.append(out);masks.append(counts==0);times.append(groups)
    out=nn.utils.rnn.pad_sequence(xs,batch_first=True,padding_value=0.)
    mask=nn.utils.rnn.pad_sequence(masks,batch_first=True,padding_value=True)
    t=nn.utils.rnn.pad_sequence(times,batch_first=True,padding_value=-1)
    return out,mask,repeat(t,'b t -> b t c',c=channels)

def mtm_modules(sources,deps):
    # Only timm's official stochastic-depth function is needed; avoid importing
    # torchvision/image model registries into this clinical sequence task.
    path=deps/'timm/layers/drop.py';tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['drop_path','DropPath']]
    ns=dict(torch=torch,nn=nn);exec(compile(tree,str(path),'exec'),ns)
    fake=types.ModuleType('timm');fake.__path__=[]
    layers=types.ModuleType('timm.layers');layers.DropPath=ns['DropPath']
    sys.modules['timm']=fake;sys.modules['timm.layers']=layers
    package=types.ModuleType('tracehr_official_mtm');package.__path__=[str(sources/'mtm/mtm')]
    sys.modules['tracehr_official_mtm']=package
    utils=load_module('tracehr_official_mtm.utils',sources/'mtm/mtm/utils.py')
    sampling=load_module('tracehr_official_mtm.sampling',sources/'mtm/mtm/sampling.py')
    model=load_module('tracehr_official_mtm.mtm',sources/'mtm/mtm/mtm.py')
    return model,utils,sampling

class SequenceClassifier(nn.Module):
    def __init__(self,family,sources,deps,f0_dim,config,device='cuda'):
        super().__init__();self.family=family
        if family=='kedgn':
            path=sources/'kedgn/model.py';tree=ast.parse(path.read_text())
            tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef)]
            ns=dict(torch=torch,nn=nn,F=F,rearrange=rearrange,repeat=repeat,reduce=reduce)
            exec(compile(tree,str(path),'exec'),ns)
            self.model=ns['KEDGN'](DEVICE=device,num_of_variables=32,num_of_timestamps=100,d_static=f0_dim,n_class=2,**config)
        else:
            module,utils,sampling=mtm_modules(sources,deps)
            self.background=nn.Sequential(nn.Linear(f0_dim,32),nn.ReLU())
            self.model=module.MTM(num_chn=32,d_static=32,num_cls=2,**config)
            # Exact second grid; the positional formula is unchanged.
            self.model.rpe=utils.precompute_rpe(self.model.d_model,max_len=86402)
            self.model.ape=utils.precompute_ape(self.model.d_model,max_len=86402)
            for sub in self.model.modules():
                if isinstance(sub,sampling.Downsample):sub.forward=types.MethodType(sparse_mtm_downsample,sub)

    def forward(self,batch,semantics=None):
        x,obs,t,secs,lengths,interval,background=batch
        if self.family=='kedgn':
            return self.model(torch.cat([x,obs.float()],-1),background,interval,lengths[:,None],t/24.,semantics)
        return self.model(x.masked_fill(~obs,torch.nan),~obs,secs,self.background(background))

def collate_numeric(sequence,indices,background,mean,scale,device='cuda'):
    offsets=sequence['offsets'];lens=np.array([offsets[i+1]-offsets[i] for i in indices],dtype=np.int64)
    # A masked placeholder allows native heads to retain all-empty patients.
    lens=np.maximum(lens,1);maxlen=int(lens.max());b=len(indices)
    x=np.zeros((b,maxlen,32),np.float32);obs=np.zeros_like(x,dtype=bool)
    t=np.zeros((b,maxlen),np.float32);secs=np.full((b,maxlen),-1,np.int64);interval=np.zeros_like(x)
    for bi,i in enumerate(indices):
        lo,hi=int(offsets[i]),int(offsets[i+1]);n=hi-lo
        if n:
            x[bi,:n]=sequence['values'][lo:hi];obs[bi,:n]=sequence['observed'][lo:hi]
            t[bi,:n]=sequence['times'][lo:hi];secs[bi,:n]=sequence['seconds'][lo:hi]
            for v in range(32):
                at=np.flatnonzero(obs[bi,:n,v]);observed_time=t[bi,at]
                if len(at)==1:interval[bi,at,v]=12.
                elif len(at)>1:
                    gaps=np.diff(observed_time)
                    interval[bi,at,v]=(np.r_[gaps[0],gaps]+np.r_[gaps,gaps[-1]])/2
        else:secs[bi,0]=0
    bg=(background[indices].toarray()-mean)/scale
    return tuple(torch.as_tensor(a,device=device) for a in [x,obs,t,secs,lens,interval,bg.astype(np.float32)])
