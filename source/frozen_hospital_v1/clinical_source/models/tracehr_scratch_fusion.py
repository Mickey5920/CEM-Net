"""Random initialization of all retained encoders and the direct fusion head."""
import copy
import sys
import torch
from torch import nn
from models.clinical_tracehr_fullstream import CachedC
from models.tracehr_multiview import SourceTimeMemory
from models.tracehr_module_fusion import ModuleFusion


class FreshA(nn.Module):
    def __init__(self,input_dim,residual_width):
        super().__init__();self.anchor=CachedC(input_dim);self.residual=CachedC(input_dim,residual_width)

    def forward(self,phi):
        a,b=self.anchor.net(phi),self.residual.net(phi)
        return torch.stack([a,b],1),self.anchor.head(a)+self.residual.head(b)


class FreshT(nn.Module):
    def __init__(self,n_types,n_sources):
        super().__init__();self.temporal=SourceTimeMemory(n_types,n_sources)
        self.query=nn.Parameter(torch.randn(2,128)*.02)
        self.attention=nn.MultiheadAttention(128,4,dropout=.1,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,1))

    def forward(self,events,n):
        memory,valid,_=self.temporal(events,n)
        u,_=self.attention(self.query[None].expand(n,-1,-1),memory,memory,key_padding_mask=~valid,need_weights=False)
        return memory,valid,self.head(u).squeeze(-1)


def fixed_statistics_to_buffers(module):
    """PyTabKit stores fitted preprocessing statistics as non-trainable Variables."""
    names=[]
    for prefix,child in module.named_modules():
        for name,parameter in list(child._parameters.items()):
            if parameter is not None and not parameter.requires_grad:
                del child._parameters[name];child.register_buffer(name,parameter.detach())
                names.append(prefix+'.'+name)
    return names


def realmlp_prediction_layer(model):
    candidates=[name for name,layer in model.named_modules()
                if hasattr(layer,'weight') and isinstance(layer.weight,torch.Tensor)
                and layer.weight.ndim>=2 and tuple(layer.weight.shape[-2:])==(256,2)]
    assert len(candidates)==1,('Ambiguous RealMLP output projection',candidates)
    return candidates[0]


def fresh_realmlp(raw,y,seed,config):
    sys.modules['lightning']=None
    from pytabkit.models.training.nn_creator import NNCreator
    from pytabkit.models.data.data import DictDataset,TensorInfo
    from pytabkit.models.alg_interfaces.base import SplitIdxs,InterfaceResources
    cfg=copy.deepcopy(config)
    for key in ('tmp_folder','device','random_state','val_metric_name'):cfg.pop(key,None)
    cfg.update(allow_single_split_opt=False,p_drop_sched='constant',wd_sched='constant',lr_sched='constant',init_ram_limit_gb=1.)
    tensors=dict(x_cont=torch.as_tensor(raw),x_cat=torch.empty((len(raw),0),dtype=torch.long),y=torch.as_tensor(y[:,None],dtype=torch.long))
    infos=dict(x_cont=TensorInfo(feat_shape=[raw.shape[1]]),x_cat=TensorInfo(cat_sizes=[]),y=TensorInfo(cat_sizes=[2]))
    ds=DictDataset(tensors,infos)
    splits=[SplitIdxs(torch.arange(len(raw))[None],None,None,seed,[seed],0)]
    creator=NNCreator(**cfg);creator.setup_from_dataset(ds,splits,InterfaceResources(4,[]))
    # This constructs data-fitted normalizers and randomly initialized network parameters;
    # it performs zero optimizer steps and reads no trained weights.
    model=creator.create_model(ds,splits)
    fixed=fixed_statistics_to_buffers(model);fixed+=['static.'+s for s in fixed_statistics_to_buffers(creator.static_model)]
    prehead=realmlp_prediction_layer(model)
    assert all(v.requires_grad for v in model.parameters())
    return model,creator.static_model,dict(config=cfg,fixed_statistics=fixed,prehead_module=prehead)


class ScratchFusion(nn.Module):
    def __init__(self,data,branches,fusion,seed,r_config,v_config,head_width=256,fixed_query=False,compress_members=False):
        super().__init__();self.branches,self.phi_dim=branches,data.phi_dim;self.initialization={}
        # Branch initialization is independent of which other branches are retained.
        for e in branches:
            if hasattr(data,'initialization_progress'):data.initialization_progress(e)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+1000*('AVRT'.index(e)+1))
                if e=='A':self.a=FreshA(data.phi_dim,58 if data.database=='mimic' else 53)
                elif e=='V':
                    from tabm import TabM
                    self.v=nn.ModuleList([TabM.make(n_num_features=data.input_dim,d_out=1,k=v_config['k'],
                        d_block=v_config['width'],n_blocks=v_config['depth'],dropout=v_config['dropout']) for _ in range(2)])
                elif e=='R':
                    self.r=nn.ModuleList();self.r_static=nn.ModuleList()
                    raw=data.initialization_raw()
                    for task in range(2):
                        net,static,info=fresh_realmlp(raw,data.targets['train']['y'][:,task],seed+3000+task,r_config)
                        self.r.append(net);self.r_static.append(static);self.initialization[f'R{task}']=info
                    del raw
                else:self.t=FreshT(data.n_types,data.n_sources)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+5000)
            self.fusion=ModuleFusion(branches,fusion,n_temporal=data.n_sources*8,head_width=head_width,
                                     fixed_query=fixed_query,compress_members=compress_members)
        assert all(p.requires_grad for p in self.parameters()),'Unexpected frozen learnable parameter'

    def forward(self,batch):
        raw=batch['raw'];n=len(raw);hidden,valid,aux={},{},{}
        if 'A' in self.branches:hidden['A'],aux['A']=self.a(raw[:,:self.phi_dim])
        if 'V' in self.branches:
            h,z=[],[]
            for net in self.v:
                found=[];hook=net.output.register_forward_pre_hook(lambda module,args:found.append(args[0]))
                try:pred=net(batch['v'])
                finally:hook.remove()
                h.append(found[0]);z.append(pred.squeeze(-1))
            hidden['V']=torch.cat(h,1);aux['V']=torch.stack(z,-1)
        if 'R' in self.branches:
            h,z=[],[]
            for task,(net,static) in enumerate(zip(self.r,self.r_static)):
                x=static(dict(x_cont=raw[None],x_cat=torch.empty((1,n,0),device=raw.device,dtype=torch.long)))
                output_layer=net.get_submodule(self.initialization[f'R{task}']['prehead_module'])
                found=[];hook=output_layer.register_forward_pre_hook(lambda module,args:found.append(args[0]['x_cont']))
                try:pred=net(x)['x_cont']
                finally:hook.remove()
                assert pred.shape==(1,n,2);h.append(found[0][0]);z.append(pred[0])
            hidden['R']=torch.stack(h,1);aux['R']=torch.stack(z,1)
        if 'T' in self.branches:hidden['T'],valid['T'],aux['T']=self.t(batch['events'],n)
        for e in self.branches:
            if e not in valid:valid[e]=torch.ones(hidden[e].shape[:2],device=raw.device,dtype=torch.bool)
        return self.fusion(hidden,valid),aux

    def optimizer_groups(self):
        common=[];groups=[]
        for name,p in self.named_parameters():
            assert p.requires_grad
            if name.startswith('r.'):
                factors=getattr(p,'hyper_factors',{})
                groups.append(dict(params=[p],lr=.04*factors.get('lr',1.),weight_decay=.02*factors.get('wd',1.),betas=(.9,.95)))
            else:common.append(p)
        groups.insert(0,dict(params=common,lr=5e-4,weight_decay=1e-4,betas=(.9,.999)))
        return groups
