"""Faithful native encoders; verified pre-head tensors, explicit last-block tuning."""
from pathlib import Path
import sys
import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.clinical_tracehr_fullstream import CachedC, CachedResidual
from models.tracehr_multiview import MultiView
from scripts.tracehr_module_common import read, sha, native_paths


class NativeEncoders(nn.Module):
    def __init__(self, project, database, seed, data, branches='AVRT'):
        super().__init__()
        from tabm import TabM
        self.branches, self.paths = branches, native_paths(project, database, seed)
        self.native_identity = {}
        self.r_objects = []
        self.stage = 'frozen'
        for e in branches:
            folder = self.paths[e]
            self.native_identity[e] = {p.name: sha(p) for p in folder.iterdir()
                                      if p.name.startswith('best') or p.name in ('run_manifest.json','preprocessing.joblib')}
        if 'A' in branches:
            ck = torch.load(self.paths['A']/'best.pt', map_location='cpu', weights_only=False)
            assert ck['seed'] == seed and ck['cache_sha256'] == data.input_hashes['phi']
            assert ck['values_sha256'] == data.input_hashes['values'] and ck['role']=='stat'
            self.a = CachedResidual(CachedC(data.phi_dim), data.phi_dim, 0, 58 if database=='mimic' else 53)
            self.a.load_state_dict(ck['model_state_dict'])
        if 'V' in branches:
            meta = read(self.paths['V']/'run_manifest.json');cfg = meta['recipe']
            assert meta['seed']==seed and all(meta[k+'_sha256']==v for k,v in
                [('cache',data.input_hashes['phi']),('values',data.input_hashes['values'])])
            self.v_prep = joblib.load(self.paths['V']/'preprocessing.joblib')
            self.v = nn.ModuleList()
            for j in range(2):
                model = TabM.make(n_num_features=data.phi_dim+832,d_out=1,k=cfg['k'],d_block=cfg['width'],
                                  n_blocks=cfg['depth'],dropout=cfg['dropout'])
                ck = torch.load(self.paths['V']/f'best_task{j}.pt',map_location='cpu',weights_only=False)
                model.load_state_dict(ck['model_state_dict']);self.v.append(model)
        if 'R' in branches:
            sys.modules['lightning'] = None
            meta = read(self.paths['R']/'run_manifest.json')
            assert meta['seed']==seed and meta['cache_sha256']==data.input_hashes['phi'] and meta['values_sha256']==data.input_hashes['values']
            self.r = nn.ModuleList()
            self.r_static = nn.ModuleList()
            for j in range(2):
                obj = joblib.load(self.paths['R']/f'best_task{j}.joblib')
                assert obj.alg_interface_.model.config.get('n_ens',1)==1
                self.r.append(obj.alg_interface_.model.model)
                self.r_static.append(obj.alg_interface_.model.creator.static_model)
                self.r_objects.append(obj)
                layers = list(self.r[-1].tfms[1].tfms)
                assert len(layers)==4 and layers[-1].tfms[0].weight.shape[-2:]==(256,2)
            self.r_verified = False
        if 'T' in branches:
            ck = torch.load(self.paths['T']/'best.pt',map_location='cpu',weights_only=False)
            assert ck['identity']['mode']=='temporal' and ck['identity']['seed']==seed
            assert ck['identity']['inputs']==data.provenance
            self.t = MultiView(data.n_types,data.n_sources,'temporal',bins=8)
            self.t.load_state_dict(ck['model'])
        self.set_stage('frozen')

    def tune_modules(self):
        result = []
        if 'A' in self.branches: result += [self.a.residual.net[4], self.a.residual.net[5], self.a.residual.net[6], self.a.residual.head]
        if 'V' in self.branches:
            for v in self.v: result += [v.backbone.blocks[-1], v.output]
        if 'R' in self.branches:
            for r in self.r: result += [r.tfms[1].tfms[-2],r.tfms[1].tfms[-1]]
        if 'T' in self.branches:
            result += [self.t.temporal.interaction,self.t.temporal.norm1,self.t.temporal.ffn,self.t.temporal.norm2,
                       self.t.aux_attention,self.t.aux_head]
        return result

    def set_stage(self, stage):
        assert stage in ('frozen','finetune')
        self.stage = stage
        self.zero_grad(set_to_none=True)
        self.requires_grad_(False)
        if stage == 'finetune':
            for module in self.tune_modules():module.requires_grad_(True)
            if 'T' in self.branches:
                for p in (self.t.temporal.source_bias,self.t.temporal.time_bias,self.t.aux_query):p.requires_grad_(True)
        self.train(False)

    def train(self, mode=True):
        # Frozen prefixes remain deterministic; cached representations are only used in frozen stage.
        super().train(False)
        self.training = mode
        if mode and self.stage=='finetune':
            for module in self.tune_modules():module.train(True)
        return self

    def verify_r_converter(self, raw):
        import pandas as pd
        for obj in self.r_objects:
            converted = obj.x_converter_.transform(pd.DataFrame(raw)).tensors
            assert converted['x_cat'].shape == (len(raw),0)
            assert np.array_equal(converted['x_cont'].numpy(), raw), 'R conversion is not identity'
        self.r_verified = True

    def forward(self, phi, raw, events):
        device = next(self.parameters()).device
        hidden, valid, aux = {}, {}, {}
        n = len(raw)
        if 'A' in self.branches:
            x = torch.as_tensor(phi,device=device)
            hc, hs = self.a.anchor.net(x), self.a.residual.net(x)
            hidden['A'] = torch.stack([hc,hs],1)
            aux['A'] = self.a.anchor.head(hc)+self.a.residual.head(hs)
        if 'V' in self.branches:
            p=self.v_prep;z=(phi-p['phi_center'])*p['phi_factor'];z/=np.sqrt(1+(z/3)**2)
            z[:,p['binary']]=phi[:,p['binary']]
            numeric=p['numeric_quantile'].transform(raw[:,phi.shape[1]:]).astype(np.float32)
            x=torch.as_tensor(np.concatenate([z,numeric],1),device=device)
            h, logits = [], []
            for model in self.v:
                captured=[]
                hook=model.output.register_forward_pre_hook(lambda module,args:captured.append(args[0]))
                try: pred=model(x)
                finally: hook.remove()
                h.append(captured[0]);logits.append(pred.squeeze(-1))
            hidden['V']=torch.cat(h,1);aux['V']=torch.stack(logits,-1)
        if 'R' in self.branches:
            if not self.r_verified:self.verify_r_converter(raw)
            h, logits = [], []
            for net,static in zip(self.r,self.r_static):
                batch=dict(x_cont=torch.as_tensor(raw,device=device)[None],x_cat=torch.empty((1,n,0),dtype=torch.long,device=device))
                batch=static(batch)
                captured=[]
                hook=net.tfms[1].tfms[-1].register_forward_pre_hook(lambda module,args:captured.append(args[0]['x_cont']))
                try:pred=net(batch)['x_cont']
                finally:hook.remove()
                assert pred.shape==(1,n,2)
                h.append(captured[0][0]);logits.append(pred[0])
            hidden['R']=torch.stack(h,1);aux['R']=torch.stack(logits,1)
        if 'T' in self.branches:
            captured=[]
            hook=self.t.temporal.register_forward_hook(lambda module,args,output:captured.append(output))
            try:
                # In temporal mode the static context is unused, so placeholders do not alter the function.
                result=self.t(torch.zeros((n,192),device=device),torch.zeros((n,1024),device=device),events)
            finally:hook.remove()
            memory,mask,_=captured[0]
            hidden['T']=memory;valid['T']=mask;aux['T']=result['logits']
        for e in self.branches:
            if e not in valid:valid[e]=torch.ones(hidden[e].shape[:2],dtype=torch.bool,device=device)
        return hidden,valid,aux


def auxiliary_loss(aux, y, pos_weight):
    losses={}
    if 'A' in aux:losses['A']=F.binary_cross_entropy_with_logits(aux['A'],y,pos_weight=pos_weight)
    if 'V' in aux:losses['V']=F.binary_cross_entropy_with_logits(aux['V'],y[:,None].expand_as(aux['V']))
    if 'R' in aux:losses['R']=F.cross_entropy(aux['R'].reshape(-1,2),y.long().reshape(-1))
    if 'T' in aux:losses['T']=F.binary_cross_entropy_with_logits(aux['T'],y)
    return losses


def native_probabilities(aux):
    result={}
    for e,x in aux.items():
        result[e] = x.softmax(-1)[...,1] if e=='R' else x.sigmoid().mean(1) if e=='V' else x.sigmoid()
    return result
