"""Official model classes plus explicit patient-isolation/memory adaptations.

Official source files remain untouched and are pinned by source manifest. AST
extraction avoids running upstream CLI/data/test-set entry points on import.
"""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional, Union
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

def module_from_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def official_nca(sources,dim,config):
    utils=module_from_file('tracehr_nca_utils',sources/'modernnca/TALENT/model/lib/tabr/utils.py')
    path=sources/'modernnca/TALENT/model/models/modernNCA.py'
    tree=ast.parse(path.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef)]
    ns=dict(torch=torch,nn=nn,F=F,Optional=Optional,make_module=utils.make_module)
    exec(compile(tree,str(path),'exec'),ns)
    return ns['ModernNCA'](d_in=dim,d_num=dim,d_out=2,**config)

def nca_encode(model,x,chunk=256,grad=True):
    # The registered recipes use n_blocks=0. Chunking must not change BN stats.
    assert model.n_blocks==0
    def encode(v):
        if model.num_embeddings is not None:v=model.num_embeddings(v).flatten(1)
        return model.encoder(v)
    return torch.cat([checkpoint(encode,b,use_reentrant=False) if grad and torch.is_grad_enabled() else encode(b)
                      for b in x.split(chunk)])

def nca_training(model,x,y,patient_ids,query_indices,chunk=256):
    q=query_indices
    eligible=torch.ones(len(x),dtype=torch.bool,device=x.device);eligible[q]=False
    indices=torch.where(eligible)[0]
    # Same stochastic candidate mechanism as upstream; no context from dev.
    take=torch.randperm(len(indices))[:int(len(indices)*model.sample_rate)].to(x.device)
    c=indices[take]
    qe=nca_encode(model,x[q],chunk);ce=nca_encode(model,x[c],chunk)
    ce=torch.cat([qe,ce]);ci=torch.cat([q,c])
    distances=torch.cdist(qe,ce,p=2)/model.T
    mask=patient_ids[q,None]==patient_ids[ci][None,:]
    assert (~mask).sum(1).min()>0
    distances=distances.masked_fill(mask,torch.inf)
    return torch.log(F.softmax(-distances,-1)@F.one_hot(y[ci],2).to(x.dtype)+1e-7)

class ExactPatientIndex:
    """Exact squared-L2 GPU backend with patient exclusion before top-k.

    The official TabR Model forward and selected-neighbor gradient path are
    unchanged. This replaces only FAISS search, whose GPU build is unavailable
    in the reused environment. Search covers the entire training candidate bank.
    """
    def __init__(self):self.query_ids=self.candidate_ids=None
    def reset(self):self.candidates=None
    def add(self,candidates):self.candidates=candidates
    def search(self,q,k):
        c=self.candidates
        d=(q.square().sum(1,keepdim=True)-2*q@c.T+c.square().sum(1)[None]).clamp_min_(0)
        if self.query_ids is not None:
            assert self.candidate_ids is not None and len(self.query_ids)==len(q)
            d.masked_fill_(self.query_ids[:,None]==self.candidate_ids[None,:],torch.inf)
        distances,index=torch.topk(d,k,dim=-1,largest=False,sorted=True)
        assert torch.isfinite(distances).all(),'Too few distinct-patient candidates'
        return distances,index

def official_tabr(sources,dim,config):
    import delu
    path=sources/'tabr/lib/deep.py';tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.level)]
    ns=dict(TaskType=object,is_oom_exception=lambda e:isinstance(e,torch.cuda.OutOfMemoryError))
    exec(compile(tree,str(path),'exec'),ns)
    lib=SimpleNamespace(**ns)
    path=sources/'tabr/bin/tabr.py';tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Model']
    ns=dict(torch=torch,Tensor=Tensor,nn=nn,F=F,math=math,Literal=Literal,Optional=Optional,Union=Union,lib=lib,delu=delu)
    exec(compile(tree,str(path),'exec'),ns)
    model=ns['Model'](n_num_features=dim,n_bin_features=0,cat_cardinalities=[],n_classes=2,
        memory_efficient=True,candidate_encoding_batch_size=2048,**config)
    model.search_index=ExactPatientIndex()
    return model,lib

def tabr_apply(model,x,y,patient_ids,q,train):
    if train:
        eligible=torch.ones(len(x),dtype=torch.bool,device=x.device);eligible[q]=False
        ci=torch.where(eligible)[0]
        model.search_index.query_ids=patient_ids[q]
        model.search_index.candidate_ids=patient_ids[torch.cat([q,ci])]
        return model(x_={'num':x[q]},y=y[q],candidate_x_={'num':x[ci]},candidate_y=y[ci],context_size=96,is_train=True).squeeze(-1)
    model.search_index.query_ids=model.search_index.candidate_ids=None
    return model(x_={'num':q},y=None,candidate_x_={'num':x},candidate_y=y,context_size=96,is_train=False).squeeze(-1)

def official_excel(sources,dim,config):
    path=sources/'excelformer/lib/deep.py'
    tree=ast.parse(path.read_text());tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    import numpy as np,typing as ty
    import copy
    ns=dict(torch=torch,nn=nn,F=F,np=np,math=math,Tensor=Tensor,ty=ty,optim=torch.optim,deepcopy=copy.deepcopy)
    exec(compile(tree,str(path),'exec'),ns)
    augmentation=module_from_file('tracehr_excel_augment',sources/'excelformer/lib/aug_utils.py')
    for name in ['batch_dim_shuffle','batch_feat_shuffle','mixup_data']:ns[name]=getattr(augmentation,name)
    path=sources/'excelformer/bin/excel_former.py';tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if not(isinstance(n,ast.Import) and any(v.name=='lib' for v in n.names))]
    context=dict(lib=SimpleNamespace(**ns))
    exec(compile(tree,str(path),'exec'),context)
    return context['ExcelFormer'](d_numerical=dim,d_out=1,categories=None,**config)
