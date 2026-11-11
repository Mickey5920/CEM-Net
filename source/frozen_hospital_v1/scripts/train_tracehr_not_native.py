"""Actual native-entry audit and F0. Instrumentation preserves native RNG/update order."""
import argparse
import copy
import gc
import hashlib
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.environ['TRACEHR_TABULAR_DEPS'])
sys.modules['lightning'] = None
import joblib
import numpy as np
import torch
from torch import nn
from models.tracehr_centered_native import FullNative, native_parameter_hash
from models.tracehr_unified_next import fixed_to_buffers
from scripts.tracehr_execution_refine_common import Data, protocol as original_protocol, stopping
CURRENT_SEED=42
def protocol(variant):
    result=original_protocol(variant);result['seed']=CURRENT_SEED
    return result
from scripts.tracehr_backbone_refine_runtime import Optimizer
from scripts.train_tracehr_native_baseline import native_fit, verify_reference
from scripts.tracehr_native_transfer_common import old_baseline
from scripts.tracehr_multiview_common import atomic, read, sha, checked_output
from scripts.tracehr_oof_common import all_scores
from scripts.train_tracehr_unified import seed_all, rng_state, restore_rng


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def tensor_drift(a, b):
    assert a.keys() == b.keys()
    return max((float((a[n]-b[n]).abs().max()) for n in a), default=0.)


class ShortRunComplete(Exception):
    pass


class NativeTrace:
    def __init__(self, args, out, data, identity):
        self.a,self.out,self.data,self.identity = args,out,data,identity
        self.steps,self.rows,self.history = 0,[],[]
        self.started=time.time(); self.rng=np.random.default_rng(args.seed)
        self.max_gradient_drift=self.max_parameter_drift=self.max_logit_drift=0.
        self.epoch_loss=0.;self.epoch_samples=0;self.epoch_started=time.time()

    def install(self):
        from pytabkit.models.training.lightning_modules import TabNNModule
        from pytabkit.models.data import data as data_module
        self.cls,self.dm=TabNNModule,data_module
        self.old_step,self.old_val=TabNNModule.training_step,TabNNModule.on_validation_epoch_end
        self.old_iter,self.old_rand=data_module.ParallelDictDataLoader.__iter__,data_module.batch_randperm
        trace=self

        def rand(*a,**kw):
            result=trace.old_rand(*a,**kw); trace.permutation=result
            return result

        def iterator(loader):
            for i,batch in enumerate(trace.old_iter(loader)):
                if loader.shuffle:
                    begin,end=loader.sep_idxs[i:i+2]
                    trace.indices=loader.idxs.gather(1,trace.permutation[:,begin:end])
                yield batch

        def step(module,batch,batch_idx):
            return trace.step(module,batch,batch_idx)

        def validation(module):
            return trace.validation(module)

        data_module.batch_randperm=rand
        data_module.ParallelDictDataLoader.__iter__=iterator
        TabNNModule.training_step=step;TabNNModule.on_validation_epoch_end=validation

    def uninstall(self):
        self.cls.training_step,self.cls.on_validation_epoch_end=self.old_step,self.old_val
        self.dm.ParallelDictDataLoader.__iter__,self.dm.batch_randperm=self.old_iter,self.old_rand

    def make_shadow(self,module):
        # All state is copied immediately before the FIRST update, never a learned checkpoint.
        net,static,hp=copy.deepcopy((module.model,module.creator.static_model,module.hp_manager))
        for n,p in net.named_parameters():
            p.context=copy.deepcopy(dict(module.model.named_parameters())[n].context)
        fixed_to_buffers(net);fixed_to_buffers(static)
        shadow=FullNative.__new__(FullNative);nn.Module.__init__(shadow)
        shadow.net,shadow.static,shadow.hp=net,static,hp
        shadow.members=1;shadow.register_parameter('input_members',None)
        shadow.info=dict(config=copy.deepcopy(module.config))
        self.shadow=shadow
        self.shadow_opt=Optimizer(shadow,protocol('F0'))
        assert native_parameter_hash(shadow.net)==native_parameter_hash(module.model)

    def step(self,module,batch,batch_idx):
        if self.steps==0:
            reference=self.a.project/'runs/tracehr_native_transfer_20261014/mimic/v2/screen/jobs'/f'B0_t{self.a.endpoint}_s42/initialization.json'
            initial=native_parameter_hash(module.model)
            assert initial==self.a.fresh_initial_hash
            if self.a.seed==42:
                assert initial==read(reference)['initial_trainable_sha256']
            else:
                assert initial!=read(reference)['initial_trainable_sha256']
            if self.a.pilot_record:
                pilot=read(self.a.pilot_record/'completion.json')
                assert pilot['identity']['initial_weights_sha256']==initial
            atomic(self.out/'seed_audit.json',dict(complete=True,seed=self.a.seed,
                initial_weights_sha256=initial,fresh_initialization_hash=self.a.fresh_initial_hash,
                internal_seeds=self.a.fresh_internal_seeds,
                initial_parameters_match_same_seed_fresh_native=True,
                seed42_historical_comparison='matched' if self.a.seed==42 else 'different as required',
                capture_optimizer_updates=0,rng_restored_before_fit=True))
            self.identity['initial_weights_sha256']=initial
            self.identity['common_weights_sha256']=initial
            self.identity['parameters']=sum(p.numel() for p in module.model.parameters() if p.requires_grad)
            self.identity['members']=1
            atomic(self.out/'run_manifest.json',self.identity)
            atomic(self.out/'execution_audit.json',dict(initialization_matches_same_seed_native=True,
                  reference=str(reference),reference_sha256=sha(reference),initial_weights_sha256=initial,
                  input_dim=self.data.input_dim,train_rows=len(self.data.raw['train']),
                  batch_size=module.train_dl.batch_size,drop_last=module.train_dl.drop_last,
                  batches_per_epoch=len(module.train_dl),samples_per_epoch=module.train_dl.get_num_iterated_samples(),
                  training_sampler='actual official Torch-device batch_randperm, native RNG stream',
                  entry='original sklearn fit -> Lightning trainer; observation hooks only',
                  initialization_from_trained_checkpoint=False,learned_parameters_all_trainable=True))
            if self.a.action=='audit':self.make_shadow(module)
        audit=self.a.action=='audit'
        if audit:
            before_rng=rng_state(self.rng);epoch_float=module.progress.epoch_float
            native_grads={};native_logits={}
            old_backward=module.manual_backward
            def backward(loss,*a,**kw):
                old_backward(loss,*a,**kw)
                for n,p in module.model.named_parameters():
                    if p.requires_grad:
                        assert p.grad is not None and torch.isfinite(p.grad).all(),n
                        native_grads[n]=p.grad.detach().clone()
            module.manual_backward=backward
            handle=module.model.register_forward_hook(lambda m,i,o:native_logits.update(value=o['x_cont'].detach().clone()))
        try:
            loss=self.old_step(module,batch,batch_idx)
        finally:
            if audit:
                module.manual_backward=old_backward;handle.remove()
        self.steps+=1
        self.epoch_loss+=float(loss.detach())*batch['y'].shape[-2]
        self.epoch_samples+=batch['y'].shape[-2]
        if audit:
            after_rng=rng_state(self.rng)
            assert self.indices.min()>=0 and self.indices.max()<len(self.data.raw['train'])
            raw=torch.as_tensor(np.array(self.data.raw['train'][self.indices[0].cpu().numpy()],copy=True),device='cuda')
            self.shadow.train();self.shadow_opt.zero_grad();self.shadow_opt.schedule(epoch_float)
            restore_rng(before_rng,self.rng)
            # Static preprocessing must match the actual original dataloader, including fixed buffers.
            with torch.no_grad():
                transformed=self.shadow.static(dict(x_cont=raw[None],x_cat=batch['x_cat']))['x_cont']
            preprocessing_drift=float((transformed-batch['x_cont']).abs().max())
            logits=self.shadow(raw).transpose(0,1)
            shadow_loss=module.criterion(logits,batch['y']).sum()
            shadow_loss.backward()
            grads={n:p.grad for n,p in self.shadow.net.named_parameters()}
            grad_drift=tensor_drift(native_grads,grads)
            self.shadow_opt.step()
            param_drift=tensor_drift({n:p for n,p in module.model.named_parameters() if p.requires_grad},
                                     dict(self.shadow.net.named_parameters()))
            logit_drift=float((logits-native_logits['value']).abs().max())
            loss_drift=abs(float(shadow_loss.detach())-float(loss.detach()))
            self.max_gradient_drift=max(self.max_gradient_drift,grad_drift)
            self.max_parameter_drift=max(self.max_parameter_drift,param_drift)
            self.max_logit_drift=max(self.max_logit_drift,logit_drift)
            row=dict(update=self.steps,batch_indices_sha256=digest(self.indices),epoch_float=epoch_float,
                     cuda_rng_before_sha256=digest(before_rng['cuda'][0]),
                     preprocessing_max_drift=preprocessing_drift,logit_max_drift=logit_drift,
                     loss_drift=loss_drift,gradient_max_drift=grad_drift,parameter_max_drift=param_drift,
                     all_learnable_parameters_connected=True)
            self.rows.append(row)
            restore_rng(after_rng,self.rng)
            atomic(self.out/'continuous_update_trace.json',self.rows)
            assert max(preprocessing_drift,grad_drift,param_drift,logit_drift,loss_drift)<=1e-6,row
        if self.steps==1 or self.steps%20==0:
            atomic(self.out/'progress.json',dict(status='continuous_native_audit' if audit else 'native_training',
                  epoch=module.progress.epoch+1,step=batch_idx+1,updates=self.steps,
                  best_ap=max((r['dev_ap'] for r in self.history),default=None),
                  elapsed_seconds=time.time()-self.started))
        if self.a.action in ('audit','probe') and self.steps>=20:
            raise ShortRunComplete()
        return loss

    def validation(self,module):
        pred=module._postprocess_ens_pred(torch.cat(module.val_preds,dim=-2))[0].softmax(-1)[:,1].detach().cpu().numpy()[:,None]
        labels=self.data.targets['dev']['y'][:,self.a.endpoint:self.a.endpoint+1]
        scores=all_scores(labels,pred)
        self.old_val(module)
        epoch=module.progress.epoch
        row=dict(epoch=epoch,dev_ap=scores['mean_ap'],metrics=scores,
                 train_loss=self.epoch_loss/max(1,self.epoch_samples),epoch_seconds=time.time()-self.epoch_started)
        self.history.append(row)
        best=max(r['dev_ap'] for r in self.history)
        if scores['mean_ap']==best:
            np.savez_compressed(self.out/'selected_native_dev.npz',probs=pred,labels=labels,ids=self.data.targets['dev']['ids'],epoch=epoch)
        wrapper=module.optimizers().optimizer
        tmp=self.out/'latest.pt.writing'
        torch.save(dict(model=module.model.state_dict(),static=module.creator.static_model.state_dict(),
                        optimizer=wrapper.opt.state_dict(),progress=vars(module.progress),
                        rng=rng_state(self.rng),history=self.history,identity=self.identity),tmp)
        os.replace(tmp,self.out/'latest.pt')
        atomic(self.out/'history.json',self.history)
        stop=stopping(self.history,protocol('F0'))
        atomic(self.out/'progress.json',dict(status='epoch_complete',epoch=epoch,dev_ap=scores['mean_ap'],
              best_ap=best,best_epoch=max(self.history,key=lambda r:r['dev_ap'])['epoch'],stopping=stop,
              epoch_seconds=row['epoch_seconds'],elapsed_seconds=time.time()-self.started))
        self.epoch_loss=0.;self.epoch_samples=0;self.epoch_started=time.time()
        if self.a.action=='smoke' or stop['stop']:module.trainer.should_stop=True


def main():
    parser=argparse.ArgumentParser()
    for name in ('project','output','preparation','pilot-record','start-barrier'):
        parser.add_argument('--'+name,type=Path,required=name in ('project','output','preparation'))
    parser.add_argument('--database',choices=['mimic'],required=True)
    parser.add_argument('--variant',choices=['B0'],default='B0')
    parser.add_argument('--action',choices=['audit','probe','smoke','train'],required=True)
    parser.add_argument('--endpoint',type=int,choices=[0,1],required=True)
    parser.add_argument('--seed',type=int,choices=[43,44],required=True)
    parser.add_argument('--expected-gib',type=float,required=True)
    a=parser.parse_args();a.project=a.project.resolve(strict=True)
    global CURRENT_SEED
    CURRENT_SEED=a.seed
    out=checked_output(a.project,a.output);assert not out.exists()
    assert shutil.disk_usage(a.project).free>(a.expected_gib+10)*2**30
    out.mkdir(parents=True)
    for k in ('TMPDIR','XDG_CACHE_HOME','TORCH_HOME','MPLCONFIGDIR'):
        checked_output(a.project,os.environ[k]).mkdir(parents=True,exist_ok=True)
    atomic(out/'storage.json',dict(project=str(a.project),code=str(ROOT.resolve()),output=str(out),
          cache=os.environ['XDG_CACHE_HOME'],temp=os.environ['TMPDIR'],expected_gib=a.expected_gib))
    atomic(out/'progress.json',dict(status='loading_verified_inputs'))
    data=Data(a.project,a.database,a.preparation)
    torch.set_num_threads(4);seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
    identity=dict(database=a.database,variant='B0',endpoint=a.endpoint,seed=a.seed,protocol=protocol('F0'),
                  input_manifest_sha256=sha(a.preparation/'completion.json'),source_manifest_sha256=sha(ROOT/'source_manifest.json'),
                  from_scratch=True,test_loaded=False,frozen_learnable_parameters=0,input_dim=data.input_dim)
    if a.pilot_record:
        pilot=read(a.pilot_record/'completion.json')
        for key in ('database','variant','endpoint','seed','source_manifest_sha256','input_manifest_sha256'):
            assert pilot['identity'][key]==identity[key]
        assert pilot['complete']
    if a.start_barrier:
        atomic(out/'progress.json',dict(status='waiting_coexistence_barrier'))
        deadline=time.time()+1200
        while not a.start_barrier.exists():
            assert time.time()<deadline
            time.sleep(.5)
    started=time.time();atomic(out/'training_started.json',dict(started_at=started,pid=os.getpid()))
    atomic(out/'progress.json',dict(status='randomly_initializing_native'))
    # Independent same-seed native initialization, before any optimizer step.
    # Preserve every RNG stream and discard this untrained object before fitting.
    capture_rng=np.random.default_rng(a.seed)
    saved_rng=rng_state(capture_rng)
    captured=FullNative(data.raw,{s:data.targets[s]['y'] for s in ('train','dev')},a.endpoint,out/'fresh_native_init_tmp',a.seed)
    a.fresh_initial_hash=captured.info['native_initial_trainable_sha256']
    a.fresh_internal_seeds=captured.info['internal_seeds']
    atomic(out/'fresh_initialization.json',captured.info)
    del captured;gc.collect();torch.cuda.empty_cache();restore_rng(saved_rng,capture_rng)
    trace=NativeTrace(a,out,data,identity);trace.install()
    target={s:data.targets[s]['y'] for s in ('train','dev')}
    model=None
    try:
        model,pred,_,best_epoch,_=native_fit(data.raw,target,out,a.endpoint,a.seed,256,'cuda:0',None)
    except ShortRunComplete:
        assert a.action in ('audit','probe') and trace.steps==20
    finally:
        trace.uninstall()
    if a.action in ('audit','probe'):
        audit=a.action=='audit'
        if audit:assert len(trace.rows)==20
        result=dict(complete=True,pilot=True,audit=audit,identity=identity,from_scratch=True,test_loaded=False,
                    frozen_learnable_parameters=0,updates=trace.steps,
                    max_gradient_drift=trace.max_gradient_drift,max_parameter_drift=trace.max_parameter_drift,
                    max_logit_drift=trace.max_logit_drift,continuous_native_entry_parity=audit,
                    effective_batch=256,selected_microbatch=256,evaluation_microbatch=256,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                    checkpoint_bytes=identity['parameters']*4*3+4*2**20,
                    seconds=time.time()-started,finished_at=time.time())
    else:
        assert model is not None
        atomic(out/'history.json',trace.history)
        model.to('cpu');joblib.dump(model,out/'best.joblib');del model;gc.collect();torch.cuda.empty_cache()
        loaded=joblib.load(out/'best.joblib');loaded.to('cuda:0');replay=loaded.predict_proba(data.raw['dev'])[:,1]
        drift=float(np.max(np.abs(pred-replay)));assert drift<=1e-5
        labels=target['dev'][:,a.endpoint:a.endpoint+1];scores=all_scores(labels,pred[:,None])
        assert abs(scores['mean_ap']-max(r['dev_ap'] for r in trace.history))<1e-7
        np.savez_compressed(out/'predictions_dev.npz',probs=pred[:,None],labels=labels,ids=data.targets['dev']['ids'],endpoint=a.endpoint)
        atomic(out/'metrics.json',scores)
        reproduction=verify_reference(data,old_baseline(a.project,'mimic','B0'),a.endpoint,None)
        reproduction.update(scope='seed42 background and input identity only; new seed is not expected to reproduce its AP',
                            seed=a.seed)
        atomic(out/'reproduction.json',reproduction)
        result=dict(complete=True,pilot=a.action=='smoke',**identity,epochs=len(trace.history),best_epoch=best_epoch,
                    dev_ap=scores['mean_ap'],metrics=scores,seconds=time.time()-started,finished_at=time.time(),
                    stopping_status=stopping(trace.history,protocol('F0')),reproduction=reproduction,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                    checkpoint_file='best.joblib',checkpoint_sha256=sha(out/'best.joblib'),
                    predictions_sha256=sha(out/'predictions_dev.npz'),
                    checkpoint_replay=dict(complete=True,max_probability_drift=drift))
    atomic(out/'completion.json',result)
    atomic(out/'progress.json',dict(status='complete',updates=trace.steps,dev_ap=result.get('dev_ap'),
                                   best_epoch=result.get('best_epoch'),epochs=result.get('epochs')))


if __name__=='__main__':main()
