"""Reuse the audited fitting/testing lifecycle with the fixed STraTS recipe."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import core_seed_tabular_entry as worker
from scripts.cem_replacement_common import expected,protocol,make_data,atomic,sha
from models.cem_replacement_strats import make_model,AuthorAdamW

worker.expected=expected;worker.protocol=protocol;worker.make_data=make_data;worker.make_model=make_model
worker.optimizer=lambda m,s:AuthorAdamW(m.parameters(),s['lr'])

def step(model,opt,data,ix,endpoint,micro):
    model.train();opt.zero_grad(set_to_none=True);total=0.
    yall=data.targets['train']['y'][:,endpoint];weight=torch.tensor(float((1-yall).sum()/yall.sum()),device='cuda')
    for start in range(0,len(ix),micro):
        sub=ix[start:start+micro];z=worker.logits(model,data.batch('train',sub,'cuda'))
        y=torch.as_tensor(yall[sub],device='cuda',dtype=torch.float32)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y,pos_weight=weight)*len(sub)/len(ix)
        assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(),.3,error_if_nonfinite=True)
    opt.step();return total
worker.step=step

original_initialize=worker.initialize
def initialize(a,out,data):
    # The generic initialization identity reads this key, without modifying inputs.
    model,identity=original_initialize(a,out,data)
    return model,identity
worker.initialize=initialize

def probe(a,out,data):
    m,identity=initialize(a,out,data);spec=identity['protocol'];opt=AuthorAdamW(m.parameters());rng=np.random.default_rng(a.seed)
    initial={n:p.detach().cpu().clone() for n,p in m.named_parameters()};tick=time.time();times=[]
    lengths=np.diff(data.raw['train']['offsets']);largest=np.argsort(lengths)[-16:]
    for i in range(20):
        ix=largest if i==0 else rng.permutation(len(lengths))[:16];begin=time.perf_counter()
        loss=step(m,opt,data,ix,a.endpoint,1);torch.cuda.synchronize();times.append(time.perf_counter()-begin)
        atomic(out/'progress.json',dict(status='training_probe',updates=i+1,loss=loss))
    moved={n:float((p.detach().cpu()-initial[n]).abs().max()) for n,p in m.named_parameters()};del initial
    unchanged=[n for n,x in moved.items() if x==0]
    assert unchanged==['demo_emb.0.weight'],unchanged
    atomic(out/'gradient_check.json',dict(all_connected=True,relative_update=moved,
        analytically_cancelled_parameters=unchanged,reason='No additional demographic inputs; the fixed zero demo coordinate has zero gradient.'))
    torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=worker.rng_state(rng)),out/'representative_checkpoint.pt')
    ix=rng.permutation(len(lengths))[:16];step(m,opt,data,ix,a.endpoint,1)
    reference=worker.digest(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=worker.rng_state(rng)))
    ck=torch.load(out/'representative_checkpoint.pt',map_location='cpu',weights_only=False)
    m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);worker.restore_rng(ck['rng'],rng);del ck
    ix2=rng.permutation(len(lengths))[:16];assert np.array_equal(ix,ix2);step(m,opt,data,ix2,a.endpoint,1)
    assert worker.digest(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=worker.rng_state(rng)))==reference
    before=worker.digest(m.state_dict());m.eval();ev=time.time()
    with torch.no_grad():
        for split in ('train','dev'):
            selected=np.argsort(np.diff(data.raw[split]['offsets']))[-16:]
            for i in selected:assert torch.isfinite(m(*data.batch(split,[i],'cuda'))).all()
    assert before==worker.digest(m.state_dict())
    atomic(out/'path_audit.json',dict(complete=True,continuous_updates=20,resume_exact=True,
        full_length_max=int(lengths.max()),longest_train_and_validation_evaluated=True,
        median_step_seconds=float(np.median(times[1:])),longest_batch_seconds=times[0],
        estimated_epoch_train_seconds=float(np.median(times[1:])*np.ceil(len(lengths)/16)),
        evaluation_preserves_state=True,analytically_cancelled_parameters=unchanged,test_loaded=False))
    atomic(out/'completion.json',dict(complete=True,pilot=True,identity=identity,from_scratch=True,test_loaded=False,
        checkpoint_bytes=(out/'representative_checkpoint.pt').stat().st_size,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,seconds=time.time()-tick))
    atomic(out/'progress.json',dict(status='complete'))
worker.probe=probe

if __name__=='__main__':worker.main()
