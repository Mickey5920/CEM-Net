"""Guarded control of the authorized 30 Medformer fits and 30 fixed tests."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.cem_medformer_common import RUN,PARENT,SEEDS,recipe,read,save,sha
from scripts.launch_ema_cross import HOSTS,ssh

PILOT='cem_med_transformer_pilot_20261103'


def build(a,out):
    parent=ROOT/'runs'/PARENT/f'local_{a.database}_v3'
    files=read(parent/'source_bundle.json');inherited=read(parent/'source_manifest.json')
    c=read(parent/'build-receipt.json')['contract']
    cache=c['hourly_cache']
    info=ssh(a.host,"import json,hashlib,shutil;from pathlib import Path;p=Path("+repr(HOSTS[a.host][1])+");f=p/"+repr(cache)+";r={s:json.loads((f/(s+'_completion.json')).read_text()) for s in ('train','dev','test') if (f/(s+'_completion.json')).exists()};assert {'train','dev'}<=set(r) and all(z['complete'] for z in r.values());print(json.dumps(dict(receipts={s:hashlib.sha256((f/(s+'_completion.json')).read_bytes()).hexdigest() for s in r},free_gib=shutil.disk_usage(p).free/2**30,cache_gib=sum(z['bytes'] for z in r.values())/2**30)))",120)
    provenance=read(ROOT/'work'/PILOT/'vendor/provenance.json')
    c.update(run=RUN,snapshot='v1',stage='medformer',expected_gib=a.expected_gib,
        variants=['Medformer'],deferred_variants=['MedSpaformer'],seeds=list(SEEDS),formal_fits=10,
        recipes={'Medformer':recipe()},matrix=[dict(variant='Medformer',seed=s,endpoint=e) for s in SEEDS for e in (0,1)],
        input_receipts=info['receipts'],shared_input_cache_gib=info['cache_gib'],
        parent_source_manifest_sha256=sha(parent/'source_manifest.json'),
        selection='SWA development AP only; lock all 10 fitted checkpoints before test',
        new_training_authorization='Medformer batch 128; three databases x two endpoints x five seeds = 30 fresh fits and 30 fixed tests',
        inference=dict(candidate='original five-seed CEM-Net',family_size=6,active_contrasts=6,
            old_families_unchanged=True,split_separate=True,
            bootstrap='20000 identical patient-cluster draws; reuse all existing CEM AP arrays; calculate only Medformer arrays',
            no_new_calibration=True,secondary_selection_unchanged=True),
        upstream={'medformer':provenance['sources']['medformer']},
        adaptation='Same read-only 24h/224-channel CNN inputs and learned event adapter; official full classification graph, fixed 2/4/8h patches, no waveform augmentation; AP replaces author F1 checkpoint selection')
    for key in ('preflight_parent_manifest','preflight_parent_output','adaptation_authorization'):
        c.pop(key,None)
    additions=['scripts/cem_medformer_common.py','models/cem_medformer.py','scripts/train_cem_medformer.py','scripts/cem_medformer_queue.py']
    for name in additions:
        assert name not in files
        b=(ROOT/name).read_bytes();compile(b,name,'exec');files[name]=base64.b64encode(b).decode()
    for name,h in provenance['files'].items():
        if not name.startswith('medformer/'):continue
        path=ROOT/'work'/PILOT/'vendor'/name;assert sha(path)==h
        files[path.relative_to(ROOT).as_posix()]=base64.b64encode(path.read_bytes()).decode()
    files[f'work/{RUN}/config.json']=base64.b64encode(json.dumps(c,indent=2).encode()).decode()
    manifest={n:hashlib.sha256(base64.b64decode(v)).hexdigest() for n,v in files.items()}
    assert all(manifest[n]==h for n,h in inherited.items())
    assert not (out/'deploy-receipt.json').exists(),'Never rebuild a deployed snapshot'
    save(out/'source_bundle.json',files);save(out/'source_manifest.json',manifest)
    result=dict(complete=True,source_manifest_sha256=sha(out/'source_manifest.json'),files=len(files),
        inherited_files=len(inherited),contract=c,storage=info)
    save(out/'build-receipt.json',result)
    return dict(complete=True,formal_fits=10,source_manifest_sha256=result['source_manifest_sha256'],files=len(files))


REMOTE=r'''
import base64,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
d=PAYLOAD;p=Path(d['project']).resolve(strict=True);source=p/d['source'];out=p/d['output'];host=p/'runs'/d['run']/('host_'+d['host'])
def read(f):return json.loads(Path(f).read_text())
def sha(f):
 h=hashlib.sha256()
 with Path(f).open('rb') as s:
  for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(f,x):
 f.parent.mkdir(parents=True,exist_ok=True);q=f.with_name(f.name+'.writing');q.write_text(json.dumps(x,indent=2));os.replace(q,f)
def env():
 deps=p/'work/tracehr_tabular_20261004/deps';temp=p/'work'/d['run']/'tmp';cache=p/'cache'/d['run']
 for x in (temp,cache):x.mkdir(parents=True,exist_ok=True);assert x.resolve().is_relative_to(p)
 paths=[source,deps,*deps.glob('*.whl'),p/'work/cem_evidence_completion_20261025/test_runtime_v1/deps']
 return dict(os.environ,TRACEHR_PROJECT=str(p),TRACEHR_DATABASE=d['database'],TRACEHR_TABULAR_DEPS=str(deps),
  CEM_COMPLETION_STAGE='medformer',PYTHONPATH=os.pathsep.join(map(str,paths)),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
  PYTHONNOUSERSITE='1',NUMPY_MADVISE_HUGEPAGE='0',TMPDIR=str(temp),TEMP=str(temp),TMP=str(temp),
  XDG_CACHE_HOME=str(cache/'runtime'),TORCH_HOME=str(cache/'torch'),MPLCONFIGDIR=str(cache/'matplotlib'),
  OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
if d['mode']=='deploy':
 assert not source.exists() and not (out.parent/'verification.json').exists()
 assert shutil.disk_usage(p).free>(10+d['expected_gib'])*2**30
 for rel in (d['source'],d['output'],'cache/'+d['run'],'work/'+d['run']+'/tmp'):
  path=p/rel;path.mkdir(parents=True,exist_ok=True);assert path.resolve().is_relative_to(p)
 for n,b in d['files'].items():
  f=source/n;assert f.resolve().is_relative_to(source.resolve());f.parent.mkdir(parents=True,exist_ok=True)
  raw=base64.b64decode(b)
  if n.endswith('.py'):compile(raw,n,'exec')
  assert not f.exists();f.write_bytes(raw)
 (source/'source_manifest.json').write_bytes(base64.b64decode(d['manifest_bytes']))
 assert sha(source/'source_manifest.json')==d['source_sha256']
 result=dict(complete=True,files=len(d['files']),source_manifest_sha256=d['source_sha256'])
elif d['mode']=='verify':
 assert not (out.parent/'verification.json').exists()
 assert sha(source/'source_manifest.json')==d['source_sha256']
 for n,h in read(source/'source_manifest.json').items():assert sha(source/n)==h,n
 os.environ.update(env());os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=os.environ['PYTHONPATH'].split(os.pathsep)
 import numpy as np,torch
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 from scripts.cem_medformer_common import Data,expected,recipe,SEEDS
 from models.cem_medformer import make_model
 from scripts.mimiciii_ema_runtime import digest
 from scripts.cem_sequence_comparator_common import input_checks
 from scripts.cem_medformer_queue import Policy
 from scripts.cem_evidence_host_queue import policy_checks
 checks=input_checks()+policy_checks();cfg=expected();data=Data(p)
 old=p/'work/cem_sequence_comparators_20261102'/d['database']/'v3'
 assert sha(old/'source_manifest.json')==cfg['parent_source_manifest_sha256']
 for split in cfg['input_receipts']:
  receipt=data.folder/(split+'_completion.json');assert sha(receipt)==cfg['input_receipts'][split]
  r=read(receipt)
  # Hash existing artifacts once; no reconstruction or learned input cache.
  for name,h in r['files'].items():
   if name!=split+'_ft.npy':assert sha(data.folder/name)==h,name
 assert not np.intersect1d(data.targets['train']['ids'][:,0],data.targets['dev']['ids'][:,0]).size
 seeds={}
 for seed in SEEDS:
  m=make_model(seed,data);first=digest(m.state_dict());other=make_model(seed,data)
  assert digest(other.state_dict())==first;seeds[str(seed)]=first;del other,m
 assert len(set(seeds.values()))==5
 m=make_model(48,data);batch=data.batch('train',np.arange(4),'cpu');m.eval()
 logits=m(*batch);direct=m.backbone(m.adapter(*batch),None,None,None)
 torch.testing.assert_close(logits,direct,atol=0,rtol=0);assert logits.shape==(4,2)
 loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1,0,1]));loss.backward()
 assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in m.parameters())
 # Empty event bags remain valid without deleting eligible stays.
 empty=(torch.zeros(2,24,96),torch.empty(0,dtype=torch.long),torch.empty(0),torch.zeros(49,dtype=torch.long))
 assert torch.isfinite(m(*empty)).all()
 policy=Policy(d['host']);assert 3 not in policy.candidates([dict(index=3)],0) if d['host']=='eicu' else True
 result=dict(complete=True,source_manifest_sha256=d['source_sha256'],formal_fits=10,test_evaluations=10,
  train_rows=len(data.targets['train']['y']),dev_rows=len(data.targets['dev']['y']),input_dim=data.input_dim,
  num_types=data.num_types,parameters=sum(x.numel() for x in m.parameters()),initialization_hashes=seeds,
  cpu_checks=checks+['author_graph_forward','all_trainable_gradients','empty_history_finite','all_five_seeds_rebuild_exact','shared_inputs_hashes'],
  existing_CNN_files_unchanged=True,test_loaded=False,verified_at=time.time())
 save(out.parent/'verification.json',result)
elif d['mode']=='register':
 v=read(out.parent/'verification.json');assert v['complete'] and v['source_manifest_sha256']==d['source_sha256']
 c=read(source/('work/'+d['run']+'/config.json'));file=host/'ready'/(d['database']+'_medformer.json');assert not file.exists()
 result=dict(database=d['database'],stage='medformer',stage_order=1,source=str(source),output=str(out),
  source_manifest_sha256=d['source_sha256'],matrix=c['matrix'],variants=c['variants'],
  entries=dict(tabular='scripts/train_cem_medformer.py'),expected_gib=d['expected_gib'],
  retention_multiplier=3,min_available_ram_gib=12,cpu_min_available_ram_gib=16,probe_gpu_mib=4096,
  probe_requirement_basis='Medformer FP32 batch 128, official full graph and SWA; no performance selection')
 save(file,result)
elif d['mode']=='start':
 assert not (host/'start_attempt.json').exists() and not (host/'state.json').exists();host.mkdir(parents=True,exist_ok=True)
 attempt=host/'start_attempt.json';save(attempt,dict(command=[sys.executable,'-B',str(source/'scripts/cem_medformer_queue.py'),'--project',str(p),'--host',d['host']],
  cwd=str(source),receipt=str(host/'start_exit.json'),job_id='cem_medformer_'+d['host'],source_manifest_sha256=d['source_sha256']))
 with (host/'queue.log').open('x') as log:
  child=subprocess.Popen([sys.executable,'-B',str(source/'scripts/tracehr_recovery_attempt.py'),'--spec',str(attempt)],cwd=source,env=env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 result=dict(complete=True,wrapper_pid=child.pid,formal_training_launched=False);save(host/'launch.json',result)
elif d['mode']=='release':
 s=read(out/'state.json');assert s['status']=='awaiting_path_review';assert not (out/'formal_release.json').exists()
 assert s['counts']['pending']==10 and s['projected_artifact_gib']<s['expected_gib']
 audits={};reports=[]
 for j in s['jobs'].values():
  if j['action']!='probe':continue
  assert j['status']=='completed' and read(j['exit_receipt'])['returncode']==0
  folder=Path(j['output']);r=read(folder/'path_audit.json');grad=read(folder/'gradient_check.json')
  assert r['complete'] and r['continuous_updates']==20 and r['resume_exact'] and r['resume_raw_optimizer_swa_rng_exact']
  assert r['swa_arithmetic_average_verified'] and r['effective_batch']==r['physical_batch']==128 and not r['test_loaded']
  assert grad['complete'] and grad['all_trainable_parameters_updated']
  audits[j['id']]=sha(folder/'path_audit.json');reports.append(dict(id=j['id'],audit=r,resources=j['result']))
 assert len(audits)==2
 result=dict(complete=True,source_manifest_sha256=d['source_sha256'],audit_sha256=audits,
  projected_artifact_gib=s['projected_artifact_gib'],reports=reports,released_at=time.time())
 save(out/'formal_release.json',result)
else:
 result=read(host/'state.json') if (host/'state.json').exists() else dict(status='not_started',groups={})
 active=Path(result.get('active_scheduler_receipt',host/'start_exit.json'))
 result['exit_receipt']=read(active) if active.exists() else None;result['active_scheduler_receipt']=str(active)
 if result.get('pid'):result['queue_alive']=Path('/proc/'+str(result['pid'])).exists()
 for g in result['groups'].values():
  for j in g['jobs'].values():
   folder=Path(j['output'])
   if (folder/'progress.json').exists() and j['status'] in ('running','failed'):j['progress']=read(folder/'progress.json')
   if j['status']=='failed':j['log_tail']=Path(j['log']).read_text(errors='replace')[-7000:]
 result['observed_at']=time.time()
print(json.dumps(result))
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--database',choices=('mimiciii','mimic','eicu'),required=True)
    parser.add_argument('--mode',choices=('build','deploy','verify','register','start','release','status'),required=True)
    parser.add_argument('--expected-gib',type=float,default=20)
    a=parser.parse_args();a.host='eicu' if a.database=='eicu' else 'mimic'
    out=Path(os.environ['PROJECT_RUN_DIR']).resolve(strict=True)
    assert out==ROOT/'runs'/RUN/f'local_{a.database}_v1'
    if a.mode=='build':result=build(a,out)
    else:
        d=dict(mode=a.mode,project=HOSTS[a.host][1],run=RUN,host=a.host,database=a.database,
            source=f'work/{RUN}/{a.database}/v1',output=f'runs/{RUN}/{a.database}/v1/screen',expected_gib=a.expected_gib)
        if a.mode!='status':d['source_sha256']=read(out/'build-receipt.json')['source_manifest_sha256']
        if a.mode=='deploy':d.update(files=read(out/'source_bundle.json'),manifest_bytes=base64.b64encode((out/'source_manifest.json').read_bytes()).decode())
        result=ssh(a.host,REMOTE.replace('PAYLOAD',repr(d)),timeout=650 if a.mode=='verify' else 180)
        save(out/(a.mode+'-receipt.json'),result)
    print(json.dumps(result))


if __name__=='__main__':
    import contextlib,traceback
    out=Path(os.environ['PROJECT_RUN_DIR'])
    with (out/'entry.log').open('a',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:main()
        except BaseException as exc:save(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
