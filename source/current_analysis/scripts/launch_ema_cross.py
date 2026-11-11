"""Storage-guarded, immutable launch of the authorized two-database batch."""
import argparse,base64,hashlib,json,os,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.evaluate_mimiciii_fixed_test import sha,read,save
from scripts.launch_mimiciii_pm_mini import CONTROL as BASE_CONTROL
RUN='tracehr_ema_cross_20261024'
PARENT='352e58aeb8015b0e8c1a6c6c6fb3163af5474288e8cde3a68cf1a836de265f3c'
ENTRIES=('scripts/ema_cross_common.py','scripts/train_ema_cross.py','scripts/ema_cross_queue.py')
HOSTS=dict(mimic=('6000ada_1','/data/Luoxm/Disease-Ranking-Hawkes-Mamba-0624','/home/gem/anaconda3/envs/lxm_Disease_Prediction/bin/python'),
           eicu=('5880_1','/home/data/lxm/TRACEHR','/home/data/lxm/TRACEHR/envs/www2027/bin/python'))

def ssh(db,code,timeout=650):
    host,_,python=HOSTS[db]
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,shlex.quote(python)+' -B -'],
                     input=code,text=True,capture_output=True,timeout=timeout)
    if r.returncode:raise RuntimeError(r.stderr[-7000:]+r.stdout[-1000:])
    return json.loads(r.stdout)

INVENTORY=r'''
import json,hashlib,shutil,subprocess
from pathlib import Path
d=PAYLOAD;p=Path(d['project']);db=d['database'];refs=d['controls']
def read(f):return json.loads(Path(f).read_text())
def sha(f):
 h=hashlib.sha256()
 with Path(f).open('rb') as s:
  for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
protected={};references={}
for key,r in refs.items():
 folder=Path(r['source']);c=read(folder/'completion.json');m=read(folder/'run_manifest.json');assert c['complete']
 assert sha(folder/'predictions_dev.npz')==r['predictions_sha256']
 for f in folder.iterdir():
  if f.name in ('completion.json','run_manifest.json','predictions_dev.npz','history.json','initialization_observation.json') or f.suffix in ('.pt','.joblib'):
   protected[str(f)]=sha(f)
 if key.startswith('C5_noT_noA_noR_untied'):
  s,e=r['seed'],r['endpoint'];h=read(folder/'history.json');assert len(h)==r['epochs']==(52 if db=='mimic' else 256)
  assert m['initial_weights_sha256']==r['initial_weights_sha256'] and sha(folder/'best.pt')==r['checkpoint_sha256']
  for ep in (1,5,8,10,12,15,18,20):assert (folder/f'epoch_{ep:03d}_model.pt').is_file()
  references[f'noEvidence_s{s}_t{e}']=dict(r,output=str(folder),protocol=m['protocol'])
prep=p/'runs/tracehr_structure_20261014'/db/'g_v1/preparation';pr=read(prep/'completion.json')
raw=Path(pr['input_pointer']['path']);assert pr['complete'] and not pr['test_loaded']
for f in (prep/'completion.json',prep/'preprocessing.npz',raw/'completion.json',raw/'train_targets.npz',raw/'dev_targets.npz'):
 protected[str(f)]=sha(f)
origin=read(raw/'completion.json')
transform=p/'runs/tracehr_tabular_20261004'/db/('round1_r2' if db=='mimic' else 'round1_r3')/'runs'/f'{db}24_tabm_cfg1_full_s42/preprocessing.joblib'
assert sha(transform)==origin['identity']['recipe']['v_preprocessing'];protected[str(transform)]=sha(transform)
test=p/'runs/tracehr_pm_test_20261021'/db/'features';fm=read(test/'completion.json')
assert fm['complete'] and fm['development_feature_replay_passed'] and fm['fit_split']=='train' and fm['patient_overlap']==0
protected[str(test/'completion.json')]=sha(test/'completion.json');test_artifacts={}
for n in ('test_raw.npy','test_targets.npz'):
 h=sha(test/n);assert h==fm['artifacts'][n]['sha256'];test_artifacts[n]=h;protected[str(test/n)]=h
assert len(references)==6
free=shutil.disk_usage(p).free/2**30;assert free>d['expected_gib']+10
print(json.dumps(dict(complete=True,references=references,protected=protected,input_preparation=str(prep.relative_to(p)),
 raw_inputs=str(raw),input_dim=pr['input_dim'],train_rows=pr['train_rows'],dev_rows=pr['dev_rows'],
 tabm_preprocessing=str(transform),test_inputs=str(test.relative_to(p)),test_artifacts=test_artifacts,
 free_gib=free,gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.total','--format=csv,noheader,nounits'],text=True))))
'''

VERIFY=r'''
elif mode=='verify':
 os.environ.update(env());sys.path[:0]=os.environ['PYTHONPATH'].split(os.pathsep);sys.modules['lightning']=None
 assert sha(source/'source_manifest.json')==d['source_sha256']
 for n,h in read(source/'source_manifest.json').items():assert sha(source/n)==h,n
 from scripts.ema_cross_common import expected,protocol,matrix,budget,Data,VARIANT,bind_ema_runtime
 c=expected()
 for path,h in c['protected'].items():assert sha(path)==h,path
 assert len(matrix())==12 and len({(r['variant'],r['seed'],r['endpoint']) for r in matrix()})==12
 for row in matrix():
  if row['variant']!=VARIANT:continue
  ref=c['references'][f"noEvidence_s{row['seed']}_t{row['endpoint']}"]
  assert protocol(VARIANT,row['seed'])==ref['protocol']
  assert len(read(Path(ref['output'])/'history.json'))==budget(row['seed'],row['endpoint'])
 data=Data(p,d['database'],p/c['input_preparation'])
 import torch,tabm
 from tabm import TabM
 torch.set_num_threads(4)
 tests=bind_ema_runtime().cpu_checks()[:3]
 mini=TabM.make(n_num_features=5,d_out=1,k=32,d_block=256,n_blocks=3,dropout=.2,arch_type='tabm-mini')
 mini.eval();assert mini(torch.zeros(4,5)).shape==(4,32,1)
 assert all(x.requires_grad for x in mini.parameters())
 assert type(mini.backbone).__name__!='MLPBackboneBatchEnsemble'
 result=dict(complete=True,source_manifest_sha256=d['source_sha256'],formal_fits=12,
  cpu_checks=tests+['unchanged_inherited_source_and_controls','original_fullEMA_recipe_and_epoch_budgets','official_mini_architecture'],
  protected_files=len(c['protected']),input_dim=data.input_dim,train_rows=len(data.targets['train']['y']),
  test_loaded=False,optimizer_updates=0,expected_gib=d['expected_gib'],tabm_package=tabm.__file__,
  formal_release_requires='twelve bounded per-seed identity/update/resume pilots, resource estimate; no AP criterion')
 save(base/'verification.json',result)
elif mode=='release':
 s=read(queue/'state.json');sys.path.insert(0,str(source))
 from scripts.tracehr_recovery_queue import same_process
 assert same_process(s['pid'],s['process_identity']) and s['status']=='awaiting_path_review'
 assert s['counts']['pending']==12 and s['probe_counts']['completed']==12
 assert not (queue/'formal_release.json').exists()
 digests={};reports=[]
 for j in s['jobs'].values():
  if j['action']!='probe':continue
  assert j['status']=='completed' and read(j['exit_receipt'])['returncode']==0
  path=Path(j['output'])/'path_audit.json';a=read(path)
  assert a['complete'] and a['continuous_updates']==20
  if j['variant']=='fullEMA':
   assert a['all_raw_parameters_gradients_optimizer_rng_loss_exact'] and a['evaluation_preserves_raw_and_rng'] and a['resume_raw_optimizer_ema_rng_exact']
  else:assert a['all_parameters_updated'] and a['resume_exact']
  digests[j['id']]=sha(path)
  reports.append(dict(id=j['id'],peak_reserved_mib=j['result']['peak_reserved_mib'],seconds=j['result']['seconds']))
 assert len(reports)==12 and s['projected_artifact_gib']<d['expected_gib']
 result=dict(complete=True,source_manifest_sha256=d['source_sha256'],audit_sha256=digests,released_at=time.time(),
  formal_fits=12,projected_artifact_gib=s['projected_artifact_gib'],reports=reports)
 save(queue/'formal_release.json',result)
'''
CONTROL=BASE_CONTROL.replace("source=p/'work'/d['run']/'v1';base=p/'runs'/d['run']/'v1'",
 "source=p/'work'/d['run']/'v1';base=p/'runs'/d['run']/d['database']/'v1'")
CONTROL=CONTROL.replace('t_m3pm','t_ema_cross').replace("TRACEHR_DATABASE='mimiciii'","TRACEHR_DATABASE=d['database']")
CONTROL=CONTROL.replace('mimiciii_pm_mini_queue.py','ema_cross_queue.py')
CONTROL=CONTROL.replace("'--project',str(p),'--source'","'--project',str(p),'--database',d['database'],'--source'")
CONTROL=CONTROL.replace('formal_fits=18','formal_fits=12').replace('test_evaluations_after_all_fits=18','test_evaluations_after_all_fits=12')
CONTROL=CONTROL.replace("job_id='mimiciii_pm_mini_18'","job_id='ema_cross_'+d['database']")
start=CONTROL.index("elif mode=='verify':");end=CONTROL.index("elif mode=='start':",start)
CONTROL=CONTROL[:start]+VERIFY.strip()+'\n'+CONTROL[end:]
CONTROL=CONTROL.replace('   rows.append(row)', '''   gpath=Path(j['output'])/'gradient_latest.json'
   if gpath.exists():
    g=read(gpath);u=g['groups']['relative_update'];row['update_evidence']=dict(epoch=g['epoch'],updated_tensors=sum(v>0 for v in u.values()),total_tensors=len(u))
   if j['status']=='running':
    owner=Path(j['exit_receipt']).with_name(Path(j['exit_receipt']).stem+'_owner.json')
    if owner.exists():
     own=read(owner);pid=own['child_pid'];row['child_live']=same_process(pid,own['child_identity'])
     if row['child_live']:
      row['process']=subprocess.check_output(['ps','-p',str(pid),'-o','pid=,etimes=,pcpu=,rss=,stat=,wchan:24='],text=True).strip()
    if row.get('progress',{}).get('status')=='randomly_initializing_native':row['initialization_log_tail']=Path(j['log']).read_text(errors='replace')[-1500:]
   rows.append(row)''')
HANDOFF=r'''
elif mode=='handoff-evaluation':
 import signal
 os.environ.update(env());sys.path[:0]=os.environ['PYTHONPATH'].split(os.pathsep);sys.modules['lightning']=None
 from scripts.tracehr_recovery_queue import same_process
 assert sha(source/'source_manifest.json')==d['source_sha256']
 runtime=p/'work'/d['run']/'evaluation_runtime_v1';assert runtime.resolve().is_relative_to(p);runtime.mkdir(parents=True,exist_ok=True)
 assert not (base/'evaluation-entry-handoff.json').exists() and not (base/'scheduler_v2_attempt.json').exists()
 hashes={}
 for n,b64 in d['runtime_files'].items():
  data=base64.b64decode(b64);compile(data,n,'exec');f=runtime/n
  if f.exists():assert f.read_bytes()==data
  else:f.write_bytes(data)
  hashes[n]=sha(f)
 # Reproduce the two interface errors without touching any checkpoint or worker.
 import torch
 from types import SimpleNamespace
 from models.tracehr_noa_validation import NoAAblation
 raw=torch.zeros(2,3);obj=SimpleNamespace(main_residual_enabled=False,base=lambda x:x)
 try:NoAAblation._forward(obj,raw,False);raise AssertionError('expected dictionary interface failure')
 except (IndexError,TypeError):pass
 assert torch.equal(NoAAblation._forward(obj,{'raw':raw},False),raw)
 with torch.no_grad():
  with torch.enable_grad():
   x=torch.ones(1,requires_grad=True);assert torch.autograd.grad((x*x).sum(),x)[0].item()==2
  assert not torch.is_grad_enabled()
 save(runtime/'envelope.json',dict(complete=True,files=hashes,training_source_unchanged=d['source_sha256'],
  change='Only pending tests: enable gradients for zero-update constructor audit; wrap raw tensors as the required input dictionary'))
 for attempt in range(20):
  s=read(queue/'state.json');pid=s['pid'];ident=s['process_identity'];assert same_process(pid,ident)
  assert s['evaluation_counts']['pending']==12 and s['evaluation_counts']['running']==s['evaluation_counts']['completed']==s['evaluation_counts']['failed']==0
  os.kill(pid,signal.SIGSTOP);time.sleep(.1);s=read(queue/'state.json')
  if not any(j['status']=='launching' for j in s['jobs'].values()):break
  os.kill(pid,signal.SIGCONT);time.sleep(.3)
 else:raise RuntimeError('could not obtain a stable scheduler boundary')
 try:
  assert same_process(pid,ident) and s['pid']==pid and s['status']!='needs_attention'
  save(runtime/'predecessor-state.json',s)
  live=[dict(id=j['id'],pid=j['pid'],identity=j['process_identity'],exit_receipt=j['exit_receipt']) for j in s['jobs'].values() if j['status']=='running']
  for j in live:assert same_process(j['pid'],j['identity']) or Path(j['exit_receipt']).exists()
  os.kill(pid,signal.SIGTERM)
 finally:
  if same_process(pid,ident):os.kill(pid,signal.SIGCONT)
 for _ in range(100):
  if not same_process(pid,ident) and (base/'start_exit.json').exists():break
  time.sleep(.1)
 assert not same_process(pid,ident) and read(base/'start_exit.json')['returncode']==-signal.SIGTERM
 for j in live:assert same_process(j['pid'],j['identity']) or Path(j['exit_receipt']).exists()
 handoff=dict(complete=True,old_queue_pid=pid,old_queue_identity=ident,test_jobs_started=0,
  predecessor_state_sha256=sha(runtime/'predecessor-state.json'),training_source_sha256=d['source_sha256'],
  runtime_envelope_sha256=sha(runtime/'envelope.json'),workers_preserved=live,reason='Confirmed pending-test input/autograd interface correction',time=time.time())
 save(base/'evaluation-entry-handoff.json',handoff)
 command=[d['python'],'-B',str(runtime/'ema_cross_scheduler_handoff.py'),'--project',str(p),'--source',str(source),
  '--output',str(queue),'--runtime',str(runtime),'--database',d['database'],'--expected-gib',str(d['expected_gib'])]
 spec=base/'scheduler_v2_attempt.json';save(spec,dict(command=command,cwd=str(source),receipt=str(base/'scheduler_v2_exit.json'),
  job_id='ema_cross_'+d['database']+'_scheduler_v2',source_manifest_sha256=d['source_sha256']))
 with (base/'scheduler_v2.log').open('x') as log:
  child=subprocess.Popen([d['python'],'-B',str(source/'scripts/tracehr_recovery_attempt.py'),'--spec',str(spec)],cwd=source,
   env=env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 result=dict(complete=True,wrapper_pid=child.pid,training_workers_preserved=len(live),source_unchanged=d['source_sha256'],
  runtime_sha256=hashes,queue=str(queue),tests_started=0)
'''
CONTROL=CONTROL.replace('else:raise ValueError(mode)',HANDOFF.strip()+'\nelse:raise ValueError(mode)')
CONTROL=CONTROL.replace(" if (base/'start_exit.json').exists():result['queue_exit']=read(base/'start_exit.json')",
 " if (base/'start_exit.json').exists():result['predecessor_queue_exit']=read(base/'start_exit.json')\n if result.get('active_scheduler_receipt') and Path(result['active_scheduler_receipt']).exists():result['queue_exit']=read(result['active_scheduler_receipt'])")

COMPLETION_RECOVERY=r'''
elif mode=='recover-completion':
 import signal,importlib.util
 os.environ.update(env());sys.path[:0]=os.environ['PYTHONPATH'].split(os.pathsep);sys.modules['lightning']=None
 from scripts.tracehr_recovery_queue import same_process
 assert sha(source/'source_manifest.json')==d['source_sha256']
 for n,h in read(source/'source_manifest.json').items():assert sha(source/n)==h,n
 assert not (base/'completion-recovery-handoff.json').exists() and not (base/'scheduler_v3_attempt.json').exists()
 runtime=p/'work'/d['run']/'completion_runtime_v1';runtime.mkdir(parents=True,exist_ok=True)
 hashes={}
 for n,b64 in d['runtime_files'].items():
  data=base64.b64decode(b64);compile(data,n,'exec');f=runtime/n
  if f.exists():assert f.read_bytes()==data
  else:f.write_bytes(data)
  hashes[n]=sha(f)
 spec=importlib.util.spec_from_file_location('completion_recovery',runtime/'ema_cross_completion_recovery.py')
 recovery=importlib.util.module_from_spec(spec);spec.loader.exec_module(recovery)
 # Reproduce only the failing serialization expression; no model or optimizer is run.
 identity={'parameter_source':'EMA'}
 try:dict(**identity,parameter_source='EMA');raise AssertionError('expected duplicate-key exception')
 except TypeError as exc:assert 'parameter_source' in str(exc)
 fixed=dict(identity);fixed.update(parameter_source='EMA');assert fixed==identity
 s=read(queue/'state.json');assert s['phase']=='formal' and s['evaluation_counts']['pending']==12
 assert s['counts']['completed']==0 and s['counts']['pending']==6
 assert all(j['status']=='pending' for j in s['jobs'].values() if j['action']=='train' and j['variant']=='TabM-mini')
 audits={}
 for j in s['jobs'].values():
  if j['status']=='failed':audits[j['id']]=recovery.audit_job(source,j)
 save(runtime/'preflight-audit.json',dict(complete=True,reproduced_serialization_error=True,source_unchanged=d['source_sha256'],failed_jobs=audits))
 save(runtime/'envelope.json',dict(complete=True,files=hashes,training_source_unchanged=d['source_sha256'],
  change='Zero-update finalization of exact terminal duplicate-key error, unchanged pending fits, preserved live workers; no weight/history rewrites'))
 pid=s['pid'];ident=s['process_identity'];was_live=same_process(pid,ident)
 old_receipt=Path(s['active_scheduler_receipt'])
 if was_live:
  for attempt in range(20):
   assert same_process(pid,ident);os.kill(pid,signal.SIGSTOP);time.sleep(.1);s=read(queue/'state.json')
   if not any(j['status']=='launching' for j in s['jobs'].values()):break
   os.kill(pid,signal.SIGCONT);time.sleep(.3)
  else:raise RuntimeError('Could not obtain stable scheduler boundary')
 else:
  assert old_receipt.exists() and read(old_receipt)['returncode']==2 and len(audits)==6
 try:
  assert s['pid']==pid and s['process_identity']==ident and s['evaluation_counts']['pending']==12
  save(runtime/'predecessor-state.json',s)
  live=[dict(id=j['id'],pid=j['pid'],identity=j['process_identity'],exit_receipt=j['exit_receipt']) for j in s['jobs'].values() if j['status']=='running']
  for j in live:assert same_process(j['pid'],j['identity']) or Path(j['exit_receipt']).exists()
  if was_live:os.kill(pid,signal.SIGTERM)
 finally:
  if was_live and same_process(pid,ident):os.kill(pid,signal.SIGCONT)
 if was_live:
  for _ in range(100):
   if not same_process(pid,ident) and old_receipt.exists():break
   time.sleep(.1)
  assert not same_process(pid,ident) and read(old_receipt)['returncode']==-signal.SIGTERM
 for j in live:assert same_process(j['pid'],j['identity']) or Path(j['exit_receipt']).exists()
 handoff=dict(complete=True,old_queue_pid=pid,old_queue_identity=ident,old_scheduler_receipt=str(old_receipt),
  old_scheduler_exit_sha256=sha(old_receipt),training_source_sha256=d['source_sha256'],
  predecessor_state_sha256=sha(runtime/'predecessor-state.json'),runtime_envelope_sha256=sha(runtime/'envelope.json'),
  preflight_audit_sha256=sha(runtime/'preflight-audit.json'),workers_preserved=live,
  completed_optimization_jobs_needing_finalization=list(audits),test_jobs_started=0,time=time.time())
 save(base/'completion-recovery-handoff.json',handoff)
 command=[d['python'],'-B',str(runtime/'ema_cross_completion_recovery.py'),'--mode','queue','--project',str(p),
  '--source',str(source),'--output',str(queue),'--runtime',str(runtime),'--database',d['database'],'--expected-gib',str(d['expected_gib'])]
 attempt=base/'scheduler_v3_attempt.json';save(attempt,dict(command=command,cwd=str(source),receipt=str(base/'scheduler_v3_exit.json'),
  job_id='ema_cross_'+d['database']+'_scheduler_v3',source_manifest_sha256=d['source_sha256']))
 with (base/'scheduler_v3.log').open('x') as log:
  child=subprocess.Popen([d['python'],'-B',str(source/'scripts/tracehr_recovery_attempt.py'),'--spec',str(attempt)],
   cwd=source,env=env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 result=dict(complete=True,wrapper_pid=child.pid,training_workers_preserved=len(live),zero_update_finalizations=len(audits),
  source_unchanged=d['source_sha256'],runtime_sha256=hashes,queue=str(queue),tests_started=0)
'''
CONTROL=CONTROL.replace('else:raise ValueError(mode)',COMPLETION_RECOVERY.strip()+'\nelse:raise ValueError(mode)')
COMPLETION_RECOVERY_V2=COMPLETION_RECOVERY.replace("mode=='recover-completion'","mode=='recover-completion-v2'").replace('completion_runtime_v1','completion_runtime_v2').replace('completion-recovery-handoff.json','completion-recovery-v2-handoff.json').replace('scheduler_v3','scheduler_v4')
COMPLETION_RECOVERY_V2=COMPLETION_RECOVERY_V2.replace("assert s['counts']['completed']==0 and s['counts']['pending']==6","assert s['counts']['completed']==0")
COMPLETION_RECOVERY_V2=COMPLETION_RECOVERY_V2.replace("  if j['status']=='failed':audits[j['id']]=recovery.audit_job(source,j)","""  if j.get('completion_pending'):
   assert j['status'] in ('pending','failed') and not (Path(j['output'])/'completion.json').exists()
   if j['status']=='failed':
    rc=read(j['exit_receipt']);assert rc['returncode']==1 and not same_process(j['pid'],j['process_identity']) and not same_process(rc['child_pid'],rc['child_identity'])
    log=Path(j['log']).read_text();assert \"if k!='epoch_seconds_with_checkpoint'}\" in log and log.rstrip().endswith('AssertionError')
   audits[j['id']]=recovery.audit_job(source,j['training_attempt'])
  elif j['status']=='failed':audits[j['id']]=recovery.audit_job(source,j)""")
CONTROL=CONTROL.replace('else:raise ValueError(mode)',COMPLETION_RECOVERY_V2.strip()+'\nelse:raise ValueError(mode)')
DIAGNOSE=r'''
elif mode=='diagnose-completion':
 os.environ.update(env());sys.path[:0]=os.environ['PYTHONPATH'].split(os.pathsep);sys.modules['lightning']=None
 import torch
 s=read(queue/'state.json');rows=[]
 def differences(a,b,prefix=''):
  if isinstance(a,dict) and isinstance(b,dict):
   return sum((differences(a.get(k),b.get(k),prefix+'/'+str(k)) for k in a.keys()|b.keys()),[])
  if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
   return [dict(path=prefix,lengths=[len(a),len(b)])] if len(a)!=len(b) else sum((differences(x,y,prefix+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b))),[])
  return [] if a==b else [dict(path=prefix,left_type=type(a).__name__,right_type=type(b).__name__,left=a,right=b)]
 for j in s['jobs'].values():
  if j['status']!='failed':continue
  folder=Path(j['output']);ck=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
  h=read(folder/'history.json');saved=ck['state']['history'];raw_differences=[];normalized_differences=[]
  for i,(left,right) in enumerate(zip(saved,h)):
   left={k:v for k,v in left.items() if k!='epoch_seconds_with_checkpoint'};right={k:v for k,v in right.items() if k!='epoch_seconds_with_checkpoint'}
   raw_differences+=differences(left,right,str(i+1));normalized_differences+=differences(json.loads(json.dumps(left)),right,str(i+1))
  rows.append(dict(id=j['id'],epochs=len(h),raw_count=len(raw_differences),normalized_count=len(normalized_differences),
   raw_examples=raw_differences[:5],normalized_examples=normalized_differences[:5]))
 result=dict(complete=True,diagnosis=rows,optimizer_updates=0)
'''
CONTROL=CONTROL.replace('else:raise ValueError(mode)',DIAGNOSE.strip()+'\nelse:raise ValueError(mode)')
CONTROL=CONTROL.replace("   rows.append(row)","""   if j.get('completion_pending'):
    row['completion_recovery']=True
    f=queue/('finalization_v2' if result.get('completion_handoff') else 'finalization')/j['id']/'output/progress.json'
    if f.exists():row['finalization_progress']=read(f)
   history=Path(j['output'])/'history.json'
   if j['action']=='train' and history.exists():
    h=read(history);row['history_tail']=[{k:r[k] for k in ('epoch','dev_ap','epoch_seconds_with_checkpoint','epoch_seconds','seconds','train_seconds','evaluation_seconds') if k in r} for r in h[-5:]]
    if h:
     best=max(h,key=lambda r:r['dev_ap']);times=[r.get('epoch_seconds_with_checkpoint',r.get('seconds')) for r in h[-20:]]
     row['history_summary']=dict(completed_epochs=len(h),best_ap=best['dev_ap'],best_epoch=best['epoch'],
      recent_epoch_seconds=sum(times)/len(times) if all(t is not None for t in times) else None)
   rows.append(row)""")

IDLE_GPU_HANDOFF=r'''
elif mode=='handoff-idle-gpus':
 import signal,importlib.util
 assert d['database']=='eicu'
 os.environ.update(env());sys.path[:0]=os.environ['PYTHONPATH'].split(os.pathsep);sys.modules['lightning']=None
 from scripts.tracehr_recovery_queue import same_process
 assert sha(source/'source_manifest.json')==d['source_sha256']
 for n,h in read(source/'source_manifest.json').items():assert sha(source/n)==h,n
 assert not (base/'idle-gpu-handoff.json').exists() and not (base/'scheduler_v5_attempt.json').exists()
 runtime=p/'work'/d['run']/'idle_gpu_runtime_v1';runtime.mkdir(parents=True,exist_ok=True)
 hashes={}
 for n,b64 in d['runtime_files'].items():
  data=base64.b64decode(b64);compile(data,n,'exec');f=runtime/n
  if f.exists():assert f.read_bytes()==data
  else:f.write_bytes(data)
  hashes[n]=sha(f)
 spec=importlib.util.spec_from_file_location('idle_gpu_admission',runtime/'ema_cross_idle_gpu_queue.py')
 policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
 recovery=p/'work'/d['run']/'completion_runtime_v2'
 recovered=policy.load_recovery_class(source,recovery)
 assert all(callable(getattr(recovered,n)) for n in ('poll','launch','validate','phase','main'))
 devices=policy.inventory(p);assert len(devices)==8 and {r['index'] for r in devices}==set(range(8))
 save(runtime/'envelope.json',dict(complete=True,files=hashes,training_source_unchanged=d['source_sha256'],
  recovery_envelope_sha256=sha(recovery/'envelope.json'),visible_gpus=list(range(8)),fallback_gpus=[4,5,6],
  idle_preference=[2,3],max_workers_per_gpu=2,continuous_idle_seconds=10,
  user_authorization='Use completely idle GPUs first, including 5880_1 GPU2/3; otherwise use originally specified GPUs',
  initial_inventory=devices,change='Pending-job GPU admission only; preserve live fits, fit matrix and scientific settings'))
 for attempt in range(20):
  s=read(queue/'state.json');pid=s['pid'];ident=s['process_identity'];assert same_process(pid,ident)
  assert s['database']=='eicu' and s['gpus']==[4,5,6] and s['status']=='running' and s['phase']=='formal'
  assert s['counts']['failed']==0 and s['evaluation_counts']['pending']==12
  os.kill(pid,signal.SIGSTOP);time.sleep(.1);s=read(queue/'state.json')
  if not any(j['status']=='launching' for j in s['jobs'].values()):break
  os.kill(pid,signal.SIGCONT);time.sleep(.3)
 else:raise RuntimeError('Could not obtain stable scheduler boundary')
 try:
  assert same_process(pid,ident) and s['pid']==pid and s['process_identity']==ident
  assert s['counts']['failed']==0 and s['evaluation_counts']['pending']==12
  assert sum(j['action']=='train' for j in s['jobs'].values())==12
  save(runtime/'predecessor-state.json',s)
  live=[dict(id=j['id'],pid=j['pid'],identity=j['process_identity'],gpu=j['gpu'],exit_receipt=j['exit_receipt']) for j in s['jobs'].values() if j['status']=='running']
  for j in live:assert same_process(j['pid'],j['identity']) or Path(j['exit_receipt']).exists()
  old_receipt=Path(s['active_scheduler_receipt']);assert old_receipt==base/'scheduler_v4_exit.json'
  os.kill(pid,signal.SIGTERM)
 finally:
  if same_process(pid,ident):os.kill(pid,signal.SIGCONT)
 for _ in range(100):
  if not same_process(pid,ident) and old_receipt.exists():break
  time.sleep(.1)
 assert not same_process(pid,ident) and read(old_receipt)['returncode']==-signal.SIGTERM
 for j in live:assert same_process(j['pid'],j['identity']) or Path(j['exit_receipt']).exists()
 handoff=dict(complete=True,old_queue_pid=pid,old_queue_identity=ident,old_scheduler_receipt=str(old_receipt),
  old_scheduler_exit_sha256=sha(old_receipt),predecessor_state_sha256=sha(runtime/'predecessor-state.json'),
  training_source_sha256=d['source_sha256'],runtime_envelope_sha256=sha(runtime/'envelope.json'),
  workers_preserved=live,formal_fits=12,new_scientific_tasks=0,scientific_settings_changed=False,time=time.time())
 save(base/'idle-gpu-handoff.json',handoff)
 command=[d['python'],'-B',str(runtime/'ema_cross_idle_gpu_queue.py'),'--project',str(p),'--source',str(source),
  '--output',str(queue),'--runtime',str(recovery),'--policy-runtime',str(runtime),'--database','eicu','--expected-gib',str(d['expected_gib'])]
 attempt=base/'scheduler_v5_attempt.json';save(attempt,dict(command=command,cwd=str(source),receipt=str(base/'scheduler_v5_exit.json'),
  job_id='ema_cross_eicu_scheduler_v5',source_manifest_sha256=d['source_sha256']))
 with (base/'scheduler_v5.log').open('x') as log:
  child=subprocess.Popen([d['python'],'-B',str(source/'scripts/tracehr_recovery_attempt.py'),'--spec',str(attempt)],
   cwd=source,env=env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 result=dict(complete=True,wrapper_pid=child.pid,training_workers_preserved=len(live),source_unchanged=d['source_sha256'],
  runtime_sha256=hashes,queue=str(queue),formal_fits=12,new_scientific_tasks=0,gpu_policy='idle-first',initial_inventory=devices)
'''
CONTROL=CONTROL.replace('else:raise ValueError(mode)',IDLE_GPU_HANDOFF.strip()+'\nelse:raise ValueError(mode)')


def build(out,db):
    parent=ROOT/'runs/mimiciii_ema_pm_20261024/local_v1'
    files=read(parent/'source_bundle.json');manifest=read(parent/'source_manifest.json')
    assert sha(parent/'source_manifest.json')==PARENT
    assert {n:hashlib.sha256(base64.b64decode(b)).hexdigest() for n,b in files.items()}==manifest
    agg=read(ROOT/'runs/tracehr_fte_binding_20261022'/(db+'_dev_results_v2/aggregates.json'))
    assert agg['complete']
    controls={k:v for k,v in agg['records'].items() if k.startswith(('C5_noT_noA_noR_untied_','B0_','tabm_'))}
    assert len(controls)==18
    remote=HOSTS[db][1];expected_gib=40 if db=='mimic' else 48
    inv=ssh(db,INVENTORY.replace('PAYLOAD',repr(dict(project=remote,database=db,controls=controls,expected_gib=expected_gib)),1))
    save(out/'inventory-receipt.json',inv)
    c=dict(inv,parent_source_sha256=PARENT,controls=controls,database=db,formal_fits=12,seeds=[48,49,50],
           variants=['fullEMA','TabM-mini'],endpoints=[0,1],gpus=[0,1,4] if db=='mimic' else [4,5,6],workers_per_gpu=2,
           expected_gib=expected_gib,output='runs/'+RUN+'/'+db+'/v1',cache='cache/'+RUN,temp='work/t_ema_cross',
           original_preprocessing=True,hyperparameter_search=False,subsequent_ablations_authorized=False,
           test_policy='Lock all twelve dev-selected checkpoints in this database before fixed forward-only tests',
           patient_bootstrap_replicates=4000,baseline_family_level=1-.05/12,noEMA_family_level=1-.05/4)
    for n in (*ENTRIES,'work/'+RUN+'/'+db+'_expected.json'):
        assert n not in files,n
        b=json.dumps(c,indent=2).encode() if n.endswith('_expected.json') else (ROOT/n).read_bytes()
        if n.endswith('.py'):compile(b,n,'exec')
        files[n]=base64.b64encode(b).decode()
    manifest={n:hashlib.sha256(base64.b64decode(b)).hexdigest() for n,b in files.items()}
    (out/'source_manifest.json').write_bytes(json.dumps(manifest,indent=2).encode());save(out/'source_bundle.json',files)
    result=dict(complete=True,database=db,inherited_files=len(files)-4,source_files=len(files),
                source_manifest_sha256=sha(out/'source_manifest.json'),contract=c)
    save(out/'build-receipt.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--database',choices=('mimic','eicu'),required=True)
    p.add_argument('--mode',choices=('build','deploy','verify','start','status','release','handoff-evaluation','recover-completion','recover-completion-v2','diagnose-completion','handoff-idle-gpus'),required=True);a=p.parse_args()
    out=Path(os.environ['PROJECT_RUN_DIR']).resolve();assert out==(ROOT/'runs'/RUN/('local_'+a.database+'_v1')).resolve()
    if a.mode=='build':result=build(out,a.database)
    else:
        b=read(out/'build-receipt.json');_,remote,python=HOSTS[a.database]
        d=dict(mode=a.mode,project=remote,run=RUN,database=a.database,python=python,expected_gib=b['contract']['expected_gib'],source_sha256=b['source_manifest_sha256'])
        if a.mode=='deploy':d['files']=read(out/'source_bundle.json')
        if a.mode=='handoff-evaluation':d['runtime_files']={n:base64.b64encode((ROOT/'scripts'/n).read_bytes()).decode() for n in ('ema_cross_evaluation_entry.py','ema_cross_scheduler_handoff.py')}
        if a.mode in ('recover-completion','recover-completion-v2'):d['runtime_files']={'ema_cross_completion_recovery.py':base64.b64encode((ROOT/'scripts/ema_cross_completion_recovery.py').read_bytes()).decode()}
        if a.mode=='handoff-idle-gpus':d['runtime_files']={'ema_cross_idle_gpu_queue.py':base64.b64encode((ROOT/'scripts/ema_cross_idle_gpu_queue.py').read_bytes()).decode()}
        result=ssh(a.database,CONTROL.replace('PAYLOAD',repr(d),1));save(out/(a.mode+'-receipt.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('contract','jobs')},ensure_ascii=False))
    if a.mode=='status':
        for j in result.get('jobs',[]):
            if j['status'] in ('running','failed'):print(json.dumps(j,ensure_ascii=False))

if __name__=='__main__':main()
