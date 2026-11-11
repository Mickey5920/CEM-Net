"""Storage-guarded deployment of the authorized, bounded ICDE evidence batch."""
import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.launch_ema_cross import HOSTS, ssh

RUN = 'cem_icde_evidence_20261028'
PRIOR = 'cem_evidence_completion_20261025'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.writing')
    q.write_text(json.dumps(x, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(q, p)

INVENTORY = r'''
import importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
p=Path(PAYLOAD['project']).resolve(strict=True)
deps=p/'work/tracehr_tabular_20261004/deps'
sys.path[:0]=[str(deps),*[str(x) for x in deps.glob('*.whl')]]
packages={}
for name in ('numpy','torch','sklearn','lightgbm','tabm','rtdl_num_embeddings','pyarrow'):
 try:
  mod=__import__(name);packages[name]={'version':getattr(mod,'__version__',None),'file':getattr(mod,'__file__',None)}
 except Exception as exc:packages[name]={'error':repr(exc)}
mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
files={}
for rel in ('configs/trace_transport_site_holdout_v1.json',
 'runs/cem_evidence_completion_20261025/host_mimic/start_exit.json',
 'runs/cem_evidence_completion_20261025/host_eicu/test_runtime_v1_exit.json',
 'runs/cem_observation_context_20261027/host_eicu_v2/start_exit.json'):
 f=p/rel
 if f.is_file():files[rel]=json.loads(f.read_text())
paths={}
for rel in ('data','runs','work','cache','work/tracehr_tabular_20261004/deps'):
 f=(p/rel).resolve(strict=True);paths[rel]={'resolved':str(f),'free_gib':shutil.disk_usage(f).free/2**30}
print(json.dumps({'complete':True,'project':str(p),'packages':packages,'paths':paths,
 'available_ram_gib':mem['MemAvailable']/2**30,'cpu_count':os.cpu_count(),'loadavg':os.getloadavg(),
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True),
 'prior_receipts':files,'training_launched':False}))
'''

INSPECT = r'''
import json,os
from pathlib import Path
p=Path(PAYLOAD['project']);result={}
for db,c in PAYLOAD['contracts'].items():
 r={}
 for key in ('input_preparation','raw_inputs','baseline_preparation','native_input_preparation'):
  if not c.get(key):continue
  f=Path(c[key]);f=f if f.is_absolute() else p/f
  r[key]={'path':str(f),'exists':f.exists()}
  if f.is_dir():
   r[key]['files']=[{'name':q.name,'bytes':q.stat().st_size} for q in f.iterdir() if q.is_file()]
   if (f/'completion.json').exists():r[key]['completion']=json.loads((f/'completion.json').read_text())
 result[db]=r
for label,rel in [('site_metadata','data/EicuExperiment4ClinicalV2SiteHoldout/24h/fold0/metadata.json'),
 ('fullstream','cache/tracehr_fullstream_20261003/eicu24'),
 ('native','cache/cem_evidence_completion_20261025/native_observations'),
 ('training_events','runs/tracehr_asof_full_eicu_20261003/raw'),
 ('training_numeric','runs/tracehr_numeric_full_eicu_20261003'),
 ('pyarrow_isolation','work/cem_evidence_completion_20261025/test_runtime_v1/deps')]:
 f=p/rel
 if f.is_file():
  v=json.loads(f.read_text());result[label]={k:v[k] for k in ('hospital_ids_by_split','hospital_overlap','patient_overlap','subject_overlap','split_by','split_seed','formal_build_config') if k in v}
 elif f.is_dir():result[label]={'path':str(f),'files':[{'name':q.name,'bytes':q.stat().st_size if q.is_file() else None} for q in f.iterdir()]}
 else:result[label]={'exists':False,'path':str(f)}
print(json.dumps(result))
'''

def build(out, db, stage, expected_gib, snapshot):
    parent=ROOT/'runs'/PRIOR/f'local_{db}_tabular_v1'
    files=read(parent/'source_bundle.json');inherited=read(parent/'source_manifest.json')
    c=read(parent/'build-receipt.json')['contract']
    c.update(run=RUN,database=db,stage=stage,expected_gib=expected_gib,snapshot=snapshot,
             native_source_cache=f'cache/{PRIOR}/native_observations/{db}')
    if stage=='basic' and snapshot=='v2':
        predecessor=ROOT/'runs'/RUN/f'local_{db}_basic_v1/build-receipt.json'
        c['execution_predecessor']=dict(source_manifest_sha256=read(predecessor)['source_manifest_sha256'],
            result='CPU preflight failed before any probe or formal fit: missing native vocabulary cache pointer',
            change='supply the existing vocabulary cache pointer; feature mask and scientific recipe unchanged')
    if stage=='lgbm':
        matrix=[dict(variant='LightGBM_full',seed=s,endpoint=e) for s in (48,49,50) for e in (0,1)]
    elif stage=='basic':
        strongest={'mimiciii':('TabM_PLE_basic','TabM_mini_basic'),'mimic':('TabM_PLE_basic',)*2,'eicu':('TabM_mini_PLE_basic',)*2}[db]
        matrix=[dict(variant=v,seed=s,endpoint=e) for s in (48,49,50) for e in (0,1) for v in ('CEM_basic',strongest[e])]
    else:
        assert db=='eicu'
        matrix=[dict(variant=v,seed=s,endpoint=e) for s in (48,49,50) for e in (0,1) for v in ('CEM_hospital','TabM_mini_PLE_hospital','LightGBM_hospital')]
        summary=read(ROOT/'runs'/RUN/'local_hospital_inputs/status.json')
        prep=summary['completion.json'];assert prep['complete'] and summary['exit.json']['returncode']==0
        base=f'runs/{RUN}/eicu/hospital_preparation_v2';project=HOSTS['eicu'][1]
        c.update(input_preparation=base,raw_inputs=base,test_inputs=base,baseline_preparation=base+'/tabular',
            input_dim=prep['input_dim'],train_rows=prep['train_rows'],dev_rows=prep['dev_rows'],
            tabm_preprocessing=project+'/'+base+'/tabm_preprocessing.joblib',
            tabm_preprocessing_sha256=prep['artifacts']['tabm_preprocessing.joblib']['sha256'],
            test_manifest={n:prep['artifacts'][n] for n in ('test_raw.npy','test_targets.npz')},
            hospital_preparation=prep,preparation_mode='fixed hospital-component split; refit train-only transforms; unchanged model recipes')
    c['matrix']=matrix;c['formal_fits']=len(matrix);c['variants']=sorted({x['variant'] for x in matrix})
    c['preserved_parent_manifest_sha256']=sha(parent/'source_manifest.json')
    added=['scripts/cem_icde72_common.py','scripts/train_cem_icde72_lgbm.py','scripts/cem_icde72_queue.py','scripts/cem_evidence_host_queue.py']
    if stage in ('basic','hospital'):added+=['models/cem_icde72_mask.py','scripts/cem_icde72_neural.py','scripts/train_cem_icde72_cem.py','scripts/train_cem_icde72_tabular.py']
    if stage=='hospital':added+=['scripts/cem_icde72_hospital.py','scripts/prepare_cem_icde72_hospital.py']
    for n in added:
        b=(ROOT/n).read_bytes();compile(b,n,'exec')
        if n in files:assert base64.b64decode(files[n])==b,n
        files[n]=base64.b64encode(b).decode()
    config=f'work/{RUN}/config.json';assert config not in files
    files[config]=base64.b64encode(json.dumps(c,indent=2).encode()).decode()
    manifest={n:hashlib.sha256(base64.b64decode(b)).hexdigest() for n,b in files.items()}
    assert all(manifest[n]==h for n,h in inherited.items())
    assert not (out/'source_manifest.json').exists(), 'A built snapshot is immutable'
    save(out/'source_bundle.json',files);save(out/'source_manifest.json',manifest)
    r=dict(complete=True,source_manifest_sha256=sha(out/'source_manifest.json'),inherited_files=len(inherited),files=len(files),contract=c)
    save(out/'build-receipt.json',r);return {k:v for k,v in r.items() if k!='contract'}

REMOTE = r'''
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
 for x in (temp,cache):x.mkdir(parents=True,exist_ok=True)
 paths=[source,deps,*deps.glob('*.whl'),p/'work/cem_evidence_completion_20261025/test_runtime_v1/deps']
 return dict(os.environ,TRACEHR_PROJECT=str(p),TRACEHR_DATABASE=d['database'],TRACEHR_TABULAR_DEPS=str(deps),
  CEM_COMPLETION_STAGE='tabular',PYTHONPATH=os.pathsep.join(map(str,paths)),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
  PYTHONNOUSERSITE='1',NUMPY_MADVISE_HUGEPAGE='0',TMPDIR=str(temp),TEMP=str(temp),TMP=str(temp),
  XDG_CACHE_HOME=str(cache/'runtime'),TORCH_HOME=str(cache/'torch'),MPLCONFIGDIR=str(cache/'matplotlib'),
  OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
if d['mode']=='deploy':
 assert not (out/'state.json').exists() and not (out.parent/'verification.json').exists()
 assert shutil.disk_usage(p).free>(10+d['expected_gib'])*2**30
 for rel in (d['source'],d['output'],'cache/'+d['run'],'work/'+d['run']+'/tmp'):
  path=p/rel;path.mkdir(parents=True,exist_ok=True);assert path.resolve().is_relative_to(p)
 for n,b in d['files'].items():
  f=source/n;assert f.resolve().is_relative_to(source.resolve());f.parent.mkdir(parents=True,exist_ok=True);b=base64.b64decode(b)
  if n.endswith('.py'):compile(b,n,'exec')
  assert not f.exists() or f.read_bytes()==b,n
  f.write_bytes(b)
 manifest_bytes=base64.b64decode(d['manifest_bytes']);assert json.loads(manifest_bytes)==d['hashes']
 (source/'source_manifest.json').write_bytes(manifest_bytes);assert sha(source/'source_manifest.json')==d['source_sha256']
 result=dict(complete=True,files=len(d['files']),source_manifest_sha256=d['source_sha256'])
elif d['mode']=='verify':
 assert not (out.parent/'verification.json').exists()
 assert sha(source/'source_manifest.json')==d['source_sha256']
 for n,h in read(source/'source_manifest.json').items():assert sha(source/n)==h,n
 os.environ.update(env());sys.path[:0]=os.environ['PYTHONPATH'].split(os.pathsep);sys.modules['lightning']=None
 from scripts.cem_icde72_common import RawData,expected
 c=expected();data=RawData(p)
 assert len(c['matrix'])==c['formal_fits'];assert len(data.targets['train']['y'])==c['train_rows']
 for s in ('train','dev'):
  import numpy as np
  for i in range(0,len(data.raw[s]),1024):assert np.isfinite(data.raw[s][i:i+1024]).all()
 from scripts.cem_evidence_host_queue import policy_checks
 checks=policy_checks()+['complete_float32_input_finite','patient_disjoint_train_validation','unchanged_parent_source_files']
 if d['stage']=='lgbm':
  import lightgbm;assert lightgbm.__version__=='4.6.0';checks+=['existing_LightGBM_4.6.0','fixed_published_recipe_no_search']
 else:
  from scripts.cem_icde72_neural import cpu_checks
  checks+=cpu_checks(p)
 result=dict(complete=True,source_manifest_sha256=d['source_sha256'],formal_fits=len(c['matrix']),test_evaluations=len(c['matrix']),
  input_dim=data.input_dim,train_rows=len(data.targets['train']['y']),dev_rows=len(data.targets['dev']['y']),
  cpu_checks=checks,optimizer_updates=0,test_loaded=False,verified_at=time.time())
 save(out.parent/'verification.json',result)
elif d['mode']=='register':
 v=read(out.parent/'verification.json');assert v['complete'] and v['source_manifest_sha256']==d['source_sha256']
 c=read(source/('work/'+d['run']+'/config.json'));file=host/'ready'/(d['database']+'_'+d['stage']+'.json');assert not file.exists()
 result=dict(database=d['database'],stage=d['stage'],stage_order={'lgbm':1,'basic':2,'hospital':3}[d['stage']],source=str(source),output=str(out),
  source_manifest_sha256=d['source_sha256'],matrix=c['matrix'],variants=c['variants'],
  entries=dict(tree='scripts/train_cem_icde72_lgbm.py',cem='scripts/train_cem_icde72_cem.py',tabular='scripts/train_cem_icde72_tabular.py'),
  expected_gib=d['expected_gib'],retention_multiplier=60 if d['stage']=='lgbm' else 13,min_available_ram_gib=24,
  cpu_min_available_ram_gib=40 if d['database']=='eicu' else 24,probe_gpu_mib=24576,
  probe_requirement_basis='conservative original bounded-probe requirement, refined only from measured same-path evidence')
 if d['stage']=='basic':
  peaks=[]
  for cell in c['matrix']:
   if cell['seed']!=48:continue
   base=cell['variant'].removesuffix('_basic');stage='controls_v1' if base=='CEM' else 'tabular_v1'
   variant='EMA_NRE_MLP' if base=='CEM' else base
   if base=='TabM_mini':continue
   receipt=p/f'runs/cem_evidence_completion_20261025/{d["database"]}/{stage}/screen/pilots/probe_{variant}_t{cell["endpoint"]}_s48/completion.json'
   r=read(receipt);assert r['complete'] and r['pilot']
   peaks.append(dict(path=str(receipt),sha256=sha(receipt),peak_reserved_mib=r['peak_reserved_mib']))
  assert peaks
  result['probe_gpu_mib']=max(8192,int(max(x['peak_reserved_mib'] for x in peaks)*1.5+2048))
  result['probe_requirement_basis']=dict(precedent='same full parameter layout, batch and dtype; evidence mask adds only index buffers',audited_peaks=peaks,headroom_factor=1.5,additional_mib=2048)
 save(file,result)
elif d['mode']=='storage-handoff':
 s=read(host/'state.json');assert not (host/'storage-handoff.json').exists()
 jobs=[j for g in s['groups'].values() for j in g['jobs'].values()]
 if any(j['status']=='running' for j in jobs) or any(g['status']!='complete' for g in s['groups'].values() if g['stage']=='lgbm'):
  result=dict(complete=False,waiting_for='existing workers to finish; no processes changed')
 else:
  assert all(g['phase']=='probes' and g['status']=='storage_review_required' for g in s['groups'].values() if g['stage']=='basic')
  assert all(j['status']=='completed' for j in jobs if j['action']=='probe')
  assert shutil.disk_usage(p).free>(sum(160 if g['stage']=='basic' else g['expected_gib'] for g in s['groups'].values())+10)*2**30
  pid=s['pid'];cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
  assert str(p/'work'/d['run']) in cmd and 'scripts/cem_icde72_queue.py' in cmd and '--host '+d['host'] in cmd
  # Only an idle scheduler is handed off; all child receipts already succeeded.
  assert all(read(j['exit_receipt'])['returncode']==0 for j in jobs if j['status']=='completed')
  os.kill(pid,15)
  for _ in range(50):
   if (host/'start_exit.json').exists():break
   time.sleep(.1)
  assert read(host/'start_exit.json')['returncode']==-15
  final=read(host/'state.json');assert all(j['status']!='running' for g in final['groups'].values() for j in g['jobs'].values())
  predecessor=host/'storage-predecessor-state.json';predecessor.write_bytes((host/'state.json').read_bytes())
  protected={}
  for g in final['groups'].values():
   protected[str(Path(g['source'])/'source_manifest.json')]=sha(Path(g['source'])/'source_manifest.json')
   for j in g['jobs'].values():
    if j['status']=='completed':
     for f in Path(j['output']).iterdir():
      if f.is_file():protected[str(f)]=sha(f)
     protected[j['exit_receipt']]=sha(j['exit_receipt'])
  runtime=p/'work'/d['run']/('storage_runtime_'+d['host']);runtime.mkdir(parents=True,exist_ok=True)
  entry=runtime/'continue.py';assert not entry.exists();entry.write_bytes(base64.b64decode(d['handoff_code']))
  save(host/'storage-handoff.json',dict(complete=True,active_workers=0,predecessor_pid=pid,state_sha256=sha(predecessor),protected=protected,
   code_sha256=sha(entry),scientific_changes=False,change='raise basic-stage storage declarations 48 to 160 GiB; preserve every job and checkpoint',created_at=time.time()))
  spec=dict(command=[sys.executable,'-B',str(entry),'--project',str(p),'--host',d['host']],cwd=str(source),receipt=str(host/'storage_runtime_exit.json'),job_id='cem72_storage_'+d['host'],source_manifest_sha256=sha(entry))
  save(host/'storage_runtime_attempt.json',spec)
  with (host/'storage_runtime.log').open('x') as log:
   child=subprocess.Popen([sys.executable,'-B',str(source/'scripts/tracehr_recovery_attempt.py'),'--spec',str(host/'storage_runtime_attempt.json')],cwd=source,env=env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  result=dict(complete=True,wrapper_pid=child.pid,protected_files=len(protected),completed_jobs_restarted=0,scientific_changes=False)
elif d['mode']=='start':
 assert not (host/'start_attempt.json').exists() and not (host/'state.json').exists();host.mkdir(parents=True,exist_ok=True)
 wrapper=source/'scripts/tracehr_recovery_attempt.py';assert wrapper.exists()
 command=[sys.executable,'-B',str(source/'scripts/cem_icde72_queue.py'),'--project',str(p),'--host',d['host']]
 attempt=host/'start_attempt.json';save(attempt,dict(command=command,cwd=str(source),receipt=str(host/'start_exit.json'),
  job_id='cem72_host_'+d['host'],source_manifest_sha256=d['source_sha256']))
 with (host/'queue.log').open('x') as log:
  child=subprocess.Popen([sys.executable,'-B',str(wrapper),'--spec',str(attempt)],cwd=source,env=env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 result=dict(complete=True,wrapper_pid=child.pid,source_manifest_sha256=d['source_sha256'],authorized_formal_fits=36,formal_training_launched=False)
 save(host/'launch.json',result)
elif d['mode']=='release':
 s=read(out/'state.json');assert s['status']=='awaiting_path_review';assert not (out/'formal_release.json').exists()
 assert s['counts']['pending']==len(s['matrix']) and s['projected_artifact_gib']<s['expected_gib']
 audits={};reports=[]
 for j in s['jobs'].values():
  if j['action']!='probe':continue
  assert j['status']=='completed' and read(j['exit_receipt'])['returncode']==0
  folder=Path(j['output']);r=read(folder/'path_audit.json');assert r['complete']
  if j['device']=='cpu':assert r['continuous_tree_updates']==20 and r['reload_prediction_exact']
  else:
   assert r['continuous_updates']==20 and r.get('resume_raw_optimizer_ema_rng_exact',r.get('resume_exact',False))
   grad=read(folder/'gradient_check.json');assert all(v>0 for v in grad['relative_update'].values())
  audits[j['id']]=sha(folder/'path_audit.json');reports.append(dict(id=j['id'],peak_reserved_mib=j['result']['peak_reserved_mib'],peak_rss_gib=j['result'].get('peak_rss_gib'),seconds=j['result']['seconds']))
 result=dict(complete=True,source_manifest_sha256=d['source_sha256'],audit_sha256=audits,projected_artifact_gib=s['projected_artifact_gib'],reports=reports,released_at=time.time())
 save(out/'formal_release.json',result)
elif d['mode']=='status':
 result=read(host/'state.json');exitfile=Path(result.get('active_scheduler_receipt',host/'start_exit.json'));result['exit_receipt']=read(exitfile) if exitfile.exists() else None
 repair=host/'finalization'
 if (repair/'observer_attempt.json').exists():
  recovery={n:read(repair/n) for n in ('observer_progress.json','observer_exit.json','exit.json') if (repair/n).exists()}
  owner=repair/'observer_exit_owner.json'
  if owner.exists():
   sys.path.insert(0,str(source))
   from scripts.tracehr_recovery_queue import same_process
   record=read(owner);recovery['observer_pid']=record['child_pid'];recovery['observer_alive']=same_process(record['child_pid'],record['child_identity'])
  result['execution_record_repair']=recovery
 for g in result['groups'].values():
  for j in g['jobs'].values():
   q=Path(j['output'])/'progress.json'
   if q.exists() and j['status'] in ('running','failed'):j['progress']=read(q)
   if j['status']=='failed':j['log_tail']=Path(j['log']).read_text(errors='replace')[-5000:]
else:raise ValueError(d['mode'])
print(json.dumps(result))
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['inventory','inspect','build','deploy','verify','register','start','status','release','storage-handoff'], required=True)
    parser.add_argument('--host', choices=['mimic', 'eicu'], required=True)
    parser.add_argument('--database',choices=['mimiciii','mimic','eicu'])
    parser.add_argument('--stage',choices=['lgbm','basic','hospital'])
    parser.add_argument('--snapshot',choices=['v1','v2'],default='v1')
    parser.add_argument('--expected-gib',type=float,default=4.)
    a = parser.parse_args()
    out = Path(os.environ['PROJECT_RUN_DIR']).resolve()
    assert out.is_relative_to(ROOT)
    _, project, _ = HOSTS[a.host]
    if a.mode not in ('inventory','inspect'):
        assert a.database and a.stage
        assert (a.database=='eicu')==(a.host=='eicu')
        if a.mode=='build':result=build(out,a.database,a.stage,a.expected_gib,a.snapshot)
        else:
            built=read(out/'build-receipt.json')
            assert built['contract']['expected_gib']==a.expected_gib
            payload=dict(project=project,host=a.host,run=RUN,mode=a.mode,database=a.database,stage=a.stage,
                source=f'work/{RUN}/{a.database}/{a.stage}_{a.snapshot}',output=f'runs/{RUN}/{a.database}/{a.stage}_{a.snapshot}/screen',
                source_sha256=built['source_manifest_sha256'],expected_gib=a.expected_gib)
            if a.mode=='deploy':payload.update(files=read(out/'source_bundle.json'),hashes=read(out/'source_manifest.json'),manifest_bytes=base64.b64encode((out/'source_manifest.json').read_bytes()).decode())
            if a.mode=='storage-handoff':payload['handoff_code']=base64.b64encode((ROOT/'scripts/continue_cem_icde72_queue.py').read_bytes()).decode()
            code=REMOTE.replace('PAYLOAD',repr(payload),1);compile(code,'remote_cem72','exec')
            result=ssh(a.host,code,timeout=900);save(out/(a.mode+'-receipt.json'),result)
        if a.mode=='status':
            compact={k:v for k,v in result.items() if k not in ('groups','gpu_inventory','idle_observations')}
            compact['groups']={k:{n:g.get(n) for n in ('status','phase','counts','probe_counts','evaluation_counts','projected_artifact_gib')} for k,g in result['groups'].items()}
            compact['active']=[{k:v for k,v in j.items() if k in ('id','status','progress','error','log_tail','gpu','started_at')} for g in result['groups'].values() for j in g['jobs'].values() if j['status'] in ('running','failed')]
            print(json.dumps(compact));return
        print(json.dumps(result));return
    payload={'project':project}
    if a.mode=='inspect':
        payload['contracts']={}
        for db in (['eicu'] if a.host=='eicu' else ['mimiciii','mimic']):
            c=read(ROOT/'runs'/PRIOR/f'local_{db}_controls_v1/build-receipt.json')['contract']
            b=read(ROOT/'runs'/PRIOR/f'local_{db}_tabular_v1/build-receipt.json')['contract']
            payload['contracts'][db]={**c,**b}
    result = ssh(a.host, (INVENTORY if a.mode=='inventory' else INSPECT).replace('PAYLOAD', repr(payload)), timeout=120)
    save(out/(a.mode+'-receipt.json'), result)
    print(json.dumps(result if a.mode=='inventory' else {'complete':True,'receipt':str(out/(a.mode+'-receipt.json')),'keys':list(result)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
