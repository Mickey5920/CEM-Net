"""Freeze and launch only the approved 60 matched-control fits."""
import argparse
import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.launch_ema_cross import HOSTS, ssh
from scripts.launch_cem_icde72 import read, save

RUN = 'cem_matched_controls_20261105'
ADDED = ('models/cem_matched_tabm.py', 'scripts/cem_matched_windows.py', 'scripts/cem_matched_common.py',
         'scripts/train_cem_matched_controls.py', 'scripts/cem_matched_queue.py', 'scripts/prepare_cem_matched_controls.py',
         'scripts/tracehr_value_features.py', 'scripts/cem_icde72_queue.py')


def build(a, out):
    parent = ROOT/'runs/cem_five_seed_20261102'/f'local_{a.database}_v3'
    files = read(parent/'source_bundle.json'); previous = read(parent/'source_manifest.json')
    inherited = read(parent/'build-receipt.json')['contract']
    c = dict(run=RUN, database=a.database, parent=inherited, expected_gib=a.expected_gib,
             window_preparation=f'runs/{RUN}/{a.database}/{a.snapshot}/windows',
             snapshot=a.snapshot, seeds=[48, 49, 50, 51, 52], variants=['matched_tabm', 'window_summary'],
             matrix=[dict(variant=v, seed=s, endpoint=e) for s in (48,49,50,51,52) for e in (0,1) for v in ('matched_tabm','window_summary')],
             epoch_budgets={'mimiciii': {'0': [120]*5, '1': [52,120,52,120,52]}, 'mimic': {'0': [52]*5, '1': [52]*5}, 'eicu': {'0': [256]*5, '1': [256]*5}},
             inference=dict(primary='AP difference: original CEM minus each control', contrasts=12,
                            seeds=[48,49,50,51,52], patient_cluster_bootstrap_replicates=20000,
                            averaging='per-seed metric differences, then mean', confidence=[.95, 1-.05/12],
                            multiplicity='one new 12-comparison family; Holm p and Bonferroni simultaneous intervals',
                            secondary=['AUROC','Brier','logloss']),
             authorization='two fixed controls; six tasks; five seeds; exactly 60 new fits across three databases; original scores immutable',
             protocol=dict(fresh_initialization=True, no_hyperparameter_search=True, fixed_original_response_mask=True,
                           batch=256, microbatch=128, evaluation_batch=256, precision='FP32',
                           selection='EMA validation AP inside fixed cell budget', schedule_reference_epochs=256,
                           original_model_unchanged=True, original_results_unchanged=True,
                           no_extra_ablation_or_transfer=True))
    db = a.database
    if db == 'mimiciii':
        normalization = ROOT/'runs/mimiciii_adaptation_20261024/v1/preparation/normalization.json'
        keys = read(normalization.parent/'completion.json')['lab_keys']
        inputs = [(ROOT/'runs/mimiciii_adaptation_20261024/v1/events/numeric.parquet',
                   f'cache/cem_evidence_completion_20261025/native_observations/{db}/numeric.parquet', ['train', 'dev', 'test'])]
    else:
        normalization = ROOT/'runs'/f'tracehr_values_full_{db}_20261003'/'normalization.json'
        keys = read(normalization.parent/'completion.json')['keys']
        train_remote = f'cache/{RUN}/measurements/{db}/train_dev.parquet' if db == 'mimic' else 'cache/cem_icde_evidence_20261028/hospital_inputs_v1/numeric.parquet'
        inputs = [(ROOT/'runs'/f'tracehr_numeric_full_{db}_20261003'/'events.parquet', train_remote, ['train', 'dev']),
                  (ROOT/'runs/tracehr_pm_test_20261021'/f'{db}_features/numeric/events.parquet',
                   f'cache/cem_evidence_completion_20261025/native_observations/{db}/numeric.parquet', ['test'])]
    c['measurements'] = dict(keys=keys, normalization=read(normalization),
                              normalization_sha256=hashlib.sha256(normalization.read_bytes()).hexdigest(),
                              sources=[dict(path=remote, local_path=str(local), splits=splits, bytes=local.stat().st_size,
                                            sha256=hashlib.sha256(local.read_bytes()).hexdigest()) for local, remote, splits in inputs])
    if a.snapshot == 'v4':
        c['window_preparation'] = f'runs/{RUN}/{db}/v3/windows'
    if a.snapshot != 'v1':
        c['previous_execution'] = dict(snapshots=['v1', 'v2', 'v3'], change='complete queue dependency bundle; reuse verified v3 preparation and unchanged scientific code',
                                       optimizer_updates=0, formal_fits_started=0, original_failed_attempts_preserved=True)
    for name in ADDED:
        assert name not in files
        content = (ROOT/name).read_bytes(); compile(content, name, 'exec')
        files[name] = base64.b64encode(content).decode()
    files[f'work/{RUN}/config.json'] = base64.b64encode(json.dumps(c, indent=2).encode()).decode()
    manifest = {n: hashlib.sha256(base64.b64decode(b)).hexdigest() for n, b in files.items()}
    assert all(manifest[n] == h for n, h in previous.items())
    assert not (out/'source_manifest.json').exists()
    save(out/'source_bundle.json', files); save(out/'source_manifest.json', manifest)
    result = dict(complete=True, source_manifest_sha256=hashlib.sha256((out/'source_manifest.json').read_bytes()).hexdigest(),
                  files=len(files), inherited_files=len(previous), contract=c)
    save(out/'build-receipt.json', result)
    return {k: v for k, v in result.items() if k != 'contract'}


REMOTE = r'''
import base64,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
d=PAYLOAD;p=Path(d['project']).resolve(strict=True);source=p/d['source'];base=p/d['base'];out=base/'screen';host=p/'runs'/d['run']/('host_'+d['host'])
def read(f):return json.loads(Path(f).read_text())
def sha(f):
 h=hashlib.sha256()
 with Path(f).open('rb') as s:
  for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(f,x):
 f.parent.mkdir(parents=True,exist_ok=True);q=f.with_name(f.name+'.writing');q.write_text(json.dumps(x,indent=2));os.replace(q,f)
def env():
 dep=p/'work/tracehr_tabular_20261004/deps';tmp=p/'work'/d['run']/'tmp'/d['database'];cache=p/'cache'/d['run']/d['database']
 for path in (tmp,cache):path.mkdir(parents=True,exist_ok=True);assert path.resolve().is_relative_to(p)
 paths=[source,dep,*dep.glob('*.whl'),p/'work/cem_evidence_completion_20261025/test_runtime_v1/deps']
 return dict(os.environ,TRACEHR_PROJECT=str(p),TRACEHR_DATABASE=d['database'],TRACEHR_TABULAR_DEPS=str(dep),
  PYTHONPATH=os.pathsep.join(map(str,paths)),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',
  NUMPY_MADVISE_HUGEPAGE='0',TMPDIR=str(tmp),TEMP=str(tmp),TMP=str(tmp),XDG_CACHE_HOME=str(cache/'runtime'),
  TORCH_HOME=str(cache/'torch'),MPLCONFIGDIR=str(cache/'matplotlib'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
  OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
def launch(label,command,folder):
 folder.mkdir(parents=True,exist_ok=True)
 spec=folder/(label+'_attempt.json');receipt=folder/(label+'_exit.json');log=folder/(label+'.log')
 assert not spec.exists() and not receipt.exists() and not log.exists()
 save(spec,dict(command=command,cwd=str(source),receipt=str(receipt),job_id=label,source_manifest_sha256=d['source_sha256']))
 with log.open('x') as f:
  proc=subprocess.Popen([sys.executable,'-B',str(source/'scripts/tracehr_recovery_attempt.py'),'--spec',str(spec)],cwd=source,env=env(),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 return dict(pid=proc.pid,spec=str(spec),exit_receipt=str(receipt),log=str(log))
for path in (source,base,out,host):assert path.resolve().is_relative_to(p) and path!=p
if d['mode']=='deploy':
 assert not source.exists() and not base.exists()
 assert shutil.disk_usage(p).free/2**30>d['expected_gib']+10
 source.mkdir(parents=True);base.mkdir(parents=True)
 for name,b64 in d['files'].items():
  f=source/name;assert f.resolve().is_relative_to(source);f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(base64.b64decode(b64));assert sha(f)==d['hashes'][name]
 (source/'source_manifest.json').write_bytes(base64.b64decode(d['manifest_bytes']))
 assert sha(source/'source_manifest.json')==d['source_sha256']
 runtime=env()
 save(base/'storage.json',dict(project=str(p),source=str(source),outputs=[str(base),str(host)],cache=runtime['XDG_CACHE_HOME'],temp=runtime['TMPDIR'],expected_gib=d['expected_gib'],free_gib=shutil.disk_usage(p).free/2**30))
 result=dict(complete=True,source=str(source),files=len(d['files']),storage=read(base/'storage.json'))
elif d['mode']=='stage-inputs':
 result={}
 for item in d['measurements']['sources']:
  f=p/item['path'];assert f.resolve().is_relative_to(p)
  if f.exists():assert f.stat().st_size==item['bytes'] and sha(f)==item['sha256'];result[item['path']]=True
  else:f.parent.mkdir(parents=True,exist_ok=True);assert shutil.disk_usage(p).free>10*2**30+item['bytes'];result[item['path']]=False
elif d['mode']=='prepare':
 assert sha(source/'source_manifest.json')==d['source_sha256']
 command=[sys.executable,'-B',str(source/'work/tracehr_group_v1_validation_20261023/memory_entry.py'),str(source/'scripts/prepare_cem_matched_controls.py'),'--project',str(p),'--output',str(base/'preflight')]
 result=launch('prepare_'+d['database'],command,base)
elif d['mode']=='adopt-preflight':
 previous=p/'runs'/d['run']/d['database']/'v3';oldsource=p/'work'/d['run']/d['database']/'v3'
 v=read(previous/'verification.json');assert v['complete'] and read(previous/('prepare_'+d['database']+'_exit.json'))['returncode']==0
 old=read(oldsource/'source_manifest.json');current=read(source/'source_manifest.json')
 for name,h in old.items():
  if name!='work/'+d['run']+'/config.json':assert current[name]==h and sha(source/name)==h,name
 c=read(source/('work/'+d['run']+'/config.json'));oc=read(oldsource/('work/'+d['run']+'/config.json'))
 for name in ['parent','measurements','matrix','epoch_budgets','inference','protocol','window_preparation']:assert c[name]==oc[name],name
 assert sha(p/c['window_preparation']/'completion.json')==v['windows_sha256']
 result=dict(v,source_manifest_sha256=d['source_sha256'],preflight_reuse=dict(path=str(previous/'verification.json'),sha256=sha(previous/'verification.json'),reason='only queue dependency added; scientific code and inputs identical'))
 save(base/'verification.json',result);save(base/'preflight_reuse.json',result['preflight_reuse'])
elif d['mode']=='register':
 r=read(base/('prepare_'+d['database']+'_exit.json')) if not (base/'preflight_reuse.json').exists() else dict(returncode=0)
 v=read(base/'verification.json');assert r['returncode']==0 and v['complete'] and v['source_manifest_sha256']==d['source_sha256']
 c=read(source/('work/'+d['run']+'/config.json'));file=host/'ready'/(d['database']+'_matched.json')
 previous_ready=None
 if file.exists():
  assert not (host/'state.json').exists() and read(host/'scheduler_exit.json')['returncode']!=0
  previous_ready=sha(file)
 result=dict(database=d['database'],stage='matched',stage_order=1,source=str(source),output=str(out),source_manifest_sha256=d['source_sha256'],
  matrix=c['matrix'],variants=c['variants'],entries={k:'scripts/train_cem_matched_controls.py' for k in ('tree','cem','tabular')},
  expected_gib=d['expected_gib'],retention_multiplier=3,min_available_ram_gib=24,cpu_min_available_ram_gib=24,probe_gpu_mib=16384,
  probe_requirement_basis='fixed 128 microbatch, FP32; bounded measured probe required')
 if previous_ready:result['superseded_ready_sha256']=previous_ready
 save(file,result)
elif d['mode']=='start':
 assert not (host/'state.json').exists()
 command=[sys.executable,'-B',str(source/'scripts/cem_matched_queue.py'),'--project',str(p),'--host',d['host']]
 result=launch('scheduler_'+d['snapshot'],command,host)
elif d['mode']=='release':
 s=read(out/'state.json');assert s['status']=='awaiting_path_review' and not (out/'formal_release.json').exists()
 assert s['counts']['pending']==20 and s['projected_artifact_gib']<d['expected_gib']
 audits={};reports=[]
 for j in s['jobs'].values():
  if j['action']!='probe':continue
  assert j['status']=='completed' and read(j['exit_receipt'])['returncode']==0
  folder=Path(j['output']);a=read(folder/'path_audit.json');assert a['complete'] and a['continuous_updates']==20 and a['resume_raw_optimizer_ema_rng_exact']
  assert a['all_parameters_updated'] and a['evaluation_preserves_raw_and_rng'] and not j['result']['test_loaded']
  audits[j['id']]=sha(folder/'path_audit.json');reports.append(dict(id=j['id'],peak_reserved_mib=j['result']['peak_reserved_mib'],median_step_seconds=j['result']['median_step_seconds']))
 assert len(audits)==4
 result=dict(complete=True,source_manifest_sha256=d['source_sha256'],audit_sha256=audits,projected_artifact_gib=s['projected_artifact_gib'],reports=reports,released_at=time.time())
 save(out/'formal_release.json',result)
elif d['mode']=='status':
 result=dict(database=d['database'],preflight_complete=(base/'verification.json').exists())
 pe=base/('prepare_'+d['database']+'_exit.json')
 if pe.exists():result['prepare_exit']=read(pe)
 else:
  pr=base/'preflight/progress.json'
  if pr.exists():result['prepare_progress']=read(pr)
 if not result['preflight_complete']:
  log=base/('prepare_'+d['database']+'.log')
  if log.exists():result['prepare_log_tail']=log.read_text(errors='replace')[-3500:]
 if (out/'state.json').exists():
  s=read(out/'state.json');result.update({k:s.get(k) for k in ['status','phase','counts','probe_counts','evaluation_counts','projected_artifact_gib']})
  result['active']=[]
  for j in s['jobs'].values():
   if j['status'] not in ('running','failed'):continue
   v={k:j.get(k) for k in ('id','status','gpu','error')};pr=Path(j['output'])/'progress.json'
   if pr.exists():v['progress']=read(pr)
   if j['status']=='failed':v['log_tail']=Path(j['log']).read_text(errors='replace')[-3500:]
   result['active'].append(v)
 if (host/'state.json').exists():
  h=read(host/'state.json');result['host_status']=h['status'];result['admission_wait']=h.get('admission_wait')
else:raise ValueError(d['mode'])
print(json.dumps(result))
'''


def main():
    p = argparse.ArgumentParser(); p.add_argument('--mode', required=True, choices=('build', 'deploy', 'stage-inputs', 'prepare', 'adopt-preflight', 'register', 'start', 'release', 'status'))
    p.add_argument('--database', required=True, choices=('mimiciii', 'mimic', 'eicu'))
    p.add_argument('--expected-gib', type=float, default=32); p.add_argument('--snapshot', default='v1')
    a = p.parse_args(); out = Path(os.environ['PROJECT_RUN_DIR']).resolve()
    assert out.is_relative_to(ROOT)
    if a.mode == 'build':
        result = build(a, out)
    else:
        built = read(out/'build-receipt.json'); assert built['contract']['expected_gib'] == a.expected_gib
        host = 'eicu' if a.database == 'eicu' else 'mimic'
        d = dict(project=HOSTS[host][1], host=host, run=RUN, mode=a.mode, database=a.database,
                 source=f'work/{RUN}/{a.database}/{a.snapshot}', base=f'runs/{RUN}/{a.database}/{a.snapshot}',
                 source_sha256=built['source_manifest_sha256'], expected_gib=a.expected_gib, snapshot=a.snapshot)
        if a.mode == 'deploy':
            d.update(files=read(out/'source_bundle.json'), hashes=read(out/'source_manifest.json'),
                     manifest_bytes=base64.b64encode((out/'source_manifest.json').read_bytes()).decode())
        if a.mode == 'stage-inputs':
            d['measurements'] = built['contract']['measurements']
        result = ssh(host, REMOTE.replace('PAYLOAD', repr(d), 1), 120)
        if a.mode == 'stage-inputs':
            for item in d['measurements']['sources']:
                if not result[item['path']]:
                    destination = HOSTS[host][1]+'/'+item['path']
                    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', item['local_path'], HOSTS[host][0]+':'+destination+'.part'], check=True, timeout=180)
                    promote = ('import json,hashlib,os;from pathlib import Path;'
                               f'p=Path({destination!r});q=Path({(destination+".part")!r});'
                               f'assert not p.exists() and q.stat().st_size=={item["bytes"]};'
                               f'assert hashlib.sha256(q.read_bytes()).hexdigest()=={item["sha256"]!r};'
                               'os.replace(q,p);print(json.dumps({"complete":True}))')
                    ssh(host, promote, 55)
            result = ssh(host, REMOTE.replace('PAYLOAD', repr(d), 1), 120)
            assert all(result.values())
        if a.mode != 'status':
            save(out/(a.mode+'-receipt.json'), result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
