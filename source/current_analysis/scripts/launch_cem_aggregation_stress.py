"""Deploy, inspect and collect only the approved fixed-weight aggregation test."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.launch_ema_cross import HOSTS, ssh

RUN = 'cem_aggregation_stress_20261106'
DBS = ('mimiciii', 'mimic', 'eicu')
PROTOCOL = dict(run=RUN, seeds=[48,49,50,51,52], databases=list(DBS), endpoints=[0,1],
    conditions=['lab_drop25','lab_drop50','lab_delay2h','lab_delay6h'],
    methods=['duration','window'], training_runs=0, checkpoints=60,
    total_stress_cases=240, expected_reused_cases=72, expected_new_cases=168,
    primary='difference of AP changes: (duration perturbed-clean) minus (window perturbed-clean)',
    secondary='perturbed AP duration minus window', family_size=24, families=['degradation','absolute'],
    bootstrap_replicates=20000, bootstrap_rng_seed=20261027, resampling='whole patients with all stays',
    seed_aggregation='per-seed AP/paired contrast, then average; no prediction pooling',
    intervals=[.95, 1-.05/24], p_values='null-centered paired bootstrap; Holm separately in each full 24-contrast family',
    fixed_deletion='reuse archived deterministic hash; nested 25/50 percent masks shared across views and models',
    normalization='reuse each model original training-fitted transforms; never refit',
    time_rules='original laboratory availability, 24h cutoff; unchanged specimen times and original MIMIC-III proxy',
    execution='CPU FP32, four threads per batch, batch 256, four independent batches in parallel; exact serial/parallel replay',
    original_model_results_and_manuscript_unchanged=True)


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name+'.writing'); q.write_text(json.dumps(value, indent=2)); os.replace(q,p)


REMOTE = r'''
import base64,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
d=PAYLOAD;p=Path(d['project']).resolve(strict=True);db=d['database'];run=d['run']
source=p/'work'/run/db/d['version'];out=p/'runs'/run/db/d['version'];cache=p/'cache'/run/db;tmp=p/'work'/run/'tmp'/db
matched=p/'work/cem_matched_controls_20261105'/db/'v4'
def read(f):return json.loads(Path(f).read_text())
def sha(f):
 h=hashlib.sha256()
 with Path(f).open('rb') as s:
  for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(f,x):
 f.parent.mkdir(parents=True,exist_ok=True);q=f.with_name(f.name+'.writing');q.write_text(json.dumps(x,indent=2));os.replace(q,f)
for path in (source,out,cache,tmp):assert path.resolve().is_relative_to(p) and path!=p
if d['mode']=='deploy':
 assert not source.exists() and not out.exists(),'Existing run: inspect, never launch a second writer'
 assert shutil.disk_usage(p).free>(10+d['expected_gib'])*2**30
 mem=dict((line.split(':')[0],int(line.split()[1])) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line and line.split()[1].isdigit())
 assert mem['MemAvailable']>24*2**20
 for path in (source,out,cache,tmp):path.mkdir(parents=True,exist_ok=True)
 for name,data in d['files'].items():
  f=source/name;assert f.resolve().is_relative_to(source);f.write_bytes(base64.b64decode(data));assert sha(f)==d['hashes'][name]
 save(source/'source_manifest.json',d['hashes'])
 cfg=read(matched/'work/cem_matched_controls_20261105/config.json')
 inputs=[matched,p/cfg['parent']['input_preparation'],Path(cfg['parent']['raw_inputs']),p/cfg['parent']['test_inputs'],p/cfg['window_preparation'],p/'runs/cem_icde_evidence_20261028'/db/'stress_inputs_v2',p/'cache/cem_evidence_completion_20261025/native_observations'/db]
 assert all(x.exists() and x.resolve().is_relative_to(p) for x in inputs)
 result=dict(complete=True,project=str(p),source=str(source),output=str(out),cache=str(cache),temp=str(tmp),
  input_paths=[str(x.resolve()) for x in inputs],expected_gib=d['expected_gib'],free_gib=shutil.disk_usage(p).free/2**30,
  available_memory_gib=mem['MemAvailable']/2**20,cpus=os.cpu_count(),load_average=os.getloadavg(),
  source_manifest_sha256=sha(source/'source_manifest.json'),scope='resolved declared paths; fcntl writer lock; no arbitrary-write sandbox')
 save(out/'storage.json',result)
elif d['mode']=='start':
 assert not (out/'attempt.json').exists() and not (out/'exit.json').exists()
 for name,h in read(source/'source_manifest.json').items():assert sha(source/name)==h
 dep=p/'work/tracehr_tabular_20261004/deps'
 env=dict(os.environ,TRACEHR_PROJECT=str(p),TRACEHR_DATABASE=db,TRACEHR_TABULAR_DEPS=str(dep),
  PYTHONPATH=os.pathsep.join(map(str,[matched,dep,*dep.glob('*.whl'),p/'work/cem_evidence_completion_20261025/test_runtime_v1/deps'])),
  PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='',
  NUMPY_MADVISE_HUGEPAGE='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='1',
  TMPDIR=str(tmp),TMP=str(tmp),TEMP=str(tmp),XDG_CACHE_HOME=str(cache/'runtime'),TORCH_HOME=str(cache/'torch'),
  MPLCONFIGDIR=str(cache/'matplotlib'),WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 command=[sys.executable,'-B',str(matched/'work/tracehr_group_v1_validation_20261023/memory_entry.py'),str(source/'worker.py'),
  '--project',str(p),'--database',db,'--source',str(matched),'--output',str(out),'--cache',str(cache),'--temp',str(tmp),'--expected-gib',str(d['expected_gib'])]
 if d.get('resume_version'):command+=['--resume-from',str(p/'runs'/run/db/d['resume_version'])]
 spec=dict(command=command,cwd=str(source),receipt=str(out/'exit.json'),job_id=run+'_'+db,
  source_manifest_sha256=sha(source/'source_manifest.json'))
 save(out/'attempt.json',spec)
 with (out/'run.log').open('x') as log:
  child=subprocess.Popen([sys.executable,'-B',str(matched/'scripts/tracehr_recovery_attempt.py'),'--spec',str(out/'attempt.json')],cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 result=dict(started=True,wrapper_pid=child.pid,attempt=str(out/'attempt.json'),output=str(out),training_runs=0)
 save(out/'launch.json',result)
elif d['mode']=='status':
 result=dict(database=db,output=str(out),exists=out.exists())
 for name in ('exit_owner.json','exit.json','progress.json','duration_cpu_pilot.json','window_cpu_pilot.json','parallel_pilot.json','completion.json'):
  if (out/name).exists():result[name]=read(out/name)
 if (out/'run.log').exists():result['log_tail']=(out/'run.log').read_text(errors='replace')[-3500:]
 if (out/'predictions').exists():
  completed=[read(x) for x in (out/'predictions').glob('*/completion.json')]
  result['completed_models']=len(completed)
  result['new_inferences']=sum(x['new_cases'] for x in completed)
  result['reused_inferences']=sum(x['reused_cases'] for x in completed)
elif d['mode']=='collect':
 assert read(out/'exit.json')['returncode']==0 and read(out/'completion.json')['complete']
 receipt=read(out/'completion.json')
 for name,h in receipt['files'].items():assert sha(out/name)==h
 allowed=['verified-results.json','paired_draws.npz','completion.json','exit.json','storage.json']
 allowed += [n for n in ('duration_cpu_pilot.json','window_cpu_pilot.json','parallel_pilot.json','reused_preparation.json') if (out/n).exists()]
 result=dict(complete=True,files={n:base64.b64encode((out/n).read_bytes()).decode() for n in allowed},hashes={n:sha(out/n) for n in allowed})
else:raise ValueError(d['mode'])
print(json.dumps(result))
'''


def aggregate(base):
    import numpy as np
    import importlib.util
    spec=importlib.util.spec_from_file_location('aggregate_worker',ROOT/'scripts/cem_aggregation_stress.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    ns=dict(np=np)
    worker.load_functions(ROOT/'scripts/cem_icde72_significance.py',('centered_p','holm'),ns)
    results=[];draws={};inputs={}
    for db in DBS:
        folder=base/(db+'_results_v1');receipt=read(folder/'completion.json')
        assert read(folder/'exit.json')['returncode']==0
        for n,h in receipt['files'].items():assert sha(folder/n)==h;inputs[str(folder/n)]=h
        r=read(folder/'verified-results.json');assert r['complete'] and r['models']==20 and r['new_inferences']==56 and r['reused_inferences']==24
        results.append(r)
        with np.load(folder/'paired_draws.npz') as a:
            for k in a.files:draws[f'{db}:{k}']=a[k]
    paired=[dict(row) for r in results for row in r['paired']]
    for family in ('degradation','absolute'):
        subset=[r for r in paired if r['estimand']==family];assert len(subset)==24
        for row,p in zip(subset,ns['holm']([r['p_bootstrap_centered'] for r in subset])):
            row['p_holm']=p
            values=draws[f"{row['database']}:{row['estimand']}:{row['endpoint']}:{row['condition']}"]
            np.testing.assert_array_equal(np.quantile(values,[.05/48,1-.05/48]),row['adjusted_ci'])
            assert ns['centered_p'](values,row['mean'])==row['p_bootstrap_centered']
    out=base/'paired_results_v1';out.mkdir(exist_ok=True)
    result=dict(complete=True,protocol=PROTOCOL,new_inferences=168,reused_inferences=72,training_runs=0,
        rows=[row for r in results for row in r['rows']],paired=paired,inputs=inputs,
        original_files_unchanged=all(r['original_files_unchanged'] for r in results),
        cpu_replay_max_probability_drift=max(r['cpu_replay_max_probability_drift'] for r in results),
        cpu_replay_max_AP_drift=max(r['cpu_replay_max_AP_drift'] for r in results))
    if (out/'completion.json').exists():assert read(out/'verified-results.json')==result
    else:
        save(out/'verified-results.json',result)
        with (out/'paired_draws.npz').open('xb') as f:np.savez_compressed(f,**draws)
        import csv
        export=[]
        for cell in result['rows']:
            row={k:cell[k] for k in ('database','endpoint','condition')}
            row['outcome']='mortality' if cell['endpoint']==0 else 'LOS'
            for method in ('duration','window'):
                for setting in ('clean','perturbed','change'):
                    for stat in ('mean','sample_SD'):
                        row[f'{method}_{setting}_{stat}']=cell['methods'][method][setting][stat]
            for estimand in ('degradation','absolute'):
                pair=next(x for x in paired if all(x[k]==cell[k] for k in ('database','endpoint','condition')) and x['estimand']==estimand)
                for stat in ('mean','sample_SD','p_bootstrap_centered','p_holm'):row[f'{estimand}_{stat}']=pair[stat]
                for interval in ('ci95','adjusted_ci'):
                    row[f'{estimand}_{interval}_low'],row[f'{estimand}_{interval}_high']=pair[interval]
            export.append(row)
        with (out/'comparisons.csv').open('x',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(export[0]));writer.writeheader();writer.writerows(export)
        save(out/'completion.json',dict(complete=True,files={n:sha(out/n) for n in ('verified-results.json','paired_draws.npz','comparisons.csv')}))
    return dict(complete=True,new_inferences=168,reused_inferences=72,families=2,contrasts=48)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('deploy','start','status','collect','aggregate'),required=True)
    p.add_argument('--database',choices=DBS);p.add_argument('--version',default='v1');p.add_argument('--expected-gib',type=float,default=2)
    p.add_argument('--resume-version')
    a=p.parse_args();base=Path(os.environ['PROJECT_RUN_DIR']).resolve()
    assert base==ROOT/'runs'/RUN
    if a.mode=='aggregate':print(json.dumps(aggregate(base)));return
    assert a.database
    host='eicu' if a.database=='eicu' else 'mimic'
    payload=dict(project=HOSTS[host][1],database=a.database,mode=a.mode,run=RUN,version=a.version,expected_gib=a.expected_gib,resume_version=a.resume_version)
    if a.mode=='deploy':
        mapping={'worker.py':'cem_aggregation_stress.py','prior_statistics.py':'cem_icde72_statistics.py','prior_significance.py':'cem_icde72_significance.py'}
        files={n:(ROOT/'scripts'/name).read_bytes() for n,name in mapping.items()}
        for n,b in files.items():compile(b,n,'exec')
        files['protocol.json']=json.dumps(PROTOCOL,indent=2).encode()
        payload['files']={n:base64.b64encode(b).decode() for n,b in files.items()}
        payload['hashes']={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    result=ssh(host,REMOTE.replace('PAYLOAD',repr(payload),1),timeout=55)
    if a.mode=='collect':
        out=base/(a.database+'_results_v1');out.mkdir(exist_ok=True)
        for name,data in result['files'].items():
            content=base64.b64decode(data);assert hashlib.sha256(content).hexdigest()==result['hashes'][name]
            target=out/name
            if target.exists():assert target.read_bytes()==content
            else:target.write_bytes(content)
        save(out/'collection-receipt.json',dict(complete=True,hashes=result['hashes']))
        result=dict(complete=True,database=a.database,files=len(result['files']))
    elif a.mode!='status':save(base/('local_'+a.database+'_'+a.version)/(a.mode+'-receipt.json'),result)
    print(json.dumps(result))


if __name__=='__main__':main()
