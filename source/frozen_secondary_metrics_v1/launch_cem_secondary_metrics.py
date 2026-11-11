"""Run the authorized secondary-metric analysis on existing CPU predictions."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.launch_ema_cross import HOSTS,ssh

RUN='cem_los_secondary_metrics_20261110'
DBS=('mimiciii','mimic','eicu')
FILES=('cem_secondary_metrics.py','cem_probability_intervals.py')


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2),encoding='utf8')


REMOTE=r'''
import base64,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
d=PAYLOAD;p=Path(d['project']).resolve(strict=True);mode=d['mode'];db=d['database']
source=p/'work'/d['run']/d['version'];out=p/'runs'/d['run']/db
tmp=p/'work'/d['run']/'tmp'/db;cache=p/'cache'/d['run']/db
for path in (source,out,tmp,cache):
 assert path.resolve().is_relative_to(p)
 path.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(p).free/2**30>10.25
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
 PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(tmp),TEMP=str(tmp),TMP=str(tmp),XDG_CACHE_HOME=str(cache))
if mode=='deploy':
 for name,b in d['files'].items():
  dest=source/name;content=base64.b64decode(b)
  assert not dest.exists() or dest.read_bytes()==content,name
  if not dest.exists():dest.write_bytes(content)
 result=dict(complete=True,source=str(source),output=str(out),temp=str(tmp),cache=str(cache),expected_gib=.25,
             project=str(p),free_gib=shutil.disk_usage(p).free/2**30,source_manifest_sha256=sha(source/'source_manifest.json'))
elif mode in ('self-test','pilot','analyze'):
 assert sha(source/'source_manifest.json')==d['source_manifest_sha256']
 for name,h in read(source/'source_manifest.json').items():assert sha(source/name)==h,name
 command=[sys.executable,'-B',str(source/'cem_secondary_metrics.py'),'--project',str(p),'--database',db,'--output',str(out),'--mode',mode]
 if mode in ('self-test','pilot'):
  r=subprocess.run(command,cwd=source,env=env,capture_output=True,text=True)
  assert r.returncode==0,r.stderr[-5000:]+r.stdout[-1000:]
  result=json.loads(r.stdout)
  if mode=='pilot':
   info=read(out/'pilot.json');result.update(estimated_seconds=info['seconds']*100,output_bytes=sum(f.stat().st_size for f in out.iterdir() if f.is_file()))
 else:
  assert (out/'pilot.json').exists() and not (out/'analyze_attempt.json').exists()
  # Existing project wrapper records process ownership and terminal exit status.
  state=read(p/'runs/cem_core_five_seed_20261107'/('host_eicu' if db=='eicu' else 'host_mimic')/'state.json')
  wrapper=Path(next(g['source'] for g in state['groups'].values() if g['database']==db))/'scripts/tracehr_recovery_attempt.py'
  spec=out/'analyze_attempt.json';receipt=out/'analyze_exit.json'
  spec.write_text(json.dumps(dict(command=command,cwd=str(source),receipt=str(receipt),job_id=d['run']+'_'+db,source_manifest_sha256=d['source_manifest_sha256']),indent=2))
  with (out/'analyze.log').open('x') as log:
   proc=subprocess.Popen([sys.executable,'-B',str(wrapper),'--spec',str(spec)],cwd=wrapper.parents[1],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  result=dict(started=True,wrapper_pid=proc.pid,output=str(out),gpu_training=False)
elif mode=='status':
 result=dict(database=db,complete=(out/'verified-results.json').exists(),progress=read(out/'progress.json') if (out/'progress.json').exists() else None,
             exit=read(out/'analyze_exit.json') if (out/'analyze_exit.json').exists() else None)
elif mode=='collect':
 exit_record=read(out/'analyze_exit.json');assert exit_record['returncode']==0
 result=read(out/'verified-results.json');assert result['complete'] and not result['pilot'] and result['replicates']==20000
 assert sha(out/'paired_draws.npz')==result['paired_draws_sha256']
 result=dict(result=result,pilot=read(out/'pilot.json'),draws=base64.b64encode((out/'paired_draws.npz').read_bytes()).decode(),exit=exit_record)
else:raise ValueError(mode)
print(json.dumps(result))
'''


def aggregate():
    results=[read(ROOT/'runs'/RUN/db/'verified-results.json') for db in DBS]
    rows=[v for r in results for v in r['paired']]
    assert len(rows)==24
    ordered=sorted(range(len(rows)),key=lambda i:rows[i]['p_bootstrap_centered'])
    maximum=0.
    for rank,i in enumerate(ordered):
        maximum=max(maximum,(24-rank)*rows[i]['p_bootstrap_centered'])
        rows[i]['p_holm']=min(1.,maximum)
    result=dict(complete=True,seeds=[48,49,50,51,52],family_count=24,replicates=20000,paired=rows,
                summaries={r['database']:r['summaries'] for r in results},rows=[v for r in results for v in r['rows']],
                selected_baselines={r['database']:r['selected_baselines'] for r in results},
                inference=results[0]['inference'],model_fits=0,model_inference=0,calibrator_fits=0,
                original_results_unchanged=True,source_results={db:sha(ROOT/'runs'/RUN/db/'verified-results.json') for db in DBS})
    dest=ROOT/'runs'/RUN/'paired_results_v1/verified-results.json'
    assert not dest.exists()
    save(dest,result)
    print(json.dumps(dict(complete=True,contrasts=len(rows),output=str(dest))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('build','deploy','self-test','pilot','analyze','status','collect','aggregate'),required=True)
    p.add_argument('--database',choices=DBS,default='mimiciii')
    p.add_argument('--version',default='source_v2');a=p.parse_args()
    assert a.version in ('source_v1','source_v2')
    local=ROOT/'runs'/RUN/a.version
    if a.mode=='build':
        local.mkdir(parents=True,exist_ok=True)
        for name in FILES:
            content=(ROOT/'scripts'/name).read_bytes();dest=local/name
            assert not dest.exists() or dest.read_bytes()==content
            if not dest.exists():dest.write_bytes(content)
        save(local/'source_manifest.json',{n:sha(local/n) for n in FILES})
        print(json.dumps(dict(complete=True,source=str(local))))
    elif a.mode=='aggregate':aggregate()
    else:
        host='eicu' if a.database=='eicu' else 'mimic'
        payload=dict(mode=a.mode,database=a.database,project=HOSTS[host][1],run=RUN,version=a.version,source_manifest_sha256=sha(local/'source_manifest.json'))
        if a.mode=='deploy':payload['files']={name:base64.b64encode((local/name).read_bytes()).decode() for name in (*FILES,'source_manifest.json')}
        value=ssh(host,'PAYLOAD='+repr(payload)+'\n'+REMOTE,timeout=180)
        out=ROOT/'runs'/RUN/a.database
        if a.mode=='collect':
            save(out/'verified-results.json',value['result']);save(out/'pilot.json',value['pilot']);save(out/'analyze_exit.json',value['exit'])
            (out/'paired_draws.npz').write_bytes(base64.b64decode(value['draws']))
            assert sha(out/'paired_draws.npz')==value['result']['paired_draws_sha256']
            print(json.dumps(dict(complete=True,database=a.database,output=str(out))))
        else:
            save(out/(a.mode+'-receipt.json'),value)
            print(json.dumps(value))
