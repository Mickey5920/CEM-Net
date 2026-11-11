"""Prepare and collect independent fixed-prediction baseline statistics.

No training entry point is exposed. Start requires both completed host queues;
same-host statistical writers are serialized. Downloads contain summaries only.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.launch_ema_cross import HOSTS, ssh
from scripts import cem_replacement_statistics as analysis

RUN = analysis.RUN
FILES = {'analysis.py':'cem_replacement_statistics.py',
         'prior_statistics.py':'cem_icde72_statistics.py',
         'prior_significance.py':'cem_icde72_significance.py'}
ALLOWED = ('verified-results.json','RESULTS.md')


def bundle():
    files = {name:(ROOT/'scripts'/relative).read_bytes() for name,relative in FILES.items()}
    for name,content in files.items():
        compile(content,name,'exec')
    return files, {name:hashlib.sha256(content).hexdigest() for name,content in files.items()}


def host_readiness(host):
    code = r'''
import hashlib,json
from pathlib import Path
d=PAYLOAD;p=Path(d['project']);folder=p/'runs'/d['run']/('host_'+d['host'])
state=json.loads((folder/'state.json').read_text())
exitfile=Path(state.get('active_scheduler_receipt',folder/'start_exit.json'))
exitrecord=json.loads(exitfile.read_text()) if exitfile.exists() else None
groups={}
for g in state['groups'].values():
 counts={action:sum(j['action']==action and j['status']=='completed' for j in g['jobs'].values()) for action in ('train','evaluate')}
 groups[g['database']]=dict(status=g['status'],**counts)
ready=(set(groups)==set(d['databases']) and exitrecord is not None and exitrecord['returncode']==0
       and all(g['status']=='complete' and g['train']==g['evaluate']==12 for g in groups.values()))
print(json.dumps(dict(ready=ready,groups=groups,exit_path=str(exitfile),returncode=exitrecord['returncode'] if exitrecord else None,
 exit_sha256=hashlib.sha256(exitfile.read_bytes()).hexdigest() if exitrecord else None)))
'''.replace('PAYLOAD',repr(dict(project=HOSTS[host][1],host=host,run=RUN,
                               databases=['eicu'] if host=='eicu' else ['mimiciii','mimic'])))
    return ssh(host,code,timeout=120)


REMOTE = r'''
import base64,fcntl,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
d=PAYLOAD;p=Path(d['project']).resolve(strict=True)
source=p/'work'/d['run']/('statistics_'+d['db']+'_v1')
out=p/'runs'/d['run']/(d['db']+'_statistics_v1')
temp=p/'work'/d['run']/'tmp_statistics';cache=p/'cache'/d['run']
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
receipt=out/'exit.json';logfile=out/'analysis.log'
if d['mode'] in ('check','audit','start'):
 for f in (source,out,temp,cache):
  f.mkdir(parents=True,exist_ok=True);assert f.resolve().is_relative_to(p)
 assert shutil.disk_usage(p).free>10.5*2**30
 for name,encoded in d['files'].items():
  content=base64.b64decode(encoded);f=source/name
  assert f.resolve().is_relative_to(source.resolve()) and hashlib.sha256(content).hexdigest()==d['manifest'][name]
  assert not f.exists() or f.read_bytes()==content,'Frozen analysis source changed'
  if not f.exists():
   with f.open('xb') as stream:stream.write(content)
 manifest=source/'source_manifest.json'
 if manifest.exists():assert read(manifest)==d['manifest']
 else:
  with manifest.open('x') as stream:json.dump(d['manifest'],stream,indent=2)
 deps=p/'work/tracehr_tabular_20261004/deps'
 env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[deps,*deps.glob('*.whl'),p/'work/cem_evidence_completion_20261025/test_runtime_v1/deps'])),
  OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='1',
  PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',NUMPY_MADVISE_HUGEPAGE='0',CUDA_VISIBLE_DEVICES='',
  TMPDIR=str(temp),TEMP=str(temp),TMP=str(temp),XDG_CACHE_HOME=str(cache))
 command=[sys.executable,'-B',str(source/'analysis.py'),'--project',str(p),'--database',d['db']]
 if d['mode'] in ('check','audit'):
  command+=['--self-test' if d['mode']=='check' else '--audit-only']
  run=subprocess.run(command,env=env,text=True,capture_output=True,timeout=600)
  assert run.returncode==0,run.stderr[-6000:]
  result=json.loads(run.stdout)
 else:
  assert all(r['ready'] for r in d['global_readiness'].values()),'All 36 fits/tests and both hosts are required'
  host=p/'runs'/d['run']/('host_'+d['host']);state=read(host/'state.json')
  active_exit=Path(state.get('active_scheduler_receipt',host/'start_exit.json'))
  expected=d['global_readiness'][d['host']]
  assert str(active_exit)==expected['exit_path'] and sha(active_exit)==expected['exit_sha256'] and read(active_exit)['returncode']==0
  group=next(g for g in state['groups'].values() if g['database']==d['db']);assert group['status']=='complete'
  assert not (out/'attempt.json').exists() and not (out/'completion.json').exists(),'Existing attempt/result; do not duplicate'
  if d['host']=='mimic':
   other=p/'runs'/d['run']/(('mimic' if d['db']=='mimiciii' else 'mimiciii')+'_statistics_v1')
   if (other/'attempt.json').exists():
    assert (other/'exit.json').exists() and read(other/'exit.json')['returncode']==0 and read(other/'completion.json')['complete'],'Wait for the other database analysis on this host'
  with (out/'writer.lock').open('a+') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   wrapper=Path(group['source'])/'scripts/tracehr_recovery_attempt.py'
   expected_wrapper=read(Path(group['source'])/'source_manifest.json')['scripts/tracehr_recovery_attempt.py']
   assert sha(wrapper)==expected_wrapper
   spec=dict(command=command,cwd=str(source),receipt=str(receipt),job_id='replacement_statistics_'+d['db'],
             source_manifest_sha256=sha(manifest),global_readiness=d['global_readiness'])
   with (out/'attempt.json').open('x') as stream:json.dump(spec,stream,indent=2)
  # The exclusive attempt prevents another launch; release the lock before the
  # child acquires it, so process scheduling cannot create a false lock failure.
  with logfile.open('x') as log:
   child=subprocess.Popen([sys.executable,'-B',str(wrapper),'--spec',str(out/'attempt.json')],cwd=source,env=env,
                          stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  result=dict(started=True,wrapper_pid=child.pid,source_manifest_sha256=sha(manifest),training_launched=False,global_readiness=d['global_readiness'])
  with (out/'launch.json').open('x') as stream:json.dump(result,stream,indent=2)
elif d['mode']=='status':
 result={name:read(out/name) for name in ('progress.json','completion.json','launch.json','exit.json') if (out/name).exists()}
 result['attempt_exists']=(out/'attempt.json').exists()
 if receipt.exists() and read(receipt)['returncode']!=0:result['log_tail']=logfile.read_text(errors='replace')[-6000:]
else:
 done=read(out/'completion.json');assert done['complete'] and read(receipt)['returncode']==0
 assert set(done['files'])=={'verified-results.json','RESULTS.md'}
 for name,digest in done['files'].items():assert sha(out/name)==digest
 result=dict(receipt=done,exit=read(receipt),files={name:base64.b64encode((out/name).read_bytes()).decode() for name in done['files']})
print(json.dumps(result))
'''


def aggregate_local(out):
    results, inputs = [], {}
    for db in analysis.DATABASES:
        folder = ROOT/'runs'/RUN/(db+'_results_v1')
        receipt = analysis.read(folder/'collection-receipt.json')
        assert receipt['exit']['returncode']==0
        path = folder/'verified-results.json'
        digest = analysis.sha(path)
        assert digest==receipt['receipt']['files']['verified-results.json']
        inputs[str(path)]=digest
        results.append(analysis.read(path))
    ns = dict(np=analysis.np)
    analysis.functions(ROOT/'scripts/cem_icde72_significance.py',('holm',),ns)
    combined = analysis.aggregate(results,ns)
    combined['inputs']=inputs
    if (out/'completion.json').exists():
        assert analysis.read(out/'verified-results.json')==combined,'Existing aggregate changed'
        for name,digest in analysis.read(out/'completion.json')['files'].items():
            assert analysis.sha(out/name)==digest
        return dict(complete=True,reused=True)
    analysis.save(out/'verified-results.json',combined)
    (out/'RESULTS.md').write_text(analysis.report(combined),encoding='utf-8')
    analysis.save(out/'completion.json',dict(complete=True,files={n:analysis.sha(out/n) for n in ALLOWED}))
    return dict(complete=True,verified_new_fits=36,verified_fixed_tests=36,comparisons=48,
                families=4,family_count=12,overall_paper_task_complete=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--database',choices=analysis.DATABASES)
    p.add_argument('--mode',choices=('check','audit','start','status','results','aggregate'),required=True)
    a=p.parse_args()
    out=Path(os.environ['PROJECT_RUN_DIR']).resolve(strict=True)
    expected=ROOT/'runs'/RUN/('paired_results_v1' if a.mode=='aggregate' else str(a.database)+'_results_v1')
    assert out==expected
    if a.mode=='aggregate':
        print(json.dumps(aggregate_local(out)))
        return
    assert a.database
    host='eicu' if a.database=='eicu' else 'mimic'
    d=dict(project=HOSTS[host][1],run=RUN,db=a.database,host=host,mode=a.mode)
    if a.mode in ('check','audit','start'):
        raw,manifest=bundle()
        d.update(files={name:base64.b64encode(content).decode() for name,content in raw.items()},manifest=manifest)
    if a.mode=='start':
        readiness={h:host_readiness(h) for h in ('mimic','eicu')}
        assert all(r['ready'] for r in readiness.values()),json.dumps(readiness)
        d['global_readiness']=readiness
    result=ssh(host,REMOTE.replace('PAYLOAD',repr(d)),timeout=650 if a.mode=='audit' else 120)
    if a.mode=='results':
        assert set(result['files'])==set(ALLOWED)
        for name,encoded in result['files'].items():
            content=base64.b64decode(encoded)
            assert hashlib.sha256(content).hexdigest()==result['receipt']['files'][name]
            destination=out/name
            if destination.exists():assert destination.read_bytes()==content,'Existing downloaded result changed'
            else:destination.write_bytes(content)
        analysis.save(out/'collection-receipt.json',{k:v for k,v in result.items() if k!='files'})
        result=dict(complete=True,database=a.database,overall_paper_task_complete=False)
    else:
        analysis.save(out/(a.mode+'-receipt.json'),result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
