"""One-writer CNN/FT collection without waiting for unrelated training jobs."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import cem_replacement_statistics as prior
sys.modules['prior_utilities']=prior
from scripts import cem_sequence_partition_statistics as analysis
from scripts import collect_cem_replacements as original
from scripts.launch_ema_cross import HOSTS,ssh
RUN=analysis.RUN
ALLOWED=original.ALLOWED


def bundle():
    mapping={'analysis.py':'cem_sequence_partition_statistics.py',
             'prior_utilities.py':'cem_replacement_statistics.py',
             'prior_statistics.py':'cem_icde72_statistics.py','prior_significance.py':'cem_icde72_significance.py'}
    files={name:(ROOT/'scripts'/path).read_bytes() for name,path in mapping.items()}
    for name,b in files.items():compile(b,name,'exec')
    manifests={db:prior.read(ROOT/'runs'/RUN/f'local_{db}_v3/build-receipt.json')['source_manifest_sha256'] for db in analysis.DATABASES}
    files['training_manifests.json']=json.dumps(manifests,indent=2).encode()
    return files,{n:hashlib.sha256(b).hexdigest() for n,b in files.items()}


READINESS=r'''
import json,hashlib,time
from pathlib import Path
d=PAYLOAD;p=Path(d['project']);host=p/'runs'/d['run']/('host_'+d['host']+'_v3')
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
s=read(host/'state.json');proofpath=host/'partition_v1/handoff.json';proof=read(proofpath)
assert proof['complete'] and proof['workers_stopped']==proof['training_restarts']==0
assert sha(proofpath)==s['partition_handoff_sha256'] and sha(proof['runtime_path'])==proof['runtime_sha256']
exitpath=host/'partition_v1/exit.json';assert str(exitpath)==s['active_scheduler_receipt']
owner=read(host/'partition_v1/exit_owner.json');assert owner['child_pid']==s['pid'] and owner['source_manifest_sha256']==proof['runtime_sha256']
proc=Path('/proc/'+str(s['pid']));alive=proc.exists();identity=None
if alive:
 stat=(proc/'stat').read_text().rsplit(')',1)[1].split();alive=stat[0] not in ('Z','X')
 if alive:assert str(proof['runtime_path']) in (proc/'cmdline').read_bytes().decode(errors='replace')
 identity=dict(pid=s['pid'],ticks=stat[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
 if alive:assert identity['ticks']==owner['child_identity']['start_ticks'] and identity['boot']==owner['child_identity']['boot']
rc=read(exitpath) if exitpath.exists() else None
if rc:assert rc['child_pid']==owner['child_pid'] and rc['child_identity']==owner['child_identity']
groups={}
for g in s['groups'].values():
 stages={}
 for stage,variants in [('cnn',('TCN','ModernTCN','LITEMV')),('ft',('FTTransformer',))]:
  counts={a:{z:sum(j['action']==a and j['variant'] in variants and j['status']==z for j in g['jobs'].values()) for z in ('pending','running','completed','failed')} for a in ('train','evaluate')}
  f=Path(g['output'])/'partition_receipts'/(stage+'.json');ready=f.exists()
  if ready:
   c=read(f);assert c['complete'] and c['variants']==list(variants) and c['handoff_sha256']==sha(proofpath)
   assert counts['train']['completed']==counts['evaluate']['completed']==10*len(variants)
  stages[stage]=dict(ready=ready,closure_path=str(f),closure_sha256=sha(f) if ready else None,**counts)
 groups[g['database']]=dict(status=g['status'],partitions=stages,probe_failed=sum(j['action']=='probe' and j['status']=='failed' for j in g['jobs'].values()))
assert set(groups)==set(('eicu',) if d['host']=='eicu' else ('mimiciii','mimic'))
print(json.dumps(dict(groups=groups,alive=alive,identity=identity,returncode=rc['returncode'] if rc else None,
 exit_path=str(exitpath),exit_sha256=sha(exitpath) if rc else None,observed_at=time.time(),handoff_sha256=sha(proofpath))))
'''


def host_readiness(host):
    return ssh(host,READINESS.replace('PAYLOAD',repr(dict(project=HOSTS[host][1],host=host,run=RUN)),1),120)


def remote_program():
    code=original.REMOTE
    def change(a,b):
        nonlocal code
        assert a in code,a
        code=code.replace(a,b,1)
    change("('statistics_'+d['db']+'_v1')","('statistics_'+d['db']+'_partition_v1')")
    change("(d['db']+'_statistics_v1')","(d['db']+'_'+d['stage']+'_statistics_v1')")
    change("'--database',d['db']]","'--database',d['db'],'--stage',d['stage']]")
    start=code.index("  assert all(r['ready'] for r in d['global_readiness'].values())")
    end=code.index("  with (out/'writer.lock').open('a+') as lock:",start)
    code=code[:start]+r'''
  expected=d['readiness'];assert expected['ready']
  closure=Path(expected['closure_path']);assert sha(closure)==expected['closure_sha256']
  closed=read(closure);assert closed['complete'] and closed['partition']==d['stage']
  group=closed['group'];assert group['database']==d['db'] and group['status']=='partition_complete'
  assert not (out/'attempt.json').exists() and not (out/'completion.json').exists(),'Existing attempt/result; do not duplicate'
  # One CPU analysis on a host, across both databases and both partitions.
  hostlock=(p/'runs'/d['run']/('host_'+d['host']+'_v3')/'partition_statistics_launch.lock').open('a+')
  fcntl.flock(hostlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for db in (('eicu',) if d['host']=='eicu' else ('mimiciii','mimic')):
   for stage in ('cnn','ft'):
    other=p/'runs'/d['run']/(db+'_'+stage+'_statistics_v1')
    if other!=out and (other/'attempt.json').exists():
     assert (other/'exit.json').exists() and read(other/'exit.json')['returncode']==0 and read(other/'completion.json')['complete'],'Wait for the active same-host analysis'
''' +code[end:]
    change("global_readiness=d['global_readiness']","partition_readiness=d['readiness']")
    change("global_readiness=d['global_readiness']","partition_readiness=d['readiness']")
    change("job_id='replacement_statistics_'+d['db']","job_id='sequence_'+d['stage']+'_statistics_'+d['db']")
    return code


REMOTE=remote_program()


def status(db,stage):
    host='eicu' if db=='eicu' else 'mimic'
    d=dict(project=HOSTS[host][1],run=RUN,db=db,host=host,mode='status',stage=stage)
    return ssh(host,REMOTE.replace('PAYLOAD',repr(d),1),120)


def result_folder(db,stage):
    return ROOT/'runs'/RUN/(db+'_'+stage+'_results_v1')


def aggregate_local(out,stage):
    api=dict(np=analysis.np)
    prior.functions(ROOT/'scripts/cem_icde72_significance.py',('holm',),api)
    inputs={}
    if stage=='all':
        hosts={host:host_readiness(host) for host in ('mimic','eicu')}
        for state in hosts.values():
            assert not state['alive'] and state['returncode']==0
            assert all(g['status']=='complete' and all(v['ready'] for v in g['partitions'].values()) for g in state['groups'].values())
        stages=[]
        for part in ('cnn','ft'):
            folder=ROOT/'runs'/RUN/(part+'_paired_results_v1')
            receipt=prior.read(folder/'completion.json')
            assert prior.sha(folder/'verified-results.json')==receipt['files']['verified-results.json']
            stages.append(prior.read(folder/'verified-results.json'));inputs[str(folder/'verified-results.json')]=receipt['files']['verified-results.json']
        result=analysis.aggregate_full(*stages,api)
        result['host_exits']={h:{k:s[k] for k in ('exit_path','exit_sha256','returncode','handoff_sha256')} for h,s in hosts.items()}
    else:
        analysis.STAGE=stage;analysis.VARIANTS=analysis.STAGES[stage];values=[]
        for db in analysis.DATABASES:
            folder=result_folder(db,stage);receipt=prior.read(folder/'collection-receipt.json')
            assert receipt['exit']['returncode']==0
            h=prior.sha(folder/'verified-results.json');assert h==receipt['receipt']['files']['verified-results.json']
            inputs[str(folder/'verified-results.json')]=h;values.append(prior.read(folder/'verified-results.json'))
        result=analysis.aggregate(values,api)
    result['inputs']=inputs
    if (out/'completion.json').exists():
        assert prior.read(out/'verified-results.json')==result
        return dict(complete=True,reused=True)
    prior.save(out/'verified-results.json',result)
    (out/'RESULTS.md').write_text(analysis.report(result),encoding='utf-8')
    prior.save(out/'completion.json',dict(complete=True,files={n:prior.sha(out/n) for n in ALLOWED}))
    return dict(complete=True,partition=stage,verified_new_fits=result['verified_new_fits'],verified_fixed_tests=result['verified_fixed_tests'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--database',choices=analysis.DATABASES)
    p.add_argument('--stage',choices=('cnn','ft','all'),required=True)
    p.add_argument('--mode',choices=('check','audit','start','status','results','aggregate'),required=True)
    a=p.parse_args();out=Path(os.environ['PROJECT_RUN_DIR']).resolve(strict=True)
    expected=ROOT/'runs'/RUN/(a.stage+'_paired_results_v1') if a.mode=='aggregate' else result_folder(a.database,a.stage)
    assert out==expected
    if a.mode=='aggregate':
        result=aggregate_local(out,a.stage)
    else:
        assert a.database and a.stage!='all'
        host='eicu' if a.database=='eicu' else 'mimic'
        d=dict(project=HOSTS[host][1],run=RUN,db=a.database,host=host,stage=a.stage,mode=a.mode)
        if a.mode in ('check','audit','start'):
            raw,manifest=bundle();d.update(files={n:base64.b64encode(b).decode() for n,b in raw.items()},manifest=manifest)
        if a.mode=='start':
            state=host_readiness(host);assert state['returncode'] in (None,0)
            d['readiness']=state['groups'][a.database]['partitions'][a.stage]
            assert d['readiness']['ready']
        result=ssh(host,REMOTE.replace('PAYLOAD',repr(d),1),650 if a.mode in ('audit','check') else 120)
        if a.mode=='results':
            assert set(result['files'])==set(ALLOWED)
            for name,encoded in result['files'].items():
                b=base64.b64decode(encoded);assert hashlib.sha256(b).hexdigest()==result['receipt']['files'][name]
                dest=out/name
                if dest.exists():assert dest.read_bytes()==b
                else:dest.write_bytes(b)
            prior.save(out/'collection-receipt.json',{k:v for k,v in result.items() if k!='files'})
            result=dict(complete=True,database=a.database,partition=a.stage)
        else:prior.save(out/(a.mode+'-receipt.json'),result)
    print(json.dumps(result))


if __name__=='__main__':
    import contextlib,traceback
    out=Path(os.environ['PROJECT_RUN_DIR'])
    with (out/'entry.log').open('a',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:main()
        except BaseException as exc:
            prior.save(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
