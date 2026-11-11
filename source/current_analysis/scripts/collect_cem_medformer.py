"""Independent one-writer collector; no training entry point."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import cem_replacement_statistics as prior
sys.modules['prior_utilities']=prior
from scripts import cem_medformer_statistics as analysis
from scripts import collect_cem_replacements as collector
from scripts.launch_ema_cross import HOSTS,ssh
RUN=analysis.RUN


def bundle():
    mapping={'analysis.py':'cem_medformer_statistics.py','prior_utilities.py':'cem_replacement_statistics.py',
             'prior_statistics.py':'cem_icde72_statistics.py','prior_significance.py':'cem_icde72_significance.py'}
    files={name:(ROOT/'scripts'/relative).read_bytes() for name,relative in mapping.items()}
    for name,b in files.items():compile(b,name,'exec')
    manifests={db:prior.read(ROOT/'runs'/RUN/f'local_{db}_v1/build-receipt.json')['source_manifest_sha256'] for db in analysis.DATABASES}
    files['training_manifests.json']=json.dumps(manifests,indent=2).encode()
    return files,{n:hashlib.sha256(b).hexdigest() for n,b in files.items()}


def host_readiness(host):
    code=r'''
import json,hashlib,time
from pathlib import Path
d=PAYLOAD;p=Path(d['project']);out=p/'runs'/d['run']/('host_'+d['host'])
s=json.loads((out/'state.json').read_text());exitpath=Path(s.get('active_scheduler_receipt',out/'start_exit.json'))
assert exitpath.resolve().is_relative_to(out.resolve())
assert exitpath==out/'start_exit.json'
owner=json.loads((out/'start_exit_owner.json').read_text());assert owner['child_pid']==s['pid']
rc=json.loads(exitpath.read_text()) if exitpath.exists() else None
groups={g['database']:dict(status=g['status'],**{a:{status:sum(j['action']==a and j['status']==status for j in g['jobs'].values()) for status in ('pending','running','completed','failed')} for a in ('train','evaluate','probe')}) for g in s['groups'].values()}
proc=Path('/proc/'+str(s['pid']));alive=proc.exists();identity=None
if alive:
 stat=(proc/'stat').read_text().rsplit(')',1)[1].split();alive=stat[0] not in ('Z','X')
 cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
 if alive:assert d['run'] in cmd and 'cem_medformer_queue.py' in cmd,'Scheduler identity mismatch'
 identity=dict(pid=s['pid'],ticks=stat[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
expected=['eicu'] if d['host']=='eicu' else ['mimiciii','mimic']
assert set(groups)==set(expected)
ready=not alive and rc is not None and rc['returncode']==0 and all(g['status']=='complete' and g['train']['completed']==g['evaluate']['completed']==10 for g in groups.values())
cnn=p/'runs/cem_sequence_comparators_20261102'
slot_ready=True
for db in expected:
 f=cnn/(db+'_cnn_statistics_v1')
 if not (f/'exit.json').exists() or not (f/'completion.json').exists():slot_ready=False
 else:
  assert json.loads((f/'exit.json').read_text())['returncode']==0,'Existing CNN statistics failed'
  assert json.loads((f/'completion.json').read_text())['complete']
print(json.dumps(dict(ready=ready,groups=groups,alive=alive,identity=identity,exit_path=str(exitpath),returncode=rc['returncode'] if rc else None,
 exit_sha256=hashlib.sha256(exitpath.read_bytes()).hexdigest() if rc else None,observed_at=time.time(),statistics_slot_ready=slot_ready)))
'''
    return ssh(host,code.replace('PAYLOAD',repr(dict(project=HOSTS[host][1],run=RUN,host=host))),120)


def aggregate_local(out):
    results=[];inputs={}
    for db in analysis.DATABASES:
        p=ROOT/'runs'/RUN/(db+'_results_v1');receipt=prior.read(p/'collection-receipt.json')
        assert receipt['exit']['returncode']==0
        h=prior.sha(p/'verified-results.json');assert h==receipt['receipt']['files']['verified-results.json']
        inputs[str(p/'verified-results.json')]=h;results.append(prior.read(p/'verified-results.json'))
    ns=dict(np=analysis.np);prior.functions(ROOT/'scripts/cem_icde72_significance.py',('holm',),ns)
    result=analysis.aggregate(results,ns);result['inputs']=inputs
    if (out/'completion.json').exists():
        assert prior.read(out/'verified-results.json')==result;return dict(complete=True,reused=True)
    prior.save(out/'verified-results.json',result);(out/'RESULTS.md').write_text(analysis.report(result),encoding='utf-8')
    prior.save(out/'completion.json',dict(complete=True,files={n:prior.sha(out/n) for n in collector.ALLOWED}))
    return dict(complete=True,verified_new_fits=30,verified_fixed_tests=30,comparisons=12,seeds=5,overall_paper_task_complete=False)


collector.RUN=RUN;collector.analysis=analysis;collector.bundle=bundle;collector.host_readiness=host_readiness
collector.aggregate_local=aggregate_local
collector.REMOTE=collector.REMOTE.replace('All 36 fits/tests','All 30 Medformer fits/tests')
collector.REMOTE=collector.REMOTE.replace("assert all(r['ready'] for r in d['global_readiness'].values())", "assert all(r['ready'] for r in d['global_readiness'].values()) and d['global_readiness'][d['host']]['statistics_slot_ready']")
if __name__=='__main__':
    import os,traceback,contextlib
    out=Path(os.environ['PROJECT_RUN_DIR'])
    with (out/'entry.log').open('a',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:collector.main()
        except BaseException as exc:
            prior.save(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise


