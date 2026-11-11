"""Use the existing one-writer collector for the 30-fit five-seed extension."""
import base64, hashlib, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import cem_replacement_statistics as prior
sys.modules['prior_utilities']=prior
from scripts import cem_five_seed_statistics as analysis
from scripts import collect_cem_replacements as collector
from scripts.launch_ema_cross import HOSTS,ssh
RUN=analysis.RUN

def bundle():
    mapping={'analysis.py':'cem_five_seed_statistics.py','prior_utilities.py':'cem_replacement_statistics.py',
             'prior_statistics.py':'cem_icde72_statistics.py','prior_significance.py':'cem_icde72_significance.py'}
    files={name:(ROOT/'scripts'/relative).read_bytes() for name,relative in mapping.items()}
    for name,b in files.items():compile(b,name,'exec')
    manifests={db:prior.read(ROOT/'runs'/RUN/f'local_{db}_v3/build-receipt.json')['source_manifest_sha256'] for db in analysis.DATABASES}
    files['training_manifests.json']=json.dumps(manifests,indent=2).encode()
    return files,{n:hashlib.sha256(b).hexdigest() for n,b in files.items()}

def host_readiness(host):
    code=r'''
import json,hashlib
from pathlib import Path
p=Path(PAYLOAD['project']);out=p/'runs'/PAYLOAD['run']/('host_'+PAYLOAD['host'])
s=json.loads((out/'state.json').read_text());exitpath=out/'start_exit.json'
rc=json.loads(exitpath.read_text()) if exitpath.exists() else None
groups={g['database']:dict(status=g['status'],train=sum(j['action']=='train' and j['status']=='completed' for j in g['jobs'].values()),evaluate=sum(j['action']=='evaluate' and j['status']=='completed' for j in g['jobs'].values())) for g in s['groups'].values()}
expected={'eicu':10} if PAYLOAD['host']=='eicu' else {'mimiciii':12,'mimic':8}
ready=set(groups)==set(expected) and rc is not None and rc['returncode']==0 and all(groups[db]['status']=='complete' and groups[db]['train']==groups[db]['evaluate']==n for db,n in expected.items())
print(json.dumps(dict(ready=ready,groups=groups,exit_path=str(exitpath),returncode=rc['returncode'] if rc else None,exit_sha256=hashlib.sha256(exitpath.read_bytes()).hexdigest() if rc else None)))
'''
    return ssh(host,code.replace('PAYLOAD',repr(dict(project=HOSTS[host][1],run=RUN,host=host))),120)

def aggregate_local(out):
    results=[];inputs={}
    for db in analysis.DATABASES:
        p=ROOT/'runs'/RUN/(db+'_results_v1');receipt=prior.read(p/'collection-receipt.json')
        assert receipt['exit']['returncode']==0
        h=prior.sha(p/'verified-results.json');assert h==receipt['receipt']['files']['verified-results.json']
        inputs[str(p/'verified-results.json')]=h;results.append(prior.read(p/'verified-results.json'))
    result=analysis.aggregate(results);result['inputs']=inputs
    if (out/'completion.json').exists():
        assert prior.read(out/'verified-results.json')==result
        return dict(complete=True,reused=True)
    prior.save(out/'verified-results.json',result);(out/'RESULTS.md').write_text(analysis.report(result),encoding='utf-8')
    prior.save(out/'completion.json',dict(complete=True,files={n:prior.sha(out/n) for n in collector.ALLOWED}))
    return dict(complete=True,verified_new_fits=30,verified_fixed_tests=30,comparisons=18,seeds=5,overall_paper_task_complete=False)

collector.RUN=RUN;collector.analysis=analysis;collector.bundle=bundle;collector.host_readiness=host_readiness
collector.aggregate_local=aggregate_local
collector.REMOTE=collector.REMOTE.replace('All 36 fits/tests','All 30 fits/tests').replace('scripts/tracehr_recovery_attempt.py','scripts/five_seed_attempt.py').replace("'replacement_statistics_'","'five_seed_statistics_'")
if __name__=='__main__':collector.main()
