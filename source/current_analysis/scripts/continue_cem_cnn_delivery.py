"""Complete the existing CNN analyses after FT cancellation; no new sampling logic."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import collect_cem_sequence_partitions as collection
from scripts.cem_sequence_partition_runtime import read,save,sha,RUN
from scripts.continue_cem_sequence_partitions import phase
OUTPUT=ROOT/'runs'/RUN/'auto_cnn_v1'
DATABASES=('mimiciii','mimic','eicu')
SOURCES=('continue_cem_cnn_delivery.py','cem_sequence_cnn_runtime.py','collect_cem_sequence_partitions.py',
 'continue_cem_sequence_partitions.py','cem_sequence_partition_statistics.py','cem_sequence_partition_runtime.py',
 'collect_cem_replacements.py','cem_replacement_statistics.py','cem_icde72_statistics.py','cem_icde72_significance.py','launch_ema_cross.py')


def readiness_code():
    code=collection.READINESS
    start=code.index("exitpath=host/'partition_v1/exit.json'")
    end=code.index("proc=Path('/proc/'+str(s['pid']))",start)
    code=code[:start]+r'''
retirepath=host/'ft_retirement_v1/handoff.json';retire=read(retirepath)
assert retire['complete'] and retire['cnn_workers_stopped']==retire['training_restarts']==0
assert sha(retirepath)==s['retirement_handoff_sha256'] and sha(retire['runtime_path'])==retire['runtime_sha256']
assert retire['parent_runtime_sha256']==proof['runtime_sha256']
assert sha(retire['predecessor_exit_path'])==retire['predecessor_exit_sha256'] and read(retire['predecessor_exit_path'])['returncode']==-15
exitpath=host/'ft_retirement_v1/exit.json';assert str(exitpath)==s['active_scheduler_receipt']
owner=read(host/'ft_retirement_v1/exit_owner.json');assert owner['child_pid']==s['pid'] and owner['source_manifest_sha256']==retire['runtime_sha256']
''' +code[end:]
    code=code.replace("str(proof['runtime_path']) in (proc/'cmdline')","str(retire['runtime_path']) in (proc/'cmdline')")
    code=code.replace("[('cnn',('TCN','ModernTCN','LITEMV')),('ft',('FTTransformer',))]","[('cnn',('TCN','ModernTCN','LITEMV'))]")
    code=code.replace(" groups[g['database']]=dict(status=g['status'],partitions=stages,", " assert not any(j['variant']=='FTTransformer' and j['status'] in ('pending','running') for j in g['jobs'].values())\n groups[g['database']]=dict(status=g['status'],partitions=stages,")
    code=code.replace("handoff_sha256=sha(proofpath))))","handoff_sha256=sha(proofpath),retirement_handoff_sha256=sha(retirepath))))")
    return code


READINESS=readiness_code()


def host_readiness(host):
    return collection.ssh(host,READINESS.replace('PAYLOAD',repr(dict(project=collection.HOSTS[host][1],host=host,run=RUN)),1),120)


collection.host_readiness=host_readiness


def hashes():return {n:sha(ROOT/'scripts'/n) for n in SOURCES}


def starts(states,ready):
    result=[]
    for dbs in (('mimiciii','mimic'),('eicu',)):
        if any(states[db]=='running' for db in dbs):continue
        choices=[db for db in dbs if states[db]=='pending' and ready[db]]
        if choices:result.append(choices[0])
    return result


def validate_host(s):
    assert s['returncode'] in (None,0)
    for g in s['groups'].values():
        assert g['status'] not in ('needs_attention','failed','storage_review_required') and not g['probe_failed']
        assert set(g['partitions'])=={'cnn'}
        for action in ('train','evaluate'):
            c=g['partitions']['cnn'][action];assert sum(c.values())==30 and not c['failed']


def invoke(mode,db=None):
    assert mode in ('start','results','aggregate','finalize')
    out=ROOT/'runs'/RUN/('cnn_final_results_v1' if mode=='finalize' else 'cnn_paired_results_v1' if mode=='aggregate' else db+'_cnn_results_v1')
    command=[sys.executable,'-B','D:/Codex/storage-tools/storage_guard.py','--project',str(ROOT),
      '--run','sequence_cnn_'+(db or 'all')+'_'+mode,'--expected-gib','0.5','--output',str(out),
      '--cache',str(ROOT/'cache'/RUN),'--temp',str(ROOT/'work'/RUN/'tmp_local'),'--',sys.executable,
      '-X','utf8','-B',str(ROOT/'scripts/continue_cem_cnn_delivery.py'),'--mode','collect','--operation',mode]
    if db:command+=['--database',db]
    with (OUTPUT/((db or 'all')+'_'+mode+'.log')).open('xb') as log:
        r=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=900,
          creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    assert r.returncode==0,'Collector failure; no automatic retry'
    name='completion.json' if mode in ('aggregate','finalize') else 'collection-receipt.json' if mode=='results' else 'start-receipt.json'
    assert (out/name).exists()


def final_adjustment(result,api):
    r=copy.deepcopy(result);assert r['partition']=='cnn' and r['verified_new_fits']==r['verified_fixed_tests']==90
    for split in ('dev','test'):
        rows=[x for x in r['paired'] if x['split']==split];assert len(rows)==18
        ps=api['holm']([x['p_bootstrap_centered'] for x in rows]+[1.]*12)
        for row,p in zip(rows,ps[:18]):row['p_holm']=float(p)
    r.update(holm_status='18 observed contrasts plus 12 retained slots for cancelled FT and deferred TVNet',
      retired_methods=['FTTransformer'],deferred_methods=['TVNet'],pending_family_methods=[],family_count=30,
      inference='Five paired fixed fits; 20000 common patient-cluster draws. Retain the original 30-comparison allowance per split: 18 CNN comparisons and 12 slots assigned p=1 for cancelled FT and deferred TVNet. Report nominal and Bonferroni intervals, approximate null-centered Bootstrap p, and full-allowance Holm p. Exact five-seed signed-rank is supplemental (minimum two-sided p=.0625). Intervals condition on fitted models, not the training population or historical selection. Existing CEM AP caches and prior secondary selections remain unchanged.')
    return r


def finalize(hosts):
    assert all(not s['alive'] and s['returncode']==0 for s in hosts.values())
    source=OUTPUT.parent/'cnn_paired_results_v1';receipt=read(source/'completion.json')
    for n,h in receipt['files'].items():assert sha(source/n)==h
    api={'np':collection.analysis.np};collection.prior.functions(ROOT/'scripts/cem_icde72_significance.py',('holm',),api)
    result=final_adjustment(read(source/'verified-results.json'),api)
    result['inputs']={str(source/'verified-results.json'):sha(source/'verified-results.json')}
    result['host_exits']={h:{k:s[k] for k in ('exit_path','exit_sha256','returncode','retirement_handoff_sha256')} for h,s in hosts.items()}
    out=OUTPUT.parent/'cnn_final_results_v1';assert Path(os.environ['PROJECT_RUN_DIR']).resolve()==out.resolve()
    assert not (out/'completion.json').exists()
    save(out/'verified-results.json',result);(out/'RESULTS.md').write_text(collection.analysis.report(result),encoding='utf-8')
    save(out/'completion.json',dict(complete=True,files={n:sha(out/n) for n in collection.ALLOWED}))
    return sha(out/'verified-results.json')


def run():
    assert Path(os.environ['PROJECT_RUN_DIR']).resolve()==OUTPUT.resolve()
    old=OUTPUT.parent/'auto_partitions_v1';oldattempt=read(old/'attempt.json')
    assert read(old/'exit.json')['returncode']==1 and not (old/'completion.json').exists()
    for n,h in oldattempt['source_hashes'].items():assert sha(ROOT/'scripts'/n)==h
    attempt=dict(pid=os.getpid(),parent_pid=os.getppid(),started_at=time.time(),source_hashes=hashes(),
      predecessor_attempt_sha256=sha(old/'attempt.json'),predecessor_exit_sha256=sha(old/'exit.json'),
      interval_seconds=60,training_commands_exposed=False,automatic_retries=False,retired_methods=['FTTransformer'])
    with (OUTPUT/'attempt.json').open('x') as f:json.dump(attempt,f,indent=2)
    rc=1
    try:
        identities={};missing={};errors=0
        while True:
            assert hashes()==attempt['source_hashes']
            try:
                hosts={h:host_readiness(h) for h in ('mimic','eicu')}
                states={db:phase(collection.status(db,'cnn')) for db in DATABASES}
            except (RuntimeError,subprocess.TimeoutExpired) as exc:
                errors+=1;save(OUTPUT/'progress.json',dict(stage='transport_wait',errors=errors,error=str(exc),time=time.time()))
                if errors>=3:raise
                time.sleep(60);continue
            errors=0;ready={}
            for h,s in hosts.items():
                validate_host(s)
                if s['alive']:assert identities.setdefault(h,s['identity'])==s['identity']
                elif s['returncode'] is None:
                    first=missing.setdefault(h,time.time());assert time.time()-first<180
                for db,g in s['groups'].items():ready[db]=g['partitions']['cnn']['ready']
            for db in starts(states,ready):invoke('start',db);states[db]=phase(collection.status(db,'cnn'))
            for db,state in states.items():
                if state=='complete' and not (collection.result_folder(db,'cnn')/'collection-receipt.json').exists():invoke('results',db)
            save(OUTPUT/'progress.json',dict(stage='cnn_only_after_user_cancelled_ft',hosts=hosts,statistics=states,time=time.time()))
            if all(state=='complete' for state in states.values()) and all(not s['alive'] and s['returncode']==0 for s in hosts.values()):
                if not (OUTPUT.parent/'cnn_paired_results_v1/completion.json').exists():invoke('aggregate')
                invoke('finalize');digest=sha(OUTPUT.parent/'cnn_final_results_v1/verified-results.json')
                save(OUTPUT/'completion.json',dict(complete=True,statistics_complete=True,paper_update_pending=True,
                  verified_fits=90,verified_tests=90,retired_methods=['FTTransformer'],aggregate_sha256=digest,finished_at=time.time()))
                rc=0;return
            time.sleep(60)
    except BaseException as exc:
        save(OUTPUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
    finally:save(OUTPUT/'exit.json',dict(pid=os.getpid(),started_at=attempt['started_at'],finished_at=time.time(),returncode=rc))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('run','collect','status'),required=True)
    p.add_argument('--operation',choices=('start','results','aggregate','finalize'));p.add_argument('--database',choices=DATABASES);a=p.parse_args()
    if a.mode=='run':run()
    elif a.mode=='collect':
        assert a.operation
        if a.operation=='finalize':finalize({h:host_readiness(h) for h in ('mimic','eicu')})
        else:
            sys.argv=[sys.argv[0],'--mode',a.operation,'--stage','cnn']+(['--database',a.database] if a.database else [])
            collection.main()
    else:print(json.dumps({n:read(OUTPUT/n) for n in ('attempt.json','progress.json','failure.json','exit.json','completion.json') if (OUTPUT/n).exists()}))
