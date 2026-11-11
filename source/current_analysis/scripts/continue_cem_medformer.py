"""Release verified bounded paths, then continue fixed statistics automatically."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import collect_cem_medformer as collection
from scripts.cem_medformer_common import RUN,read,save,sha
OUTPUT=ROOT/'runs'/RUN/'auto_statistics_v1'
SOURCES=('scripts/continue_cem_medformer.py','scripts/collect_cem_medformer.py',
    'scripts/cem_medformer_statistics.py','scripts/launch_cem_medformer.py','scripts/cem_medformer_common.py',
    'scripts/collect_cem_replacements.py','scripts/cem_replacement_statistics.py',
    'scripts/cem_icde72_statistics.py','scripts/cem_icde72_significance.py','scripts/launch_ema_cross.py')


def hashes():return {n:sha(ROOT/n) for n in SOURCES}


def training_ready(status):
    if status['returncode'] is not None:assert status['returncode']==0,'Training host failure'
    for g in status['groups'].values():
        for action in ('train','evaluate'):assert sum(g[action].values())==10
        for action in ('train','evaluate','probe'):assert not g[action]['failed'],'Producer failure; no automatic retry'
        assert g['status'] not in ('needs_attention','storage_review_required','failed')
    return status['ready']


def statistics_status(db):
    host='eicu' if db=='eicu' else 'mimic'
    d=dict(project=collection.HOSTS[host][1],run=RUN,db=db,host=host,mode='status')
    return collection.ssh(host,collection.collector.REMOTE.replace('PAYLOAD',repr(d)),120)


def phase(status):
    if 'exit.json' in status:
        assert status['exit.json']['returncode']==0 and status.get('completion.json',{}).get('complete'),'Statistics failure'
        return 'complete'
    if status.get('attempt_exists'):return 'running'
    assert not status.get('completion.json') and not status.get('launch.json')
    return 'pending'


def starts(states,slots):
    result=[]
    if slots['eicu'] and states['eicu']=='pending':result.append('eicu')
    if slots['mimic']:
        if states['mimiciii']=='pending':
            assert states['mimic']!='running';result.append('mimiciii')
        elif states['mimiciii']=='complete' and states['mimic']=='pending':result.append('mimic')
    return result


def invoke(mode,db=None):
    assert mode in ('release','start','results','aggregate')
    release=mode=='release'
    out=ROOT/'runs'/RUN/(f'local_{db}_v1' if release else 'paired_results_v1' if mode=='aggregate' else db+'_results_v1')
    script='launch_cem_medformer.py' if release else 'collect_cem_medformer.py'
    command=[sys.executable,'-B','D:/Codex/storage-tools/storage_guard.py','--project',str(ROOT),
        '--run','medformer_auto_'+(db or 'all')+'_'+mode,'--expected-gib','0.5','--output',str(out),
        '--cache',str(ROOT/'cache'/RUN),'--temp',str(ROOT/'work'/RUN/'tmp_local'),
        '--',sys.executable,'-X','utf8','-B',str(ROOT/'scripts'/script),'--mode',mode]
    if db is not None:command+=['--database',db]
    with (OUTPUT/((db or 'all')+'_'+mode+'.log')).open('xb') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=900,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    assert result.returncode==0,mode+' failed; no automatic retry'
    receipt=out/('completion.json' if mode=='aggregate' else 'collection-receipt.json' if mode=='results' else mode+'-receipt.json')
    assert receipt.exists(),str(receipt)


def run():
    assert Path(os.environ['PROJECT_RUN_DIR']).resolve()==OUTPUT.resolve()
    for db in collection.analysis.DATABASES:
        local=ROOT/'runs'/RUN/f'local_{db}_v1'
        assert read(local/'verify-receipt.json')['complete'] and read(local/'register-receipt.json')['variants']==['Medformer']
        assert read(ROOT/'runs'/RUN/(db+'_results_v1/check-receipt.json'))['complete']
    attempt=dict(pid=os.getpid(),parent_pid=os.getppid(),started_at=time.time(),source_hashes=hashes(),
        interval_seconds=60,automatic_retries=False,only_new_training_release='Two verified 20-step paths per database',
        independent_of_CNN=True,statistics_wait_for_existing_CNN_CPU_writers=True)
    with (OUTPUT/'attempt.json').open('x') as stream:json.dump(attempt,stream,indent=2)
    rc=1
    try:
        identities={};missing={};errors=0;released=set()
        while True:
            assert hashes()==attempt['source_hashes'],'Frozen continuation source changed'
            try:hosts={h:collection.host_readiness(h) for h in ('mimic','eicu')}
            except (RuntimeError,subprocess.TimeoutExpired) as exc:
                errors+=1;save(OUTPUT/'progress.json',dict(stage='transport_wait',errors=errors,error=str(exc),time=time.time()))
                if errors>=3:raise
                time.sleep(60);continue
            errors=0;ready={h:training_ready(s) for h,s in hosts.items()}
            for h,s in hosts.items():
                if s['alive']:assert identities.setdefault(h,s['identity'])==s['identity'],'Queue identity changed'
                if not s['alive'] and s['returncode'] is None:
                    first=missing.setdefault(h,time.time());assert time.time()-first<180,'Missing genuine queue exit'
                for db,g in s['groups'].items():
                    if g['status']=='awaiting_path_review' and db not in released:
                        # Launcher validates both real exits, gradients, SWA and exact recovery.
                        invoke('release',db);released.add(db)
            if not all(ready.values()):
                save(OUTPUT/'progress.json',dict(stage='waiting_for_30_fits_and_30_tests',hosts=hosts,time=time.time()))
                time.sleep(60);continue
            states={db:phase(statistics_status(db)) for db in collection.analysis.DATABASES}
            slots={h:s['statistics_slot_ready'] for h,s in hosts.items()}
            for db in starts(states,slots):invoke('start',db);states[db]=phase(statistics_status(db))
            for db,status in states.items():
                if status=='complete' and not (ROOT/'runs'/RUN/(db+'_results_v1/collection-receipt.json')).exists():invoke('results',db)
            save(OUTPUT/'progress.json',dict(stage='statistics',databases=states,CPU_slots=slots,time=time.time()))
            if all(s=='complete' for s in states.values()):
                invoke('aggregate');out=ROOT/'runs'/RUN/'paired_results_v1'
                for n,h in read(out/'completion.json')['files'].items():assert sha(out/n)==h
                result=read(out/'verified-results.json');assert result['verified_new_fits']==result['verified_fixed_tests']==30
                save(OUTPUT/'completion.json',dict(complete=True,statistics_complete=True,paper_update_pending=True,
                    aggregate_sha256=sha(out/'verified-results.json'),finished_at=time.time()));rc=0;return
            time.sleep(60)
    except BaseException as exc:
        save(OUTPUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
    finally:save(OUTPUT/'exit.json',dict(pid=os.getpid(),started_at=attempt['started_at'],finished_at=time.time(),returncode=rc))


def self_test():
    pending=dict(eicu='pending',mimiciii='pending',mimic='pending')
    assert starts(pending,dict(eicu=True,mimic=True))==['eicu','mimiciii']
    assert starts(pending,dict(eicu=False,mimic=False))==[]
    assert starts(dict(eicu='complete',mimiciii='complete',mimic='pending'),dict(eicu=True,mimic=True))==['mimic']
    assert starts(dict(eicu='running',mimiciii='running',mimic='pending'),dict(eicu=True,mimic=True))==[]
    assert phase(dict(attempt_exists=True))=='running' and phase(dict(attempt_exists=False))=='pending'
    for s in ({'exit.json':dict(returncode=1)},{'exit.json':dict(returncode=0)}):
        try:phase(s)
        except AssertionError:pass
        else:raise AssertionError('Incomplete statistics accepted')
    g=dict(status='running',train=dict(pending=9,running=1,completed=0,failed=0),evaluate=dict(pending=10,running=0,completed=0,failed=0),probe=dict(failed=0))
    assert not training_ready(dict(returncode=None,ready=False,groups={'eicu':g}))
    g['probe']['failed']=1
    try:training_ready(dict(returncode=None,ready=False,groups={'eicu':g}))
    except AssertionError:pass
    else:raise AssertionError('Producer failure accepted')
    return dict(complete=True,checks=10,CNN_CPU_priority=True,duplicate_statistics_prevented=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('check','run','status'),required=True);a=p.parse_args()
    if a.mode=='run':run()
    elif a.mode=='check':save(Path(os.environ['PROJECT_RUN_DIR'])/'continuation-check.json',self_test())
    else:print(json.dumps({n:read(OUTPUT/n) for n in ('attempt.json','progress.json','failure.json','exit.json','completion.json') if (OUTPUT/n).exists()}))
