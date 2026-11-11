"""Continue independent CNN delivery, then FT, without exposing training commands."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import collect_cem_sequence_partitions as collection
from scripts.cem_sequence_partition_runtime import read,save,sha,RUN

OUTPUT=ROOT/'runs'/RUN/'auto_partitions_v1'
SOURCES=('continue_cem_sequence_partitions.py','collect_cem_sequence_partitions.py',
 'cem_sequence_partition_statistics.py','cem_sequence_partition_runtime.py',
 'collect_cem_replacements.py','cem_replacement_statistics.py','cem_icde72_statistics.py',
 'cem_icde72_significance.py','launch_ema_cross.py')
DATABASES=('mimiciii','mimic','eicu')


def hashes():return {n:sha(ROOT/'scripts'/n) for n in SOURCES}


def phase(status):
    if 'exit.json' in status:
        assert status['exit.json']['returncode']==0 and status.get('completion.json',{}).get('complete'),'Statistics execution/audit failure'
        return 'complete'
    if status.get('attempt_exists'):return 'running'
    assert not status.get('completion.json') and not status.get('launch.json')
    return 'pending'


def validate_host(s):
    assert s['returncode'] in (None,0),'Training host failure'
    for g in s['groups'].values():
        assert g['status'] not in ('failed','needs_attention','storage_review_required') and not g['probe_failed']
        for part,n in (('cnn',30),('ft',10)):
            for action in ('train','evaluate'):
                c=g['partitions'][part][action]
                assert sum(c.values())==n and c['failed']==0,'Producer failure; no automatic retry'


def starts(states,ready):
    chosen=[]
    for databases in (('mimiciii','mimic'),('eicu',)):
        if any(states[db,part]=='running' for db in databases for part in ('cnn','ft')):continue
        for part in ('cnn','ft'):
            # Reserve the host's first analyses for its CNN partitions.
            if part=='ft' and any(states[db,'cnn']!='complete' for db in databases):break
            candidates=[(db,part) for db in databases if states[db,part]=='pending' and ready[db,part]]
            if candidates:chosen.append(candidates[0]);break
    return chosen


def invoke(mode,part,db=None):
    assert mode in ('start','results','aggregate')
    out=ROOT/'runs'/RUN/(part+'_paired_results_v1' if mode=='aggregate' else db+'_'+part+'_results_v1')
    command=[sys.executable,'-B','D:/Codex/storage-tools/storage_guard.py','--project',str(ROOT),
      '--run','sequence_partition_'+(db or 'all')+'_'+part+'_'+mode,'--expected-gib','0.5','--output',str(out),
      '--cache',str(ROOT/'cache'/RUN),'--temp',str(ROOT/'work'/RUN/'tmp_local'),'--',sys.executable,
      '-X','utf8','-B',str(ROOT/'scripts/collect_cem_sequence_partitions.py'),'--mode',mode,'--stage',part]
    if db:command+=['--database',db]
    with (OUTPUT/((db or 'all')+'_'+part+'_'+mode+'.log')).open('xb') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=900,
          creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    assert result.returncode==0,mode+' failed; no automatic retry'
    name='completion.json' if mode=='aggregate' else 'collection-receipt.json' if mode=='results' else 'start-receipt.json'
    assert (out/name).exists(),str(out/name)


def prepare():
    old=OUTPUT.parent/'auto_statistics_v2';a=read(old/'attempt.json')
    for n,h in a['source_hashes'].items():assert sha(ROOT/n)==h
    assert not (old/'completion.json').exists()
    proof=dict(complete=True,old_attempt_sha256=sha(old/'attempt.json'),old_source_hashes=a['source_hashes'],
      authorization='CNN delivery independent of FT; no worker interruption or training changes',time=time.time())
    with (OUTPUT/'monitor-handoff.json').open('x') as f:json.dump(proof,f,indent=2)


def run():
    assert Path(os.environ['PROJECT_RUN_DIR']).resolve()==OUTPUT.resolve()
    proof=read(OUTPUT/'monitor-handoff.json');old=OUTPUT.parent/'auto_statistics_v2'
    assert proof['complete'] and sha(old/'attempt.json')==proof['old_attempt_sha256']
    assert read(old/'exit.json')['returncode']==1 and not (old/'completion.json').exists()
    assert 'partition_v1' in read(old/'failure.json')['traceback'] or 'AssertionError' in read(old/'failure.json')['traceback']
    for n,h in proof['old_source_hashes'].items():assert sha(ROOT/n)==h
    for db in DATABASES:
        # Prior 120-fit collector was never permitted to start before this handoff.
        h='eicu' if db=='eicu' else 'mimic'
        check="from pathlib import Path\nimport json\np=Path("+repr(collection.HOSTS[h][1])+")/'runs'/"+repr(RUN)+"/"+repr(db+'_statistics_v1')+"\nassert not (p/'attempt.json').exists()\nprint(json.dumps({'complete':True}))"
        collection.ssh(h,check,120)
        for part in ('cnn','ft'):assert phase(collection.status(db,part))=='pending'
    attempt=dict(pid=os.getpid(),parent_pid=os.getppid(),started_at=time.time(),source_hashes=hashes(),
      interval_seconds=60,training_commands_exposed=False,automatic_retries=False,
      predecessor_exit_sha256=sha(old/'exit.json'),predecessor_failure_sha256=sha(old/'failure.json'))
    with (OUTPUT/'attempt.json').open('x') as f:json.dump(attempt,f,indent=2)
    rc=1
    try:
        identities={};missing={};errors=0
        while True:
            assert hashes()==attempt['source_hashes'],'Frozen continuation source changed'
            try:
                hosts={h:collection.host_readiness(h) for h in ('mimic','eicu')}
                statuses={(db,part):collection.status(db,part) for db in DATABASES for part in ('cnn','ft')}
            except (RuntimeError,subprocess.TimeoutExpired) as exc:
                errors+=1;save(OUTPUT/'progress.json',dict(stage='transport_wait',errors=errors,error=str(exc),time=time.time()))
                if errors>=3:raise
                time.sleep(60);continue
            errors=0;ready={}
            for h,s in hosts.items():
                validate_host(s)
                if s['alive']:assert identities.setdefault(h,s['identity'])==s['identity'],'Queue identity changed'
                elif s['returncode'] is None:
                    first=missing.setdefault(h,time.time());assert time.time()-first<180,'Missing real host exit'
                for db,g in s['groups'].items():
                    for part,v in g['partitions'].items():ready[db,part]=v['ready']
            states={key:phase(s) for key,s in statuses.items()}
            for db,part in starts(states,ready):
                invoke('start',part,db);states[db,part]=phase(collection.status(db,part))
            for (db,part),state in states.items():
                if state=='complete' and not (collection.result_folder(db,part)/'collection-receipt.json').exists():invoke('results',part,db)
            for part,total in (('cnn',90),('ft',30)):
                marker=OUTPUT/(part+'_completion.json')
                if not marker.exists() and all(states[db,part]=='complete' for db in DATABASES):
                    invoke('aggregate',part);out=OUTPUT.parent/(part+'_paired_results_v1')
                    for n,h in read(out/'completion.json')['files'].items():assert sha(out/n)==h
                    r=read(out/'verified-results.json');assert r['verified_new_fits']==r['verified_fixed_tests']==total
                    save(marker,dict(complete=True,partition=part,paper_update_pending=True,
                      aggregate_sha256=sha(out/'verified-results.json'),finished_at=time.time()))
            save(OUTPUT/'progress.json',dict(stage='independent_cnn_and_ft',hosts=hosts,
              statistics={db+':'+part:state for (db,part),state in states.items()},
              cnn_ready_for_paper=(OUTPUT/'cnn_completion.json').exists(),time=time.time()))
            if all(state=='complete' for state in states.values()) and all(not s['alive'] and s['returncode']==0 for s in hosts.values()):
                invoke('aggregate','all');out=OUTPUT.parent/'all_paired_results_v1'
                save(OUTPUT/'completion.json',dict(complete=True,statistics_complete=True,paper_update_pending=True,
                  aggregate_sha256=sha(out/'verified-results.json'),finished_at=time.time()));rc=0;return
            time.sleep(60)
    except BaseException as exc:
        save(OUTPUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
    finally:save(OUTPUT/'exit.json',dict(pid=os.getpid(),started_at=attempt['started_at'],finished_at=time.time(),returncode=rc))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('prepare','run','status'),required=True);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='run':run()
    else:print(json.dumps({n:read(OUTPUT/n) for n in ('attempt.json','progress.json','failure.json','exit.json','cnn_completion.json','completion.json') if (OUTPUT/n).exists()}))
