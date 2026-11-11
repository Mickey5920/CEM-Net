"""Four bounded path pilots, twelve fresh fits, then twelve fixed tests per database."""
import argparse, math, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.ema_cross_queue import Queue as ParentQueue, process_identity, same_process
from scripts.ema_cross_idle_gpu_queue import inventory
from scripts.ema_pm_cross_common import *

class Queue(ParentQueue):
    def __init__(self,a):
        import fcntl
        self.a,self.fcntl=a,fcntl; self.project=a.project.resolve(strict=True); self.source=a.source.resolve(strict=True)
        self.out=a.output.resolve(); self.cache=self.project/'cache'/RUN; self.temp=self.project/'work'/RUN/'tmp'
        for path in (self.out,self.source,self.cache,self.temp): assert path.is_relative_to(self.project) and path!=self.project
        self.out.mkdir(parents=True,exist_ok=True)
        self.owner=(self.out/'owner.lock').open('a+'); fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (self.out/'state.json').exists(),'Never duplicate a queue'
        for n,h in read(self.source/'source_manifest.json').items(): assert sha(self.source/n)==h,n
        check=read(self.out.parent/'verification.json')
        assert check['complete'] and check['formal_fits']==12 and check['source_manifest_sha256']==sha(self.source/'source_manifest.json')
        self.handles={}; self.locks={}; self.idle_since={}; self.leases={}
        self.state=dict(database=a.database,pid=os.getpid(),process_identity=process_identity(os.getpid()),
            source_manifest_sha256=sha(self.source/'source_manifest.json'),phase='probes',status='preflight',jobs={},started_at=time.time(),
            gpus=[r['index'] for r in inventory(self.project)],fallback_gpus=expected()['fallback_gpus'],
            slots_per_gpu=2,max_workers_per_gpu=2,max_concurrent_workers=6,expected_gib=a.expected_gib,formal_fits=12,
            hyperparameter_search=False,min_free_gpu_mib=24576,
            gpu_policy='continuously idle first, then acquired idle leases, then previously designated pool')
        for row in matrix():
            name=f"{row['variant']}_t{row['endpoint']}_s{row['seed']}"
            actions=[('train','','jobs'),('evaluate','test_','test')]
            if row['seed']==48: actions.insert(0,('probe','probe_','pilots'))
            for action,prefix,folder in actions:
                key=prefix+name
                self.state['jobs'][key]=dict(row,id=key,pilot=action=='probe',action=action,
                    pilot_id=pilot_name(row['variant'],row['endpoint']),fit_id=name,status='pending',output=str(self.out/folder/key))
        self.save()

    def phase(self):
        jobs=list(self.state['jobs'].values())
        if any(j['status']=='failed' for j in jobs): self.state['status']='needs_attention'; return
        if self.state['phase']=='probes' and all(j['status']=='completed' for j in jobs if j['action']=='probe'):
            estimates=[self.state['jobs'][j['pilot_id']]['result']['checkpoint_bytes']*13*1.2/2**30 for j in jobs if j['action']=='train']
            estimate=1.+sum(estimates)+sum(j['result']['checkpoint_bytes']/2**30 for j in jobs if j['action']=='probe')
            assert estimate<self.a.expected_gib,('storage estimate exceeds declaration',estimate)
            self.state['projected_artifact_gib']=estimate
            if not (self.out/'formal_release.json').exists(): self.state['status']='awaiting_path_review'; return
            r=read(self.out/'formal_release.json')
            assert r['complete'] and r['source_manifest_sha256']==self.state['source_manifest_sha256']
            for j in jobs:
                if j['action']=='probe': assert sha(Path(j['output'])/'path_audit.json')==r['audit_sha256'][j['id']]
            self.state.update(phase='formal',status='running',formal_released_at=time.time())
        if self.state['phase']=='formal' and all(j['status']=='completed' for j in jobs if j['action']=='train'):
            atomic(self.out/'fixed-checkpoint-lock.json',dict(complete=True,created_at=time.time(),test_used_for_selection=False,
                selections={j['id']:dict(checkpoint_sha256=j['result']['checkpoint_sha256'],best_epoch=j['result']['best_epoch'],
                    dev_ap=j['result']['dev_ap']) for j in jobs if j['action']=='train'}))
            self.state.update(phase='evaluation',status='evaluating',all_12_fits_successfully_exited=True)
        if self.state['phase']=='evaluation' and all(j['status']=='completed' for j in jobs if j['action']=='evaluate'):
            self.state.update(status='complete',finished_at=time.time())

    def launch(self,j,gpu,slot):
        assert gpu in self.state['gpus'] and slot in (0,1); out=Path(j['output']); assert not out.exists()
        cmd=[sys.executable,'-B',str(self.source/'work/tracehr_group_v1_validation_20261023/memory_entry.py'),
            str(self.source/'scripts/train_ema_pm_cross.py'),'--project',str(self.project),'--database',self.a.database,
            '--output',str(out),'--variant',j['variant'],'--seed',str(j['seed']),'--endpoint',str(j['endpoint']),
            '--action',j['action'],'--expected-gib',str(self.a.expected_gib)]
        if j['action']=='train': cmd+=['--pilot-record',self.state['jobs'][j['pilot_id']]['output']]
        if j['action']=='evaluate': cmd+=['--fit-output',self.state['jobs'][j['fit_id']]['output']]
        spec,receipt,log=[self.out/(j['id']+s) for s in ('_attempt.json','_exit.json','.log')]
        atomic(spec,dict(command=cmd,cwd=str(self.source),receipt=str(receipt),job_id=j['id'],source_manifest_sha256=self.state['source_manifest_sha256']))
        j.update(status='launching',gpu=gpu,slot=slot,command=cmd,log=str(log),exit_receipt=str(receipt)); self.save()
        with log.open('x') as f:
            child=subprocess.Popen([sys.executable,'-B',str(self.source/'scripts/tracehr_recovery_attempt.py'),'--spec',str(spec)],
                cwd=self.source,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        self.handles[j['id']]=child
        j.update(status='running',pid=child.pid,process_identity=process_identity(child.pid),started_at=time.time()); self.save()

    def admit(self):
        if self.state['status'] in ('complete','needs_attention','awaiting_path_review'): return
        running=[j for j in self.state['jobs'].values() if j['status']=='running']
        if len(running)>=6: self.state['admission_wait']='six-worker memory cap'; return
        if any(self.progress(j).get('status') in (None,'loading_verified_inputs','randomly_initializing_native') for j in running):
            self.state['admission_wait']='serialized initialization'; return
        action={'probes':'probe','formal':'train','evaluation':'evaluate'}[self.state['phase']]
        pending=[j for j in self.state['jobs'].values() if j['status']=='pending' and j['action']==action]
        if not pending: return
        now=time.time(); rows={r['index']:r for r in inventory(self.project)}
        for gpu,row in rows.items():
            if row['fully_idle'] and not any(j['gpu']==gpu for j in running): self.idle_since.setdefault(gpu,now)
            else: self.idle_since.pop(gpu,None)
        preferred=([2,3] if self.a.database=='eicu' else [0,1,4])+self.state['gpus']
        preferred=list(dict.fromkeys(preferred))
        idle=[g for g in preferred if g in self.idle_since and now-self.idle_since[g]>=10]
        if self.idle_since and not idle: self.state['admission_wait']='verifying continuously idle GPU'; return
        owned=[g for g in self.leases if g not in idle and g in rows and all(p['project_worker'] for p in rows[g]['processes'])]
        fallback=[g for g in self.state['fallback_gpus'] if g not in idle+owned and g in rows]
        j=pending[0]
        self.state['min_free_gpu_mib']=24576 if action=='probe' else max(8192,math.ceil(self.state['jobs'][j['pilot_id']]['result']['peak_reserved_mib']*1.3+2048))
        for gpu in idle+owned+fallback:
            on_gpu=[r for r in running if r['gpu']==gpu]
            if len(on_gpu)>=(1 if action=='probe' else 2) or sum(p['project_worker'] for p in rows[gpu]['processes'])>=2: continue
            slot=next(s for s in (0,1) if not any(r['slot']==s for r in on_gpu))
            if not self.resources(gpu): continue
            current=next(r for r in inventory(self.project) if r['index']==gpu)
            if gpu in idle and not current['fully_idle']: continue
            if gpu in owned and not all(p['project_worker'] for p in current['processes']): continue
            if sum(p['project_worker'] for p in current['processes'])>=2: continue
            reason='fully_idle' if gpu in idle else 'owned_idle_lease' if gpu in owned else 'authorized_fallback'
            if reason=='fully_idle': self.leases[gpu]=dict(acquired_at=now,evidence=current,idle_seconds=now-self.idle_since[gpu])
            j['gpu_admission']=dict(reason=reason,evidence=current,observed_at=time.time(),idle_lease=self.leases.get(gpu))
            self.state['idle_gpu_leases']=self.leases
            self.launch(j,gpu,slot); return
        self.state['admission_wait']='waiting for idle or authorized capacity'

    def validate(self,j):
        import numpy as np
        from sklearn.metrics import average_precision_score
        out=Path(j['output']); r=read(out/'completion.json'); i=r.get('identity',r); assert r['complete']
        for k,v in dict(database=self.a.database,variant=j['variant'],seed=j['seed'],endpoint=j['endpoint'],
            source_manifest_sha256=self.state['source_manifest_sha256']).items(): assert i[k]==v,k
        assert i['protocol']==protocol(j['variant'],j['seed']) and i['ema_rule']==EMA_RULE
        assert i['members']==8 and i['epoch_budget']==budget(j['seed'],j['endpoint'])
        a=read(out/'connection_audit.json'); obs=read(out/'initialization_observation.json')
        ref=expected()['full_ema'][f"fullEMA_t{j['endpoint']}_s{j['seed']}"]
        assert a['full_initial_weights_sha256']==ref['initial_weights_sha256']
        assert i['initial_weights_sha256']==obs['initial_weights_sha256']
        assert a['P_retained']==j['variant'].endswith('M') and a['M_retained']==j['variant'].endswith('P')
        assert a['original_static_buffers_exact'] and a['deletion']['retained_initials_exact'] and not a['learned_checkpoint_loaded']
        assert i['parameters']==ref['parameters']-a['deletion']['removed_parameters']
        assert not a['evidence_branch_present'] and not a['auxiliary_present']
        if j['action']=='evaluate':
            assert r['optimizer_updates']==0 and not r['parameters_updated'] and r['dev_replay_max_drift']<=1e-5
            assert r['model_sha256']==self.state['jobs'][j['fit_id']]['result']['checkpoint_sha256']
            assert r['parameter_source']=='EMA'
            split='test'; target_path=self.project/expected()['test_inputs']/'test_targets.npz'
        else:
            assert r['pilot']==j['pilot'] and r['from_scratch'] and not r['test_loaded'] and r['frozen_learnable_parameters']==0
            if j['pilot']:
                a=read(out/'path_audit.json'); assert a['complete'] and a['continuous_updates']==20
                assert a['all_raw_parameters_gradients_optimizer_rng_loss_exact'] and a['evaluation_preserves_raw_and_rng'] and a['resume_raw_optimizer_ema_rng_exact']
                assert all(v>0 for v in read(out/'gradient_check.json')['relative_update'].values())
                return r
            h=read(out/'history.json'); assert len(h)==r['epochs']==budget(j['seed'],j['endpoint'])
            assert sha(out/'best.pt')==r['checkpoint_sha256'] and r['checkpoint_replay']['max_probability_drift']<=1e-5
            assert r['optimizer_updates']==r['ema_updates']==len(h)*math.ceil(expected()['train_rows']/256)
            assert r['best_epoch']==1+int(np.argmax([v['dev_ap'] for v in h])) and r['parameter_source']=='EMA'
            for ep in EARLY:
                d=read(out/f'epoch_{ep:03d}_diagnostics.json'); assert sha(out/f'epoch_{ep:03d}_model.pt')==d['checkpoint_sha256']
                groups=d['gradient']['groups']['loss_paths']['groups']
                assert all(g['main_connected']==g['tensors'] and g['aux_connected']==0 for g in groups.values())
                if ep>1: assert all(v>0 for v in d['gradient']['groups']['relative_update'].values())
            split='dev'; target_path=Path(expected()['raw_inputs'])/'dev_targets.npz'
        pred=out/f'predictions_{split}.npz'; assert sha(pred)==r['predictions_sha256']
        z=dict(np.load(pred)); target=dict(np.load(target_path))
        np.testing.assert_array_equal(z['ids'],target['ids']); np.testing.assert_array_equal(z['labels'][:,0],target['y'][:,j['endpoint']])
        ap=float(average_precision_score(z['labels'][:,0],z['probs'][:,0]))
        assert abs(ap-(r['dev_ap'] if split=='dev' else r['metrics']['mean_ap']))<1e-12
        return r

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('project','source','output'): p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--database',choices=('mimic','eicu'),required=True); p.add_argument('--expected-gib',type=float,required=True)
    q=Queue(p.parse_args()); q.main()
    if q.state['status']!='complete': raise SystemExit(2)
