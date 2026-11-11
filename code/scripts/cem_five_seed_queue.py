"""A single host owner; original GPU lease policy plus bounded CPU tree slots."""
import argparse, fcntl, math, os, shutil, time
from pathlib import Path
from scripts import cem_evidence_host_queue as prior
from scripts.cem_evidence_host_queue import read, sha, save, inventory

prior.RUN='cem_five_seed_20261102'

class Queue(prior.Queue):
    def __init__(self,a):
        super().__init__(a)
        self.state['expected_groups']=1 if a.host=='eicu' else 2
        self.state['authorized_formal_fits']=10 if a.host=='eicu' else 20
        self.state['max_cpu_workers']=1

    def discover(self):
        for file in sorted((self.out/'ready').glob('*.json')):
            key=file.stem
            if key in self.state['groups']:
                assert sha(file)==self.state['groups'][key]['ready_sha256'];continue
            d=read(file);source=Path(d['source']);out=Path(d['output'])
            assert source.resolve().is_relative_to(self.project) and out.resolve().is_relative_to(self.project)
            assert sha(source/'source_manifest.json')==d['source_manifest_sha256']
            for n,h in read(source/'source_manifest.json').items():assert sha(source/n)==h,n
            v=read(out.parent/'verification.json')
            assert v['complete'] and v['source_manifest_sha256']==d['source_manifest_sha256']
            assert v['formal_fits']==len(d['matrix']) and not (out/'state.json').exists()
            assert len({(j['variant'],j['seed'],j['endpoint']) for j in d['matrix']})==len(d['matrix'])
            g=dict(**d,ready_sha256=sha(file),phase='probes',status='preflight',jobs={})
            for cell in d['matrix']:
                v,s,e=cell['variant'],cell['seed'],cell['endpoint'];name=f'{v}_t{e}_s{s}'
                actions=[('train','','jobs'),('evaluate','test_','test')]
                if s==51:actions.insert(0,('probe','probe_','pilots'))
                for action,prefix,folder in actions:
                    jid=prefix+name
                    g['jobs'][jid]=dict(id=jid,**cell,action=action,status='pending',output=str(out/folder/jid),
                        pilot_id=f'probe_{v}_t{e}_s51',fit_id=name,device='cpu' if v.startswith('LightGBM') else 'gpu')
            self.state['groups'][key]=g

    def reap(self):
        for key,g in self.state['groups'].items():
            for j in g['jobs'].values():
                if j['status']!='running':continue
                receipt=Path(j['exit_receipt'])
                if not receipt.exists():
                    h=self.handles.get((key,j['id']))
                    if h is not None and h.poll() is not None:j.update(status='failed',error='Wrapper exited without a receipt')
                    continue
                rc=read(receipt)
                try:
                    assert rc['returncode']==0,f"Child exit {rc['returncode']}"
                    out=Path(j['output']);r=read(out/'completion.json');identity=r.get('identity',r)
                    assert r['complete'] and identity['source_manifest_sha256']==g['source_manifest_sha256']
                    for n in ('variant','seed','endpoint'):assert identity[n]==j[n],n
                    if j['action']=='probe':
                        audit=read(out/'path_audit.json');assert audit['complete']
                        if j['device']=='cpu':assert audit['continuous_tree_updates']==20 and audit['reload_prediction_exact']
                        else:
                            assert audit['continuous_updates']==20
                            assert audit.get('resume_raw_optimizer_ema_rng_exact',audit.get('resume_exact',False))
                        assert not r['test_loaded'] and r['from_scratch']
                    elif j['action']=='train':
                        assert not r['pilot'] and r['from_scratch'] and not r['test_loaded']
                        assert sha(out/r['checkpoint_file'])==r['checkpoint_sha256']
                        assert sha(out/'predictions_dev.npz')==r['predictions_sha256']
                        assert r['checkpoint_replay']['max_probability_drift']<=1e-5
                    else:
                        assert r['optimizer_updates']==0 and not r['parameters_updated'] and r['dev_replay_max_drift']<=1e-5
                        assert r['model_sha256']==g['jobs'][j['fit_id']]['result']['checkpoint_sha256']
                        assert sha(out/'predictions_test.npz')==r['predictions_sha256']
                    j.update(status='completed',result=r,finished_at=rc['finished_at'])
                except Exception as exc:j.update(status='failed',error=repr(exc),finished_at=time.time())
                h=self.handles.pop((key,j['id']),None)
                if h is not None:h.poll()

    def env(self,g,gpu):
        env=super().env(g,gpu)
        pyarrow=self.project/'work/cem_evidence_completion_20261025/test_runtime_v1/deps'
        if pyarrow.is_dir():env['PYTHONPATH']+=os.pathsep+str(pyarrow)
        if gpu<0:env['CUDA_VISIBLE_DEVICES']=''
        env['CEM_ICDE_STAGE']=g['stage']
        env['CEM_FIVE_VERIFICATION']=str(Path(g['output']).parent/'verification.json')
        return env

    def launch(self,key,g,j,gpu,slot,evidence):
        # Different model families share the scheduler, never their training code.
        copy=dict(g,entry=g['entries']['tree' if j['device']=='cpu' else 'cem' if j['variant'].startswith('CEM') else 'tabular'])
        super().launch(key,copy,j,gpu,slot,evidence)

    def admit(self,rows):
        running=[j for g in self.state['groups'].values() for j in g['jobs'].values() if j['status']=='running']
        initializing=(None,'loading_verified_inputs','randomly_initializing_native','preparing_inputs')
        if any(self.progress(j).get('status') in initializing for j in running):
            self.state['admission_wait']='serialized initialization';return
        mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
        by={r['index']:r for r in rows};now=time.time()
        groups=sorted(self.state['groups'].items(),key=lambda v:(v[1]['stage_order'],v[1].get('last_launch',0)))
        for key,g in groups:
            if g['status'] not in ('preflight','running','evaluating'):continue
            action={'probes':'probe','formal':'train','evaluation':'evaluate'}[g['phase']]
            pending=[j for j in g['jobs'].values() if j['status']=='pending' and j['action']==action]
            for j in pending:
                if shutil.disk_usage(self.project).free/2**30<10+g['expected_gib']:continue
                if j['device']=='cpu':
                    if sum(x['device']=='cpu' for x in running)>=self.state['max_cpu_workers']:continue
                    peak=0 if action=='probe' else g['jobs'][j['pilot_id']]['result']['peak_rss_gib']
                    needed=max(g['cpu_min_available_ram_gib'],peak*1.25+8)
                    if mem['MemAvailable']/2**30<needed:continue
                    self.launch(key,g,j,-1,0,dict(device='CPU',threads=4,available_ram_gib=mem['MemAvailable']/2**30,required_ram_gib=needed))
                    g['last_launch']=time.time();self.state['admission_wait']=None;return
                if mem['MemAvailable']/2**30<g.get('min_available_ram_gib',24):continue
                needed=g['probe_requirements_mib'][j['variant']] if action=='probe' else max(8192,math.ceil(g['jobs'][j['pilot_id']]['result']['peak_reserved_mib']*1.3+2048))
                for gpu in self.idle.candidates(rows,now):
                    row=by[gpu];active=[x for x in running if x['gpu']==gpu]
                    if len(active)>=2 or sum(p['project_worker'] for p in row['processes'])>=2 or row['free_mib']<needed:continue
                    if action=='probe' and active:continue
                    if gpu not in self.locks:
                        path=self.project/'work/tracehr_fullstream_20261003/locks'/f'gpu{gpu}.lock';path.parent.mkdir(parents=True,exist_ok=True)
                        h=path.open('a+')
                        try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:h.close();continue
                        self.locks[gpu]=h
                    current=next(x for x in inventory(self.project) if x['index']==gpu)
                    if current['free_mib']<needed or sum(p['project_worker'] for p in current['processes'])>=2:continue
                    if gpu not in self.idle.preferred and gpu not in self.idle.leases and not current['fully_idle']:continue
                    evidence=self.idle.acquire(gpu,current,now)
                    evidence.update(required_free_mib=needed,probe_requirement_basis=g.get('probe_requirement_basis'))
                    slot=next(s for s in (0,1) if not any(x['slot']==s for x in active))
                    self.launch(key,g,j,gpu,slot,evidence);g['last_launch']=time.time();self.state['admission_wait']=None;return
        self.state['admission_wait']='waiting for verified tasks or measured resources'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--host',choices=('mimic','eicu'),required=True)
    Queue(p.parse_args()).main()
