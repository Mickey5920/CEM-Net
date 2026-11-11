"""Immutable, storage-guarded extension of Table VI from three to five seeds."""
import argparse, base64, hashlib, json, math, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.launch_ema_cross import HOSTS, ssh
from scripts.launch_cem_icde72 import read, save
from scripts.launch_cem_replacements import REMOTE as OLD_REMOTE
RUN = 'cem_five_seed_20261102'
def sha_file(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
CELLS = {'mimiciii': {0: ('CEM','TabM_PLE','TabM_mini_PLE'), 1: ('CEM','TabM_mini','LightGBM_full')},
         'mimic': {0: ('CEM','TabM_PLE'), 1: ('CEM','TabM_PLE')},
         'eicu': {0: ('CEM','TabM_mini_PLE'), 1: ('CEM','TabM_mini_PLE','TabM_PLE')}}

def replace_once(text, before, after):
    assert text.count(before) == 1, (before[:100], text.count(before))
    return text.replace(before, after, 1)

def generated(files):
    def original(name): return base64.b64decode(files['scripts/'+name+'.py']).decode()
    made = {}
    m = base64.b64decode(files['models/tracehr_noa_main_confirmation.py']).decode()
    m = replace_once(m, 'seed in (42,43,44,45,46,47,48,49,50)', 'seed in (42,43,44,45,46,47,48,49,50,51,52)')
    made['models/five_seed_noa_main_confirmation.py'] = m
    m = base64.b64decode(files['models/tracehr_noa_validation.py']).decode()
    m = replace_once(m, 'seed in (48,49,50)', 'seed in (48,49,50,51,52)')
    m = replace_once(m, 'from models.tracehr_noa_main_confirmation import', 'from models.five_seed_noa_main_confirmation import')
    made['models/five_seed_noa_validation.py'] = m
    route = base64.b64decode(files['scripts/cem_v1_group_confirm_seed_route.py']).decode()
    route = replace_once(route, 'original_init_source=textwrap.dedent(inspect.getsource(original.EvidenceStructure.__init__))',
        'from scripts import cem_v1_group_confirm_seed_route as legacy\noriginal_init_source=legacy.original_init_source')
    route = replace_once(route, "new='assert variant in VARIANTS and seed in (48, 49, 50)'", "new='assert variant in VARIANTS and seed in (48, 49, 50, 51, 52)'")
    made['scripts/five_seed_constructor_route.py'] = route
    runtime = original('ema_internal_runtime')
    runtime = runtime.replace('from scripts.ema_internal_common import EARLY, budget, atomic, read, sha',
        'from scripts.cem_five_seed_common import EARLY, budget, budget_stop, atomic, read, sha')
    runtime = runtime.replace('if a.seed == 48:', 'if a.seed == 51:').replace('if a.seed==48 else None','if a.seed==51 else None')
    runtime = runtime.replace('budget(48, a.endpoint)', 'budget(51, a.endpoint)').replace('path_pilot_seed=48','path_pilot_seed=51')
    runtime = replace_once(runtime, "observation['full_initial_tensors_exact'] and observation['deletion']['retained_initials_exact']", "observation['full_initial_tensors_exact'] and observation['original_static_buffers_exact']")
    runtime = replace_once(runtime, "    del opt, ema; gc.collect()", "        stop = budget_stop(history, spec)\n        atomic(out/'budget_decision.json', dict(**stop, epoch=epoch, source='raw z0 Validation AP; frozen original stopping rule', EMA_used_for_budget=False, test_used=False))\n        if stop['stop']: break\n    limit = len(history)\n    del opt, ema; gc.collect()")
    runtime = replace_once(runtime, "stopping_status=dict(stop=True, reason='predeclared full EMA epoch budget', epoch_budget=limit),", "stopping_status=read(out/'budget_decision.json'),")
    runtime = runtime.replace("termination='fixed_original_budget'", "termination='inherited_original_raw_stopping_rule'")
    made['scripts/five_seed_cem_runtime.py'] = runtime
    entry = original('train_cem_evidence_controls').replace('from scripts.cem_evidence_common import *','from scripts.cem_five_seed_common import *\ndef variants(): return (\'CEM\',)')
    made['scripts/five_seed_cem_entry.py'] = entry
    entry = original('train_cem_additional_baseline').replace('a.seed in (48,49,50)', 'a.seed in (51,52)').replace('if a.seed==48:', 'if a.seed==51:')
    made['scripts/five_seed_tabular_entry.py'] = entry
    entry = (ROOT/'scripts/train_cem_icde72_lgbm.py').read_text()
    entry = entry.replace('choices=(48,49,50)', 'choices=(51,52)').replace('resource_path_seed=48','resource_path_seed=51')
    entry = entry.replace('work/cem_icde_evidence_20261028/config.json',f'work/{RUN}/config.json')
    made['scripts/five_seed_lgbm_entry.py'] = entry
    host = (ROOT/'scripts/cem_evidence_host_queue.py').read_text().replace('scripts/tracehr_recovery_attempt.py','scripts/five_seed_attempt.py')
    made['scripts/cem_evidence_host_queue.py'] = host
    queue = (ROOT/'scripts/cem_icde72_queue.py').read_text().replace("prior.RUN='cem_icde_evidence_20261028'",f"prior.RUN='{RUN}'")
    queue = queue.replace("3 if a.host=='eicu' else 4", "1 if a.host=='eicu' else 2").replace("self.state['authorized_formal_fits']=36", "self.state['authorized_formal_fits']=10 if a.host=='eicu' else 20")
    queue = queue.replace('if s==48:', 'if s==51:').replace("_s48'", "_s51'")
    queue = queue.replace("g['probe_gpu_mib'] if action=='probe'", "g['probe_requirements_mib'][j['variant']] if action=='probe'")
    queue = replace_once(queue, "        return env", "        env['CEM_FIVE_VERIFICATION']=str(Path(g['output']).parent/'verification.json')\n        return env")
    made['scripts/cem_five_seed_queue.py'] = queue
    made['scripts/five_seed_attempt.py'] = (ROOT/'scripts/tracehr_recovery_attempt.py').read_text()
    for n in ('cem_five_seed_common','cem_five_seed_entry','cem_icde72_common'):
        made['scripts/'+n+'.py'] = (ROOT/'scripts'/ (n+'.py')).read_text()
    for n, text in made.items():
        assert n not in files, n
        compile(text, n, 'exec')
    return made

def build(a, out):
    parent = ROOT/'runs/cem_evidence_completion_20261025'/f'local_{a.database}_tabular_v1'
    files = read(parent/'source_bundle.json'); inherited = read(parent/'source_manifest.json')
    c = read(parent/'build-receipt.json')['contract']
    replay = read(ROOT/'runs'/RUN/'preflight/budget-replay-inventory.json')
    assert all(r['raw_rule_first_stop']==r['recorded_epochs'] for host in replay.values() for r in host['budget_replay'])
    matrix = [dict(variant=v,seed=s,endpoint=e) for s in (51,52) for e,vs in CELLS[a.database].items() for v in vs]
    assert len(matrix) == {'mimiciii':12,'mimic':8,'eicu':10}[a.database]
    c.update(run=RUN, database=a.database, stage='five_seed', snapshot=a.snapshot, seeds=[51,52], variants=sorted({j['variant'] for j in matrix}),
        matrix=matrix, formal_fits=len(matrix), expected_gib=a.expected_gib,
        budget_inheritance=dict(rule='original raw-model Validation AP stopping computed on the identical raw trajectory within EMA training',
            boundaries=[52,120,256], min_delta=.0001, patience=17, eicu_fixed_epochs=256,
            EMA_AP_used_for_budget=False, test_used=False, extra_predecessor_fits=0,
            original_18_budgets_replayed=True, replay_sha256=hashlib.sha256((ROOT/'runs'/RUN/'preflight/budget-replay-inventory.json').read_bytes()).hexdigest()),
        selection='unchanged EMA z0 Validation AP; fixed raw budget rule; unchanged model-specific baseline selection',
        authorization='Only CEM and all existing Table VI references; 30 new fits and 30 locked tests, seeds 51/52',
        inference=dict(old_three_seed_results_immutable=True, seeds=[48,49,50,51,52], bootstrap_replicates=20000,
            reuse_old_patient_indices_and_AP_distributions=True, no_pooled_predictions=True,
            contrasts_per_split=9, family='nine preselected Table VI contrasts; original conservative 54-comparison Bonferroni coverage retained',
            adjusted_alpha_denominator=54, old_analysis_families_unchanged=True,
            tests='centered paired patient-cluster Bootstrap approximation; Bonferroni p; exact five-seed signed-rank descriptive supplement',
            best_reference_not_reselected=True, no_extra_calibration=True))
    if a.snapshot in ('v2','v3'):
        predecessors={}
        for version in (('v1',) if a.snapshot=='v2' else ('v1','v2')):
            previous=ROOT/'runs'/RUN/f'local_{a.database}_{version}'
            predecessors[version]=dict(source_manifest_sha256=read(previous/'build-receipt.json')['source_manifest_sha256'],
                failures={f.name:sha_file(f) for f in previous.glob('verify-failure-*.json')})
        c['execution_predecessor']=dict(versions=predecessors,formal_fits_started=0,test_evaluations_started=0,
            change='extend legacy seed-ID admission; preserve every constructor calculation; verify seed-dependent initialization buffers against independent same-seed reconstruction rather than seed48')
    resources = read(ROOT/'runs'/RUN/'preflight/resource-reuse.json')
    c['reused_probe_resources'] = [r for r in resources if r['db']==a.database]
    c['probe_requirements_mib'] = {v:max(8192, math.ceil(max([r['peak_reserved_mib'] for r in resources if r['db']==a.database and r['variant']==v] or [4600])*1.3+2048)) for v in c['variants']}
    if a.database=='mimiciii':
        c['recipes']['TabM_mini'] = dict(epochs=256,batch=256,micro=256,k=32,d_block=256,n_blocks=3,dropout=.2,
            optimizer='AdamW',lr=.0005,weight_decay=.001,patience=16,selection='validation AP',precision='float32',
            loss='mean member BCE',tail='retained',architecture='tabm-mini',embedding=None)
    for n, text in generated(files).items(): files[n]=base64.b64encode(text.encode()).decode()
    files[f'work/{RUN}/config.json'] = base64.b64encode(json.dumps(c,indent=2).encode()).decode()
    manifest={n:hashlib.sha256(base64.b64decode(b)).hexdigest() for n,b in files.items()}
    assert all(manifest[n]==h for n,h in inherited.items())
    assert not (out/'source_manifest.json').exists()
    save(out/'source_bundle.json',files); save(out/'source_manifest.json',manifest)
    result=dict(complete=True,source_manifest_sha256=hashlib.sha256((out/'source_manifest.json').read_bytes()).hexdigest(),
        files=len(files), inherited_files=len(inherited),contract=c)
    save(out/'build-receipt.json',result)
    return {k:v for k,v in result.items() if k!='contract'}

VERIFY = '''elif d['mode']=='verify':
 assert not (out.parent/'verification.json').exists()
 assert sha(source/'source_manifest.json')==d['source_sha256']
 for n,h in read(source/'source_manifest.json').items():assert sha(source/n)==h,n
 os.environ.update(env());sys.path[:0]=os.environ['PYTHONPATH'].split(os.pathsep)
 from scripts.cem_five_seed_common import verify
 result=verify(p);result.update(source_manifest_sha256=d['source_sha256'],verified_at=time.time())
 save(out.parent/'verification.json',result)
'''
REGISTER = '''elif d['mode']=='register':
 v=read(out.parent/'verification.json');assert v['complete'] and v['source_manifest_sha256']==d['source_sha256']
 c=read(source/('work/'+d['run']+'/config.json'));file=host/'ready'/(d['database']+'_five_seed.json');assert not file.exists()
 result=dict(database=d['database'],stage='five_seed',stage_order=1,source=str(source),output=str(out),
  source_manifest_sha256=d['source_sha256'],matrix=c['matrix'],variants=c['variants'],
  entries={k:'scripts/cem_five_seed_entry.py' for k in ('tree','cem','tabular')},
  expected_gib=d['expected_gib'],retention_multiplier=13,min_available_ram_gib=24,cpu_min_available_ram_gib=24,
  probe_requirements_mib=c['probe_requirements_mib'],
  probe_gpu_mib=8192,probe_requirement_basis='same full CEM / TabM / PLE paths and dimensions as completed runs; measured bounded probes before release')
 save(file,result)
'''
RELEASE = '''elif d['mode']=='release':
 s=read(out/'state.json');assert s['status']=='awaiting_path_review';assert not (out/'formal_release.json').exists()
 c=read(source/('work/'+d['run']+'/config.json'))
 assert s['counts']['pending']==len(c['matrix']) and s['projected_artifact_gib']<s['expected_gib']
 audits={};reports=[]
 for j in s['jobs'].values():
  if j['action']!='probe':continue
  assert j['status']=='completed' and read(j['exit_receipt'])['returncode']==0
  folder=Path(j['output']);r=read(folder/'path_audit.json');assert r['complete']
  if j['device']=='cpu':assert r['continuous_tree_updates']==20 and r['reload_prediction_exact']
  elif j['variant']=='CEM':
   assert r['continuous_updates']==20 and r['all_raw_parameters_gradients_optimizer_rng_loss_exact']
   assert r['evaluation_preserves_raw_and_rng'] and r['resume_raw_optimizer_ema_rng_exact']
   assert all(v>0 for v in read(folder/'gradient_check.json')['relative_update'].values())
  else:assert r['continuous_updates']==20 and r['resume_exact'] and r['all_parameters_updated']
  assert not j['result']['test_loaded'] and j['result']['from_scratch']
  audits[j['id']]=sha(folder/'path_audit.json');reports.append(dict(id=j['id'],resources=j['result']))
 result=dict(complete=True,source_manifest_sha256=d['source_sha256'],audit_sha256=audits,
  projected_artifact_gib=s['projected_artifact_gib'],reports=reports,released_at=time.time())
 save(out/'formal_release.json',result)
'''
REMOTE = OLD_REMOTE[:OLD_REMOTE.index("elif d['mode']=='verify':")] + VERIFY + REGISTER + OLD_REMOTE[OLD_REMOTE.index("elif d['mode']=='start':"):OLD_REMOTE.index("elif d['mode']=='allocator-handoff':")] + RELEASE + OLD_REMOTE[OLD_REMOTE.index("elif d['mode']=='status':"):]
REMOTE = REMOTE.replace("CEM_COMPLETION_STAGE='sequence'", "CEM_COMPLETION_STAGE='five_seed'")
REMOTE = REMOTE.replace('scripts/cem_replacement_queue.py','scripts/cem_five_seed_queue.py').replace('scripts/tracehr_recovery_attempt.py','scripts/five_seed_attempt.py').replace("'cem_replacements_'", "'cem_five_seed_'")

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['build','deploy','verify','register','start','release','status'])
    p.add_argument('--database',required=True,choices=list(CELLS));p.add_argument('--expected-gib',type=float,required=True)
    p.add_argument('--snapshot',choices=('v1','v2','v3'),default='v3')
    a=p.parse_args();a.host='eicu' if a.database=='eicu' else 'mimic'
    out=Path(os.environ['PROJECT_RUN_DIR']).resolve();assert out.is_relative_to(ROOT)
    if a.mode=='build': result=build(a,out)
    else:
        built=read(out/'build-receipt.json');assert built['contract']['expected_gib']==a.expected_gib
        d=dict(project=HOSTS[a.host][1],host=a.host,run=RUN,mode=a.mode,database=a.database,
            source=f'work/{RUN}/{a.database}/{a.snapshot}',output=f'runs/{RUN}/{a.database}/{a.snapshot}/screen',
            source_sha256=built['source_manifest_sha256'],expected_gib=a.expected_gib)
        if a.mode=='deploy':d.update(files=read(out/'source_bundle.json'),hashes=read(out/'source_manifest.json'),manifest_bytes=base64.b64encode((out/'source_manifest.json').read_bytes()).decode())
        code=REMOTE.replace('PAYLOAD',repr(d),1);compile(code,'five_seed_remote','exec')
        try:
            result=ssh(a.host,code,1800)
        except Exception as exc:
            import time
            save(out/(a.mode+'-failure-'+str(time.time_ns())+'.json'),dict(error=repr(exc),mode=a.mode,database=a.database))
            raise
        save(out/(a.mode+'-receipt.json'),result)
    if a.mode=='status':
        result=dict(status=result['status'],exit_receipt=result.get('exit_receipt'),groups={k:{n:g.get(n) for n in ('status','counts','probe_counts','evaluation_counts','projected_artifact_gib')} for k,g in result['groups'].items()},
            active=[{n:j.get(n) for n in ('id','status','gpu','progress','error','log_tail')} for g in result['groups'].values() for j in g['jobs'].values() if j['status'] in ('running','failed')])
    print(json.dumps(result))
if __name__=='__main__':main()
