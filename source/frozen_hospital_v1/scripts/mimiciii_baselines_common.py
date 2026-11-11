"""Fixed input identity and per-method contracts; no fit or test selection here."""
import hashlib,json,pickle
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN='mimiciii_baselines_20261024'
SEEDS=(48,49,50)
CLINICAL=('mhp','amhp','s2p2','decoupled_mtpp_ode','ehrmamba','hymate','xtsformer','core_behrt','resmlp')
NUMERIC=('noEvidence','B0','TabM')
TASKS=('icu_mortality_after_24h','icu_los_gt_7d_after_24h_eligible')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def expected():return read(ROOT/'work'/RUN/'expected.json')
def matrix():
    return [dict(variant=v,seed=s,endpoints=([0,1] if v in CLINICAL else [e]),endpoint=(-1 if v in CLINICAL else e))
        for s in SEEDS for v in NUMERIC+CLINICAL for e in ((-1,) if v in CLINICAL else (0,1))]
def verify_inputs(project):
    import numpy as np
    from scripts.mimiciii_fixed_common import Data
    c=expected();p=Path(project)/c['preparation'];d=Data(Path(project)/c['numeric_preparation'])
    for n,a in c['inputs'].items():assert sha(p/n)==a['sha256'] and (p/n).stat().st_size==a['bytes'],n
    assert not any(x.name.startswith('test') for x in p.iterdir())
    counts={}
    for split in ('train','dev'):
        with (p/(split+'.pkl')).open('rb') as f:r=pickle.load(f)
        assert r['tasks']==list(TASKS) and r['dim_process']==d.num_types
        rows=r[split]
        ids=np.array([[a[k] for k in ('subject_id','hadm_id','stay_id')] for a in rows])
        y=np.array([[a['labels'][t] for t in TASKS] for a in rows])
        np.testing.assert_array_equal(ids,d.targets[split]['ids']);np.testing.assert_array_equal(y,d.targets[split]['y'])
        assert all(a['length']==len(a['type_event'])<=512 for a in rows)
        assert all(0<=v<d.num_types for a in rows for v in a['type_event'])
        assert all(0<=t<=24 for a in rows for t in a['time_since_start'])
        counts[split]=dict(stays=len(rows),empty=sum(a['length']==0 for a in rows),labels=y.sum(0).tolist())
    assert len(matrix())==45 and sum(len(j['endpoints']) for j in matrix())==72
    return dict(complete=True,counts=counts,ids_labels_match=True,test_loaded=False)
def numeric_protocol(variant,seed):
    if variant=='noEvidence':
        from scripts.mimiciii_fixed_common import protocol
        p=protocol('V1_group_rate','mimiciii',seed)
        p.update(variant=variant,structure=variant,description='delete the entire evidence-enhancement branch',
            evidence_enabled=False,additional_channels=0,main_residual_enabled=False)
        return p
    if variant=='B0':
        from scripts.tracehr_execution_refine_common import protocol
        return dict(protocol('F0'),seed=seed,database='mimiciii')
    assert variant=='TabM'
    return dict(variant=variant,seed=seed,database='mimiciii',epochs=256,batch=256,k=32,d_block=256,
        n_blocks=3,dropout=.2,optimizer='AdamW',lr=.0005,weight_decay=.001,patience=16,
        selection='endpoint dev AP',precision='float32',loss='mean member BCE',tail='retained')
