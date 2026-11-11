"""Complete the failed MIMIC-IV cost measurement in its original feature runtime.

Read existing local source tables only; never replace model inputs or predictions.
"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='2'
os.environ['NUMEXPR_NUM_THREADS']='1'
import gc,hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits,threadpool_info

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.tracehr_exact_statistics import EventStatistics
from scripts.tracehr_value_features import simple_summary,duration_summary
from scripts.build_tracehr_retention import configure_dataframe_runtime

RUN='cem_icde_evidence_20261028'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def reconstruct(folder,ids,vocab,keys,norm,sources):
    phi=4*len(vocab)+2*len(sources)+4
    a=np.zeros((len(ids),phi+832),np.float32)
    a[:,:phi]=EventStatistics(len(vocab),sources,24.).features()
    lookup={int(s):i for i,s in enumerate(ids[:,2])};allowed=list(lookup)
    source_index={s:i for i,s in enumerate(sources)};total=dict(categorical=0,numeric=0)
    for path in sorted((folder/'events/raw').glob('*.parquet')):
        f=pd.read_parquet(path,filters=[('stay_id','in',allowed),('time','<=',24.)]).drop_duplicates(['stay_id','time','token'])
        mapped=f.token.map(vocab);missing=mapped.isna()
        mapped.loc[missing]=(f.loc[missing,'source']+'::UNK').map(vocab)
        assert mapped.notna().all()
        f['type_id']=mapped.astype('int64');f['source_id']=f.source.map(source_index)
        for sid,g in f.groupby('stay_id',sort=False):
            a[lookup[int(sid)],:phi]=EventStatistics(len(vocab),sources,24.).add(g.type_id.to_numpy(),g.time.to_numpy(),g.source_id.to_numpy()).features()
        total['categorical']+=len(f)
    f=pd.read_parquet(folder/'numeric/events.parquet',filters=[('stay_id','in',allowed),('available_h','<=',24.)])
    f['key']=f.lab.astype(str)+'|'+f.unit;f=f[f.key.isin(keys)].copy()
    f=f.groupby(['key','stay_id','available_h'],as_index=False,observed=True).agg(value=('value','median'),sample_h=('sample_h','max'))
    total['numeric']=len(f)
    for j,key in enumerate(keys):
        g=f[f.key.eq(key)].sort_values(['stay_id','available_h'],kind='stable').copy();s=norm[key]
        g['z']=np.clip((g.value-s['train_median'])/s['train_iqr_or_std'],-5.,5.)
        for sid,h in g.groupby('stay_id',sort=False):
            t=h.available_h.to_numpy();z=h.z.to_numpy();row=lookup[int(sid)]
            a[row,phi+j*11:phi+(j+1)*11]=simple_summary(t,z,h.sample_h.to_numpy())
            a[row,phi+352+j*15:phi+352+(j+1)*15]=duration_summary(t,z)
    return a,total

def main():
    out=Path(os.environ['PROJECT_RUN_DIR']).resolve()
    assert out==ROOT/'runs'/RUN/'mimic_secondary_v1'
    dest=out/'pipeline_original_runtime_v1';dest.mkdir(exist_ok=True)
    assert not (dest/'completion.json').exists(),'Completed timing must not be rerun'
    configure_dataframe_runtime()
    folder=ROOT/'runs/tracehr_pm_test_20261021/mimic_features'
    phi=ROOT/'runs/tracehr_phi_full_mimic_20261003';values=ROOT/'runs/tracehr_values_full_mimic_20261003'
    pm=read(phi/'completion.json');vm=read(values/'completion.json');test=read(folder/'completion.json')
    vocab=read(phi/'token_to_id.json');norm=read(values/'normalization.json');keys=vm['keys'];sources=pm['sources']
    files={folder/n:v['sha256'] for n,v in test['artifacts'].items()}
    files.update({phi/'token_to_id.json':test['train_vocabulary_sha256'],values/'normalization.json':test['train_value_scales_sha256'],folder/'numeric/events.parquet':read(folder/'numeric/completion.json')['events_sha256']})
    eventmeta=read(folder/'events/completion.json')
    files.update({folder/'events/raw'/n:v['sha256'] for n,v in eventmeta['artifacts'].items()})
    started=time.perf_counter()
    for p,digest in files.items():assert sha(p)==digest,str(p)
    hash_seconds=time.perf_counter()-started
    raw=np.load(folder/'test_raw.npy',mmap_mode='r');ids=np.load(folder/'test_targets.npz')['ids'];rows=[]
    with threadpool_limits(limits=2):
        pools=threadpool_info()
        for n in (128,512,2048):
            ix=np.unique(np.linspace(0,len(raw)-1,min(n,len(raw)),dtype=int));times=[]
            for repeat in range(3):
                began=time.perf_counter();a,total=reconstruct(folder,ids[ix],vocab,keys,norm,sources);elapsed=time.perf_counter()-began
                np.testing.assert_array_equal(a,raw[ix])
                times.append(elapsed);del a;gc.collect()
                print(json.dumps(dict(rows=len(ix),repeat=repeat+1,exact=True,seconds=elapsed)),flush=True)
            rows.append(dict(stays=len(ix),categorical_records=total['categorical'],numerical_records=total['numeric'],dense_dimension=raw.shape[1],repeats=3,seconds_excluding_integrity_hash=times,median_seconds=float(np.median(times)),integrity_verification_seconds=[0.,0.,0.],raw_reconstruction_exact=True))
    for p,digest in files.items():assert sha(p)==digest,'Source changed during timing'
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        cpu_name=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    result=dict(complete=True,database='mimic',source_sha256=sha(Path(__file__)),rows=rows,threads=2,fit_transforms=False,optimizer_updates=0,
        cpu=cpu_name,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,threadpools=pools,
        reference_raw_sha256=test['artifacts']['test_raw.npy']['sha256'],reference_targets_sha256=test['artifacts']['test_targets.npz']['sha256'],
        train_normalization_sha256=test['train_value_scales_sha256'],train_vocabulary_sha256=test['train_vocabulary_sha256'],
        feature_functions_sha256={n:sha(ROOT/'scripts'/n) for n in ('tracehr_exact_statistics.py','tracehr_value_features.py')},
        source_table_hashes={str(p.relative_to(ROOT)):v for p,v in files.items()},integrity_verification_seconds=hash_seconds,
        runtime_repair=dict(reason='Original Windows MKL feature producer and remote Linux OpenBLAS differ on one near-zero slope; exact equality retained by using the original producer runtime.',remote_mismatched_elements=1,remote_checked_elements=25616384,remote_max_absolute_difference=6.991092309281775e-17,normalized_numeric_input_sha256='5ad710d3c4780628b98116851b4da7f3ca8d1e34e56b3f7067d6c97d235c012c',normalized_input_groups=53283),
        scope='Filtered existing source-table reads, aggregation, frozen normalization and dense materialization on the original local feature-production CPU; model inference was measured separately on the remote GPU.',
        cache_policy='OS cache not flushed. Source hashes checked before and after all measurements; verification cost reported separately.',
        limitation='MIMIC-IV feature-pipeline timing uses a different CPU/runtime from the remote database timings and GPU inference; no direct cross-host speed comparison. Three fixed subset sizes do not establish distributed scalability.',patient_records_exported=False)
    payload=json.dumps(result,indent=2).encode('utf-8')
    with (dest/'completion.json').open('xb') as f:f.write(payload)
    print(json.dumps(dict(complete=True,receipt=str(dest/'completion.json'),sha256=hashlib.sha256(payload).hexdigest())))

if __name__=='__main__':main()
