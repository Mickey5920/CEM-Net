"""Training-only lab/unit selection and scaling; matched simple and duration summaries."""
import argparse,gc,json,os,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_tracehr_retention import atomic,sha,configure_dataframe_runtime
from scripts.tracehr_value_features import duration_summary,simple_summary,TAUS

def main():
    p=argparse.ArgumentParser();p.add_argument('--database',choices=['mimic','eicu'],required=True)
    p.add_argument('--numeric',type=Path,required=True);p.add_argument('--pilot-stays',type=int,default=0)
    a=p.parse_args();configure_dataframe_runtime();out=Path(os.environ['PROJECT_RUN_DIR']).resolve()
    assert out.is_relative_to(ROOT) and not (out/'completion.json').exists();start=time.monotonic()
    done=json.loads((a.numeric/'completion.json').read_text());assert done['complete'] and done['source_scan_complete']
    assert sha(a.numeric/'events.parquet')==done['events_sha256']
    prefix='tracehr_fullstream_features_20261003' if a.database=='mimic' else 'tracehr_fullstream_eicu_features_20261003_r2'
    cache=ROOT/'runs'/prefix/(a.database+'24');targets={s:dict(np.load(cache/(s+'_targets.npz'))) for s in ['train','dev']}
    tasks=json.loads((cache/'completion.json').read_text())['tasks']
    if a.pilot_stays:
        targets={s:{k:v[:a.pilot_stays] for k,v in d.items()} for s,d in targets.items()}
    train_ids=set(targets['train']['ids'][:,2]);ids=np.concatenate([targets[s]['ids'][:,2] for s in ['train','dev']]);lookup={int(s):i for i,s in enumerate(ids)}
    assert len(lookup)==len(ids) and not set(targets['train']['ids'][:,0])&set(targets['dev']['ids'][:,0])
    f=pd.read_parquet(a.numeric/'events.parquet',filters=[('stay_id','in',ids.tolist()),('available_h','<=',24.)])
    f['lab']=f.lab.astype(str);f['key']=f.lab+'|'+f.unit
    # Select at most one recorded unit for a lab, by train-stay coverage, then
    # choose 32 common lab identifiers. No cross-unit numerical pooling.
    cov=f[f.stay_id.isin(train_ids)].groupby(['lab','unit'],observed=True).stay_id.nunique().reset_index(name='stays')
    cov=cov.sort_values(['stays','lab','unit'],ascending=[False,True,True]).drop_duplicates('lab').head(32)
    keys=(cov.lab+'|'+cov.unit).tolist();assert len(keys)==32
    f=f[f.key.isin(keys)].copy();unit_kept=len(f)
    # Simultaneous available results receive a symmetric median, not an arbitrary order.
    duplicates=int(f.duplicated(['stay_id','key','available_h']).sum())
    f=f.groupby(['key','stay_id','available_h'],as_index=False,observed=True).agg(value=('value','median'),sample_h=('sample_h','max'))
    stats={};simple=np.zeros((len(ids),32*11),dtype=np.float32);state=np.zeros((len(ids),32*15),dtype=np.float32)
    for j,key in enumerate(keys):
        g=f[f.key.eq(key)].sort_values(['stay_id','available_h'],kind='stable').copy()
        tr=g[g.stay_id.isin(train_ids)].value.to_numpy();q=np.quantile(tr,[.25,.5,.75]);scale=float(q[2]-q[0])
        if scale<1e-8:scale=float(np.std(tr))
        if scale<1e-8:scale=1.
        g['z']=np.clip((g.value-q[1])/scale,-5.,5.)
        stats[key]=dict(train_median=float(q[1]),train_iqr_or_std=scale,train_events=len(tr),train_stays=int(g[g.stay_id.isin(train_ids)].stay_id.nunique()))
        for sid,h in g.groupby('stay_id',sort=False):
            ix=lookup[int(sid)];t=h.available_h.to_numpy();z=h.z.to_numpy()
            simple[ix,j*11:(j+1)*11]=simple_summary(t,z,h.sample_h.to_numpy())
            state[ix,j*15:(j+1)*15]=duration_summary(t,z)
        atomic(out/'progress.json',dict(labs_complete=j+1,labs_total=32,elapsed_seconds=time.monotonic()-start))
        if (j+1)%8==0:print(json.dumps(dict(labs_complete=j+1,seconds=time.monotonic()-start)),flush=True)
    n=len(targets['train']['ids']);artifacts={}
    for split,sl in [('train',slice(0,n)),('dev',slice(n,len(ids)))]:
        for name,x in [('simple',simple),('state',state)]:
            file=out/(split+'_'+name+'.npz');np.savez_compressed(file,x=x[sl]);artifacts[file.name]=dict(bytes=file.stat().st_size,sha256=sha(file))
        file=out/(split+'_targets.npz');np.savez_compressed(file,**targets[split]);artifacts[file.name]=dict(bytes=file.stat().st_size,sha256=sha(file))
    atomic(out/'normalization.json',stats)
    report=dict(status='complete',setting=a.database+'24',pilot=bool(a.pilot_stays),test_loaded=False,tasks=tasks,
        numeric_source_sha256=sha(a.numeric/'completion.json'),numeric_events_sha256=done['events_sha256'],
        keys=keys,widths=dict(simple=352,state=480),taus=TAUS.tolist(),basis=['1','z','z^2','relu(z-1)','relu(-z-1)'],
        path='right-continuous last available result; same-time median; no value before first result; no physiological interpolation claim',
        selected_rows_before_tie_merge=unit_kept,tie_rows_merged=duplicates,artifacts=artifacts,
        artifact_bytes=sum(v['bytes'] for v in artifacts.values()),elapsed_seconds=time.monotonic()-start,
        source_sha256={str(q.relative_to(ROOT)):sha(q) for q in [Path(__file__),ROOT/'scripts/tracehr_value_features.py']},
        normalization_sha256=sha(out/'normalization.json'),limitations=['recorded-unit semantics are database-specific; no external transfer or unit-conversion claim'])
    atomic(out/'completion.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['artifacts','source_sha256']}),flush=True)

if __name__=='__main__':main()
