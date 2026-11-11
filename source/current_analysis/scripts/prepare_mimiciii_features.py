"""Fixed, train-fitted features for the MIMIC-III zero-update acceptance."""
import argparse
import collections
import gc
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.preprocessing import QuantileTransformer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.adapt_mimiciii import save, sha, emit
from scripts.tracehr_exact_statistics import EventStatistics
from scripts.tracehr_value_features import simple_summary, duration_summary


def main(out):
    start=time.monotonic()
    done=json.loads((out/'events/completion.json').read_text())
    assert done['complete'] and not done['pilot']
    extraction_cohort=out/'extraction_cohort.parquet'
    if not extraction_cohort.exists():extraction_cohort=out/'cohort.parquet'
    assert done['cohort_sha256']==sha(extraction_cohort)
    prep=out/'preparation';assert not prep.exists();prep.mkdir()
    cohort=pd.read_parquet(out/'cohort.parquet')
    old_cohort=pd.read_parquet(extraction_cohort)
    assert set(cohort.stay_id) <= set(old_cohort.stay_id)
    old_labels=old_cohort.set_index('stay_id').loc[cohort.stay_id,['mortality','los']].to_numpy()
    np.testing.assert_array_equal(old_labels,cohort[['mortality','los']].to_numpy())
    excluded_after_extraction=len(old_cohort)-len(cohort)
    if excluded_after_extraction:
        assert excluded_after_extraction==28
        assert done['source_sha256']==sha(ROOT/'work/mimiciii_adaptation_20261024/adapter_extraction_v1.py')
    splits={s:cohort[cohort.split.eq(s)].reset_index(drop=True) for s in ('train','dev','test')}
    joined=pd.concat(splits.values(),ignore_index=True)
    ids=joined.stay_id.to_numpy(dtype=np.int64); lookup={int(v):i for i,v in enumerate(ids)}
    train=set(splits['train'].stay_id)
    n=len(ids);ntr=len(splits['train']);ndv=len(splits['dev'])
    slices={'train':slice(0,ntr),'dev':slice(ntr,ntr+ndv),'test':slice(ntr+ndv,n)}
    subjects={s:set(f.subject_id) for s,f in splits.items()}
    assert all(not subjects[a]&subjects[b] for a,b in [('train','dev'),('train','test'),('dev','test')])
    sources=['chart','input','lab','procedure_icu']; source_index={s:i for i,s in enumerate(sources)}
    counts=collections.Counter(); dedup_removed=0
    files=sorted((out/'events').glob('cat_*.parquet'));assert len(files)==32
    for part,p in enumerate(files):
        f=pd.read_parquet(p);f=f[f.stay_id.isin(lookup)];before=len(f);f=f.drop_duplicates(['stay_id','time','token']);dedup_removed+=before-len(f)
        counts.update(f.loc[f.stay_id.isin(train),'token'].value_counts().to_dict())
        if part%8==0:emit(out,stage='train_only_vocabulary',buckets=part+1,total=32)
    names=[s+'::UNK' for s in sources]+sorted(k for k,c in counts.items() if c>=20 and not k.endswith('::UNK'))
    vocab={v:i for i,v in enumerate(names)};nv=len(vocab);phi=4*nv+2*len(sources)+4;dim=phi+832
    save(prep/'token_to_id.json',vocab)
    projected=n*dim*4
    assert projected<18*2**30,('Dense feature bound exceeded',projected)
    raw={s:np.lib.format.open_memmap(prep/(s+'_raw.npy'),mode='w+',dtype=np.float32,shape=(len(f),dim)) for s,f in splits.items()}
    source_counts=np.zeros((n,4),np.int64);oov_counts=np.zeros(n,np.int64);lengths=np.zeros(n,np.int64)
    seen=np.zeros(n,bool);checks=0
    split_idx=np.concatenate([np.full(len(f),j) for j,f in enumerate(splits.values())])
    split_offset=[0,ntr,ntr+ndv];split_names=list(splits)
    empty=EventStatistics(nv,sources,24.).features()
    for arr in raw.values(): arr[:,:phi]=empty;arr[:,phi:]=0
    for part,p in enumerate(files):
        f=pd.read_parquet(p);f=f[f.stay_id.isin(lookup)].drop_duplicates(['stay_id','time','token'])
        mapped=f.token.map(vocab);missing=mapped.isna()
        mapped.loc[missing]=(f.loc[missing,'source']+'::UNK').map(vocab)
        assert mapped.notna().all() and f.time.between(0,24,inclusive='both').all()
        f['type_id']=mapped.astype('int64');f['source_id']=f.source.map(source_index);f['oov']=missing
        for sid,g in f.groupby('stay_id',sort=False):
            i=lookup[int(sid)];assert not seen[i];seen[i]=True
            ti=g.type_id.to_numpy();tt=g.time.to_numpy();ss=g.source_id.to_numpy()
            actual=EventStatistics(nv,sources,24.).add(ti,tt,ss).features()
            if i%97==0:
                # Independent count/presence checks and exact split-stream aggregation.
                count=np.bincount(ti,minlength=nv)
                np.testing.assert_array_equal(actual[nv:2*nv],count>0)
                np.testing.assert_allclose(actual[:nv],np.log1p(count),rtol=2e-7,atol=2e-7)
                mid=len(ti)//2
                streamed=EventStatistics(nv,sources,24.).add(ti[:mid],tt[:mid],ss[:mid])
                streamed.merge(EventStatistics(nv,sources,24.).add(ti[mid:],tt[mid:],ss[mid:]))
                np.testing.assert_allclose(actual,streamed.features(),rtol=2e-6,atol=2e-6)
                checks+=1
            sn=split_names[split_idx[i]];row=i-split_offset[split_idx[i]]
            raw[sn][row,:phi]=actual
            source_counts[i]=np.bincount(ss,minlength=4);lengths[i]=len(g);oov_counts[i]=int(g.oov.sum())
        emit(out,stage='categorical_features',buckets=part+1,total=32,seconds=round(time.monotonic()-start,1))
        del f;gc.collect()
    f=pd.read_parquet(out/'events/numeric.parquet');f=f[f.stay_id.isin(lookup)].copy();f['key']=f.lab+'|'+f.unit
    coverage=f[f.stay_id.isin(train)].groupby(['lab','unit'],observed=True).stay_id.nunique().reset_index(name='stays')
    selected=coverage.sort_values(['stays','lab','unit'],ascending=[False,True,True]).drop_duplicates('lab').head(32)
    keys=(selected.lab+'|'+selected.unit).tolist();assert len(keys)==32
    f=f[f.key.isin(keys)].copy();before=len(f)
    f=f.groupby(['key','stay_id','available_h'],as_index=False,observed=True).agg(value=('value','median'),sample_h=('sample_h','max'))
    merged=before-len(f);norm={};num_present=np.zeros((n,32),bool)
    labdict=pd.read_parquet(out/'lab_dictionary.parquet').set_index('itemid')
    for j,key in enumerate(keys):
        g=f[f.key.eq(key)].sort_values(['stay_id','available_h'],kind='stable').copy()
        tr=g[g.stay_id.isin(train)].value.to_numpy();q=np.quantile(tr,[.25,.5,.75]);scale=float(q[2]-q[0])
        if scale<1e-8:scale=float(np.std(tr))
        if scale<1e-8:scale=1.
        g['z']=np.clip((g.value-q[1])/scale,-5.,5.)
        lab,unit=key.split('|',1)
        norm[key]=dict(train_median=float(q[1]),train_iqr_or_std=scale,train_events=len(tr),
            train_stays=int(g[g.stay_id.isin(train)].stay_id.nunique()),label=str(labdict.loc[int(lab),'label']),unit=unit)
        for sid,h in g.groupby('stay_id',sort=False):
            i=lookup[int(sid)];s=split_names[split_idx[i]];row=i-split_offset[split_idx[i]]
            tt=h.available_h.to_numpy();z=h.z.to_numpy()
            raw[s][row,phi+j*11:phi+(j+1)*11]=simple_summary(tt,z,h.sample_h.to_numpy())
            raw[s][row,phi+352+j*15:phi+352+(j+1)*15]=duration_summary(tt,z)
            num_present[i,j]=True
        if (j+1)%4==0:emit(out,stage='laboratory_features',labs=j+1,total=32,seconds=round(time.monotonic()-start,1))
    save(prep/'normalization.json',norm)
    del f,g;gc.collect()
    # Hand-check an all-window constant path and an all-missing path.
    expected=np.tile([1.,2.,4.,1.,0.],3)
    np.testing.assert_allclose(duration_summary([0.],[2.]),expected,rtol=1e-7)
    assert not duration_summary([],[]).any() and not simple_summary([],[],[]).any()
    center=np.zeros(dim,np.float32);factor=np.ones(dim,np.float32);binary=np.zeros(dim,bool);nonconstant=np.zeros(dim,bool)
    for begin in range(0,phi,128):
        end=min(begin+128,phi);block=np.asarray(raw['train'][:,begin:end])
        q1,med,q3=np.quantile(block,[.25,.5,.75],axis=0);lo,hi=block.min(0),block.max(0)
        spread=q3-q1;spread[spread==0]=.5*(hi-lo)[spread==0]
        scale=np.divide(1.,spread,out=np.zeros_like(spread),where=spread!=0)
        center[begin:end],factor[begin:end]=med,scale
        binary[begin:end]=np.all((block==0)|(block==1),axis=0)
        nonconstant[begin:end]=lo!=hi
    phi_center,phi_factor,phi_binary=center[:phi].copy(),factor[:phi].copy(),binary[:phi].copy()
    binary[phi+np.arange(32)*11]=True
    for begin in range(phi,dim,32):
        end=min(begin+32,dim);block=np.asarray(raw['train'][:,begin:end],dtype=np.float64)
        low,med,high=np.quantile(block,[.25,.5,.75],axis=0);spread=high-low
        spread=np.where(spread>1e-6,spread,np.std(block,axis=0))
        center[begin:end]=med;factor[begin:end]=1/np.maximum(spread,1.)
        nonconstant[begin:end]=block.min(0)!=block.max(0)
    center[binary]=0;factor[binary]=1
    indices=np.flatnonzero(~binary & nonconstant)
    np.savez_compressed(prep/'preprocessing.npz',center=center,factor=factor,binary=binary,numeric_indices=indices)
    qt=QuantileTransformer(n_quantiles=min(1000,ntr),output_distribution='normal',subsample=min(10000,ntr),random_state=42,copy=True)
    qt.fit(raw['train'][:,phi:])
    joblib.dump(dict(phi_center=phi_center,phi_factor=phi_factor,binary=phi_binary,numeric_quantile=qt,phi_dim=phi,fit_split='train'),prep/'tabm_preprocessing.joblib')
    aggregate=[]
    for s,g in splits.items():
        sl=slices[s];x=raw[s]
        for begin in range(0,len(x),512): assert np.isfinite(x[begin:begin+512]).all()
        x.flush()
        target=dict(ids=g[['subject_id','hadm_id','stay_id']].to_numpy(dtype=np.int64),y=g[['mortality','los']].to_numpy(dtype=np.int64),lengths=lengths[sl])
        np.savez_compressed(prep/(s+'_targets.npz'),**target)
        np.save(prep/(s+'_event_lengths.npy'),lengths[sl])
        aggregate.append(dict(split=s,stays=len(g),patients=int(g.subject_id.nunique()),mortality_positive=int(g.mortality.sum()),los_positive=int(g.los.sum()),
            mortality_rate=float(g.mortality.mean()),los_rate=float(g.los.mean()),events=int(lengths[sl].sum()),
            oov_events=int(oov_counts[sl].sum()),oov_rate=float(oov_counts[sl].sum()/max(lengths[sl].sum(),1))))
    coverage=[]
    for system in sorted(joined.dbsource.unique()):
        ix=joined.dbsource.eq(system).to_numpy()
        coverage.append(dict(system=system,stays=int(ix.sum()),
            sources={s:dict(stays=int((source_counts[ix,j]>0).sum()),fraction=float((source_counts[ix,j]>0).mean()),events=int(source_counts[ix,j].sum())) for j,s in enumerate(sources)},
            numeric_any_stays=int(num_present[ix].any(1).sum()),mean_selected_lab_coverage=float(num_present[ix].sum(1).mean()),
            per_lab_coverage={k:float(num_present[ix,j].mean()) for j,k in enumerate(keys)}))
    artifacts={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in prep.iterdir() if p.is_file()}
    result=dict(complete=True,database='mimiciii',preparation_complete=True,fit_split='train',
        train_rows=ntr,dev_rows=ndv,test_rows=len(splits['test']),input_dim=dim,phi_dim=phi,event_num_types=nv,
        source_names=sources,lab_keys=keys,feature_widths=dict(event_statistics=phi,simple=352,state=480),
        numeric_features=len(indices),binary_features=int(binary.sum()),constant_features=int((~nonconstant).sum()),
        cohort_sha256=sha(out/'cohort.parquet'),extraction_cohort_sha256=sha(extraction_cohort),
        excluded_after_extraction=excluded_after_extraction,
        cohort_correction='Normalize MIMIC-III midnight-formatted date-only DOD before unchanged label rules; filter extracted superset; retained labels unchanged',
        source_coverage=coverage,splits=aggregate,total_events=int(lengths.sum()),cross_chunk_duplicates_removed=dedup_removed,
        numeric_tie_rows_merged=merged,empty_event_stays=int((lengths==0).sum()),fewer_than_five_events=int((lengths<5).sum()),
        stream_checks=checks,lab_boundary_checks=2,patient_split_overlap=0,all_finite=True,
        train_only_vocabulary_min_frequency=20,train_only_lab_selection=True,train_only_scaling=True,
        test_features_constructed=True,test_used_for_fit=False,test_predictions_computed=False,optimizer_updates=0,
        lab_time_semantics='specimen timestamp; result availability unknown, not as-of results',
        preprocessing_sha256=sha(prep/'preprocessing.npz'),artifacts=artifacts,
        bytes=sum(a['bytes'] for a in artifacts.values()),seconds=time.monotonic()-start,
        rss_gib=psutil.Process().memory_info().rss/2**30,
        sources_sha256={p:sha(ROOT/p) for p in ['scripts/prepare_mimiciii_features.py','scripts/tracehr_value_features.py','scripts/tracehr_exact_statistics.py','train_tracehr_tabular.py']})
    save(prep/'completion.json',result)
    emit(out,stage='features_complete',stays=n,input_dim=dim,vocabulary=nv,gib=result['bytes']/2**30,seconds=result['seconds'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();assert out==Path(os.environ['PROJECT_RUN_DIR']).resolve() and out.is_relative_to(ROOT)
    main(out)
