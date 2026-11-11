"""Reconstruct token events at documented entry/availability times.

Medication cancellation-at-export and discharge-active fields are not consulted.
No cohort outcomes or held-out test split are read.
"""
import argparse,collections,gzip,json,os,re,sys,time,zipfile
from pathlib import Path
from contextlib import ExitStack
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.extract_tracehr_numeric import identities
from scripts.build_tracehr_retention import atomic,sha,configure_dataframe_runtime
SCHEMA=pa.schema([('stay_id',pa.int64()),('time',pa.float64()),('token',pa.string()),('source',pa.string())])
MIMIC={'chart':('icu/chartevents.csv.gz','charttime'), 'input':('icu/inputevents.csv.gz','starttime'),
       'procedure_icu':('icu/procedureevents.csv.gz','starttime'),'lab':('hosp/labevents.csv.gz','charttime')}
EICU={'diagnosis':('diagnosis.csv.gz','diagnosisoffset',['diagnosisstring','icd9code']),
      'medication':('medication.csv.gz','drugorderoffset',['drugstartoffset','drugname','drughiclseqno']),
      'infusionDrug':('infusionDrug.csv.gz','infusionoffset',['drugname']),
      'treatment':('treatment.csv.gz','treatmentoffset',['treatmentstring']),
      'lab':('lab.csv.gz','labresultrevisedoffset',['labresultoffset','labname'])}

def clean(s):
    return s.fillna('').astype(str).str.strip().str.replace(r'[|,;:/\\\t\r\n]+','_',regex=True).str.replace(r'\s+','_',regex=True).str.upper().str.strip('_').str.slice(0,120)

def eicu_token(c,source):
    if source=='diagnosis':
        code=clean(c.icd9code).str.split('_').str[0];text=clean(c.diagnosisstring)
        return ('diagnosis::ICD::'+code).where(code.ne(''),('diagnosis::TEXT::'+text).where(text.ne(''),''))
    if source=='medication':
        code=clean(c.drughiclseqno);text=clean(c.drugname)
        return ('medication::HICL::'+code).where(code.ne(''),('medication::DRUG::'+text).where(text.ne(''),''))
    col,prefix={'infusionDrug':('drugname','infusion::DRUG::'),'treatment':('treatmentstring','treatment::TEXT::'),
                'lab':('labname','lab::NAME::')}[source]
    text=clean(c[col]);return (prefix+text).where(text.ne(''),'')

def main():
    p=argparse.ArgumentParser();p.add_argument('--database',choices=['mimic','eicu'],required=True)
    p.add_argument('--max-rows-per-source',type=int,default=2000000);p.add_argument('--max-output-gib',type=float,default=2)
    a=p.parse_args();configure_dataframe_runtime();out=Path(os.environ['PROJECT_RUN_DIR']).resolve()
    assert out.is_relative_to(ROOT) and not (out/'raw').exists();(out/'raw').mkdir()
    ids,proofs=identities(a.database);selected=set(ids.stay_id);start=time.monotonic()
    audit={};sources={};nbytes=0
    if a.database=='mimic':
        info=pd.read_parquet('F:/Code/Mamba-Hawkes-Process/data/processed/mimiciv_exp2_v2_full/cohort_labels_v2.parquet',columns=['subject_id','hadm_id','stay_id','intime'])
        info=info[info.stay_id.isin(selected)];assert len(info)==len(selected)
    with ExitStack() as stack:
        writers=[stack.enter_context(pq.ParquetWriter(out/'raw'/f'{i:02d}.parquet',SCHEMA,compression='zstd')) for i in range(32)]
        zf=stack.enter_context(zipfile.ZipFile('F:/Data/eicu-collaborative-research-database-2.0.zip')) if a.database=='eicu' else None
        for source,spec in (MIMIC if a.database=='mimic' else EICU).items():
            stat=collections.Counter();audit[source]=stat
            with ExitStack() as inputs:
                if a.database=='mimic':
                    path=Path('F:/Data/mimic-iv-3.1')/spec[0]
                    if path.is_file():
                        sources[source]=dict(path=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
                        handle=inputs.enter_context(gzip.open(path,'rb'))
                    else:
                        mz=inputs.enter_context(zipfile.ZipFile('F:/Data/mimic-iv-3.1.zip'))
                        member='mimic-iv-3.1/'+spec[0]
                        raw=inputs.enter_context(mz.open(member));handle=inputs.enter_context(gzip.GzipFile(fileobj=raw))
                        sources[source]=dict(archive=str(mz.filename),member=member,member_bytes=mz.getinfo(member).file_size)
                    cols=['subject_id','hadm_id','itemid',spec[1],'storetime']+(['flag'] if source=='lab' else ['stay_id'])
                else:
                    raw=inputs.enter_context(zf.open('eicu-collaborative-research-database-2.0/'+spec[0]));handle=inputs.enter_context(gzip.GzipFile(fileobj=raw))
                    sources[source]=dict(archive=str(zf.filename),member=spec[0],member_bytes=zf.getinfo('eicu-collaborative-research-database-2.0/'+spec[0]).file_size)
                    cols=['patientunitstayid',spec[1]]+spec[2]
                reader=pd.read_csv(handle,usecols=cols,dtype=str,chunksize=250000,nrows=a.max_rows_per_source or None)
                for ci,c in enumerate(reader):
                    stat['scanned']+=len(c)
                    if a.database=='mimic':
                        keys=['subject_id','hadm_id']+([] if source=='lab' else ['stay_id'])
                        for k in keys:c[k]=pd.to_numeric(c[k],errors='coerce')
                        c=c[c.hadm_id.isin(info.hadm_id)] if source=='lab' else c[c.stay_id.isin(selected)]
                        c=c.merge(info,on=keys,how='inner',validate='many_to_many' if source=='lab' else 'many_to_one')
                        sample=(pd.to_datetime(c[spec[1]],errors='coerce')-c.intime).dt.total_seconds()/3600
                        entry=(pd.to_datetime(c.storetime,errors='coerce')-c.intime).dt.total_seconds()/3600
                        token=source+'::'+c.itemid
                        if source=='lab':token=token+'::'+c.flag.fillna('normal').str.strip().str.lower().str.replace(' ','_',regex=False).replace('','normal')
                    else:
                        c['stay_id']=pd.to_numeric(c.patientunitstayid,errors='coerce');c=c[c.stay_id.isin(selected)].copy()
                        entry=pd.to_numeric(c[spec[1]],errors='coerce')/60
                        if source=='lab':sample=pd.to_numeric(c.labresultoffset,errors='coerce')/60
                        elif source=='medication':sample=(pd.to_numeric(c.drugstartoffset,errors='coerce')/60).fillna(entry)
                        else:sample=entry
                        token=eicu_token(c,source)
                    inside=sample.between(0,48,inclusive='both');stat['measurement_0_48h']+=int(inside.sum())
                    for w in [24,48]:
                        m=sample.between(0,w,inclusive='both');stat[f'measurement_by_{w}h']+=int(m.sum())
                        stat[f'unavailable_by_{w}h']+=int((m&(entry.isna()|entry.gt(w))).sum())
                    stat['entry_before_measurement']+=int((inside&entry.lt(sample)).sum())
                    t=entry.where(entry.isna(),np.maximum(sample,entry))
                    keep=inside&t.between(0,48,inclusive='both')&token.ne('')
                    f=pd.DataFrame(dict(stay_id=c.stay_id,time=t,token=token,source=source))[keep]
                    f=f.drop_duplicates(['stay_id','time','token']);stat['retained_chunk_dedup']+=len(f)
                    for part,g in f.groupby(f.stay_id.astype('int64')%32,sort=False):
                        writers[int(part)].write_table(pa.Table.from_pandas(g,schema=SCHEMA,preserve_index=False))
                    if (ci+1)%8==0:
                        nbytes=sum(q.stat().st_size for q in (out/'raw').glob('*.parquet'))
                        state=dict(database=a.database,source=source,audit={k:dict(v) for k,v in audit.items()},elapsed_seconds=time.monotonic()-start,output_bytes=nbytes)
                        atomic(out/'progress.json',state);print(json.dumps({'source':source,**dict(stat),'seconds':state['elapsed_seconds']}),flush=True)
                        if nbytes>a.max_output_gib*2**30:raise RuntimeError('Output cap exceeded')
    artifacts={q.name:dict(bytes=q.stat().st_size,sha256=sha(q)) for q in (out/'raw').glob('*.parquet')}
    report=dict(status='complete',database=a.database,pilot=bool(a.max_rows_per_source),audit={k:dict(v) for k,v in audit.items()},
        source_files=sources,development_ids=proofs,test_loaded=False,outcome_columns_loaded=False,
        artifacts=artifacts,artifact_bytes=sum(v['bytes'] for v in artifacts.values()),elapsed_seconds=time.monotonic()-start,
        source_sha256=sha(Path(__file__)),time_rule='measurement/start and documented entry must both be in window; event time=max(measurement,start/entry)',
        deviations=['medication cancelled flag ignored because no cancellation time is recorded; tokens represent recorded orders whose start is no later than landmark',
                    'no discharge-active or retrospective priority features; remaining source-content revisions without version histories cannot be fully reconstructed'])
    atomic(out/'completion.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['artifacts','development_ids']}),flush=True)

if __name__=='__main__':main()
