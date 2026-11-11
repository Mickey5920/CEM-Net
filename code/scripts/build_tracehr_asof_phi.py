"""Refit vocabulary and rebuild exact statistics from recorded-availability events."""
import argparse,collections,gc,json,os,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_tracehr_retention import atomic,sha,configure_dataframe_runtime
from scripts.tracehr_exact_statistics import EventStatistics

def main():
    p=argparse.ArgumentParser();p.add_argument('--database',choices=['mimic','eicu'],required=True)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--pilot-stays',type=int,default=0)
    a=p.parse_args();configure_dataframe_runtime();out=Path(os.environ['PROJECT_RUN_DIR']).resolve()
    assert out.is_relative_to(ROOT) and not (out/'completion.json').exists();start=time.monotonic()
    source=json.loads((a.raw/'completion.json').read_text());assert source['status']=='complete'
    assert not source['pilot'] or a.pilot_stays
    old=ROOT/'runs'/('tracehr_fullstream_features_20261003' if a.database=='mimic' else 'tracehr_fullstream_eicu_features_20261003_r2')/(a.database+'24')
    prior=json.loads((old/'completion.json').read_text());targets={s:dict(np.load(old/(s+'_targets.npz'))) for s in ['train','dev']}
    if a.pilot_stays:targets={s:{k:v[:a.pilot_stays] for k,v in d.items()} for s,d in targets.items()}
    train=set(targets['train']['ids'][:,2]);ids=np.concatenate([targets[s]['ids'][:,2] for s in ['train','dev']]);selected=set(ids)
    assert len(set(ids))==len(ids) and not set(targets['train']['ids'][:,0])&set(targets['dev']['ids'][:,0])
    snames=sorted(source['audit']);si={s:i for i,s in enumerate(snames)};counts=collections.Counter()
    def read(path):
        return pd.read_parquet(path,filters=[('stay_id','in',ids.tolist()),('time','<=',24.)]).drop_duplicates(['stay_id','time','token'])
    for q in sorted((a.raw/'raw').glob('*.parquet')):
        assert sha(q)==source['artifacts'][q.name]['sha256'];f=read(q)
        counts.update(f[f.stay_id.isin(train)].token.value_counts().to_dict())
    names=[s+'::UNK' for s in snames]+sorted(t for t,n in counts.items() if n>=20 and not t.endswith('::UNK'))
    vocab={s:i for i,s in enumerate(names)};atomic(out/'token_to_id.json',vocab);nv=len(vocab)
    phi={};lengths={};oov=0;events=0
    for part,q in enumerate(sorted((a.raw/'raw').glob('*.parquet'))):
        f=read(q);mapped=f.token.map(vocab);miss=mapped.isna();oov+=int(miss.sum())
        mapped.loc[miss]=(f.loc[miss,'source']+'::UNK').map(vocab)
        assert mapped.notna().all();f['type_id']=mapped.astype('int64');f['source_id']=f.source.map(si)
        for sid,g in f.groupby('stay_id',sort=False):
            assert int(sid) not in phi
            x=EventStatistics(nv,snames,24.).add(g.type_id.to_numpy(),g.time.to_numpy(),g.source_id.to_numpy()).features()
            phi[int(sid)]=sparse.csr_matrix(x[None]);lengths[int(sid)]=len(g);events+=len(g)
        if (part+1)%4==0:
            atomic(out/'progress.json',dict(parts_complete=part+1,stays=len(phi),events=events,elapsed_seconds=time.monotonic()-start))
            print(json.dumps(dict(parts_complete=part+1,stays=len(phi),seconds=time.monotonic()-start)),flush=True)
        del f;gc.collect()
    empty=sparse.csr_matrix(EventStatistics(nv,snames,24.).features()[None]);artifacts={}
    for split in ['train','dev']:
        rows=[phi.get(int(s),empty) for s in targets[split]['ids'][:,2]]
        x=sparse.vstack(rows,format='csr');assert np.isfinite(x.data).all()
        path=out/(split+'_phi.npz');sparse.save_npz(path,x);artifacts[path.name]=dict(bytes=path.stat().st_size,sha256=sha(path))
        target=dict(targets[split]);target['lengths']=np.array([lengths.get(int(s),0) for s in target['ids'][:,2]],dtype=np.int64)
        path=out/(split+'_targets.npz');np.savez_compressed(path,**target);artifacts[path.name]=dict(bytes=path.stat().st_size,sha256=sha(path))
    report=dict(status='complete',setting=a.database+'24',pilot=bool(a.pilot_stays),test_loaded=False,
        tasks=prior['tasks'],num_types=nv,sources=snames,window=24,widths=dict(phi=4*nv+2*len(snames)+4,first=0,pair=0),
        splits={s:len(targets[s]['ids']) for s in targets},events=events,oov=oov,empty_stays=len(ids)-len(phi),
        artifacts=artifacts,artifact_bytes=sum(v['bytes'] for v in artifacts.values()),elapsed_seconds=time.monotonic()-start,
        vocabulary_sha256=sha(out/'token_to_id.json'),vocab_scope='refit on availability-filtered train events only; minimum token frequency 20',
        input_protocol='documented recorded-availability times; known start/measurement<=landmark; latest snapshot caveat remains',
        source_sha256=sha(Path(__file__)),raw_completion_sha256=sha(a.raw/'completion.json'),
        targets_from=str(old),target_role='existing 24h development train/dev; retained empty-input stays; unchanged labels and patient split')
    atomic(out/'completion.json',report);print(json.dumps({k:v for k,v in report.items() if k!='artifacts'}),flush=True)

if __name__=='__main__':main()
