"""Train-only reconstruction for the authorized, fixed eICU hospital split."""
import argparse, base64, collections, gc, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

RUN='cem_icde_evidence_20261028'
ROOT=Path(__file__).resolve().parents[1]

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.writing');q.write_text(json.dumps(v,indent=2),encoding='utf-8');os.replace(q,p)

def stage(out):
    import numpy as np,pandas as pd
    assert not (out/'sources.json').exists()
    mapping_file=ROOT/'runs'/RUN/'local_mimic/inspect-receipt.json'
    mapping=read(mapping_file)['site_metadata']
    source=Path('F:/Code/Mamba-Hawkes-Process/data/processed/eicu_exp4_clinical_v1/cohort_labels.parquet')
    columns=['stay_id','subject_id','hospitalid','age','gender','unitdischargeoffset_num','unitdischargestatus']
    meta=pd.read_parquet(source,columns=columns).set_index('stay_id')
    files=[ROOT/'runs/tracehr_phi_full_eicu_20261003'/f'{s}_targets.npz' for s in ('train','dev')]
    files += [ROOT/'runs/tracehr_pm_test_20261021/eicu_features/test_targets.npz']
    pieces=[dict(np.load(p)) for p in files]
    ids=np.concatenate([p['ids'] for p in pieces]);y=np.concatenate([p['y'] for p in pieces])
    f=meta.loc[ids[:,2]].reset_index();assert np.array_equal(f.subject_id,ids[:,0]) and f.stay_id.is_unique
    f['hadm_id']=ids[:,1];f['mortality']=y[:,0];f['los']=y[:,1]
    hm={int(h):s for s,hs in mapping['hospital_ids_by_split'].items() for h in hs}
    f['split']=f.hospitalid.map(hm);assert f.split.notna().all()
    assert f.groupby('subject_id').split.nunique().max()==1
    f.sort_values(['split','subject_id','stay_id'],kind='stable').to_parquet(out/'cohort.parquet',index=False)
    save(out/'hospital_mapping.json',mapping)
    source_files={f'events/{p.name}':dict(local=str(p),sha256=sha(p),bytes=p.stat().st_size)
        for p in sorted((ROOT/'runs/tracehr_asof_full_eicu_20261003/raw').glob('*.parquet'))}
    assert len(source_files)==32
    for name,p in [('numeric.parquet',ROOT/'runs/tracehr_numeric_full_eicu_20261003/events.parquet'),
                   ('cohort.parquet',out/'cohort.parquet'),('hospital_mapping.json',out/'hospital_mapping.json')]:
        source_files[name]=dict(local=str(p),sha256=sha(p),bytes=p.stat().st_size)
    result=dict(complete=True,files=source_files,metadata_source_sha256=sha(source),
        target_sources={str(p):sha(p) for p in files},mapping_source_sha256=sha(mapping_file),
        stays_by_split=f.groupby('split').size().to_dict(),hospitals_by_split=f.groupby('split').hospitalid.nunique().to_dict(),
        patient_overlap=0,hospital_overlap=0,labels_reused_without_change=True,
        grouping='patient-connected hospital components, fixed pre-existing seed42 assignment',
        historical_exposure='Hospitals participated in earlier random-split studies; this is hospital-held-out fitting, not untouched external validation',
        data_used_as_predictors=['categorical events available by 24h','numeric laboratory results available by 24h'])
    save(out/'sources.json',result);return {k:v for k,v in result.items() if k not in ('files','target_sources')}

def local(a,out):
    sys.path.insert(0,str(ROOT));from scripts.launch_ema_cross import HOSTS,ssh
    if a.mode=='stage':return stage(out)
    host,project,python=HOSTS['eicu'];cache=f'{project}/cache/{RUN}/hospital_inputs_v1'
    source=f'{project}/work/{RUN}/hospital_preparation_v2';dest=f'{project}/runs/{RUN}/eicu/hospital_preparation_v2'
    if a.mode=='deploy':
        spec=read(out/'sources.json')
        r=ssh('eicu',f"import json,shutil;from pathlib import Path;p=Path({project!r}).resolve(strict=True);c=Path({cache!r});s=Path({source!r});assert shutil.disk_usage(p).free>74*2**30;[(x.mkdir(parents=True,exist_ok=True)) for x in [c/'events',s]];assert c.resolve().is_relative_to(p) and s.resolve().is_relative_to(p);print(json.dumps({{str(x.relative_to(c)):__import__('hashlib').sha256(x.read_bytes()).hexdigest() for x in c.rglob('*') if x.is_file()}}))")
        for name,v in spec['files'].items():
            assert sha(v['local'])==v['sha256']
            if name in r:assert r[name]==v['sha256'];continue
            subprocess.run(['scp','-q','-o','BatchMode=yes',v['local'],host+':'+cache+'/'+name],check=True,timeout=120)
        remote_spec=dict(spec,files={n:{k:v for k,v in x.items() if k!='local'} for n,x in spec['files'].items()})
        payload={'files':remote_spec['files'],'spec':remote_spec,'script':base64.b64encode(Path(__file__).read_bytes()).decode(),'script_sha256':sha(__file__),
                 'dependencies':{n:base64.b64encode((ROOT/n).read_bytes()).decode() for n in ('scripts/tracehr_exact_statistics.py','scripts/tracehr_value_features.py')}}
        code="""import json,base64,hashlib
from pathlib import Path
d=PAYLOAD;c=Path(CACHE);s=Path(SOURCE)
for n,r in d['files'].items():assert hashlib.sha256((c/n).read_bytes()).hexdigest()==r['sha256'],n
b=base64.b64decode(d['script']);f=s/'prepare.py';assert not f.exists() or f.read_bytes()==b
f.write_bytes(b);(c/'sources.json').write_text(json.dumps(d['spec'],indent=2))
for n,encoded in d['dependencies'].items():
 f=s/n;v=base64.b64decode(encoded);assert not f.exists() or f.read_bytes()==v;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(v)
print(json.dumps(dict(complete=True,files=len(d['files']),bytes=sum(v['bytes'] for v in d['files'].values()),script_sha256=d['script_sha256'])))""".replace('PAYLOAD',repr(payload)).replace('CACHE',repr(cache)).replace('SOURCE',repr(source))
        result=ssh('eicu',code,timeout=120);save(out/'deploy.json',result);return result
    if a.mode=='status':
        result=ssh('eicu',f"import json;from pathlib import Path;p=Path({dest!r});print(json.dumps({{n:json.loads((p/n).read_text()) for n in ['progress.json','probe.json','completion.json','exit.json'] if (p/n).is_file()}}))")
        save(out/'status.json',result)
        return {k:({n:v for n,v in r.items() if n not in ('artifacts','splits','normalization','vocabulary','sources','source_files')} if isinstance(r,dict) else r) for k,r in result.items()}
    assert a.mode in ('probe','build')
    payload=dict(project=project,source=source,dest=dest,cache=cache,mode=a.mode,python=python,expected_gib=a.expected_gib)
    code=r'''import json,os,subprocess,shutil
from pathlib import Path
d=PAYLOAD;p=Path(d['project']);out=Path(d['dest']);out.mkdir(parents=True,exist_ok=True)
parent=p/'work/cem_icde_evidence_20261028/eicu/lgbm_v1';deps=p/'work/tracehr_tabular_20261004/deps';tmp=p/'work/cem_icde_evidence_20261028/tmp_hospital'
tmp.mkdir(exist_ok=True);cache=Path(d['cache'])
assert all(x.resolve().is_relative_to(p.resolve()) for x in (out,parent,tmp,cache))
assert shutil.disk_usage(p).free>(10+d['expected_gib'])*2**30
env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[parent,deps,*deps.glob('*.whl'),p/'work/cem_evidence_completion_20261025/test_runtime_v1/deps'])),
 PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',NUMPY_MADVISE_HUGEPAGE='0',
 TMPDIR=str(tmp),TEMP=str(tmp),TMP=str(tmp),XDG_CACHE_HOME=str(cache),TORCH_HOME=str(cache/'torch'),MPLCONFIGDIR=str(cache/'mpl'),
 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='1')
command=[d['python'],'-B',str(Path(d['source'])/'prepare.py'),'--mode','remote-'+d['mode'],'--project',str(p),'--input',str(cache),'--output',str(out),'--expected-gib',str(d['expected_gib'])]
if d['mode']=='probe':
 r=subprocess.run(command,env=env,cwd=parent,capture_output=True,text=True);assert r.returncode==0,r.stderr[-6000:];print(r.stdout)
else:
 assert not (out/'start.json').exists() and not (out/'completion.json').exists()
 spec=dict(command=command,cwd=str(parent),receipt=str(out/'exit.json'),job_id='hospital_input_reconstruction',source_manifest_sha256=__import__('hashlib').sha256((Path(d['source'])/'prepare.py').read_bytes()).hexdigest())
 (out/'start.json').write_text(json.dumps(spec,indent=2))
 with (out/'build.log').open('x') as log:
  child=subprocess.Popen([d['python'],'-B',str(parent/'scripts/tracehr_recovery_attempt.py'),'--spec',str(out/'start.json')],env=env,cwd=parent,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print(json.dumps(dict(complete=True,wrapper_pid=child.pid,training_launched=False)))
'''.replace('PAYLOAD',repr(payload))
    result=ssh('eicu',code,timeout=900);save(out/(a.mode+'.json'),result);return result

def remote(a):
    import numpy as np,pandas as pd,torch,joblib
    from sklearn.preprocessing import QuantileTransformer
    from scripts.tracehr_exact_statistics import EventStatistics
    from scripts.tracehr_value_features import simple_summary,duration_summary
    from scripts.tracehr_centered_native_common import native_summary_statistics
    from scripts.cem_additional_common import transform
    torch.set_num_threads(4);sys.modules['lightning']=None
    p=a.project.resolve(strict=True);out=a.output;inp=a.input
    assert out.resolve().is_relative_to(p) and inp.resolve().is_relative_to(p)
    spec=read(inp/'sources.json')
    for n,r in spec['files'].items():assert sha(inp/n)==r['sha256'],n
    f=pd.read_parquet(inp/'cohort.parquet');assert f.stay_id.is_unique and f.groupby('subject_id').split.nunique().max()==1
    assert f.groupby('hospitalid').split.nunique().max()==1
    splits={s:f[f.split.eq(s)].sort_values(['subject_id','stay_id'],kind='stable').reset_index(drop=True) for s in ('train','dev','test')}
    allids=set(f.stay_id);train=set(splits['train'].stay_id)
    testcache=p/'cache/cem_evidence_completion_20261025/native_observations/eicu'
    firstfiles=sorted((inp/'events').glob('*.parquet'));secondfiles=sorted((testcache/'events').glob('*.parquet'))
    catfiles=list(zip(firstfiles,secondfiles));assert len(firstfiles)==len(secondfiles)==32
    sources=['diagnosis','infusionDrug','lab','medication','treatment']
    started=time.monotonic()
    def progress(stage,**extra):save(out/'progress.json',dict(stage=stage,seconds=time.monotonic()-started,**extra))
    def cat(path):
        pieces=[]
        for file in path:
            g=pd.read_parquet(file);pieces.append(g[g.stay_id.isin(allids)&g.time.between(0,24)].drop_duplicates(['stay_id','time','token']))
        a,b=pieces;overlap=set(a.stay_id)&set(b.stay_id);columns=['stay_id','time','token','source']
        if overlap:
            left=a[a.stay_id.isin(overlap)][columns].sort_values(columns,kind='stable').reset_index(drop=True)
            right=b[b.stay_id.isin(overlap)][columns].sort_values(columns,kind='stable').reset_index(drop=True)
            pd.testing.assert_frame_equal(left,right,check_dtype=False)
        return pd.concat([a,b[~b.stay_id.isin(overlap)]],ignore_index=True)
    if a.mode=='remote-probe':
        assert not (out/'probe.json').exists();counts=collections.Counter();rows=0
        for path in catfiles:
            g=cat(path);counts.update(g[g.stay_id.isin(train)].token.value_counts().to_dict());rows+=len(g)
        names=[s+'::UNK' for s in sources]+sorted(k for k,c in counts.items() if c>=20 and not k.endswith('::UNK'))
        assert len(set(names))==len(names);phi=4*len(names)+2*len(sources)+4;dim=phi+832
        # Small real feature replay; no weights or test predictions are produced.
        vocab={v:i for i,v in enumerate(names)};g=cat(catfiles[0]);checks=0
        for _,h in g[g.stay_id.isin(train)].groupby('stay_id',sort=False):
            t=h.token.map(vocab).fillna((h.source+'::UNK').map(vocab)).to_numpy(dtype=np.int64)
            s=h.source.map({v:i for i,v in enumerate(sources)}).to_numpy(dtype=np.int64)
            x=EventStatistics(len(vocab),sources,24).add(t,h.time.to_numpy(),s).features();assert np.isfinite(x).all()
            checks+=1
            if checks==256:break
        estimate=3*len(f)*dim*4/2**30+4
        result=dict(complete=True,vocabulary=vocab,phi_dim=phi,input_dim=dim,training_vocabulary_min_frequency=20,
            source_files={str(x):sha(x) for pair in catfiles for x in pair},input_metadata_sha256=sha(inp/'sources.json'),
            rows=len(f),event_rows=rows,bounded_patient_checks=checks,projected_artifact_gib=estimate,
            optimizer_updates=0,test_predictions=False,seconds=time.monotonic()-started)
        assert estimate<a.expected_gib
        save(out/'probe.json',result);return {k:v for k,v in result.items() if k not in ('vocabulary','source_files')}
    probe=read(out/'probe.json');assert probe['complete'] and probe['projected_artifact_gib']<a.expected_gib
    assert not (out/'completion.json').exists() and not (out/'train_raw.npy').exists()
    assert probe['input_metadata_sha256']==sha(inp/'sources.json')
    for pair in catfiles:
        for path in pair:assert sha(path)==probe['source_files'][str(path)]
    vocab=probe['vocabulary'];phi=probe['phi_dim'];dim=probe['input_dim'];save(out/'token_to_id.json',vocab)
    raw={s:np.lib.format.open_memmap(out/(s+'_raw.npy'),mode='w+',dtype=np.float32,shape=(len(g),dim)) for s,g in splits.items()}
    lookup={int(sid):(s,i) for s,g in splits.items() for i,sid in enumerate(g.stay_id)}
    lengths={s:np.zeros(len(g),np.int64) for s,g in splits.items()};seen=set();oov={s:0 for s in splits}
    empty=EventStatistics(len(vocab),sources,24).features()
    for x in raw.values():x[:,:phi]=empty;x[:,phi:]=0
    source_index={s:i for i,s in enumerate(sources)}
    for part,path in enumerate(catfiles):
        g=cat(path);mapped=g.token.map(vocab);missing=mapped.isna();mapped= mapped.fillna((g.source+'::UNK').map(vocab));assert mapped.notna().all()
        g['type_id']=mapped.astype('int64');g['source_id']=g.source.map(source_index);g['oov']=missing;assert g.source_id.notna().all()
        for sid,h in g.groupby('stay_id',sort=False):
            assert sid not in seen;seen.add(sid);s,i=lookup[int(sid)]
            raw[s][i,:phi]=EventStatistics(len(vocab),sources,24).add(h.type_id.to_numpy(),h.time.to_numpy(),h.source_id.to_numpy()).features()
            lengths[s][i]=len(h);oov[s]+=int(h.oov.sum())
        progress('categorical_features',buckets=part+1,total=len(catfiles));del g;gc.collect()
    paths=[inp/'numeric.parquet',testcache/'numeric.parquet']
    left,right=[pd.read_parquet(x) for x in paths]
    left=left[left.stay_id.isin(allids)&left.available_h.between(0,24)&np.isfinite(left.value)].copy()
    right=right[right.stay_id.isin(allids)&right.available_h.between(0,24)&np.isfinite(right.value)].copy()
    overlap=set(left.stay_id)&set(right.stay_id)
    columns=['stay_id','lab','unit','sample_h','available_h','value']
    if overlap:
        overlap_left=left[left.stay_id.isin(overlap)][columns].sort_values(columns,kind='stable').reset_index(drop=True)
        overlap_right=right[right.stay_id.isin(overlap)][columns].sort_values(columns,kind='stable').reset_index(drop=True)
        pd.testing.assert_frame_equal(overlap_left,overlap_right,check_dtype=False)
    g=pd.concat([left,right[~right.stay_id.isin(overlap)]],ignore_index=True);del left,right
    g['key']=g.lab+'|'+g.unit
    coverage=g[g.stay_id.isin(train)].groupby(['lab','unit'],observed=True).stay_id.nunique().reset_index(name='stays')
    selected=coverage.sort_values(['stays','lab','unit'],ascending=[False,True,True]).drop_duplicates('lab').head(32)
    keys=(selected.lab+'|'+selected.unit).tolist();assert len(keys)==32
    g=g[g.key.isin(keys)].groupby(['key','stay_id','available_h'],as_index=False,observed=True).agg(value=('value','median'),sample_h=('sample_h','max'))
    norm={}
    for j,key in enumerate(keys):
        h=g[g.key.eq(key)].sort_values(['stay_id','available_h'],kind='stable').copy();v=h[h.stay_id.isin(train)].value.to_numpy()
        q=np.quantile(v,[.25,.5,.75]);scale=float(q[2]-q[0]);scale=scale if scale>=1e-8 else float(np.std(v));scale=scale if scale>=1e-8 else 1.
        norm[key]=dict(train_median=float(q[1]),train_iqr_or_std=scale,train_events=len(v));h['z']=np.clip((h.value-q[1])/scale,-5,5)
        for sid,k in h.groupby('stay_id',sort=False):
            s,i=lookup[int(sid)];t=k.available_h.to_numpy();z=k.z.to_numpy()
            raw[s][i,phi+j*11:phi+(j+1)*11]=simple_summary(t,z,k.sample_h.to_numpy())
            raw[s][i,phi+352+j*15:phi+352+(j+1)*15]=duration_summary(t,z)
        progress('numeric_features',labs=j+1,total=32)
    save(out/'normalization.json',norm);del g,h;gc.collect()
    center=np.zeros(dim,np.float32);factor=np.ones(dim,np.float32);binary=np.zeros(dim,bool);nonconstant=np.zeros(dim,bool)
    for begin in range(0,phi,128):
        end=min(begin+128,phi);b=np.asarray(raw['train'][:,begin:end]);q1,med,q3=np.quantile(b,[.25,.5,.75],axis=0);lo,hi=b.min(0),b.max(0)
        spread=q3-q1;spread[spread==0]=.5*(hi-lo)[spread==0]
        center[begin:end]=med;factor[begin:end]=np.divide(1.,spread,out=np.zeros_like(spread),where=spread!=0)
        binary[begin:end]=np.all((b==0)|(b==1),axis=0);nonconstant[begin:end]=lo!=hi
        if begin%1024==0:progress('training_phi_scaling',columns=end,total=phi)
    pc,pf,pb=center[:phi].copy(),factor[:phi].copy(),binary[:phi].copy();binary[phi+np.arange(32)*11]=True
    for begin in range(phi,dim,32):
        end=min(begin+32,dim);b=np.asarray(raw['train'][:,begin:end],dtype=np.float64);q1,med,q3=np.quantile(b,[.25,.5,.75],axis=0)
        spread=np.where(q3-q1>1e-6,q3-q1,np.std(b,axis=0));center[begin:end]=med;factor[begin:end]=1/np.maximum(spread,1.);nonconstant[begin:end]=b.min(0)!=b.max(0)
    center[binary]=0;factor[binary]=1;indices=np.flatnonzero(~binary&nonconstant)
    prep=dict(center=center,factor=factor,binary=binary,numeric_indices=indices);np.savez_compressed(out/'preprocessing.npz',**prep)
    nc,nf,ni=native_summary_statistics(raw['train'],prep,phi);np.savez(out/'native_summary_scaling.npz',center=nc,factor=nf,indices=ni)
    fixed=p/'runs/tracehr_structure_20261014/eicu/g_v1/preparation/native_configs.json';shutil.copyfile(fixed,out/'native_configs.json')
    qt=QuantileTransformer(n_quantiles=1000,output_distribution='normal',subsample=10000,random_state=42).fit(raw['train'][:,phi:])
    trans=dict(phi_center=pc,phi_factor=pf,binary=pb,numeric_quantile=qt,phi_dim=phi,fit_split='train');joblib.dump(trans,out/'tabm_preprocessing.joblib')
    tb=out/'tabular';tb.mkdir();sizes=[]
    for s,g in splits.items():
        raw[s].flush();np.savez_compressed(out/(s+'_targets.npz'),ids=g[['subject_id','hadm_id','stay_id']].to_numpy(np.int64),y=g[['mortality','los']].to_numpy(np.int64),lengths=lengths[s],hospital_ids=g.hospitalid.to_numpy(np.int64))
        for begin in range(0,len(g),1024):assert np.isfinite(raw[s][begin:begin+1024]).all()
        if s!='test':
            z=np.lib.format.open_memmap(tb/(s+'.npy'),mode='w+',dtype=np.float32,shape=raw[s].shape)
            for begin in range(0,len(g),1024):z[begin:begin+1024]=transform(raw[s][begin:begin+1024],trans)
            z.flush();del z
        sizes.append(dict(split=s,stays=len(g),patients=int(g.subject_id.nunique()),hospitals=int(g.hospitalid.nunique()),mortality_positive=int(g.mortality.sum()),los_positive=int(g.los.sum()),events=int(lengths[s].sum()),oov_events=oov[s]))
    bins=[];x=np.load(tb/'train.npy',mmap_mode='r')
    for begin in range(0,dim,64):
        edges=torch.quantile(torch.from_numpy(np.array(x[:,begin:begin+64],copy=True)),torch.linspace(0,1,49),dim=0).T
        for q in edges:
            q=q.unique();bins.append(q if len(q)>1 else torch.stack([q[0]-.5,q[0]+.5]))
        if begin%1024==0:progress('training_PLE_bins',columns=begin,total=dim)
    torch.save(bins,tb/'bins.pt');save(tb/'completion.json',dict(complete=True,fit_split='train',test_loaded=False,optimizer_updates=0,input_dim=dim,artifacts={n:sha(tb/n) for n in ('train.npy','dev.npy','bins.pt')}))
    artifact={str(x.relative_to(out)):dict(bytes=x.stat().st_size,sha256=sha(x)) for x in out.rglob('*') if x.is_file() and x.suffix in ('.npy','.npz','.pt','.joblib')}
    for n in ('token_to_id.json','normalization.json','native_configs.json'):artifact[n]=dict(bytes=(out/n).stat().st_size,sha256=sha(out/n))
    result=dict(complete=True,preparation_complete=True,database='eicu',fit_split='train',input_dim=dim,phi_dim=phi,event_num_types=len(vocab),
        train_rows=len(splits['train']),dev_rows=len(splits['dev']),test_rows=len(splits['test']),source_names=sources,lab_keys=keys,splits=sizes,
        preprocessing_sha256=sha(out/'preprocessing.npz'),native_summary_scaling_sha256=sha(out/'native_summary_scaling.npz'),native_configs_sha256=sha(out/'native_configs.json'),
        native_config_reference=dict(path=str(fixed),sha256=sha(fixed),only_fixed_hyperparameters_reused=True),
        patient_overlap=0,hospital_overlap=0,labels_unchanged=True,train_only_vocabulary=True,train_only_lab_selection=True,train_only_scaling=True,
        input_metadata_sha256=sha(inp/'sources.json'),numeric_sources={str(x):sha(x) for x in paths},
        test_features_constructed=True,test_used_for_fit=False,test_predictions_computed=False,optimizer_updates=0,
        artifact_gib=sum(v['bytes'] for v in artifact.values())/2**30,artifacts=artifact,seconds=time.monotonic()-started)
    assert result['artifact_gib']<a.expected_gib;save(out/'completion.json',result);progress('complete')
    return {k:v for k,v in result.items() if k not in ('artifacts','splits','lab_keys')}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=['stage','deploy','probe','build','status','remote-probe','remote-build'])
    parser.add_argument('--project',type=Path);parser.add_argument('--input',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--expected-gib',type=float,default=64)
    a=parser.parse_args()
    if a.mode.startswith('remote-'):result=remote(a)
    else:
        out=Path(os.environ['PROJECT_RUN_DIR']).resolve();assert out.is_relative_to(ROOT);out.mkdir(parents=True,exist_ok=True);result=local(a,out)
    print(json.dumps(result))
