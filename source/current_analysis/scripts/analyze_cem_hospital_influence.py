"""Descriptive component influence using the already locked hospital predictions.

No model imports, training, checkpoint selection, bootstrap, or probability export.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
from sklearn.metrics import average_precision_score

RUN='cem_icde_evidence_20261028'
FIXED_SHA='9d2a99ad4c57a4e2842b30d3222265fd6e736f84edf68edbae51978bd946ba2d'
SOURCE_SHA='65a75589cf731b90a036298bc49210eb93872c89f1487381eb772f3079b0c884'
METHODS=('CEM_hospital','TabM_mini_PLE_hospital','LightGBM_hospital')
SEEDS=(48,49,50)


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def components(patients,hospitals):
    # Same root convention and iteration order as the frozen bootstrap analysis.
    parent={int(h):int(h) for h in np.unique(hospitals)}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for patient in np.unique(patients):
        hs=np.unique(hospitals[patients==patient]);a=find(int(hs[0]))
        for h in hs[1:]:parent[find(int(h))]=a
    return np.asarray([find(int(h)) for h in hospitals])


def difference(y,a,b):
    assert a.shape==b.shape==(3,len(y))
    assert set(np.unique(y))=={0,1}
    aa=np.asarray([average_precision_score(y,q) for q in a])
    bb=np.asarray([average_precision_score(y,q) for q in b])
    delta=aa-bb
    return dict(CEM_AP=aa.tolist(),baseline_AP=bb.tolist(),seed_differences=delta.tolist(),
                mean=float(delta.mean()),sample_SD=float(delta.std(ddof=1)),positive_seeds=int((delta>0).sum()))


def self_test():
    patients=np.array([1,2,2,3,4,5,6,7]);h=np.array([10,10,20,30,30,40,40,50])
    c=components(patients,h)
    assert len(np.unique(c))==4 and c[0]==c[2] and c[3]!=c[0]
    y=np.array([1,0,1,0,1,0,1,0]);a=np.tile([.9,.8,.7,.6,.5,.4,.3,.2],(3,1));b=np.flip(a,axis=1)
    for component in np.unique(c):
        keep=c!=component
        d=difference(y[keep],a[:,keep],b[:,keep])
        swapped=difference(y[keep],b[:,keep],a[:,keep])
        np.testing.assert_array_equal(d['seed_differences'],-np.asarray(swapped['seed_differences']))
        assert difference(y[keep],a[:,keep],a[:,keep])['mean']==0
        assert d['mean']==average_precision_score(y[keep],a[0,keep])-average_precision_score(y[keep],b[0,keep])
    assert difference(y,np.full_like(a,.5),np.full_like(a,.5))['mean']==0
    return ['patient_connected_hospital_union','exact_subset_AP','contrast_swap_symmetry','identical_prediction_zero','tied_scores']


def analyze(project,output):
    project=project.resolve(strict=True);output=output.resolve()
    assert output==project/'runs'/RUN/'hospital_influence_v1'
    assert output.is_relative_to(project) and shutil.disk_usage(project).free>10.5*2**30
    output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(output/'writer.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'completion.json').exists():
        done=read(output/'completion.json')
        assert done['complete'] and sha(output/'verified-results.json')==done['results_sha256']
        return done
    fixedpath=project/'runs'/RUN/'eicu_statistics_v1/verified-results.json'
    assert sha(fixedpath)==FIXED_SHA
    fixed=read(fixedpath)
    snapshot=project/'work'/RUN/'eicu/hospital_v1'
    assert sha(snapshot/'source_manifest.json')==SOURCE_SHA
    manifest=read(snapshot/'source_manifest.json');configpath=snapshot/'work'/RUN/'config.json'
    assert sha(configpath)==manifest['work/'+RUN+'/config.json']
    config=read(configpath);prep=project/config['raw_inputs']
    assert read(prep/'completion.json')==config['hospital_preparation']
    protected={fixedpath:FIXED_SHA,configpath:sha(configpath),snapshot/'source_manifest.json':SOURCE_SHA,
               prep/'completion.json':sha(prep/'completion.json')}
    result=dict(complete=True,source_sha256=sha(__file__),primary_results_sha256=FIXED_SHA,
                analysis='leave one patient-connected hospital component out of fixed validation/test predictions',
                training_executed=False,checkpoints_changed=False,bootstrap_resampled=False,
                selection_changed=False,primary_intervals_changed=False,patient_records_exported=False,
                self_test=self_test(),partitions={},rows=[],summaries=[])
    for split,expected_n in (('dev',13),('test',22)):
        targetpath=prep/(split+'_targets.npz');h=config['hospital_preparation']['artifacts'][targetpath.name]['sha256']
        assert sha(targetpath)==h;protected[targetpath]=h
        target=dict(np.load(targetpath,allow_pickle=False))
        ids,y,hosp=target['ids'],target['y'],target['hospital_ids']
        cluster=components(ids[:,0],hosp);unique=np.unique(cluster);assert len(unique)==expected_n
        ordered=sorted(unique,key=lambda c:(-int((cluster==c).sum()),int(c)))
        aliases={c:f'{split.upper()}-{i+1:02d}' for i,c in enumerate(ordered)}
        pred={}
        for method in METHODS:
            meta_path=fixedpath.parent/'bootstrap'/(f'hospital_{split}_{method}.json')
            meta=read(meta_path)['identity'];protected[meta_path]=sha(meta_path)
            assert meta['cluster_ids_sha256']==hashlib.sha256(cluster.tobytes()).hexdigest()
            assert meta['replicates']==20000 and meta['arm']=='hospital' and meta['split']==split
            for seed in SEEDS:
                for endpoint in (0,1):
                    reference=meta['predictions'][f'{seed}:{endpoint}'];p=Path(reference['path'])
                    assert p.is_relative_to(project) and sha(p)==reference['sha256']==fixed['artifacts'][str(p)]
                    protected[p]=reference['sha256'];z=dict(np.load(p,allow_pickle=False))
                    np.testing.assert_array_equal(z['ids'],ids)
                    col=endpoint if z['labels'].ndim==2 and z['labels'].shape[1]==2 else 0
                    labels=z['labels'] if z['labels'].ndim==1 else z['labels'][:,col]
                    q=z['probs'] if z['probs'].ndim==1 else z['probs'][:,col]
                    np.testing.assert_array_equal(labels,y[:,endpoint]);assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
                    expected=next(r['AP'] for r in fixed['rows'] if (r['arm'],r['split'],r['method'],r['seed'],r['endpoint'])==('hospital',split,method,seed,endpoint))
                    assert abs(average_precision_score(labels,q)-expected)<1e-12
                    pred[method,seed,endpoint]=q
        descriptions=[]
        for c in ordered:
            ix=cluster==c
            descriptions.append(dict(component=aliases[c],stays=int(ix.sum()),stay_fraction=float(ix.mean()),
                                     hospitals=len(np.unique(hosp[ix])),patients=len(np.unique(ids[ix,0])),
                                     mortality_positive=int(y[ix,0].sum()),LOS_positive=int(y[ix,1].sum())))
        weights=np.asarray([d['stay_fraction'] for d in descriptions])
        result['partitions'][split]=dict(stays=len(ids),patients=len(np.unique(ids[:,0])),hospitals=len(np.unique(hosp)),
            components=len(unique),component_details=descriptions,largest_stay_fraction=float(weights.max()),
            inverse_size_concentration=float(1/np.square(weights).sum()),
            concentration_note='Descriptive inverse squared stay shares, not an inferential degrees-of-freedom estimate')
        for baseline in METHODS[1:]:
            for endpoint in (0,1):
                a=np.stack([pred['CEM_hospital',s,endpoint] for s in SEEDS]);b=np.stack([pred[baseline,s,endpoint] for s in SEEDS])
                full=difference(y[:,endpoint],a,b)
                np.testing.assert_allclose(full['seed_differences'],fixed['paired'][split]['hospital'][baseline][str(endpoint)]['values'],rtol=0,atol=1e-12)
                rows=[]
                for c in ordered:
                    keep=cluster!=c
                    d=difference(y[keep,endpoint],a[:,keep],b[:,keep])
                    row=dict(split=split,baseline=baseline,endpoint=endpoint,omitted_component=aliases[c],
                             omitted_stays=int((~keep).sum()),retained_stays=int(keep.sum()),**d,
                             influence=d['mean']-full['mean'],same_mean_sign_as_full=bool(np.sign(d['mean'])==np.sign(full['mean'])))
                    rows.append(row)
                result['rows'].extend(rows)
                most=max(rows,key=lambda r:abs(r['influence']))
                result['summaries'].append(dict(split=split,baseline=baseline,endpoint=endpoint,full=full,
                    leave_one_out_range=[min(r['mean'] for r in rows),max(r['mean'] for r in rows)],
                    positive_leaveouts=sum(r['mean']>0 for r in rows),negative_leaveouts=sum(r['mean']<0 for r in rows),
                    sign_reversals=sum(not r['same_mean_sign_as_full'] for r in rows),components=len(rows),
                    largest_absolute_influence=most['influence'],most_influential_component=most['omitted_component']))
    assert len(result['rows'])==140 and len(result['summaries'])==8
    for p,h in protected.items():assert sha(p)==h,'Input changed: '+str(p)
    result['inputs_sha256']={str(p.relative_to(project)):h for p,h in protected.items()}
    result['limitations']='Descriptive case-mix influence at fixed weights, not refitted leave-one-hospital-out validation or a new significance test. All components remain in the primary estimate and original adjusted component-bootstrap intervals.'
    payload=json.dumps(result,indent=2).encode()
    (output/'verified-results.json').write_bytes(payload)
    done=dict(complete=True,results_sha256=hashlib.sha256(payload).hexdigest(),source_sha256=sha(__file__),
              inputs_unchanged=True,paired_leaveouts=140,per_seed_paired_leaveouts=420)
    (output/'completion.json').write_text(json.dumps(done,indent=2))
    return done


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test()))
    else:
        if args.project is None or args.output is None:parser.error('--project and --output are required')
        print(json.dumps(analyze(args.project,args.output)))
