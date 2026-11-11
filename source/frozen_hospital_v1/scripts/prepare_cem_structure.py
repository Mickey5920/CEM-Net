"""Reorganize existing train/dev evidence; no test data or new learned transforms."""
import argparse, os, shutil, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import sparse
from scripts.tracehr_multiview_common import atomic,read,sha,checked_output,inputs
from scripts.tracehr_unified_common import Data as RawData

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();project=a.project.resolve(strict=True);out=checked_output(project,a.output)
    assert not out.exists();out.mkdir(parents=True)
    assert shutil.disk_usage(project).free>12*2**30
    old=project/'runs/tracehr_structure_20261014/mimic/g_v1/preparation'
    record=read(old/'completion.json');data=RawData(project,'mimic',old)
    event=inputs(project,'mimic')['events'];em=read(event/'completion.json')
    assert em['complete'] and not em['test_loaded'] and em['window']==24
    assert sha(event/'completion.json')==record['source_inputs']['events']
    nv=em['num_types'];assert record['phi_dim']==4*nv+2*len(em['sources'])+4
    for name,desc in em['artifacts'].items():assert sha(event/name)==desc['sha256'],name
    for name in ('preprocessing.npz','native_configs.json','native_summary_scaling.npz'):
        shutil.copyfile(old/name,out/name)
    offsets=np.load(event/'cat_offsets.npy',mmap_mode='r');types=np.load(event/'cat_types.npy',mmap_mode='r');times=np.load(event/'cat_times.npy',mmap_mode='r')
    # A hard bound: each nonzero bin/type cell consumes at least one original event.
    bound=(len(types)*8+len(offsets)*16)/2**30
    assert bound<2
    artifacts={};shift=0;pilot=[];start=time.time()
    for split in ('train','dev'):
        t=dict(np.load(event/(split+'_targets.npz')))
        assert np.array_equal(t['ids'],data.targets[split]['ids']) and np.array_equal(t['y'],data.targets[split]['y'])
        n=len(t['ids']);indptr=np.zeros(n+1,np.int64);all_idx=[];all_value=[];lens=np.empty(n,np.int32)
        for i in range(n):
            lo,hi=offsets[shift+i:shift+i+2];ids=np.asarray(types[lo:hi],np.int64);ts=np.asarray(times[lo:hi])
            assert np.isfinite(ts).all() and np.all((ts>=0)&(ts<=24)) and np.all(ids<nv)
            cells=np.minimum((ts/3).astype(np.int64),7)*nv+ids
            idx,count=np.unique(cells,return_counts=True)
            aggregate=np.bincount(ids,minlength=nv)
            np.testing.assert_allclose(np.log1p(aggregate),data.raw[split][i,:nv],rtol=1e-6,atol=1e-6)
            assert np.array_equal(aggregate>0,data.raw[split][i,nv:2*nv]>0)
            lens[i]=np.count_nonzero(aggregate);indptr[i+1]=indptr[i]+len(idx)
            all_idx.append(idx.astype(np.int32));all_value.append(count.astype(np.float32))
            if split=='train' and i<128:pilot.append(len(idx)*8)
            if split=='train' and i==127:atomic(out/'bounded-preparation-estimate.json',dict(rows=128,measured_csr_payload_bytes=sum(pilot),
                projected_payload_gib=float(np.mean(pilot))*(em['train_rows']+em['dev_rows'])/2**30,hard_payload_bound_gib=bound,test_loaded=False))
            if i%5000==0:atomic(out/'progress.json',dict(status='binning_existing_events',split=split,rows=i,total=n))
        csr=sparse.csr_matrix((np.concatenate(all_value),np.concatenate(all_idx),indptr),shape=(n,8*nv))
        sparse.save_npz(out/(split+'_bin_counts.npz'),csr);np.save(out/(split+'_event_lengths.npy'),lens)
        shift+=n
        for name in (split+'_bin_counts.npz',split+'_event_lengths.npy'):
            f=out/name;artifacts[name]=dict(bytes=f.stat().st_size,sha256=sha(f))
        del csr,all_idx,all_value
    assert shift==len(offsets)-1
    record.update(event_num_types=nv,event_sources=em['sources'],event_manifest_sha256=sha(event/'completion.json'),
        structure_artifacts=artifacts,structure_input='same existing categorical and numerical train/dev evidence; fixed eight bins',
        original_preparation=str(old),original_preparation_sha256=sha(old/'completion.json'),bin_width_hours=3,
        categorical_count_reconstruction_rows=shift,preparation_complete=True,test_loaded=False)
    atomic(out/'completion.json',record);atomic(out/'progress.json',dict(status='complete',seconds=time.time()-start,artifact_gib=sum(r['bytes'] for r in artifacts.values())/2**30))
    print({'complete':True,'rows':shift,'artifacts_gib':sum(r['bytes'] for r in artifacts.values())/2**30})

if __name__=='__main__':main()
