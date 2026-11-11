"""Reuse original input matrices and fitted transforms; store event lengths only."""
import argparse, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.tracehr_multiview_common import atomic, read, sha, checked_output, inputs
from scripts.tracehr_unified_common import Data

def main():
    p=argparse.ArgumentParser()
    for n in ('project','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--database',choices=['eicu'],required=True)
    a=p.parse_args();project=a.project.resolve(strict=True);out=checked_output(project,a.output)
    assert not out.exists();out.mkdir(parents=True)
    old=project/'runs/tracehr_structure_20261014'/a.database/'g_v1/preparation'
    record=read(old/'completion.json');data=Data(project,a.database,old)
    event=inputs(project,a.database)['events'];em=read(event/'completion.json')
    assert em['complete'] and not em['test_loaded'] and em['window']==24
    assert sha(event/'completion.json')==record['source_inputs']['events']
    nv=em['num_types'];assert record['phi_dim']==4*nv+2*len(em['sources'])+4
    artifacts={}
    for name,field in [('preprocessing.npz','preprocessing_sha256'),('native_configs.json','native_configs_sha256'),
                       ('native_summary_scaling.npz','native_summary_scaling_sha256')]:
        assert sha(old/name)==record[field];shutil.copyfile(old/name,out/name)
    for split in ('train','dev'):
        t=dict(np.load(event/(split+'_targets.npz')))
        for key in ('ids','y'):assert np.array_equal(t[key],data.targets[split][key])
        n=len(t['ids']);lengths=np.empty(n,np.int32)
        for start in range(0,n,256):
            lengths[start:start+256]=(data.raw[split][start:start+256,nv:2*nv]>0).sum(1)
        f=out/(split+'_event_lengths.npy');np.save(f,lengths)
        artifacts[f.name]=dict(bytes=f.stat().st_size,sha256=sha(f))
    record.update(event_num_types=nv,event_sources=em['sources'],event_manifest_sha256=sha(event/'completion.json'),
        structure_artifacts=artifacts,structure_input='unchanged categorical and numerical evidence; active event indices only',
        original_preparation=str(old),original_preparation_sha256=sha(old/'completion.json'),
        event_presence_rows=record['train_rows']+record['dev_rows'],preparation_complete=True,test_loaded=False)
    atomic(out/'completion.json',record)
    print({'complete':True,'event_types':nv,'artifact_bytes':sum(v['bytes'] for v in artifacts.values())})

if __name__=='__main__':main()
