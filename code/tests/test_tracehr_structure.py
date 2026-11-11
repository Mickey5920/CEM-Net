import copy
import os
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
deps=Path(os.environ['TRACEHR_TABULAR_DEPS']);sys.path.insert(0,str(deps))
for wheel in deps.glob('*.whl'):sys.path.insert(0,str(wheel))
import numpy as np
import torch
from models.tracehr_centered_native import CenteredNative
from models.tracehr_structure import StructureNetwork, VariableTrajectory
from models.tracehr_unified_next import parameter_hash
from scripts.tracehr_structure_common import protocol,stopping,patient_records,reference_paths,VARIANTS
from scripts.tracehr_structure_runtime import Optimizer,gradients
from scripts.tracehr_multiview_common import read


class StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        rng=np.random.default_rng(7)
        cls.raw=rng.normal(size=(32,836)).astype(np.float32)
        cls.y=np.tile([0,1],16)
        cls.prep=dict(center=np.zeros(836,np.float32),factor=np.ones(836,np.float32),
                      binary=np.zeros(836,bool),numeric_indices=np.arange(4,836))
        prev,_=reference_paths(Path(os.environ['TRACEHR_PROJECT']),os.environ['TRACEHR_DATABASE'])
        cls.cfg=read(prev/'native_configs.json')[0]
        records=[];segments=[];counts=[]
        for i in range(32):
            r=rng.normal(size=(4,32)).astype(np.float32)
            obs=rng.uniform(size=(4,32))>.3
            x,s,c=patient_records(r,obs,np.array([0.,3.,12.,24.]),np.zeros(32),np.ones(32))
            records.append(x);segments.append(s.astype(np.int64)+i*256);counts.append(c)
        cls.batch=dict(raw=torch.tensor(cls.raw),records=torch.tensor(np.concatenate(records)),
                       segments=torch.tensor(np.concatenate(segments)),counts=torch.tensor(np.stack(counts)))

    def model(self,v):
        base=CenteredNative(self.raw,self.y,self.prep,self.cfg,42,False,32)
        return StructureNetwork(base,4,v)

    def test_matrix_and_fixed_recipe(self):
        self.assertEqual(VARIANTS,('G1','G2','G3'))
        for db in ('mimic','eicu'):
            specs=[protocol(v,db) for v in VARIANTS]
            comparable=[{k:x for k,x in s.items() if k not in ('variant','structure','description')} for s in specs]
            self.assertTrue(all(s==comparable[0] for s in comparable))
            self.assertEqual(specs[0]['lr'],.04);self.assertEqual(specs[0]['fusion_lr'],.0003)
        history=[dict(epoch=i+1,dev_ap=.5) for i in range(256)]
        self.assertTrue(stopping(history[:52],protocol('G1','mimic'))['stop'])
        self.assertFalse(stopping(history[:52],protocol('G1','eicu'))['stop'])
        self.assertTrue(stopping(history,protocol('G1','eicu'))['stop'])

    def test_records_preserve_identity_timing_and_all_observations(self):
        r=np.zeros((3,32),np.float32);m=np.zeros_like(r,bool)
        m[:,0]=True;r[:,0]=[3,1,2]
        x,s,c=patient_records(r,m,np.array([24.,0.,12.]),np.zeros(32),np.ones(32))
        np.testing.assert_array_equal(s,[0,4,7]);np.testing.assert_array_equal(x[:,0],[1,2,3])
        np.testing.assert_array_equal(x[:,3],[0,1,1]);self.assertEqual(c.sum(),3)
        self.assertEqual(int(c[1:].sum()),0)
        with self.assertRaises(AssertionError):patient_records(r,m,np.array([25.,0.,12.]),np.zeros(32),np.ones(32))

    def test_common_initialization_exact_base_path_and_real_structure_difference(self):
        hashes=[];outputs=[]
        for v in VARIANTS:
            net=self.model(v);net.eval();hashes.append(parameter_hash(net))
            with torch.no_grad():
                net.fusion_enabled=False
                self.assertTrue(torch.equal(net(self.batch),net.base(self.batch['raw'])))
                net.fusion_enabled=True;outputs.append(net(self.batch))
        self.assertEqual(len(set(hashes)),1)
        self.assertGreater(float((outputs[0]-outputs[1]).abs().max()),1e-7)
        self.assertGreater(float((outputs[1]-outputs[2]).abs().max()),1e-7)

    def test_every_group_jointly_updates_and_optimizer_restore(self):
        for v in VARIANTS:
            net=self.model(v);opt=Optimizer(net,protocol(v,'mimic'));opt.schedule(.1)
            before={n:p.detach().clone() for n,p in net.named_parameters()}
            net.train();opt.zero_grad();net.loss(net(self.batch),torch.tensor(self.y)).backward()
            report=gradients(net);self.assertEqual(set(report),{'trajectory','fusion','numerical','input_members','backbone'})
            opt.step()
            for prefix in ('base.net.','base.numerical.','base.input_members','fusion.trajectory.','fusion.'):
                self.assertTrue(any(not torch.equal(p,before[n]) for n,p in net.named_parameters() if n.startswith(prefix)),prefix)
            saved=copy.deepcopy(opt.state_dict());values=copy.deepcopy(opt.values())
            opt.schedule(70);opt.load_state_dict(saved);self.assertEqual(values,opt.values())

    def test_variable_and_patient_boundaries_and_empty_mask(self):
        net=self.model('G1');net.eval()
        batch={k:v.clone() for k,v in self.batch.items()}
        with torch.no_grad():
            a,mask=net.fusion.encode(batch)
            select=batch['segments']==0
            batch['records'][select,0]+=1
            b,_=net.fusion.encode(batch)
            self.assertTrue(torch.equal(a[1:],b[1:]))
            self.assertTrue(torch.equal(a[0,1:],b[0,1:]))
            empty=dict(raw=self.batch['raw'][:1],records=torch.empty(0,8),
                       segments=torch.empty(0,dtype=torch.long),counts=torch.zeros(1,32,8,dtype=torch.int32))
            out=net(empty);self.assertTrue(torch.isfinite(out).all())


if __name__=='__main__':unittest.main()
