import copy
import os
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
deps=Path(os.environ['TRACEHR_TABULAR_DEPS']);sys.path.insert(0,str(deps))
for wheel in deps.glob('*.whl'):sys.path.insert(0,str(wheel))
import numpy as np
import torch
from models.tracehr_native_transfer import TransferNetwork
from models.tracehr_unified_next import parameter_hash
from models.tracehr_backbone_refine import PostScaleResidual, TypedNative
from scripts.tracehr_backbone_refine_common import protocol, stopping, select_structure, old_baseline
from scripts.tracehr_backbone_refine_runtime import Optimizer
from scripts.tracehr_multiview_common import read


class RefineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        rng=np.random.RandomState(7)
        cls.raw=rng.normal(size=(64,12)).astype(np.float32)
        cls.raw[:,10:]=(cls.raw[:,10:]>0).astype(np.float32)
        cls.y=np.tile([0,1],32).astype(np.int64)
        cls.prep=dict(center=np.zeros(12,np.float32),factor=np.ones(12,np.float32),
                      binary=np.arange(12)>=10,numeric_indices=np.arange(10))
        project=Path(os.environ['TRACEHR_PROJECT']);db=os.environ['TRACEHR_DATABASE']
        cls.config=read(old_baseline(project,db,'B0')/'effective_config_task0.json')

    def test_D0_identical_initial_function_and_shared_weights(self):
        x=torch.as_tensor(self.raw[:8])
        old=TransferNetwork(12,self.prep,'C4',0,42,width=16,depth=2).eval()
        new=PostScaleResidual(12,self.prep,0,42,width=16,depth=2).eval()
        for name,p in old.named_parameters():self.assertTrue(torch.equal(p,dict(new.named_parameters())[name]),name)
        self.assertTrue(torch.equal(old(x),new(x)))
        with torch.no_grad():new.numerical.scale.mul_(2)
        self.assertFalse(torch.equal(old(x),new(x)))

    def test_native_common_initialization_member_prefix_and_joint_updates(self):
        models=[TypedNative(self.raw,self.y,self.prep,self.config,k,42,64) for k in (1,4,8)]
        self.assertEqual(len({parameter_hash(m,True) for m in models}),1)
        self.assertTrue(torch.equal(models[1].input_members,models[2].input_members[:4]))
        self.assertEqual(models[0].info['sub_split_seed'],1608637542)
        for model,v in zip(models,('D1','D2','D3')):
            opt=Optimizer(model,protocol(v));model.train();opt.schedule(.1)
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            for _ in range(2):
                opt.zero_grad();loss=model.loss(model(torch.as_tensor(self.raw)),torch.as_tensor(self.y));loss.backward()
                for name,p in model.named_parameters():
                    self.assertTrue(p.requires_grad and p.grad is not None,name)
                    self.assertTrue(torch.isfinite(p.grad).all(),name)
                opt.step()
            for name,p in model.named_parameters():self.assertFalse(torch.equal(p,before[name]),name)
            model.eval()
            self.assertEqual(tuple(model.probabilities(torch.as_tensor(self.raw)).shape),(64,1))
            self.assertEqual(model.info['hidden_member_adapters'],0)

    def test_optimizer_state_clock_restores(self):
        model=TypedNative(self.raw,self.y,self.prep,self.config,4,42,64)
        opt=Optimizer(model,protocol('D2'));opt.schedule(37.5)
        before=copy.deepcopy(opt.values());state=copy.deepcopy(opt.state_dict())
        opt.schedule(80);opt.load_state_dict(state)
        self.assertEqual(opt.values(),before)
        self.assertEqual(before['clock_reference_epochs'],256)

    def test_no_intra_cycle_stop_and_exact_boundary(self):
        rows=[dict(epoch=i+1,dev_ap=.5) for i in range(256)]
        for n in (1,17,30,51):self.assertFalse(stopping(rows[:n],protocol('D1'))['stop'])
        self.assertTrue(stopping(rows[:52],protocol('D1'))['stop'])
        rows[42]['dev_ap']=.51
        self.assertFalse(stopping(rows[:52],protocol('D1'))['stop'])
        self.assertTrue(stopping(rows[:120],protocol('D1'))['stop'])
        self.assertEqual(protocol('D3')['members'],8)
        with self.assertRaises(AssertionError):protocol('D4')
        self.assertEqual(protocol('D4','D2')['lr'],.02)
        self.assertEqual(protocol('D6','D2')['adapter_weight_decay'],.001)

    def test_common_selection_uses_both_databases(self):
        records={}
        for db in ('mimic','eicu'):
            records[db]={}
            for v in ('D0','D1','D2','D3'):
                for e in (0,1):
                    score=(.55 if db=='mimic' else .42)+(.01 if v=='D2' else 0)
                    records[db][f'{v}_t{e}_s42']=dict(database=db,variant=v,endpoint=e,seed=42,dev_ap=score,
                        complete=True,pilot=False,from_scratch=True,test_loaded=False,frozen_learnable_parameters=0,seconds=10)
        self.assertEqual(select_structure(records)['selected'],'D2')
        records['eicu']['D2_t1_s42']['dev_ap']=.2
        self.assertNotEqual(select_structure(records)['selected'],'D2')
        records['eicu']['D1_t1_s42']['test_loaded']=True
        with self.assertRaises(AssertionError):select_structure(records)


if __name__=='__main__':unittest.main()
