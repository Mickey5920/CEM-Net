import copy
import os
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,os.environ['TRACEHR_TABULAR_DEPS'])
import numpy as np
import torch
from models.tracehr_centered_native import CenteredNative
from models.tracehr_unified_next import parameter_hash
from scripts.tracehr_execution_refine_common import configure_e2
from scripts.tracehr_execution_eicu_common import protocol,VARIANTS,stopping
from scripts.tracehr_centered_native_common import old_baseline
from scripts.tracehr_backbone_refine_runtime import Optimizer
from scripts.tracehr_multiview_common import read


class RecipeTests(unittest.TestCase):
    def test_exact_matrix_and_boundary_rule(self):
        self.assertEqual(VARIANTS,('F0','E2','F1','F2','F3','F4','F5'))
        with self.assertRaises(AssertionError):protocol('F6')
        rows=[dict(epoch=i+1,dev_ap=.5) for i in range(52)]
        for v in VARIANTS:
            self.assertFalse(stopping(rows,protocol(v))['stop'])
            full=[dict(epoch=i+1,dev_ap=.5) for i in range(256)]
            self.assertTrue(stopping(full,protocol(v))['stop'])
            self.assertEqual(protocol(v)['schedule_reference_epochs'],256)
            self.assertFalse(protocol(v)['centered'])

    def test_isolated_runtime_factors_and_shared_initialization(self):
        torch.set_num_threads(2)
        rng=np.random.RandomState(7);raw=rng.normal(size=(64,12)).astype(np.float32)
        raw[:,10:]=(raw[:,10:]>0).astype(np.float32);y=np.tile([0,1],32)
        prep=dict(center=np.zeros(12,np.float32),factor=np.ones(12,np.float32),
                  binary=np.arange(12)>=10,numeric_indices=np.arange(10))
        cfg=read(old_baseline(Path(os.environ['TRACEHR_PROJECT']),'eicu','B0')/'effective_config_task0.json')
        base=CenteredNative(raw,y,prep,cfg,42,False,64);initial=parameter_hash(base)
        base.eval();x=torch.tensor(raw)
        with torch.no_grad():expected=base(x)
        reference=Optimizer(base,protocol('F4'));reference.schedule(.1)
        reference_groups={r['name']:r for r in reference.values()['groups']}
        for v in VARIANTS[1:]:
            model=copy.deepcopy(base);spec=protocol(v);configure_e2(model,spec)
            self.assertEqual(initial,parameter_hash(model))
            model.eval()
            with torch.no_grad():self.assertTrue(torch.equal(expected,model(x)))
            opt=Optimizer(model,spec);opt.schedule(.1)
            for group in opt.values()['groups']:
                old=reference_groups[group['name']]
                factor=.5 if v=='F1' else 1.
                if v=='F2' and group['name']=='numerical.frequency':factor=.3
                if v=='F3' and old['factors'].get('lr')==6.:factor=1/3
                self.assertAlmostEqual(group['lr'],old['lr']*factor,places=12)
                self.assertEqual(group['wd'],old['wd'])
            self.assertAlmostEqual(opt.adapter_base,.001 if v=='F4' else .0003)
            self.assertEqual(model.hp.hyper_base_values['p_drop'][''],.25 if v=='F5' else .15)
            model.train();opt.zero_grad();model.loss(model(x),torch.tensor(y)).backward()
            self.assertTrue(all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            opt.step()


if __name__=='__main__':unittest.main()
