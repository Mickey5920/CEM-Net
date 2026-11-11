import hashlib
import unittest
import torch
from torch import nn
from types import SimpleNamespace
from scripts.tracehr_pm_ablation_common import *
from models.tracehr_pm_ablation import remove_pm_component


class Contracts(unittest.TestCase):
    def test_scope(self):
        for db in ('mimic', 'eicu'):
            self.assertEqual(len(set(matrix(db))), 12)
            self.assertEqual({s for v,e,s in matrix(db)}, {48,49,50})
            with self.assertRaises(AssertionError): protocol(VARIANTS[0], db, 51)
            with self.assertRaises(AssertionError): protocol(PM, db, 48)

    def test_fixed_recipe(self):
        for db in ('mimic', 'eicu'):
            for v,e,s in matrix(db):
                ref = prior_protocol(PM if db=='mimic' else 'C5_noT_noA', db, s)
                spec = protocol(v, db, s)
                self.assertLessEqual({k for k in set(spec)|set(ref) if spec.get(k)!=ref.get(k)}, PROTOCOL_FIELDS)
                self.assertEqual(spec['training_objective'], '0.5*member_CE(z0)')
                self.assertTrue(all(spec[k]=='z0' for k in ('main_view','checkpoint_selection_view','stopping_view')))
                self.assertEqual(spec['training_microbatch'],128)
                self.assertEqual(spec['evaluation_microbatch'],256)
                self.assertFalse(spec['tuning'])
                if db=='eicu':
                    self.assertFalse(stopping([{'epoch':i+1,'dev_ap':.5} for i in range(255)],spec)['stop'])
                    self.assertTrue(stopping([{'epoch':i+1,'dev_ap':.5} for i in range(256)],spec)['stop'])

    def test_parent_unchanged(self):
        for name,digest in expected()['parent_manifest'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)
        from scripts import tracehr_noa_validation_runtime as runner
        self.assertEqual(runner.train.__module__,'scripts.tracehr_noa_validation_runtime')
        self.assertEqual(Optimizer.__module__,'scripts.tracehr_noa_ablation_common')

    def test_control_scope(self):
        for db in ('mimic','eicu'):
            controls=expected()['databases'][db]['controls']
            self.assertEqual(len(controls),30)
            self.assertEqual({r['variant'] for r in controls.values()},{PM,'C5_noT_noA',*BASELINES})
            for v,e,s in matrix(db):
                self.assertEqual(controls[f'{PM}_t{e}_s{s}']['seed'],s)

    def test_deletion_keeps_raw_width_and_other_parameters(self):
        for variant in VARIANTS:
            net=nn.Module();net.fusion=nn.Module();net.members=8
            net.main_residual_enabled=net.auxiliary_enabled=False
            base=nn.Module();net.base=base
            enc=nn.Module();base.numerical=enc
            enc.register_buffer('numeric_indices',torch.tensor([0,1]))
            for name,value in [('center',torch.zeros(3)),('factor',torch.ones(3)),('passthrough',torch.zeros(3,dtype=torch.bool))]:enc.register_buffer(name,value)
            enc.weight=nn.Parameter(torch.randn(2,3))
            base.input_members=nn.Parameter(torch.randn(8,9));base.net=nn.Linear(9,2)
            base.hp=SimpleNamespace();base.info={};base.members=8;base.centered=False
            count=sum(p.numel() for p in net.parameters())
            net.info=dict(reference_noA_parameter_count=count+7,removed_from_noA_names=['fusion.old'])
            original={n:p.detach().clone() for n,p in net.named_parameters()}
            remove_pm_component(net,variant)
            self.assertTrue(all(torch.equal(p,original[n]) for n,p in net.named_parameters()))
            self.assertFalse(net.main_residual_enabled);self.assertFalse(net.auxiliary_enabled)
            self.assertFalse(list(net.fusion.parameters()))
            if variant.endswith('noP'):
                raw=torch.randn(4,3);encoded=net.base.numerical(raw)
                self.assertEqual(tuple(encoded.shape),(4,9));self.assertTrue(torch.equal(encoded[:,:6],torch.zeros(4,6)))
                self.assertIsNotNone(net.base.input_members)
            else:
                self.assertIsNone(net.base.input_members);self.assertTrue(hasattr(net.base.numerical,'weight'))
            self.assertGreater(net.info['removed_from_PM'],0)

if __name__=='__main__':unittest.main()
