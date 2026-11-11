"""Seed routing, within-noT deletions, unchanged loss, and joint optimization."""
import copy
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import test_tracehr_structure as fixture
import torch
from models.tracehr_centered_native import CenteredNative
from models.tracehr_structure import StructureNetwork
from models.tracehr_context_cut import remove_hidden_context
from models.tracehr_c5_ablation import C5AblationNetwork
from models.tracehr_not_confirmation import NoTrajectoryConfirmation, VARIANTS
from models.tracehr_unified_next import parameter_hash
from scripts.tracehr_not_confirmation_common import protocol,MATRIX,PROTOCOL_FIELDS
from scripts.tracehr_c5_ablation_common import protocol as old_protocol
from scripts.tracehr_structure_runtime import Optimizer,gradients
from scripts.tracehr_auxiliary_isolation_diagnostics import path_gradients


class ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.StructureTests.setUpClass();cls.f=fixture.StructureTests()

    def c1(self,seed):
        f=self.f
        base=CenteredNative(f.raw,f.y,f.prep,copy.deepcopy(f.cfg),seed,False,32)
        return remove_hidden_context(StructureNetwork(base,4,'G2',seed),seed)

    def model(self,v,seed=42):
        return NoTrajectoryConfirmation(self.c1(seed),v,seed)

    def test_exact_authorized_matrix_and_unchanged_numeric_recipe(self):
        self.assertEqual(len(MATRIX),12);self.assertEqual(len(set(MATRIX)),12)
        for v in VARIANTS:
            for seed in (42,43,44):
                a=old_protocol('C5_noT','mimic');b=protocol(v,'mimic',seed)
                self.assertEqual({k:a[k] for k in a if k not in PROTOCOL_FIELDS},
                                 {k:b[k] for k in b if k not in PROTOCOL_FIELDS})
                self.assertEqual(b['seed'],seed)
                self.assertTrue(all(b[k]=='z0' for k in ('main_view','checkpoint_selection_view','stopping_view')))
        with self.assertRaises(AssertionError):protocol('C5_noT','eicu',43)

    def test_seed42_exact_original_function_and_continuous_updates(self):
        a=C5AblationNetwork(self.c1(42),'C5_noT',42);b=self.model('C5_noT')
        self.assertEqual(parameter_hash(a),parameter_hash(b))
        oa,ob=Optimizer(a,protocol('C5_noT')),Optimizer(b,protocol('C5_noT'))
        for step in range(3):
            for model,opt in ((a,oa),(b,ob)):
                torch.manual_seed(10042+step);model.train();opt.zero_grad();opt.schedule((step+1)/10)
                model.loss(model(self.f.batch),torch.tensor(self.f.y)).backward();opt.step()
            self.assertEqual(parameter_hash(a),parameter_hash(b))

    def test_seed_routing_and_retained_initialization(self):
        seed_hashes=[]
        for seed in (42,43,44):
            ref=self.model('C5_noT',seed);seed_hashes.append(parameter_hash(ref))
            repeat=self.model('C5_noT',seed)
            self.assertEqual(parameter_hash(ref),parameter_hash(repeat))
            old=dict(ref.named_parameters())
            for v in VARIANTS[1:]:
                b=self.model(v,seed)
                self.assertEqual(b.info['reference_noT_initial_weights_sha256'],parameter_hash(ref))
                for n,p in b.named_parameters():torch.testing.assert_close(p,old[n],rtol=0,atol=0)
                self.assertTrue(b.info['retained_noT_tensors_exact'])
            if seed==42:first={n:p.detach().clone() for n,p in ref.named_parameters()}
            else:
                for n in ('base.numerical.frequency','base.input_members','fusion.summary.0.weight','fusion.auxiliary_head.weight'):
                    self.assertFalse(torch.equal(old[n],first[n]),n)
        self.assertEqual(len(set(seed_hashes)),3)

    def test_deletion_semantics_and_no_trajectory_dispatch(self):
        ref=self.model('C5_noT').eval();p=self.model('C5_noT_noP').eval();m=self.model('C5_noT_noM').eval()
        raw=self.f.batch['raw']
        with torch.no_grad():
            encoded=ref.base.numerical(raw);without=p.base.numerical(raw)
            self.assertEqual(encoded.shape,without.shape)
            self.assertTrue(torch.equal(encoded[:,-836:],without[:,-836:]))
            self.assertEqual(float(without[:,:-836].abs().max()),0.)
            ref.base.input_members.fill_(1.)
            for a,b in zip(ref.forward_views({'raw':raw}),m.forward_views({'raw':raw})):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertIsNone(m.base.input_members)
        for model in (p,m):
            self.assertFalse(hasattr(model.fusion,'query'));self.assertEqual(model.variant,'C5_noT')

    def test_all_remaining_parameters_update_auxiliary_isolation_and_fixed_loss(self):
        for v in VARIANTS:
            model=self.model(v);opt=Optimizer(model,protocol(v));opt.schedule(.1);opt.zero_grad()
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            logits=model({'raw':self.f.batch['raw']});losses=model.path_losses(logits,torch.tensor(self.f.y))
            report=path_gradients(model,losses);self.assertTrue(report['auxiliary_base_gradient_absent'])
            self.assertNotIn('trajectory_encoder',report['groups']);self.assertNotIn('attention',report['groups'])
            self.assertTrue(torch.equal(model.loss(logits,torch.tensor(self.f.y)),.5*sum(losses)))
            model.loss(logits,torch.tensor(self.f.y)).backward();gradients(model);opt.step()
            for prefix in ('base.net.','fusion.summary.','fusion.trajectory.variable.','fusion.auxiliary_head.'):
                self.assertTrue(any(not torch.equal(p,before[n]) for n,p in model.named_parameters() if n.startswith(prefix)),(v,prefix))
            if not v.endswith('noP'):
                self.assertTrue(any(not torch.equal(p,before[n]) for n,p in model.named_parameters() if n.startswith('base.numerical.')))
            if not v.endswith('noM'):self.assertFalse(torch.equal(model.base.input_members,before['base.input_members']))


if __name__=='__main__':unittest.main()
