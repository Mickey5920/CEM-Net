"""Controlled C5 deletions: initialization, forward paths, gradients and optimizer."""
import copy
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import test_tracehr_structure as fixture
import torch
from models.tracehr_context_cut import remove_hidden_context
from models.tracehr_auxiliary_isolation import AuxiliaryIsolationNetwork
from models.tracehr_c5_ablation import C5AblationNetwork,VARIANTS
from models.tracehr_unified_next import parameter_hash
from scripts.tracehr_auxiliary_isolation_common import protocol as c5_protocol
from scripts.tracehr_c5_ablation_common import protocol,PROTOCOL_CHANGES
from scripts.tracehr_structure_runtime import Optimizer,gradients
from scripts.tracehr_auxiliary_isolation_diagnostics import path_gradients


class C5AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.StructureTests.setUpClass();cls.fixture=fixture.StructureTests()

    def c1(self):return remove_hidden_context(self.fixture.model('G2'))
    def full(self):return AuxiliaryIsolationNetwork(self.c1())
    def model(self,v):return C5AblationNetwork(self.c1(),v)

    def test_only_authorized_protocol_fields_change(self):
        for v in VARIANTS:
            a=c5_protocol('C5','mimic');b=protocol(v,'mimic')
            self.assertEqual({k for k in set(a)|set(b) if a.get(k)!=b.get(k)},PROTOCOL_CHANGES)
            self.assertEqual(b['main_view'],'z0');self.assertEqual(b['checkpoint_selection_view'],'z0')
            self.assertEqual(b['stopping_view'],'z0')
        with self.assertRaises(AssertionError):protocol(VARIANTS[0],'eicu')

    def test_retained_random_tensors_buffers_and_rng(self):
        reference=self.full();old=dict(reference.named_parameters());buffers=dict(reference.named_buffers())
        for v in VARIANTS:
            c1=self.c1();rng=torch.get_rng_state().clone();b=C5AblationNetwork(c1,v)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(b.info['reference_C5_initial_weights_sha256'],parameter_hash(reference))
            for n,p in b.named_parameters():
                ref=old[n]
                if v=='C5_noT' and n=='fusion.combine.0.weight':ref=ref[:,:64]
                torch.testing.assert_close(p,ref,rtol=0,atol=0)
            for n,p in b.named_buffers():torch.testing.assert_close(p,buffers[n],rtol=0,atol=0)
            self.assertTrue(b.info['retained_initialization_exact'])
            self.assertGreater(b.info['removed_from_C5'],0)
            self.assertTrue(all(p.requires_grad for p in b.parameters()))

    def test_no_trajectory_keeps_auxiliary_and_matches_empty_trajectory(self):
        a=self.full().eval();b=self.model('C5_noT').eval();batch=self.fixture.batch
        empty=dict(raw=batch['raw'],records=torch.empty(0,8),segments=torch.empty(0,dtype=torch.long),
                   counts=torch.zeros(32,32,8,dtype=torch.int32))
        with torch.no_grad():
            ref=a.forward_views(empty);out=b.forward_views({'raw':batch['raw']})
        for x,y in zip(ref,out):torch.testing.assert_close(x,y,rtol=2e-6,atol=2e-6)
        self.assertEqual(set(b.fusion.trajectory._modules),{'variable'})
        self.assertFalse(hasattr(b.fusion,'query'));self.assertTrue(hasattr(b.fusion,'auxiliary_head'))
        self.assertEqual(b.fusion.combine[0].weight.shape,(64,64))

    def test_no_periodic_preserves_raw_coordinates_and_native_width(self):
        a=self.full().eval();b=self.model('C5_noP').eval();x=self.fixture.batch['raw']
        with torch.no_grad():original=a.base.numerical(x);modified=b.base.numerical(x)
        self.assertEqual(original.shape,modified.shape)
        self.assertTrue(torch.equal(original[:,-836:],modified[:,-836:]))
        self.assertEqual(float(modified[:,:-836].abs().max()),0.)
        self.assertEqual(list(b.base.numerical.parameters()),[])
        self.assertEqual(set(b.info['removed_parameter_names']),
                         {'base.numerical.frequency','base.numerical.phase','base.numerical.weight','base.numerical.bias'})

    def test_no_members_is_identity_with_eight_loss_members(self):
        a=self.full().eval();b=self.model('C5_noM').eval()
        # A temporary randomly initialized comparator, never a training warm start.
        with torch.no_grad():
            a.base.input_members.fill_(1.)
            ref=a.forward_views(self.fixture.batch);out=b.forward_views(self.fixture.batch)
        for x,y in zip(ref,out):torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertIsNone(b.base.input_members);self.assertEqual(out[0].shape,(32,8,2))
        torch.testing.assert_close(out[0][:,1:],out[0][:,:1].expand(-1,7,-1),rtol=0,atol=0)
        self.assertEqual(b.info['removed_parameter_names'],['base.input_members'])

    def test_auxiliary_isolation_and_exact_dual_loss_for_each_deletion(self):
        for v in VARIANTS:
            b=self.model(v).train();y=torch.tensor(self.fixture.y);logits=b(self.fixture.batch)
            main,aux=b.path_losses(logits,y);report=path_gradients(b,(main,aux))
            self.assertTrue(report['auxiliary_base_gradient_absent'])
            absent={'C5_noT':['trajectory_encoder','attention'],'C5_noP':['periodic_frontend'],
                    'C5_noM':['input_members']}[v]
            for key in absent:self.assertNotIn(key,report['groups'])
            self.assertGreater(report['groups']['auxiliary_head']['auxiliary_norm'],0)
            self.assertTrue(torch.equal(b.loss(logits,y),.5*(main+aux)))
            self.assertTrue(all(p.grad is None for p in b.parameters()))
            b.loss(logits,y).backward();gradients(b)

    def test_every_remaining_module_updates_and_optimizer_restores(self):
        for v in VARIANTS:
            b=self.model(v);opt=Optimizer(b,protocol(v,'mimic'));opt.schedule(.1)
            before={n:p.detach().clone() for n,p in b.named_parameters()}
            b.train();opt.zero_grad();b.loss(b(self.fixture.batch),torch.tensor(self.fixture.y)).backward()
            gradients(b);opt.step()
            prefixes=['base.net.','fusion.summary.','fusion.trajectory.variable.','fusion.auxiliary_head.']
            if v!='C5_noP':prefixes.append('base.numerical.')
            if v!='C5_noM':prefixes.append('base.input_members')
            if v!='C5_noT':prefixes.append('fusion.query.')
            for prefix in prefixes:
                self.assertTrue(any(not torch.equal(p,before[n]) for n,p in b.named_parameters() if n.startswith(prefix)),(v,prefix))
            state=copy.deepcopy(opt.state_dict());values=copy.deepcopy(opt.values())
            opt.schedule(70);opt.load_state_dict(state);self.assertEqual(values,opt.values())

    def test_fixed_main_output_single_base_forward_and_inference_no_auxiliary(self):
        for v in VARIANTS:
            b=self.model(v).train();calls=[];handle=b.base.register_forward_hook(lambda *args:calls.append(1))
            z0,za=b(self.fixture.batch);handle.remove();self.assertEqual(len(calls),1)
            self.assertEqual(za.shape,(32,1,2));b.eval()
            with torch.no_grad():reference,_=b.forward_views(self.fixture.batch)
            b.fusion.encode=lambda _:self.fail('Inference read trajectory')
            b.fusion.auxiliary_head.forward=lambda _:self.fail('Inference used auxiliary head')
            with torch.no_grad():out=b({'raw':self.fixture.batch['raw']})
            torch.testing.assert_close(out,reference,rtol=1e-6,atol=1e-6)


if __name__=='__main__':unittest.main()
