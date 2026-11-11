import unittest
import torch
from models.cem_gated_event import GatedEventChannels
from models.cem_evidence_structure import EvidenceChannels
from models.cem_connection_ablation import ConnectionChannels
from models.cem_gate_inputs import InputGateChannels, VARIANTS
from scripts.cem_gate_inputs_common import matrix, protocol, u1_protocol


class GateInputTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.nv, self.phi = 11, 56
        g = torch.Generator().manual_seed(207)
        self.raw = torch.randn(3,self.phi+832,generator=g,dtype=torch.float64)
        self.raw[:,self.phi:self.phi+352:11] = 1
        self.batch = dict(raw=self.raw,event_ids=torch.tensor([[0,2,5,11],[1,3,11,11],[11,11,11,11]]),
            event_mask=torch.tensor([[1,1,1,0],[1,1,0,0],[0,0,0,0]],dtype=torch.bool))

    def model(self,v):
        return InputGateChannels(self.nv,12,v,48).double().eval()

    def call(self,m,batch=None):
        return m(self.raw,self.batch if batch is None else batch,self.phi)

    def test_exact_retained_initialization_slice_and_rng(self):
        reference = GatedEventChannels(self.nv,12,48).state_dict()
        for v in VARIANTS:
            rng = torch.get_rng_state().clone()
            actual = self.model(v).float().state_dict()
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(set(reference),set(actual))
            start = 0 if v=='W1' else 96
            for n in actual:
                expected = reference[n][:,start:start+96] if n=='gate.weight' else reference[n]
                self.assertTrue(torch.equal(actual[n],expected),(v,n))
            self.assertEqual(sum(p.numel() for p in self.model(v).gate.parameters()),97)

    def test_gate_deleted_input_has_no_effect_or_gradient(self):
        for v in VARIANTS:
            gate = self.model(v).gate
            x = torch.randn(3,8,192,dtype=torch.float64,requires_grad=True)
            a = gate(x);a.sum().backward()
            start = 0 if v=='W1' else 96
            removed = slice(96,192) if v=='W1' else slice(0,96)
            self.assertEqual(float(x.grad[...,removed].abs().max()),0.)
            self.assertGreater(float(x.grad[...,start:start+96].abs().sum()),0.)
            changed = x.detach().clone();changed[...,removed] += 100.
            torch.testing.assert_close(a,gate(changed),rtol=0,atol=0)

    def test_masked_U1_forward_gradients_and_dropout_order(self):
        for v in VARIANTS:
            for train in (False,True):
                m = self.model(v);ref = GatedEventChannels(self.nv,12,48).double()
                m.train(train);ref.train(train)
                start = 0 if v=='W1' else 96
                with torch.no_grad():
                    removed = slice(96,192) if v=='W1' else slice(0,96)
                    ref.gate.weight[:,removed].zero_()
                rng = torch.get_rng_state().clone()
                a = self.call(m);after = torch.get_rng_state().clone()
                torch.set_rng_state(rng);b = self.call(ref)
                self.assertTrue(torch.equal(after,torch.get_rng_state()))
                torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-12)
                a.square().mean().backward();b.square().mean().backward()
                for n,p in m.named_parameters():
                    target = dict(ref.named_parameters())[n].grad
                    if n=='gate.weight':target = target[:,start:start+96]
                    torch.testing.assert_close(p.grad,target,rtol=1e-9,atol=1e-12)

    def test_gate_limits_preserve_both_residual_routes(self):
        for v in VARIANTS:
            for value,ref in [(1000.,EvidenceChannels(self.nv,12,'S3',48)),
                              (-1000.,ConnectionChannels(self.nv,12,'S3_noDirect',48))]:
                m = self.model(v);ref = ref.double().eval()
                with torch.no_grad():m.gate.weight.zero_();m.gate.bias.fill_(value)
                torch.testing.assert_close(self.call(m),self.call(ref),rtol=0,atol=0)

    def test_all_retained_parameters_receive_main_gradients(self):
        for v in VARIANTS:
            m = self.model(v);self.call(m).square().mean().backward()
            for n,p in m.named_parameters():
                self.assertTrue(p.requires_grad,(v,n))
                self.assertIsNotNone(p.grad,(v,n))
                self.assertTrue(torch.isfinite(p.grad).all(),(v,n))
                self.assertGreater(float(p.grad.norm()),0.,(v,n))

    def test_missing_labs_and_padding_invariance(self):
        self.raw[:,self.phi:self.phi+352:11] = 0
        for v in VARIANTS:
            m = self.model(v);before = self.call(m)
            self.assertTrue(torch.isfinite(before).all())
            batch = dict(self.batch)
            batch['event_ids'] = torch.cat((batch['event_ids'],torch.full((3,5),11)),1)
            batch['event_mask'] = torch.cat((batch['event_mask'],torch.zeros(3,5,dtype=torch.bool)),1)
            torch.testing.assert_close(before,self.call(m,batch),rtol=1e-9,atol=1e-11)
            with torch.no_grad():m.lab_identity.weight.add_(100.)
            torch.testing.assert_close(before,self.call(m),rtol=1e-9,atol=1e-11)

    def test_scope_and_training_recipe_unchanged(self):
        self.assertEqual(set(matrix('mimic')),{(v,e,48) for v in VARIANTS for e in (0,1)})
        with self.assertRaises(AssertionError):matrix('eicu')
        for v in VARIANTS:
            with self.assertRaises(AssertionError):protocol(v,'mimic',49)
            before,after = u1_protocol('U1','mimic',48),protocol(v,'mimic',48)
            changes = {k for k in before.keys()|after.keys() if before.get(k)!=after.get(k)}
            self.assertEqual(changes,{'variant','structure','description','event_direct_gate','event_gate_input_condition'})


if __name__=='__main__':unittest.main()
