import unittest
import torch
from models.cem_gated_event import GatedEventChannels
from models.cem_evidence_structure import EvidenceChannels
from models.cem_connection_ablation import ConnectionChannels
from models.cem_gate_location import GateChannels, VARIANTS
from scripts.cem_gate_common import matrix, protocol, u1_protocol, PROTOCOL_FIELDS


class GateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.nv, self.phi = 11, 56
        g = torch.Generator().manual_seed(207)
        self.raw = torch.randn(3, self.phi+832, generator=g, dtype=torch.float64)
        self.raw[:, self.phi:self.phi+352:11] = 1
        self.batch = dict(raw=self.raw, event_ids=torch.tensor([[0,2,5,11],[1,3,11,11],[11,11,11,11]]),
            event_mask=torch.tensor([[1,1,1,0],[1,1,0,0],[0,0,0,0]], dtype=torch.bool))
        self.hidden = torch.randn(8,3,256,generator=g,dtype=torch.float64)

    def model(self, v):
        return GateChannels(self.nv,12,v,48).double().eval()

    def call(self, m, batch=None):
        return m(self.raw,self.batch if batch is None else batch,self.phi,hidden=self.hidden)

    def test_exact_retained_U1_initialization_and_rng(self):
        reference = GatedEventChannels(self.nv,12,48).state_dict()
        for v in VARIANTS:
            rng = torch.get_rng_state().clone()
            actual = self.model(v).float().state_dict()
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(set(reference)-set(actual),{'gate.weight'} if v=='V1' else set())
            self.assertEqual(set(actual)-set(reference),{'backbone_gate.weight'} if v=='V2' else set())
            for n in actual.keys()&reference.keys():
                self.assertTrue(torch.equal(actual[n],reference[n]),(v,n))

    def test_scalar_has_no_input_dependence_and_matches_constant_U1(self):
        m = self.model('V1'); ref = GatedEventChannels(self.nv,12,48).double().eval()
        with torch.no_grad(): ref.gate.weight.zero_()
        a,b = self.call(m),ref(self.raw,self.batch,self.phi)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        a.square().mean().backward();b.square().mean().backward()
        for n,p in m.named_parameters():
            torch.testing.assert_close(p.grad,dict(ref.named_parameters())[n].grad,rtol=0,atol=0)
        values = m.gate(torch.randn(3,8,192,dtype=torch.float64))
        self.assertTrue(torch.equal(values,values[0,0].expand_as(values)))

    def test_zero_backbone_gate_reproduces_U1_and_dropout_order(self):
        for train in (False,True):
            m=self.model('V2');ref=GatedEventChannels(self.nv,12,48).double()
            m.train(train);ref.train(train)
            with torch.no_grad():m.backbone_gate.weight.zero_()
            rng=torch.get_rng_state().clone()
            a=self.call(m);after=torch.get_rng_state().clone()
            torch.set_rng_state(rng);b=ref(self.raw,self.batch,self.phi)
            self.assertTrue(torch.equal(after,torch.get_rng_state()))
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            a.square().mean().backward();b.square().mean().backward()
            for n,p in ref.named_parameters():
                torch.testing.assert_close(dict(m.named_parameters())[n].grad,p.grad,rtol=1e-9,atol=1e-12)

    def test_gate_limits_reproduce_existing_routes(self):
        for v in VARIANTS:
            for value,ref in [(1000.,EvidenceChannels(self.nv,12,'S3',48)),
                              (-1000.,ConnectionChannels(self.nv,12,'S3_noDirect',48))]:
                m=self.model(v);ref=ref.double().eval()
                with torch.no_grad():
                    m.gate.bias.fill_(value)
                    if v=='V2':m.gate.weight.zero_();m.backbone_gate.weight.zero_()
                torch.testing.assert_close(self.call(m),ref(self.raw,self.batch,self.phi),rtol=0,atol=0)

    def test_all_parameters_and_current_member_hidden_receive_gradients(self):
        for v in VARIANTS:
            m=self.model(v);self.hidden.requires_grad_(True)
            self.call(m).square().mean().backward()
            for n,p in m.named_parameters():
                self.assertIsNotNone(p.grad,(v,n))
                self.assertTrue(torch.isfinite(p.grad).all(),(v,n))
                self.assertGreater(float(p.grad.norm()),0.,(v,n))
            if v=='V2':
                self.assertIsNotNone(self.hidden.grad)
                self.assertGreater(float(self.hidden.grad.norm()),0.)
                context=m.encode(self.raw,self.batch,self.phi)
                changed=self.hidden.detach().clone();changed[0].mul_(-1)
                a=m.readout(*context,hidden=self.hidden);b=m.readout(*context,hidden=changed)
                torch.testing.assert_close(a[1:],b[1:],rtol=0,atol=0)
                self.assertGreater(float((a[0]-b[0]).abs().max()),0.)

    def test_missing_labs_and_event_padding(self):
        self.batch['raw'][:,self.phi:self.phi+352:11]=0
        for v in VARIANTS:
            m=self.model(v);before=self.call(m)
            self.assertTrue(torch.isfinite(before).all())
            batch=dict(self.batch)
            batch['event_ids']=torch.cat((batch['event_ids'],torch.full((3,5),11)),1)
            batch['event_mask']=torch.cat((batch['event_mask'],torch.zeros(3,5,dtype=torch.bool)),1)
            torch.testing.assert_close(before,self.call(m,batch),rtol=1e-9,atol=1e-11)
            with torch.no_grad():m.lab_identity.weight.add_(100.)
            torch.testing.assert_close(before,self.call(m),rtol=1e-9,atol=1e-11)

    def test_authorized_scope_and_fixed_training_recipe(self):
        self.assertEqual(set(matrix('mimic')),{(v,e,48) for v in VARIANTS for e in (0,1)})
        with self.assertRaises(AssertionError):matrix('eicu')
        for v in VARIANTS:
            before,after=u1_protocol('U1','mimic',48),protocol(v,'mimic',48)
            diff={k for k in before.keys()|after.keys() if before.get(k)!=after.get(k)}
            self.assertTrue(diff<=PROTOCOL_FIELDS,diff)
            for k in before.keys()-PROTOCOL_FIELDS:self.assertEqual(before[k],after[k],k)


if __name__=='__main__':unittest.main()
