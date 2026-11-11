import unittest
import torch
from models.cem_gated_event import GatedEventChannels
from models.cem_gate_location import ScalarGate
from models.cem_gate_start import StartGate
from scripts.cem_gate_start_common import matrix, protocol, u1_protocol


class GateStartTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.nv,self.phi = 11,56
        g = torch.Generator().manual_seed(207)
        self.raw = torch.randn(3,self.phi+832,generator=g,dtype=torch.float64)
        self.raw[:,self.phi:self.phi+352:11] = 1
        self.batch = dict(raw=self.raw,event_ids=torch.tensor([[0,2,5,11],[1,3,11,11],[11,11,11,11]]),
            event_mask=torch.tensor([[1,1,1,0],[1,1,0,0],[0,0,0,0]],dtype=torch.bool))

    def channels(self, variant):
        m = GatedEventChannels(self.nv,12,48).double()
        if variant=='X1': m.gate = StartGate(m.gate)
        if variant=='V1': m.gate = ScalarGate(m.gate.bias)
        return m

    def call(self, model, batch=None):
        return model(self.raw,self.batch if batch is None else batch,self.phi)

    def test_retained_initialization_and_rng(self):
        u = self.channels('U1')
        rng = torch.get_rng_state().clone()
        x = self.channels('X1')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(set(x.state_dict())-set(u.state_dict()),{'gate.alpha'})
        for n,p in u.state_dict().items(): self.assertTrue(torch.equal(p,x.state_dict()[n]),n)
        self.assertEqual(x.gate.alpha.item(),0.)
        self.assertEqual(sum(p.numel() for p in x.gate.parameters()),194)
        self.assertTrue(all(p.requires_grad for p in x.parameters()))

    def test_initial_V1_function_gradients_and_rng(self):
        for training in (False,True):
            x,v = self.channels('X1'),self.channels('V1')
            x.train(training);v.train(training)
            rng = torch.get_rng_state().clone()
            a = self.call(x); after = torch.get_rng_state().clone()
            torch.set_rng_state(rng);b = self.call(v)
            self.assertTrue(torch.equal(after,torch.get_rng_state()))
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            a.square().mean().backward();b.square().mean().backward()
            for n,p in v.named_parameters():
                torch.testing.assert_close(p.grad,dict(x.named_parameters())[n].grad,rtol=1e-9,atol=1e-12)
            self.assertEqual(x.gate.weight.grad.abs().max().item(),0.)
            self.assertGreater(abs(x.gate.alpha.grad.item()),0.)

    def test_unit_coefficient_recovers_U1_function(self):
        x,u = self.channels('X1').eval(),self.channels('U1').eval()
        with torch.no_grad(): x.gate.alpha.fill_(1.)
        torch.testing.assert_close(self.call(x),self.call(u),rtol=1e-9,atol=1e-12)

    def test_learning_and_restore(self):
        x = self.channels('X1').eval()
        opt = torch.optim.AdamW(x.parameters(),lr=.001)
        self.assertEqual(sum(p is x.gate.alpha for g in opt.param_groups for p in g['params']),1)
        self.call(x).square().mean().backward()
        self.assertEqual(x.gate.weight.grad.norm().item(),0.)
        opt.step();opt.zero_grad()
        self.assertNotEqual(x.gate.alpha.item(),0.)
        before = x.gate.weight.detach().clone()
        self.call(x).square().mean().backward()
        self.assertGreater(x.gate.weight.grad.norm().item(),0.)
        self.assertTrue(all(p.grad is not None for p in x.parameters()))
        opt.step()
        self.assertFalse(torch.equal(before,x.gate.weight))
        clone = self.channels('X1').eval();clone.load_state_dict(x.state_dict())
        torch.testing.assert_close(self.call(x),self.call(clone),rtol=0,atol=0)

    def test_missing_labs_and_padding(self):
        self.raw[:,self.phi:self.phi+352:11] = 0
        x = self.channels('X1').eval()
        before = self.call(x);self.assertTrue(torch.isfinite(before).all())
        batch = dict(self.batch)
        batch['event_ids'] = torch.cat((batch['event_ids'],torch.full((3,5),11)),1)
        batch['event_mask'] = torch.cat((batch['event_mask'],torch.zeros(3,5,dtype=torch.bool)),1)
        torch.testing.assert_close(before,self.call(x,batch),rtol=1e-9,atol=1e-11)

    def test_scope_and_unchanged_recipe(self):
        self.assertEqual(matrix('mimic'),(('X1',0,48),('X1',1,48)))
        with self.assertRaises(AssertionError):matrix('eicu')
        with self.assertRaises(AssertionError):protocol('X1','mimic',49)
        before,after = u1_protocol('U1','mimic',48),protocol('X1','mimic',48)
        changes = {k for k in before.keys()|after.keys() if before.get(k)!=after.get(k)}
        self.assertEqual(changes,{'variant','structure','description','event_direct_gate',
            'event_gate_alpha_initial','event_gate_alpha_trainable'})


if __name__=='__main__':unittest.main()
