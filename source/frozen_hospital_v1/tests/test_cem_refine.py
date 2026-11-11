import unittest
import torch
from models.cem_evidence_structure import EvidenceChannels
from models.cem_connection_ablation import ConnectionChannels
from models.cem_gated_event import GatedEventChannels
from models.cem_refine import RefineChannels, VARIANTS, expected_new_names, expected_removed_names
from scripts.cem_refine_common import matrix, protocol, s3_protocol, PROTOCOL_FIELDS


class RefineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.nv, self.phi = 11, 56
        g = torch.Generator().manual_seed(207)
        self.raw = torch.randn(3, self.phi+832, generator=g, dtype=torch.float64)
        self.raw[:, self.phi:self.phi+352:11] = 1
        self.batch = dict(raw=self.raw, event_ids=torch.tensor([[0,2,5,11],[1,3,11,11],[11,11,11,11]]),
            event_mask=torch.tensor([[1,1,1,0],[1,1,0,0],[0,0,0,0]], dtype=torch.bool))
        self.hidden = torch.randn(8, 3, 256, generator=g, dtype=torch.float64)

    def model(self, variant):
        return (GatedEventChannels(self.nv,12,48) if variant=='U1'
                else RefineChannels(self.nv,12,variant,48)).double().eval()

    def run_model(self, model, batch=None):
        if getattr(model,'refinement',None)=='U5':
            return model(self.raw,self.batch if batch is None else batch,self.phi,hidden=self.hidden)
        return model(self.raw,self.batch if batch is None else batch,self.phi)

    def test_initialization_namespace_and_exact_kept_parameters(self):
        reference = EvidenceChannels(self.nv,12,'S3',48).state_dict()
        for v in VARIANTS:
            rng = torch.get_rng_state().clone()
            model = self.model(v).float()
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            actual = model.state_dict()
            self.assertEqual(set(reference)-set(actual),expected_removed_names(v,reference))
            self.assertEqual(set(actual)-set(reference),expected_new_names(v))
            for n in set(actual)&set(reference):
                self.assertTrue(torch.equal(actual[n],reference[n]),(v,n))

    def test_gate_limits_reproduce_both_existing_routes(self):
        for value, reference in [(1000.,EvidenceChannels(self.nv,12,'S3',48)),
                                 (-1000.,ConnectionChannels(self.nv,12,'S3_noDirect',48))]:
            model = self.model('U1'); reference = reference.double().eval()
            hook = model.gate.register_forward_hook(lambda m,a,o: torch.full_like(o,value))
            actual = self.run_model(model)
            expected = reference(self.raw,self.batch,self.phi)
            torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-11)
            actual.square().mean().backward();expected.square().mean().backward()
            for n,p in reference.named_parameters():
                torch.testing.assert_close(dict(model.named_parameters())[n].grad,p.grad,rtol=1e-7,atol=1e-10)
            hook.remove()

    def test_zero_observation_bias_and_zero_condition_reproduce_no_direct(self):
        ref = ConnectionChannels(self.nv,12,'S3_noDirect',48).double().eval()
        expected = ref(self.raw,self.batch,self.phi)
        for v,name in [('U2','observation_bias'),('U5','query_condition')]:
            model = self.model(v)
            with torch.no_grad(): getattr(model,name).weight.zero_()
            torch.testing.assert_close(self.run_model(model),expected,rtol=1e-9,atol=1e-11)

    def test_reverse_attention_reads_event_slots_and_masks_lab_queries(self):
        model = self.model('U3'); seen = {}
        def capture(m,args):
            seen['q'],seen['k'],seen['v'] = args[:3]
        model.lab_attention.register_forward_pre_hook(capture)
        self.run_model(model)
        self.assertEqual(seen['q'].shape,(3,33,96))
        self.assertEqual(seen['k'].shape,(3,32,96))
        self.assertTrue(seen['k'] is seen['v'])
        self.batch['raw'][:,self.phi:self.phi+352:11] = 0
        before = self.run_model(model)
        with torch.no_grad():
            model.lab_identity.weight.add_(100.)
            model.lab_projection[0].weight.mul_(10.)
        torch.testing.assert_close(before,self.run_model(model),rtol=1e-9,atol=1e-11)

    def test_light_interaction_matches_removal_of_self_block(self):
        model = self.model('U4')
        ref = ConnectionChannels(self.nv,12,'S3_noDirect',48).double().eval()
        ref.self_attention.register_forward_hook(lambda m,a,o:(torch.zeros_like(o[0]),o[1]))
        ref.self_norm.register_forward_hook(lambda m,a,o:a[0])
        ref.self_ffn.register_forward_hook(lambda m,a,o:a[0])
        actual,expected = self.run_model(model),ref(self.raw,self.batch,self.phi)
        torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-11)
        actual.square().mean().backward();expected.square().mean().backward()
        for n,p in model.named_parameters():
            torch.testing.assert_close(p.grad,dict(ref.named_parameters())[n].grad,rtol=1e-7,atol=1e-10)

    def test_new_paths_and_all_retained_tensors_receive_gradients(self):
        for v in VARIANTS:
            model = self.model(v)
            self.hidden.requires_grad_(True)
            self.run_model(model).square().mean().backward()
            for n,p in model.named_parameters():
                self.assertIsNotNone(p.grad,(v,n))
                self.assertTrue(torch.isfinite(p.grad).all(),(v,n))
                self.assertGreater(float(p.grad.norm()),0.,(v,n))
            if v=='U5':
                self.assertIsNotNone(self.hidden.grad)
                self.assertGreater(float(self.hidden.grad.norm()),0.)
                # The conditioning reads the individual member, not their mean.
                cross,mask = model.encode(self.raw,self.batch,self.phi)
                changed = self.hidden.detach().clone(); changed[0].mul_(-1)
                before = model.readout(cross,mask,self.hidden)
                after = model.readout(cross,mask,changed)
                torch.testing.assert_close(before[1:],after[1:],rtol=0,atol=0)
                self.assertGreater(float((before[0]-after[0]).abs().max()),0.)

    def test_empty_masks_event_padding_and_missing_lab_metadata(self):
        self.batch['raw'][:,self.phi:self.phi+352:11] = 0
        for v in VARIANTS:
            model = self.model(v)
            ref = self.run_model(model)
            self.assertTrue(torch.isfinite(ref).all())
            other = dict(self.batch)
            other['event_ids'] = torch.cat((other['event_ids'],torch.full((3,5),11)),1)
            other['event_mask'] = torch.cat((other['event_mask'],torch.zeros(3,5,dtype=torch.bool)),1)
            torch.testing.assert_close(ref,self.run_model(model,other),rtol=1e-9,atol=1e-11)

    def test_scope_and_fixed_training_recipe(self):
        self.assertEqual(set(matrix('mimic')),{(v,e,48) for v in VARIANTS for e in (0,1)})
        with self.assertRaises(AssertionError): matrix('eicu')
        for v in VARIANTS:
            before,after = s3_protocol('S3','mimic',48),protocol(v,'mimic',48)
            diff = {k for k in before.keys()|after.keys() if before.get(k)!=after.get(k)}
            self.assertTrue(diff<=PROTOCOL_FIELDS,diff)
            for k in before.keys()-PROTOCOL_FIELDS:
                self.assertEqual(before[k],after[k],k)


if __name__=='__main__': unittest.main()
