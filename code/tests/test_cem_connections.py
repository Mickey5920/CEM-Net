import unittest
import torch
from models.cem_evidence_structure import EvidenceChannels
from models.cem_connection_ablation import ConnectionChannels, VARIANTS, DIRECT_PARAMETERS
from scripts.cem_connections_common import matrix, protocol, s3_protocol


class ConnectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.nv, self.phi = 11, 56
        g = torch.Generator().manual_seed(207)
        self.raw = torch.randn(3, self.phi+832, generator=g, dtype=torch.float64)
        self.raw[:, self.phi:self.phi+352:11] = 1
        self.batch = dict(raw=self.raw, event_ids=torch.tensor([[0,2,5,11],[1,3,11,11],[11,11,11,11]]),
            event_mask=torch.tensor([[1,1,1,0],[1,1,0,0],[0,0,0,0]], dtype=torch.bool))

    def test_exact_retained_initial_values_and_deletion(self):
        ref = EvidenceChannels(self.nv, 12, 'S3', 48).state_dict()
        for v in VARIANTS:
            rng = torch.get_rng_state().clone()
            m = ConnectionChannels(self.nv, 12, v, 48)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            actual = m.state_dict()
            self.assertEqual(set(ref)-set(actual), set() if v=='S3_noSkip' else DIRECT_PARAMETERS)
            for n, p in actual.items():
                self.assertTrue(torch.equal(p, ref[n]), n)

    def test_reference_connection_interventions_match_outputs_and_gradients(self):
        for v in VARIANTS:
            ref = EvidenceChannels(self.nv, 12, 'S3', 48).double().eval()
            m = ConnectionChannels(self.nv, 12, v, 48).double().eval()
            context = {}
            if not m.event_direct_enabled:
                ref.output.register_forward_hook(lambda mod, args, out: out*0)
            if not m.interaction_event_skip_enabled:
                def capture(mod, args): context['query'] = args[0]
                ref.lab_attention.register_forward_pre_hook(capture)
                ref.lab_norm.register_forward_pre_hook(lambda mod, args: (args[0]-context['query'],))
            original, changed = ref(self.raw, self.batch, self.phi), m(self.raw, self.batch, self.phi)
            torch.testing.assert_close(original, changed, rtol=1e-9, atol=1e-11)
            original.square().mean().backward(); changed.square().mean().backward()
            for n, p in m.named_parameters():
                self.assertIsNotNone(p.grad, (v,n))
                torch.testing.assert_close(p.grad, dict(ref.named_parameters())[n].grad, rtol=1e-7, atol=1e-10)

    def test_event_query_still_receives_gradients(self):
        for v in VARIANTS:
            m = ConnectionChannels(self.nv, 12, v, 48).double().eval()
            m(self.raw, self.batch, self.phi).square().mean().backward()
            self.assertGreater(float(m.type_embedding.weight.grad.norm()), 0, v)
            self.assertGreater(float(m.lab_attention.in_proj_weight.grad[:96].norm()), 0, v)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_empty_masks_and_padding(self):
        self.batch['raw'][:, self.phi:self.phi+352:11] = 0
        for v in VARIANTS:
            m = ConnectionChannels(self.nv, 12, v, 48).double().eval()
            ref = m(self.raw, self.batch, self.phi)
            self.assertTrue(torch.isfinite(ref).all())
            other = dict(self.batch)
            other['event_ids'] = torch.cat((other['event_ids'], torch.full((3,5), 11)), 1)
            other['event_mask'] = torch.cat((other['event_mask'], torch.zeros(3,5,dtype=torch.bool)), 1)
            torch.testing.assert_close(ref, m(self.raw, other, self.phi), rtol=1e-9, atol=1e-11)

    def test_scope_and_recipe(self):
        self.assertEqual(len(matrix('mimic')), 6)
        for v in VARIANTS:
            before, after = s3_protocol('S3','mimic',48), protocol(v,'mimic',48)
            diff = {k for k in before.keys()|after.keys() if before.get(k)!=after.get(k)}
            self.assertEqual(diff, {'variant','structure','ablated_component','description','ablation_implementation',
                                   'event_direct_enabled','interaction_event_skip_enabled'})


if __name__ == '__main__': unittest.main()
