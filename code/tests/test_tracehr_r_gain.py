import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from models.tracehr_r_gain import GainResidual, apply_gain
from scripts.tracehr_r_gain_common import MATRIX, protocol, PROTOCOL_FIELDS
from scripts.tracehr_not_stability_common import protocol as old_protocol


class GainConnection(unittest.TestCase):
    def test_authorized_matrix(self):
        self.assertEqual(len(MATRIX), 6)
        self.assertEqual(len(set(MATRIX)), 6)
        self.assertEqual({s for _, _, s in MATRIX}, {43, 44, 45})
        self.assertEqual({e for _, e, _ in MATRIX}, {0, 1})

    def test_unchanged_recipe(self):
        for v, e, s in MATRIX:
            a, b = protocol(v, 'mimic', s), old_protocol('C5_noT_noA', 'mimic', s)
            self.assertLessEqual({k for k in set(a) | set(b) if a.get(k) != b.get(k)}, PROTOCOL_FIELDS)
            self.assertEqual(a['main_view'], 'z0')
            self.assertEqual(a['checkpoint_selection_view'], 'z0')
            self.assertEqual(a['training_objective'], '0.5*member_CE(z0)')

    def test_original_training_runtime(self):
        text = (Path(__file__).resolve().parents[1] / 'scripts/train_tracehr_r_gain.py').read_text()
        self.assertIn('from scripts import tracehr_noa_ablation_runtime as runner', text)
        self.assertNotIn('torch.optim.', text)

    def test_identity_and_no_independent_summary(self):
        h = torch.arange(48, dtype=torch.float64).reshape(8, 2, 3)
        d = torch.tensor([[.3, -.4, -2.], [1., 0., 2.]], dtype=torch.float64)
        self.assertTrue(torch.equal(apply_gain(h, torch.zeros_like(d)), h))
        self.assertTrue(torch.equal(apply_gain(torch.zeros_like(h), d), torch.zeros_like(h)))
        self.assertTrue(torch.equal(apply_gain(h, d), h * (1 + d[None])))

    def test_gradients(self):
        h = torch.arange(48, dtype=torch.float64).reshape(8, 2, 3).requires_grad_()
        d = torch.full((2, 3), .25, dtype=torch.float64, requires_grad=True)
        apply_gain(h, d).sum().backward()
        self.assertTrue(torch.equal(h.grad, torch.full_like(h, 1.25)))
        self.assertTrue(torch.equal(d.grad, h.detach().sum(0)))

    def test_actual_hook_preserves_other_fields(self):
        h = torch.arange(48, dtype=torch.float64).reshape(8, 2, 3)
        d = torch.full((2, 3), .25, dtype=torch.float64)
        summary = torch.ones(2, 32, 26)
        seen = []
        def encode(s):
            seen.append(s)
            return d
        dummy = SimpleNamespace(fusion_enabled=True, auxiliary_enabled=False,
                                _context=(summary, None, None, False), summary_delta=encode)
        original = {'x_cont': h, 'other': object()}
        result = GainResidual._before_head(dummy, None, (original,))[0]
        self.assertIs(seen[0], summary)
        self.assertEqual(len(seen), 1)
        self.assertTrue(torch.equal(result['x_cont'], h * 1.25))
        self.assertIs(original['x_cont'], h)
        self.assertIs(result['other'], original['other'])
        dummy.fusion_enabled = False
        self.assertIsNone(GainResidual._before_head(dummy, None, (original,)))


if __name__ == '__main__':
    unittest.main()
