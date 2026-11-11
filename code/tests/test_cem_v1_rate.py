import unittest
from scripts.cem_v1_rate_common import *


class V1RateTests(unittest.TestCase):
    def test_exact_authorized_scope(self):
        self.assertEqual(set(matrix('mimic')), {(v,e,48) for v in RATES for e in (0,1)})
        with self.assertRaises(AssertionError): matrix('eicu')
        with self.assertRaises(AssertionError): protocol('V1','mimic',48)
        with self.assertRaises(AssertionError): protocol('V1_lr_low','mimic',49)

    def test_only_branch_rate_changes(self):
        ref=v1_protocol('V1','mimic',48)
        for v,lr in RATES.items():
            spec=protocol(v,'mimic',48)
            diff={k for k in ref.keys()|spec.keys() if ref.get(k)!=spec.get(k)}
            self.assertEqual(diff, RATE_FIELDS)
            self.assertEqual(spec['fusion_lr'], lr)
            self.assertEqual(spec['structure'],'V1')
            for k in ('main_view','checkpoint_selection_view','stopping_view'): self.assertEqual(spec[k],'z0')

    def test_optimizer_update_and_schedule_methods_unchanged(self):
        for name in ('step','schedule','zero_grad','state_dict','load_state_dict'):
            self.assertIs(getattr(Optimizer,name),getattr(OriginalOptimizer,name))

    def test_declared_reference_thresholds_and_initialization(self):
        exp=expected()
        self.assertEqual(exp['formal_fits'],4)
        self.assertEqual(exp['variants'],list(VARIANTS))
        for endpoint in (0,1):
            ref=exp['controls'][f'V1_t{endpoint}_s48']
            self.assertEqual(ref['parameters'],12431071)
            self.assertEqual(len(ref['initial_weights_sha256']),64)
        self.assertEqual(exp['controls']['U1_t0_s48']['AP'],.5454013655524206)
        self.assertEqual(exp['controls']['V1_t1_s48']['AP'],.5551855407555454)


if __name__=='__main__': unittest.main()
