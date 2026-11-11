import unittest
from scripts.cem_v1_group_rate_common import *
from models.cem_gate_location import GateChannels


class V1GroupRateTests(unittest.TestCase):
    def test_exact_authorized_scope(self):
        self.assertEqual(set(matrix('mimic')),{('V1_group_rate',e,48) for e in (0,1)})
        for args in [('V1','mimic',48),('V1_group_rate','mimic',49),('V1_group_rate','eicu',48)]:
            with self.assertRaises(AssertionError):protocol(*args)

    def test_only_authorized_protocol_fields_change(self):
        ref=v1_protocol('V1','mimic',48);spec=protocol(VARIANTS[0],'mimic',48)
        self.assertEqual({k for k in ref.keys()|spec.keys() if ref.get(k)!=spec.get(k)},RATE_FIELDS)
        self.assertEqual(spec['fusion_lr'],.00030);self.assertEqual(spec['fusion_readout_lr'],.00015)
        self.assertEqual(spec['structure'],'V1')
        for k in ('main_view','checkpoint_selection_view','stopping_view'):self.assertEqual(spec[k],'z0')

    def test_actual_fusion_partition_keeps_gate_and_encoder_norms(self):
        fusion=GateChannels(20,12,'V1',48);before=parameter_hash(fusion)
        rng=torch.get_rng_state().clone();groups=split_fusion(fusion)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(before,parameter_hash(fusion))
        self.assertEqual({n for n,p in groups['readout']},set(READOUT_NAMES))
        self.assertEqual(sum(p.numel() for n,p in groups['readout']),49858)
        fast={n for n,p in groups['encoder']}
        self.assertTrue({'gate.bias','cross_norm.weight','self_norm.weight','lab_norm.weight'}<=fast)
        ids=[id(p) for members in groups.values() for n,p in members]
        self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in fusion.parameters()})

    def test_deleted_required_parameter_rejected(self):
        fusion=GateChannels(20,12,'V1',48);del fusion.extra_scale
        with self.assertRaises(AssertionError):split_fusion(fusion)

    def test_inherited_update_and_recovery(self):
        for name in ('step','zero_grad','state_dict','load_state_dict'):
            self.assertIs(getattr(Optimizer,name),getattr(OriginalOptimizer,name))

    def test_archive_and_thresholds(self):
        exp=expected();self.assertEqual(exp['formal_fits'],2);self.assertEqual(len(exp['controls']),18)
        for e in (0,1):
            ref=exp['controls'][f'V1_t{e}_s48'];self.assertEqual(ref['parameters'],12431071)
            self.assertEqual(len(ref['initial_weights_sha256']),64)
        self.assertEqual(exp['controls']['U1_t0_s48']['AP'],.5454013655524206)
        self.assertEqual(exp['controls']['V1_t1_s48']['AP'],.5551855407555454)


if __name__=='__main__':unittest.main()
