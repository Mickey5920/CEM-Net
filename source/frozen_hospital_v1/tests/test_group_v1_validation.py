import copy, inspect, unittest
import torch
from scripts.group_v1_validation_common import *
from scripts.cem_v1_group_confirm_common import Optimizer as ParentOptimizer
from scripts.cem_v1_group_rate_common import protocol as seed48_protocol
from models.group_v1_paths import delete_path, removed_names
from models.cem_gate_location import GateChannels

class ValidationTests(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(len(matrix('mimic')),12);self.assertEqual(len(matrix('eicu')),6)
        for v,d,s in [('V1_group_rate','mimic',48),('V1_noDirect','eicu',48),('V1_noInteraction','mimic',51)]:
            with self.assertRaises(AssertionError):protocol(v,d,s)
    def test_fixed_recipe(self):
        from scripts.cem_structure_common import cem_protocol
        parent=seed48_protocol('V1_group_rate','mimic',48)
        for db in ('mimic','eicu'):
            for v,e,s in matrix(db):
                spec=protocol(v,db,s);base=cem_protocol(CEM,db,s)
                self.assertFalse(spec['tuning']);self.assertEqual(spec['fusion_lr'],.00030)
                self.assertEqual(spec['fusion_readout_lr'],.00015)
                self.assertEqual({k for k in base.keys()|spec.keys() if base.get(k)!=spec.get(k)}-PROTOCOL_FIELDS,set())
                for k in ('main_view','checkpoint_selection_view','stopping_view'):self.assertEqual(spec[k],'z0')
                if db=='eicu':self.assertEqual(spec['min_epochs'],256);self.assertEqual(spec['early_stop_boundaries'],[])
                else:
                    for k in ('effective_batch','training_microbatch','evaluation_microbatch','early_stop_boundaries','fusion_lr'):
                        self.assertEqual(spec[k],parent[k])
    def test_exact_path_outputs_and_initialization(self):
        torch.set_num_threads(2)
        for seed in SEEDS:
            full=GateChannels(20,12,'V1',seed).eval();reference=dict(full.named_parameters())
            x=torch.randn(3,92+832);batch={'raw':x,'event_ids':torch.tensor([[0,1,20],[3,20,20],[1,4,7]]),
                'event_mask':torch.tensor([[1,1,0],[1,0,0],[1,1,1]],dtype=torch.bool)}
            with torch.no_grad():
                event,interaction,direct=full.encode(x,batch,92)
                expected_direct=(full.gate.bias.sigmoid()*direct).transpose(0,1)
                expected_interaction=(full.extra_scale*full.extra_output(interaction)).transpose(0,1)
                self.assertTrue(torch.equal(full(x,batch,92),expected_direct+expected_interaction))
            for variant,target in [('V1_noDirect',expected_interaction),('V1_noInteraction',expected_direct)]:
                reduced=copy.deepcopy(full);before=torch.get_rng_state().clone();delete_path(reduced,variant)
                self.assertTrue(torch.equal(before,torch.get_rng_state()))
                actual=dict(reduced.named_parameters());self.assertEqual(set(reference)-set(actual),removed_names(reference,variant))
                self.assertTrue(all(torch.equal(p,reference[n]) for n,p in actual.items()))
                with torch.no_grad():
                    self.assertTrue(torch.equal(reduced(x,batch,92),target))
                grads=torch.autograd.grad(reduced(x,batch,92).square().sum(),tuple(reduced.parameters()),allow_unused=True)
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
    def test_optimizer_coverage(self):
        for v in VARIANTS:
            f=GateChannels(20,12,'V1',48)
            if v!='V1_group_rate':delete_path(f,v)
            groups=split_fusion(f);flat=[id(p) for g in groups.values() for n,p in g]
            self.assertEqual(len(flat),len(set(flat)));self.assertEqual(set(flat),{id(p) for p in f.parameters()})
            self.assertEqual({n for n,p in groups['readout']},READOUT_NAMES-removed_names(READOUT_NAMES,v))
        for method in ('schedule','step','zero_grad','state_dict','load_state_dict'):
            self.assertEqual(inspect.getsource(getattr(Optimizer,method)),inspect.getsource(getattr(ParentOptimizer,method)))
    def test_sources_and_control_labels(self):
        exp=expected();self.assertEqual(exp['formal_fits'],{'mimic':12,'eicu':6})
        self.assertEqual(exp['primary_baselines'],['B0','tabm'])
        self.assertEqual(exp['display_names'][CEM],'去除整个证据增强分支')
        for db,number in [('mimic',24),('eicu',18)]:self.assertEqual(len(exp['controls'][db]),number)
        for name,digest in exp['parent_manifest'].items():self.assertEqual(sha(ROOT/name),digest)

if __name__=='__main__':unittest.main()
