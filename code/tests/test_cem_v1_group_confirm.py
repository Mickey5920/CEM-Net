import ast
import inspect
import unittest
from pathlib import Path
from scripts.cem_v1_group_confirm_common import *
from scripts.cem_v1_group_rate_common import protocol as parent_protocol, Optimizer as ParentOptimizer
from models.cem_gate_location import GateChannels


class GroupConfirmationTests(unittest.TestCase):
    def test_constructor_seed_admission_only(self):
        from scripts.cem_v1_group_confirm_seed_route import original_init_source, routed_init_source
        self.assertEqual(routed_init_source,original_init_source.replace('seed == 48','seed in (48, 49, 50)'))

    def test_scope(self):
        self.assertEqual(set(matrix('mimic')),{('V1_group_rate',e,s) for s in (49,50) for e in (0,1)})
        for v,d,s in [('V1_group_rate','mimic',48),('V1_group_rate','mimic',51),('V1_group_rate','eicu',49),('V1','mimic',49)]:
            with self.assertRaises(AssertionError): protocol(v,d,s)

    def test_fixed_recipe(self):
        ref=parent_protocol('V1_group_rate','mimic',48)
        for s in SEEDS:
            spec=protocol('V1_group_rate','mimic',s)
            self.assertEqual({k for k in ref.keys()|spec.keys() if ref.get(k)!=spec.get(k)}, {'seed','tuning'})
            self.assertFalse(spec['tuning'])
            self.assertEqual(spec['fusion_lr'],.00030);self.assertEqual(spec['fusion_readout_lr'],.00015)
            for k in ('main_view','checkpoint_selection_view','stopping_view'):self.assertEqual(spec[k],'z0')

    def test_partition_and_gate(self):
        fusion=GateChannels(20,12,'V1',49);before=parameter_hash(fusion)
        groups=split_fusion(fusion)
        self.assertEqual(parameter_hash(fusion),before)
        self.assertEqual({n for n,p in groups['readout']},set(READOUT_NAMES))
        self.assertEqual(sum(p.numel() for n,p in groups['readout']),49858)
        self.assertIn('gate.bias',{n for n,p in groups['encoder']})
        ids=[id(p) for members in groups.values() for n,p in members]
        self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in fusion.parameters()})

    def test_optimizer_math_unchanged(self):
        for name in ('schedule','step','zero_grad','state_dict','load_state_dict'):
            self.assertEqual(inspect.getsource(getattr(Optimizer,name)),inspect.getsource(getattr(ParentOptimizer,name)))
        a=ast.parse(inspect.getsource(ParentOptimizer));b=ast.parse(inspect.getsource(Optimizer))
        def init_body(tree):
            f=next(x for x in tree.body[0].body if isinstance(x,ast.FunctionDef) and x.name=='__init__')
            return [ast.dump(n,include_attributes=False) for n in f.body if not isinstance(n,ast.Assert)]
        self.assertEqual(init_body(a),init_body(b))

    def test_training_math_unchanged(self):
        root=Path(__file__).resolve().parents[1]
        a=(root/'scripts/cem_structure_runtime.py').read_text()
        b=(root/'scripts/cem_v1_group_confirm_runtime.py').read_text()
        anchor='    seed_all(a.seed+10000)\n    rng=np.random.default_rng(a.seed)'
        self.assertEqual(a[a.index(anchor):],b[b.index(anchor):])
        aa=ast.parse(a);bb=ast.parse(b)
        for name in ('update','predict','save','probe'):
            fa=next(n for n in aa.body if isinstance(n,ast.FunctionDef) and n.name==name)
            fb=next(n for n in bb.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(fa),ast.dump(fb))

    def test_controls_and_targets(self):
        exp=expected();self.assertEqual(exp['formal_fits'],4);self.assertEqual(len(exp['controls']),22)
        self.assertEqual(exp['primary_comparators'],[CEM,'B0','tabm'])
        for s in (48,49,50):
            for e in (0,1):
                for v in (CEM,'B0','tabm'):self.assertIn(f'{v}_t{e}_s{s}',exp['controls'])
        self.assertNotIn('U1_t0_s48',exp['controls'])


if __name__=='__main__':unittest.main()
