"""Small CPU contract tests; real graph/update/recovery is checked by bounded GPU jobs."""
import ast,unittest
from pathlib import Path
from scripts.tracehr_noa_ablation_common import MATRIX,VARIANTS,protocol,PROTOCOL_FIELDS
from scripts.tracehr_not_stability_common import protocol as parent_protocol
from scripts.tracehr_noa_ablation_queue import Queue
from scripts.audit_tracehr_noa_r_equivalence import exact,NotEquivalent
import torch
ROOT=Path(__file__).resolve().parents[1]

class Contracts(unittest.TestCase):
    def test_authorized_matrix(self):
        self.assertEqual(len(MATRIX),18);self.assertEqual(len(set(MATRIX)),18)
        self.assertEqual({s for _,_,s in MATRIX},{43,44,45})
    def test_no_recipe_change(self):
        for v,e,s in MATRIX:
            a=protocol(v,'mimic',s);b=parent_protocol('C5_noT_noA','mimic',s)
            self.assertLessEqual({k for k in set(a)|set(b) if a.get(k)!=b.get(k)},PROTOCOL_FIELDS)
            self.assertEqual(a['main_view'],'z0');self.assertEqual(a['training_objective'],'0.5*member_CE(z0)')
    def test_scientific_update_unchanged(self):
        old=ast.parse((ROOT/'scripts/tracehr_not_functional_runtime.py').read_text())
        new=ast.parse((ROOT/'scripts/tracehr_noa_ablation_runtime.py').read_text())
        for name in ('update','predict','save','probe'):
            a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b))
        a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        def body_from_seed(fn):
            index=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='seed_all')
            return [ast.dump(n) for n in fn.body[index:]]
        self.assertEqual(body_from_seed(a),body_from_seed(b))
    def test_equivalence_rejects_changed_updates(self):
        exact({'inner':torch.ones(2),'spec':{'label':'a'}},{'inner':torch.ones(2),'spec':{'label':'b'}})
        with self.assertRaises(NotEquivalent):exact({'inner':torch.ones(2)},{'inner':torch.zeros(2)})
    def test_negative_equivalence_falls_back(self):
        q=Queue.__new__(Queue)
        q.state={'jobs':{'bad':{'status':'failed'}},'phase':'r_equivalence'}
        q.phase();self.assertEqual(q.state['status'],'needs_attention')

if __name__=='__main__':unittest.main()
