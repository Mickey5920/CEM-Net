"""Changed seed-routing contracts; actual identity/update/replay checked on GPU."""
import ast,unittest
from pathlib import Path
from scripts.tracehr_pm_confirmation_common import MATRIX,protocol,authorized
from scripts.tracehr_noa_ablation_common import protocol as old_protocol
from scripts.tracehr_not_stability_common import protocol as noA_protocol
ROOT=Path(__file__).resolve().parents[1]
class Contracts(unittest.TestCase):
    def test_exact_scope(self):
        self.assertEqual(set(MATRIX),{('C5_noT_noA_noR',e,s) for s in (46,47) for e in (0,1)})
        for s in range(42,51):
            self.assertEqual(authorized('C5_noT_noA_noR',0,s,'train'),s in (46,47))
            self.assertEqual(authorized('C5_noT_noA_noR',1,s,'probe'),s==43)
        self.assertFalse(authorized('C5_noT_noA_noP',0,46,'train'))
    def test_no_model_change(self):
        old=(ROOT/'models/tracehr_noa_ablation.py').read_text()
        expected=old.replace("VARIANTS = tuple('C5_noT_noA_no'+c for c in 'PMR')", "VARIANTS = ('C5_noT_noA_noR',)").replace('seed in (43,44,45)','seed in (43,46,47)')
        self.assertEqual(ast.dump(ast.parse(expected)),ast.dump(ast.parse((ROOT/'models/tracehr_pm_confirmation.py').read_text())))
    def test_same_recipe(self):
        reference=old_protocol('C5_noT_noA_noR','mimic',43)
        for v,e,s in MATRIX:
            spec=protocol(v,'mimic',s)
            self.assertEqual({**reference,'seed':s},spec)
            self.assertEqual(spec['main_view'],spec['checkpoint_selection_view'])
            self.assertEqual(spec['stopping_view'],'z0')
            self.assertEqual(spec['training_objective'],'0.5*member_CE(z0)')
    def test_immutable_update_runtime(self):
        worker=(ROOT/'scripts/train_tracehr_pm_confirmation.py').read_text()
        self.assertIn('from scripts import tracehr_noa_ablation_runtime as runner',worker)
        self.assertNotIn('torch.optim.',worker)
        self.assertIn('a.resume=False',worker)
    def test_seed_routing_preserved(self):
        def model_ast(p):
            t=ast.parse((ROOT/p).read_text())
            return ast.dump(next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='model_for'))
        self.assertEqual(model_ast('scripts/tracehr_pm_confirmation_common.py'),model_ast('scripts/tracehr_noa_ablation_common.py'))
if __name__=='__main__':unittest.main()
