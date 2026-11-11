"""Narrow seed-only routing and immutable scientific-entry contracts."""
import ast,unittest
from pathlib import Path
from scripts.tracehr_noa_pm_completion_common import MATRIX,protocol,authorized
from scripts.tracehr_noa_ablation_common import protocol as old_protocol
ROOT=Path(__file__).resolve().parents[1]

class Contracts(unittest.TestCase):
    def test_exact_scope(self):
        self.assertEqual(set(MATRIX),{('C5_noT_noA_no'+v,e,s) for v in 'PM' for s in (46,47) for e in (0,1)})
        for v in ('C5_noT_noA_noP','C5_noT_noA_noM'):
            for s in range(42,51):
                self.assertEqual(authorized(v,0,s,'train'),s in (46,47))
                self.assertEqual(authorized(v,1,s,'probe'),s==43)
        for s in (43,46,47):self.assertFalse(authorized('C5_noT_noA_noR',0,s,'train'))
    def test_no_model_change(self):
        old=(ROOT/'models/tracehr_noa_ablation.py').read_text()
        expected=old.replace("VARIANTS = tuple('C5_noT_noA_no'+c for c in 'PMR')", "VARIANTS = tuple('C5_noT_noA_no'+c for c in 'PM')").replace('seed in (43,44,45)','seed in (43,46,47)')
        self.assertEqual(ast.dump(ast.parse(expected)),ast.dump(ast.parse((ROOT/'models/tracehr_noa_pm_completion.py').read_text())))
    def test_same_recipe(self):
        for v,e,s in MATRIX:
            spec=protocol(v,'mimic',s);reference=old_protocol(v,'mimic',43)
            self.assertEqual({**reference,'seed':s},spec)
            self.assertEqual(spec['main_view'],spec['checkpoint_selection_view'])
            self.assertEqual(spec['stopping_view'],'z0')
            self.assertEqual(spec['training_objective'],'0.5*member_CE(z0)')
    def test_immutable_update_runtime(self):
        worker=(ROOT/'scripts/train_tracehr_noa_pm_completion.py').read_text()
        reference=(ROOT/'scripts/train_tracehr_pm_confirmation.py').read_text().replace('scripts.tracehr_pm_confirmation_common import','scripts.tracehr_noa_pm_completion_common import')
        self.assertEqual(ast.dump(ast.parse(worker)),ast.dump(ast.parse(reference)))
        self.assertIn('from scripts import tracehr_noa_ablation_runtime as runner',worker)
        self.assertNotIn('torch.optim.',worker)
        self.assertIn('a.resume=False',worker)
    def test_seed_routing_preserved(self):
        def model_ast(p):
            t=ast.parse((ROOT/p).read_text())
            return ast.dump(next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='model_for'))
        self.assertEqual(model_ast('scripts/tracehr_noa_pm_completion_common.py'),model_ast('scripts/tracehr_noa_ablation_common.py'))

if __name__=='__main__':unittest.main()
