"""Decisive scope, science preservation, native-entry and validator checks."""
import ast,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tracehr_noa_main_confirmation_common import MATRIX,FAMILIES,baseline_command,expected,protocol
def source(n):return (ROOT/n).read_text()
def syntax(t):return ast.dump(ast.parse(t),include_attributes=False)
class ConfirmationTests(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(set(MATRIX),{(v,e,s) for v in ('C5_noT_noA','B0') for e in (0,1) for s in (48,49,50)})
        self.assertEqual(len(MATRIX)+len(FAMILIES)*3*2,30)
        self.assertEqual(len(MATRIX)+len(FAMILIES)*3,21)
        for f,s in [('realmlp',48),('tabm',47),('xgboost',51),('noR',48)]:
            with self.assertRaises(AssertionError):baseline_command(f,s,'unused','python')
    def test_native_argv(self):
        for family in FAMILIES:
            old=expected()['templates'][family]['command'];i=old.index('-B')
            normalized=['python',*old[i:]];normalized[normalized.index('--output')+1]='new_output'
            if '--seed' in normalized:normalized[normalized.index('--seed')+1]='48'
            else:normalized+=['--seed','48']
            self.assertEqual(baseline_command(family,48,'new_output','python'),normalized)
    def test_all_old_records_have_bound_prediction_hash(self):
        exp=expected()
        for r in exp['records'].values():
            digest=r.get('predictions_sha256',exp['baseline_records'].get(f"{r['variant']}_s{r['seed']}",{}).get('predictions_sha256'))
            self.assertEqual(len(digest or ''),64)
    def test_model_only_seed_allowlist_extended(self):
        old=source('models/tracehr_not_stability_functional.py').replace('seed in (42,43,44,45,46,47)','seed in (42,43,44,45,46,47,48,49,50)')
        self.assertEqual(syntax(old),syntax(source('models/tracehr_noa_main_confirmation.py')))
    def test_protocol_unchanged_except_seed(self):
        from scripts.tracehr_not_stability_common import protocol as old
        for variant,_,seed in MATRIX:
            ref=old(variant,'mimic',47);ref['seed']=seed
            self.assertEqual(protocol(variant,'mimic',seed),ref)
    def test_neural_workers_unchanged_except_seed_routing(self):
        old=source('scripts/train_tracehr_not_stability_native.py').replace('choices=[45,46,47]','choices=[48,49,50]')
        self.assertEqual(syntax(old),syntax(source('scripts/train_tracehr_noa_main_native.py')))
        old=source('scripts/train_tracehr_not_stability_functional.py').replace('scripts.tracehr_not_stability_common import','scripts.tracehr_noa_main_confirmation_common import').replace('choices=[43,44,45,46,47]','choices=[48,49,50]')
        old=old.replace('assert (a.variant,a.endpoint,a.seed) in MATRIX',"assert not a.resume and a.variant=='C5_noT_noA'\n    assert (a.variant,a.endpoint,a.seed) in MATRIX")
        self.assertEqual(syntax(old),syntax(source('scripts/train_tracehr_noa_main_confirmation.py')))
    def test_seed_model_factory_unchanged(self):
        def fun(t):return next(n for n in ast.parse(t).body if isinstance(n,ast.FunctionDef) and n.name=='model_for')
        old=source('scripts/tracehr_not_stability_common.py').replace('seed in (42,43,44,45,46,47)','seed in (42,43,44,45,46,47,48,49,50)')
        self.assertEqual(ast.dump(fun(old)),ast.dump(fun(source('scripts/tracehr_noa_main_confirmation_common.py'))))
    def test_native_acceptance_archives(self):
        from unittest.mock import patch
        from scripts.tracehr_noa_main_confirmation_queue import Queue
        from scripts.tracehr_recovery_queue import read
        validator=Queue.__new__(Queue);contracts={}
        for family in FAMILIES:
            r=expected()['baseline_records'][family+'_s44'];m=read(Path(r['path'])/'run_manifest.json')
            entry=baseline_command(family,48,'unused','python')[2]
            contracts[family]=dict(manifest=m,entry=entry,entry_sha256=m['source_sha256'],reference=r)
        validator.state={'contracts':contracts}
        for family in FAMILIES:
            r=expected()['baseline_records'][family+'_s44']
            j=dict(kind='baseline',family=family,seed=44,id='archived_'+family,output=r['path'],command=['archived'])
            with patch('scripts.tracehr_noa_main_confirmation_queue.baseline_command',return_value=['archived']):
                self.assertTrue(validator.validate(j)['original_stopping_verified'])
if __name__=='__main__':unittest.main()
