import hashlib, unittest
from scripts.tracehr_noa_eicu_ablation_common import *

class Contracts(unittest.TestCase):
    def test_scope_and_priority(self):
        rows=matrix('eicu')
        self.assertEqual(len(rows),18);self.assertEqual(len(set(rows)),18)
        self.assertTrue(all(v.endswith('noR') for v,e,s in rows[:6]))
        self.assertEqual({s for v,e,s in rows},{48,49,50})
        with self.assertRaises(AssertionError):matrix('mimic')
        with self.assertRaises(AssertionError):protocol('C5_noT_noA','eicu',48)
        with self.assertRaises(AssertionError):protocol(VARIANTS[0],'eicu',51)

    def test_recipe_and_stopping(self):
        for v,e,s in matrix('eicu'):
            current=protocol(v,'eicu',s);ref=parent_protocol('C5_noT_noA','eicu',s)
            diff={k for k in set(current)|set(ref) if current.get(k)!=ref.get(k)}
            self.assertLessEqual(diff,PROTOCOL_FIELDS)
            self.assertEqual(current['training_objective'],'0.5*member_CE(z0)')
            self.assertTrue(all(current[k]=='z0' for k in ('main_view','checkpoint_selection_view','stopping_view')))
            self.assertEqual(current['min_epochs'],256)
            self.assertFalse(stopping([{'epoch':i+1,'dev_ap':.5} for i in range(255)],current)['stop'])
            self.assertTrue(stopping([{'epoch':i+1,'dev_ap':.5} for i in range(256)],current)['stop'])

    def test_reuse_scope(self):
        exp=expected()
        self.assertEqual(len(exp['archive_records']),30)
        self.assertEqual(set(BASELINES),{'B0','tabm','xgboost'})
        for v,e,s in matrix('eicu'):
            r=exp['controls'][f'C5_noT_noA_t{e}_s{s}']
            self.assertEqual(r['seed'],s);self.assertEqual(r['endpoint'],e)

    def test_scientific_parent_unchanged(self):
        exp=expected()
        for name,digest in exp['parent_manifest'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)

    def test_no_new_scientific_implementation(self):
        self.assertEqual(model_for.__module__,'scripts.tracehr_noa_validation_common')
        self.assertEqual(Optimizer.__module__,'scripts.tracehr_noa_ablation_common')
        from scripts import tracehr_noa_validation_runtime as runner
        self.assertEqual(runner.train.__module__,'scripts.tracehr_noa_validation_runtime')

if __name__=='__main__':unittest.main()
