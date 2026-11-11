import unittest
from scripts.tracehr_noa_validation_common import *
from scripts.tracehr_noa_main_confirmation_common import protocol as reference_protocol
class Contracts(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(len(matrix('mimic')),18);self.assertEqual(len(matrix('eicu')),12)
        self.assertEqual({s for _,_,s in matrix('mimic')},set(SEEDS))
        self.assertFalse(any(v.endswith(('noP','noM','noR')) for v,_,_ in matrix('eicu')))
        with self.assertRaises(AssertionError):protocol('C5_noT_noA_noR','eicu',48)
        with self.assertRaises(AssertionError):protocol('C5_noT_noA','eicu',51)
    def test_deletion_recipe(self):
        for v,e,s in matrix('mimic'):
            spec=protocol(v,'mimic',s);ref=reference_protocol('C5_noT_noA','mimic',s)
            diff={k for k in set(spec)|set(ref) if spec.get(k)!=ref.get(k)}
            self.assertLessEqual(diff,PROTOCOL_FIELDS)
            self.assertEqual(spec['main_view'],'z0');self.assertEqual(spec['training_objective'],'0.5*member_CE(z0)')
    def test_eicu_protocol(self):
        for s in SEEDS:
            spec=protocol('C5_noT_noA','eicu',s);ref=reference_protocol('C5_noT_noA','mimic',s)
            self.assertEqual(spec['min_epochs'],256);self.assertEqual(spec['early_stop_boundaries'],[])
            self.assertLessEqual({k for k in spec if spec[k]!=ref.get(k)}, {'database','min_epochs','early_stop_boundaries','significant_gain_patience','early_stop_rule'})
            self.assertFalse(stopping([{'epoch':i+1,'dev_ap':.5} for i in range(120)],spec)['stop'])
            self.assertTrue(stopping([{'epoch':i+1,'dev_ap':.5} for i in range(256)],spec)['stop'])
    def test_native_stop_dispatch(self):
        spec=protocol('B0','eicu',48)
        self.assertFalse(native_stopping([{'epoch':i+1,'dev_ap':.5} for i in range(120)],spec)['stop'])
        self.assertTrue(native_stopping([{'epoch':i+1,'dev_ap':.5} for i in range(256)],spec)['stop'])
    def test_native_commands(self):
        for f in FAMILIES:
            for s in SEEDS:
                cmd=baseline_command(f,s,Path('/tmp/not_executed'),'python')
                self.assertEqual(cmd[cmd.index('--seed')+1],str(s))
                self.assertNotIn('timeout',cmd);self.assertNotIn('--pilot',cmd)
                self.assertIn('/eicu24',cmd[cmd.index('--cache')+1])
if __name__=='__main__':unittest.main()
