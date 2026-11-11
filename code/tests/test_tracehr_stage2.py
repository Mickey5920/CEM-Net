"""Scope, unchanged native argv, immutable model entry and acceptance regression."""
import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tracehr_stage2_common import NOA_MATRIX,FAMILIES,baseline_command,expected

class Stage2Tests(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(set(NOA_MATRIX),{('C5_noT_noA',e,s) for e in (0,1) for s in (45,46,47)})
        self.assertEqual(len(NOA_MATRIX)+len(FAMILIES)*3*2,24)
        for family,seed in [('realmlp',45),('tabm',44),('xgboost',48),('noR',45)]:
            with self.assertRaises(AssertionError):baseline_command(family,seed,'unused','python')

    def test_native_arguments_only_seed_output_and_timeout_envelope_change(self):
        for family in FAMILIES:
            old=expected()['templates'][family]['command'];i=old.index('-B');normalized=['python',*old[i:]]
            normalized[normalized.index('--output')+1]='new_output'
            if '--seed' in normalized:normalized[normalized.index('--seed')+1]='45'
            else:normalized+=['--seed','45']
            self.assertEqual(baseline_command(family,45,'new_output','python'),normalized)
            self.assertNotIn('--pilot',normalized)

    def test_noA_entry_and_protocol_unchanged(self):
        from scripts import train_tracehr_not_stability_functional as original
        oldmain=original.main
        from scripts import train_tracehr_stage2_noa as wrapper
        self.assertIs(wrapper.entry.main,oldmain)
        self.assertEqual(wrapper.entry.MATRIX,NOA_MATRIX)
        from scripts.tracehr_not_stability_common import protocol
        for _,_,seed in NOA_MATRIX:
            old=protocol('C5_noT_noA','mimic',44);old['seed']=seed
            self.assertEqual(protocol('C5_noT_noA','mimic',seed),old)

    def test_native_acceptance_against_three_completed_archives(self):
        from unittest.mock import patch
        from scripts.tracehr_stage2_queue import Queue
        from scripts.tracehr_recovery_queue import read
        validator=Queue.__new__(Queue);contracts={}
        for family in FAMILIES:
            r=expected()['baseline_records'][family+'_s44'];path=Path(r['path'])
            cmd=baseline_command(family,45,'unused','python');entry=cmd[2]
            m=read(path/'run_manifest.json')
            contracts[family]=dict(manifest=m,entry=entry,entry_sha256=m['source_sha256'],reference=r)
        validator.state={'contracts':contracts}
        for family in FAMILIES:
            r=expected()['baseline_records'][family+'_s44']
            job=dict(kind='baseline',family=family,seed=44,id='archived_'+family,output=r['path'],command=['archived'])
            with patch('scripts.tracehr_stage2_queue.baseline_command',return_value=['archived']):
                self.assertTrue(validator.validate(job)['original_stopping_verified'])

if __name__=='__main__':unittest.main()
