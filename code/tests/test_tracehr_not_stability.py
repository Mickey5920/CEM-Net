"""New-seed parity, exact job scope and regression of native stopping validation."""
import copy, json, os, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import test_tracehr_not_confirmation as fixture
import torch
from models.tracehr_not_functional import NoTrajectoryFunctional as OldFunctional
from models.tracehr_not_stability_functional import NoTrajectoryFunctional
from models.tracehr_not_confirmation import NoTrajectoryConfirmation
from models.tracehr_unified_next import parameter_hash
from scripts.tracehr_not_stability_common import protocol,MATRIX,VARIANTS,stopping_for
from scripts.tracehr_not_confirmation_common import protocol as old_confirm
from scripts.tracehr_not_functional_common import protocol as old_functional
from scripts.tracehr_execution_refine_common import protocol as native_protocol, stopping as native_stopping
from scripts.tracehr_structure_runtime import Optimizer,gradients
from scripts.tracehr_not_functional_diagnostics import path_gradients
from scripts.tracehr_not_stability_queue import Queue
from scripts.tracehr_recovery_queue import read

class StabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.ConfirmationTests.setUpClass();cls.f=fixture.ConfirmationTests()

    def test_exact_28_and_only_seed_changes(self):
        wanted={(v,e,s) for v in ('C5_noT','B0') for e in (0,1) for s in (45,46,47)}
        wanted|={(v,e,s) for v in VARIANTS[1:] for e in (0,1) for s in (43,44)}
        self.assertEqual(len(MATRIX),28);self.assertEqual(set(MATRIX),wanted)
        for v,e,s in MATRIX:
            p=protocol(v,'mimic',s)
            previous=native_protocol('F0') if v=='B0' else (
                old_functional(v) if v.endswith(('noA','noR')) else old_confirm(v))
            previous['seed']=s;self.assertEqual(p,previous)
        with self.assertRaises(AssertionError):protocol('C5_noT','eicu',45)
        with self.assertRaises(AssertionError):protocol('C5_noT','mimic',48)

    def test_seed42_functional_parity_through_updates(self):
        for v in ('C5_noT_noA','C5_noT_noR'):
            a=OldFunctional(self.f.c1(42),v,42);b=NoTrajectoryFunctional(self.f.c1(42),v,42)
            self.assertEqual(parameter_hash(a),parameter_hash(b))
            oa,ob=Optimizer(a,protocol(v)),Optimizer(b,protocol(v))
            for step in range(3):
                for model,opt in ((a,oa),(b,ob)):
                    torch.manual_seed(10042+step);model.train();opt.zero_grad();opt.schedule((step+1)/10)
                    model.loss(model(self.f.f.batch),torch.tensor(self.f.f.y)).backward();opt.step()
                self.assertEqual(parameter_hash(a),parameter_hash(b))

    def test_new_functional_seeds_retain_initialization_and_gradients(self):
        for s in (43,44):
            ref=NoTrajectoryConfirmation(self.f.c1(s),'C5_noT',s);params=dict(ref.named_parameters())
            for v in ('C5_noT_noA','C5_noT_noR'):
                m=NoTrajectoryFunctional(self.f.c1(s),v,s)
                self.assertEqual(m.info['reference_noT_initial_weights_sha256'],parameter_hash(ref))
                for n,p in m.named_parameters():torch.testing.assert_close(p,params[n],rtol=0,atol=0)
                opt=Optimizer(m,protocol(v,'mimic',s));opt.schedule(.1)
                before={n:p.detach().clone() for n,p in m.named_parameters()}
                logits=m(self.f.f.batch);losses=m.path_losses(logits,torch.tensor(self.f.f.y))
                g=path_gradients(m,losses);self.assertTrue(g['auxiliary_base_gradient_absent'])
                if v.endswith('noA'):
                    self.assertIsNone(losses[1]);self.assertNotIn('auxiliary_head',g['groups'])
                else:
                    for name,row in g['groups'].items():
                        if name.startswith(('summary','fusion','shared','auxiliary')):
                            self.assertEqual(row['main_connected'],0)
                            self.assertGreater(row['auxiliary_norm'],0)
                m.loss(logits,torch.tensor(self.f.f.y)).backward();gradients(m);opt.step()
                for prefix in ('base.net.','base.numerical.','base.input_members','fusion.summary.'):
                    self.assertTrue(any(not torch.equal(p,before[n]) for n,p in m.named_parameters() if n.startswith(prefix)))
                self.assertTrue(all(p.requires_grad for p in m.parameters()))

    def test_new_full_seed_routes_distinct_and_repeatable(self):
        hashes=[]
        for s in (42,43,44,45,46,47):
            a=self.f.model('C5_noT',s);b=self.f.model('C5_noT',s)
            self.assertEqual(parameter_hash(a),parameter_hash(b));hashes.append(parameter_hash(a))
        self.assertEqual(len(set(hashes)),6)

    def test_native_stopping_and_completed_receipt_validation(self):
        p=Path(os.environ['TRACEHR_PROJECT'])
        summary=read(p/'runs/tracehr_not_confirmation_20261017/results_v1/aggregates.json')
        prior=read(p/'runs/tracehr_not_confirmation_20261017/mimic/confirm_validation_v2/screen/state.json')
        for s in (43,44):
            for e in (0,1):
                key=f'B0_t{e}_s{s}';path=Path(summary['records'][key]['source'])
                c=read(path/'completion.json');history=read(path/'history.json');spec=protocol('B0','mimic',s)
                for n in range(1,len(history)+1):
                    self.assertEqual(stopping_for('B0',history[:n],spec),native_stopping(history[:n],spec))
                q=Queue.__new__(Queue)
                pilot=read(Path(prior['jobs']['probe_'+key]['output'])/'completion.json')
                q.state=dict(source_manifest_sha256=c['source_manifest_sha256'],jobs={'probe_'+key:{'result':pilot}})
                job=dict(id=key,variant='B0',endpoint=e,seed=s,pilot=False,output=str(path))
                self.assertTrue(q.validate(job)['complete'])
                for field,value in (('seed',47),('endpoint',1-e),('variant','C5_noT'),('pilot',True)):
                    bad=dict(job);bad[field]=value
                    with self.assertRaises(AssertionError):q.validate(bad)

if __name__=='__main__':unittest.main()
