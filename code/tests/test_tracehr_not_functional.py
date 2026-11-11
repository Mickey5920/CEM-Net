"""Functional intervention, retained initialization, loss scale and joint updates."""
import copy
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import test_tracehr_structure as fixture
import torch
from models.tracehr_context_cut import remove_hidden_context
from models.tracehr_not_confirmation import NoTrajectoryConfirmation
from models.tracehr_not_functional import VARIANTS,NoTrajectoryFunctional
from models.tracehr_unified_next import parameter_hash
from scripts.tracehr_not_confirmation_common import protocol as old_protocol
from scripts.tracehr_not_functional_common import protocol,MATRIX,PROTOCOL_FIELDS
from scripts.tracehr_not_functional_diagnostics import path_gradients
from scripts.tracehr_structure_runtime import Optimizer,gradients


class FunctionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.StructureTests.setUpClass();cls.f=fixture.StructureTests()

    def c1(self):return remove_hidden_context(self.f.model('G2'))
    def full(self):return NoTrajectoryConfirmation(self.c1(),'C5_noT')
    def model(self,v):return NoTrajectoryFunctional(self.c1(),v)

    def test_four_only_and_unchanged_recipe(self):
        self.assertEqual(set(MATRIX),{(v,e,42) for v in VARIANTS for e in (0,1)})
        self.assertEqual(len(MATRIX),4)
        for v in VARIANTS:
            a=old_protocol('C5_noT','mimic',42);b=protocol(v)
            changed={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
            self.assertTrue(changed <= PROTOCOL_FIELDS)
            self.assertTrue(all(b[k]=='z0' for k in ('main_view','checkpoint_selection_view','stopping_view')))
            self.assertEqual(b['lr'],.04);self.assertEqual(b['fusion_lr'],.0003)
        with self.assertRaises(AssertionError):protocol(VARIANTS[0],seed=43)

    def test_retained_parameters_buffers_and_random_state(self):
        full=self.full();initial=parameter_hash(full);params=dict(full.named_parameters());buffers=dict(full.named_buffers())
        for v in VARIANTS:
            fresh=self.c1();rng=torch.get_rng_state().clone();m=NoTrajectoryFunctional(fresh,v)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(m.info['reference_noT_initial_weights_sha256'],initial)
            for name,p in m.named_parameters():torch.testing.assert_close(p,params[name],rtol=0,atol=0)
            for name,b in m.named_buffers():torch.testing.assert_close(b,buffers[name],rtol=0,atol=0)
            self.assertEqual(m.info['removed_from_noT'],514 if v.endswith('noA') else 0)
            self.assertTrue(all(p.requires_grad for p in m.parameters()))
            self.assertFalse(m.info['learned_checkpoint_loaded'])

    def test_exact_forward_interventions_and_one_backbone(self):
        full=self.full().eval();a=self.model(VARIANTS[0]).eval();r=self.model(VARIANTS[1]).eval()
        batch={'raw':self.f.batch['raw']}
        with torch.no_grad():
            z,aux=full.forward_views(batch)
            za,absent=a.forward_views(batch);zr,auxr=r.forward_views(batch)
            full.fusion_enabled=False;base=full(batch)
        torch.testing.assert_close(z,za,rtol=0,atol=0);self.assertIsNone(absent)
        torch.testing.assert_close(base,zr,rtol=0,atol=0)
        torch.testing.assert_close(aux,auxr,rtol=0,atol=0)
        for m in (a,r):
            m.train();calls=[];h=m.base.register_forward_hook(lambda *args:calls.append(1))
            m(batch);h.remove();self.assertEqual(len(calls),1)
            m.eval();reference,_=m.forward_views(batch)
            if m.auxiliary_enabled:m.fusion.auxiliary_head.forward=lambda _:self.fail('Inference used auxiliary')
            torch.testing.assert_close(m(batch),reference,rtol=0,atol=0)

    def test_loss_scale_path_gradients_and_optimizer_coverage(self):
        y=torch.tensor(self.f.y);batch={'raw':self.f.batch['raw']}
        for v in VARIANTS:
            m=self.model(v).train();opt=Optimizer(m,protocol(v));opt.schedule(.1)
            initial={n:p.detach().clone() for n,p in m.named_parameters()}
            logits=m(batch);main,aux=m.path_losses(logits,y)
            loss=m.loss(logits,y)
            expected=.5*main if aux is None else .5*(main+aux)
            torch.testing.assert_close(loss,expected,rtol=0,atol=0)
            report=path_gradients(m,(main,aux))
            self.assertTrue(report['auxiliary_base_gradient_absent'])
            if v.endswith('noR'):
                for key in ('summary_encoder','summary_refinement','fusion_readout','shared_variable_embedding','auxiliary_head'):
                    self.assertEqual(report['groups'][key]['main_connected'],0)
                    self.assertGreater(report['groups'][key]['auxiliary_norm'],0)
            else:
                self.assertNotIn('auxiliary_head',report['groups'])
                self.assertTrue(all(row['aux_connected']==0 for row in report['groups'].values()))
            loss.backward();gradients(m);opt.step()
            for prefix in ('base.net.','base.numerical.','base.input_members','fusion.summary.','fusion.trajectory.variable.'):
                self.assertTrue(any(not torch.equal(p,initial[n]) for n,p in m.named_parameters() if n.startswith(prefix)),(v,prefix))
            if m.auxiliary_enabled:self.assertTrue(any(not torch.equal(p,initial[n]) for n,p in m.named_parameters() if n.startswith('fusion.auxiliary_head.')))
            state=copy.deepcopy(opt.state_dict());values=copy.deepcopy(opt.values())
            opt.schedule(70);opt.load_state_dict(state);self.assertEqual(opt.values(),values)

    def test_noA_gradients_match_main_only_full_model(self):
        full=self.full().train();a=self.model(VARIANTS[0]).train();y=torch.tensor(self.f.y)
        torch.manual_seed(10042);main,_=full.path_losses(full(self.f.batch),y);(.5*main).backward()
        torch.manual_seed(10042);a.loss(a(self.f.batch),y).backward()
        original=dict(full.named_parameters())
        for name,p in a.named_parameters():torch.testing.assert_close(p.grad,original[name].grad,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
