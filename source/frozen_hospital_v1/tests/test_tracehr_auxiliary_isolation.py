"""C5 initialization, no auxiliary base dependency, joint updates, and fixed z0."""
import copy
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import test_tracehr_structure as fixture
import torch
from torch.nn import functional as F
from models.tracehr_context_cut import remove_hidden_context
from models.tracehr_direct_supervision import DirectSupervisionNetwork
from models.tracehr_auxiliary_isolation import AuxiliaryIsolationNetwork, retained_parameter_hash
from models.tracehr_unified_next import parameter_hash
from scripts.tracehr_auxiliary_isolation_common import protocol, PROTOCOL_CHANGES
from scripts.tracehr_direct_supervision_common import protocol as c4_protocol
from scripts.tracehr_structure_runtime import Optimizer, gradients
from scripts.tracehr_auxiliary_isolation_diagnostics import path_gradients


class AuxiliaryIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.StructureTests.setUpClass();cls.fixture=fixture.StructureTests()

    def c1(self): return remove_hidden_context(self.fixture.model('G2'))
    def c5(self): return AuxiliaryIsolationNetwork(self.c1())

    def test_retained_initialization_and_fixed_recipe(self):
        a=self.c1();before=parameter_hash(a);count=sum(p.numel() for p in a.parameters())
        rng=torch.get_rng_state().clone();b=AuxiliaryIsolationNetwork(a)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(before,retained_parameter_hash(b))
        self.assertEqual(sum(p.numel() for p in b.parameters())-count,514)
        x=c4_protocol('C4','mimic');y=protocol('C5','mimic')
        self.assertEqual({k for k in set(x)|set(y) if x.get(k)!=y.get(k)},PROTOCOL_CHANGES)
        self.assertEqual(y['main_view'],y['checkpoint_selection_view'])
        self.assertEqual(y['stopping_view'],'z0')

    def test_main_equals_c4_and_deployment_skips_auxiliary(self):
        a=DirectSupervisionNetwork(self.c1()).eval();b=self.c5().eval()
        with torch.no_grad():
            z0,za=b.forward_views(self.fixture.batch)
            torch.testing.assert_close(z0,a(self.fixture.batch),rtol=1e-6,atol=1e-6)
            self.assertEqual(za.shape,(32,1,2))
            b.fusion.encode=lambda _:self.fail('Deployment invoked trajectory')
            b.fusion.auxiliary_head.forward=lambda _:self.fail('Deployment invoked auxiliary head')
            torch.testing.assert_close(z0,b({'raw':self.fixture.batch['raw']}),rtol=1e-6,atol=1e-6)

    def test_one_base_forward_and_matching_dropout(self):
        a=DirectSupervisionNetwork(self.c1()).train();b=self.c5().train()
        calls=[];handle=b.base.register_forward_hook(lambda *args:calls.append(1))
        torch.manual_seed(812);z0,_=b(self.fixture.batch);state=torch.get_rng_state()
        torch.manual_seed(812);ref,_=a(self.fixture.batch)
        self.assertEqual(len(calls),1);handle.remove()
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        torch.testing.assert_close(z0,ref,atol=1e-6,rtol=1e-6)

    def test_auxiliary_has_no_base_gradient_but_shared_modules_learn(self):
        b=self.c5().train();logits=b(self.fixture.batch);y=torch.tensor(self.fixture.y)
        main,aux=b.path_losses(logits,y)
        report=path_gradients(b,(main,aux))
        self.assertTrue(report['auxiliary_base_gradient_absent'])
        for key in ('periodic_frontend','input_members','backbone_and_head'):
            self.assertEqual(report['groups'][key]['aux_connected'],0)
            self.assertGreater(report['groups'][key]['main_norm'],0)
        for key in ('summary_encoder','trajectory_encoder','auxiliary_head'):
            self.assertGreater(report['groups'][key]['auxiliary_norm'],0)
        self.assertTrue(all(p.grad is None for p in b.parameters()))
        b.loss(logits,y).backward();gradients(b)

    def test_exact_loss_and_half_main_base_gradient(self):
        b=self.c5().train();y=torch.tensor(self.fixture.y);z0,za=b(self.fixture.batch)
        main=b.base.loss(z0,y);aux=F.cross_entropy(za[:,0],y.long())
        loss=b.loss((z0,za),y)
        self.assertTrue(torch.equal(loss,.5*(main+aux)))
        params=list(b.base.parameters())
        g0=torch.autograd.grad(main,params,retain_graph=True)
        total=torch.autograd.grad(loss,params)
        for x,y in zip(g0,total):torch.testing.assert_close(y,.5*x,rtol=1e-6,atol=1e-7)

    def test_full_updates_and_optimizer_restore(self):
        b=self.c5();opt=Optimizer(b,protocol('C5','mimic'));opt.schedule(.1)
        before={n:p.detach().clone() for n,p in b.named_parameters()}
        b.train();opt.zero_grad();b.loss(b(self.fixture.batch),torch.tensor(self.fixture.y)).backward()
        gradients(b);opt.step()
        for prefix in ('base.net.','base.numerical.','base.input_members','fusion.trajectory.','fusion.summary.','fusion.auxiliary_head.'):
            self.assertTrue(any(not torch.equal(p,before[n]) for n,p in b.named_parameters() if n.startswith(prefix)),prefix)
        saved=copy.deepcopy(opt.state_dict());values=copy.deepcopy(opt.values())
        opt.schedule(70);opt.load_state_dict(saved);self.assertEqual(values,opt.values())

    def test_diagnostic_gradient_query_does_not_change_training_update(self):
        a=self.c5().train();b=self.c5().train();y=torch.tensor(self.fixture.y)
        torch.manual_seed(19);za=a(self.fixture.batch);a.loss(za,y).backward()
        state=torch.get_rng_state().clone()
        torch.manual_seed(19);zb=b(self.fixture.batch)
        path_gradients(b,b.path_losses(zb,y));b.loss(zb,y).backward()
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(na,nb);torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)

    def test_empty_trajectory_finite(self):
        b=self.c5().eval()
        empty=dict(raw=self.fixture.batch['raw'][:1],records=torch.empty(0,8),
                   segments=torch.empty(0,dtype=torch.long),counts=torch.zeros(1,32,8,dtype=torch.int32))
        with torch.no_grad():z0,za=b.forward_views(empty)
        self.assertTrue(torch.isfinite(z0).all() and torch.isfinite(za).all())


if __name__=='__main__':unittest.main()
