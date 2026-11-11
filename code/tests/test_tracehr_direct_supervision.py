"""Native graph, initialization, fixed selection view and full gradient checks."""
import copy
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import test_tracehr_structure as fixture
import torch
from models.tracehr_context_cut import remove_hidden_context
from models.tracehr_trajectory_contribution import remove_trajectory
from models.tracehr_direct_supervision import DirectSupervisionNetwork
from models.tracehr_unified_next import parameter_hash
from scripts.tracehr_direct_supervision_common import protocol, PROTOCOL_CHANGES
from scripts.tracehr_context_cut_common import protocol as c1_protocol
from scripts.tracehr_structure_runtime import Optimizer, gradients


class DirectSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.StructureTests.setUpClass();cls.fixture=fixture.StructureTests()

    def c1(self):return remove_hidden_context(self.fixture.model('G2'))

    def test_random_initialization_parameters_and_recipe(self):
        a=self.c1();before=parameter_hash(a)
        count=sum(p.numel() for p in a.parameters())
        b=DirectSupervisionNetwork(a)
        self.assertEqual(before,parameter_hash(b))
        self.assertEqual(count,sum(p.numel() for p in b.parameters()))
        self.assertTrue(all(p.requires_grad for p in b.parameters()))
        x=c1_protocol('C1','mimic');y=protocol('C4','mimic')
        self.assertEqual({k for k in set(x)|set(y) if x.get(k)!=y.get(k)},PROTOCOL_CHANGES)
        self.assertEqual(y['checkpoint_selection_view'],'z0');self.assertEqual(y['stopping_view'],'z0')

    def test_both_views_equal_references_and_inference_skips_trajectory(self):
        a=self.c1();b=DirectSupervisionNetwork(self.c1());c=remove_trajectory(self.c1())
        a.eval();b.eval();c.eval()
        with torch.no_grad():
            z0,z1=b.forward_views(self.fixture.batch)
            torch.testing.assert_close(z1,a(self.fixture.batch),atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(z0,c({'raw':self.fixture.batch['raw']}),atol=1e-6,rtol=1e-6)
            b.fusion.encode=lambda _:self.fail('deployment invoked trajectory encoder')
            torch.testing.assert_close(z0,b({'raw':self.fixture.batch['raw']}),atol=1e-6,rtol=1e-6)
            self.assertEqual(z0.shape,(32,8,2))

    def test_one_backbone_forward_same_dropout_and_exact_average_loss(self):
        a=self.c1();b=DirectSupervisionNetwork(self.c1());a.train();b.train()
        calls=[]
        handle=b.base.register_forward_hook(lambda *args:calls.append(1))
        torch.manual_seed(812);z0,z1=b(self.fixture.batch);rng_after=torch.get_rng_state()
        torch.manual_seed(812);expected=a(self.fixture.batch)
        self.assertEqual(len(calls),1);handle.remove()
        self.assertTrue(torch.equal(rng_after,torch.get_rng_state()))
        torch.testing.assert_close(z1,expected,atol=1e-6,rtol=1e-6)
        y=torch.tensor(self.fixture.y)
        self.assertTrue(torch.equal(b.loss((z0,z1),y),.5*(a.base.loss(z0,y)+a.base.loss(z1,y))))

    def test_joint_gradients_updates_and_optimizer_restore(self):
        b=DirectSupervisionNetwork(self.c1());opt=Optimizer(b,protocol('C4','mimic'));opt.schedule(.1)
        before={n:p.detach().clone() for n,p in b.named_parameters()}
        b.train();opt.zero_grad();b.loss(b(self.fixture.batch),torch.tensor(self.fixture.y)).backward()
        self.assertEqual(set(gradients(b)),{'trajectory','fusion','numerical','input_members','backbone'})
        opt.step()
        for prefix in ('base.net.','base.numerical.','base.input_members','fusion.trajectory.','fusion.'):
            self.assertTrue(any(not torch.equal(p,before[n]) for n,p in b.named_parameters() if n.startswith(prefix)),prefix)
        saved=copy.deepcopy(opt.state_dict());values=copy.deepcopy(opt.values())
        opt.schedule(70);opt.load_state_dict(saved);self.assertEqual(values,opt.values())

    def test_empty_trajectory_views_coincide(self):
        b=DirectSupervisionNetwork(self.c1());b.eval()
        empty=dict(raw=self.fixture.batch['raw'][:1],records=torch.empty(0,8),
                   segments=torch.empty(0,dtype=torch.long),counts=torch.zeros(1,32,8,dtype=torch.int32))
        with torch.no_grad():z0,z1=b.forward_views(empty)
        self.assertTrue(torch.equal(z0,z1))

    def test_evaluation_keeps_main_prediction_when_auxiliary_is_better(self):
        import numpy as np
        from sklearn.metrics import average_precision_score
        from scripts.tracehr_direct_supervision_runtime import predict
        class FakeData:
            targets={'dev':{'y':np.array([[0],[1],[0],[1]])}}
            def batch(self, split, ix):return torch.tensor(ix),None
        class FakeModel:
            def eval(self):return self
            def forward_views(self, ix):
                v=torch.tensor([3.,-3.,2.,-2.])[ix]
                z=torch.stack((torch.zeros_like(v),v),-1)[:,None].repeat(1,8,1)
                return z,-z
        model=FakeModel();data=FakeData();pred=predict(model,data,2)
        expected=torch.tensor([3.,-3.,2.,-2.]).sigmoid().numpy()
        np.testing.assert_allclose(pred[:,0],expected,rtol=1e-6)
        self.assertGreater(average_precision_score(data.targets['dev']['y'][:,0],model._diagnostic_aux_probs[:,0]),
                           average_precision_score(data.targets['dev']['y'][:,0],pred[:,0]))


if __name__=='__main__':unittest.main()
