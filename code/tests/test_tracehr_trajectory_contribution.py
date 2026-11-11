"""C2 topology, retained initialization, equivalent zero-trajectory forward, joint updates."""
import unittest
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import torch
import test_tracehr_structure as fixture
from models.tracehr_context_cut import remove_hidden_context
from models.tracehr_trajectory_contribution import remove_trajectory
from scripts.tracehr_trajectory_contribution_common import protocol
from scripts.tracehr_context_cut_common import protocol as c1_protocol
from scripts.tracehr_structure_runtime import Optimizer, gradients
from models.tracehr_unified_next import parameter_hash


class ContributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.StructureTests.setUpClass();cls.fixture=fixture.StructureTests()

    def pair(self):
        a=remove_hidden_context(self.fixture.model('G2'))
        b=remove_trajectory(remove_hidden_context(self.fixture.model('G2')))
        return a,b

    def test_retained_tensors_recipe_and_no_unused_trajectory_parameters(self):
        a,b=self.pair()
        self.assertEqual(parameter_hash(a.base),parameter_hash(b.base))
        self.assertTrue(b.info['retained_initialization_exact'])
        self.assertFalse(any('trajectory' in n or 'context' in n or 'query' in n or 'key' in n or 'value' in n for n,_ in b.named_parameters()))
        x=c1_protocol('C1','mimic');y=protocol('C2','mimic')
        self.assertEqual({k for k in set(x)|set(y) if x.get(k)!=y.get(k)}, {'variant','structure','description','trajectory'})

    def test_zero_trajectory_reference_and_raw_only_input(self):
        a,b=self.pair();a.eval();b.eval()
        def zeros(batch):
            n=len(batch['raw'])
            return batch['raw'].new_zeros(n,32,8,64),torch.zeros(n,32,8,dtype=torch.bool)
        a.fusion.encode=zeros
        with torch.no_grad():
            expected=a(self.fixture.batch);actual=b({'raw':self.fixture.batch['raw']})
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6,rtol=1e-6))
        hidden=torch.randn(8,3,256,requires_grad=True);summary=torch.randn(3,32,26)
        residual=b.fusion.fuse(hidden,summary,None,None,'C2')-hidden
        self.assertTrue(torch.equal(torch.autograd.grad(residual.sum(),hidden)[0],torch.zeros_like(hidden)))

    def test_all_remaining_groups_jointly_update(self):
        _,b=self.pair();opt=Optimizer(b,protocol('C2','mimic'));opt.schedule(.1)
        before={n:p.detach().clone() for n,p in b.named_parameters()}
        b.train();opt.zero_grad();b.loss(b({'raw':self.fixture.batch['raw']}),torch.tensor(self.fixture.y)).backward()
        self.assertEqual(set(gradients(b)),{'fusion','numerical','input_members','backbone'})
        opt.step()
        for prefix in ('base.net.','base.numerical.','base.input_members','fusion.'):
            self.assertTrue(any(not torch.equal(p,before[n]) for n,p in b.named_parameters() if n.startswith(prefix)),prefix)


if __name__=='__main__':unittest.main()
