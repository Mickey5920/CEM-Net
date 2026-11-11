"""Focused evidence for a single-edge removal, without changing the G2 recipe."""
import copy
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import test_tracehr_structure as original_tests
from models.tracehr_context_cut import remove_hidden_context
from models.tracehr_structure import StructureFusion
from scripts.tracehr_context_cut_common import protocol
from scripts.tracehr_structure_common import protocol as original_protocol
from scripts.tracehr_structure_runtime import Optimizer, gradients
from models.tracehr_unified_next import parameter_hash
import torch
import unittest


class ContextCutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original_tests.StructureTests.setUpClass()
        cls.fixture=original_tests.StructureTests()

    def pair(self):
        ref=self.fixture.model('G2')
        old={k:v.clone() for k,v in ref.state_dict().items()}
        net=remove_hidden_context(self.fixture.model('G2'))
        return ref,net,old

    def test_exact_retained_initialization_and_fixed_recipe(self):
        ref,net,old=self.pair()
        self.assertEqual(set(old)-set(net.state_dict()),{'fusion.context.weight','fusion.context.bias'})
        for name,p in net.state_dict().items():self.assertTrue(torch.equal(p,old[name]),name)
        self.assertEqual(parameter_hash(ref.base),parameter_hash(net.base))
        a=original_protocol('G2','mimic');b=protocol('C1','mimic')
        for key in a:
            if key not in ('variant','structure','description'):self.assertEqual(a[key],b[key],key)
        self.assertFalse(hasattr(net.fusion,'context'))
        self.assertEqual(sum(p.numel() for p in ref.parameters())-sum(p.numel() for p in net.parameters()),16448)

    def test_deleted_edge_and_reference_forward_equivalence(self):
        ref,net,_=self.pair();ref.eval();net.eval()
        ref.fusion.context.forward=lambda x:x.new_zeros((x.shape[0],64))
        with torch.no_grad():
            self.assertTrue(torch.equal(ref(self.fixture.batch),net(self.fixture.batch)))
        hidden=torch.randn(8,3,256,requires_grad=True)
        summary=torch.randn(3,32,26);temporal=torch.randn(3,32,8,64);mask=torch.ones(3,32,8,dtype=torch.bool)
        residual=net.fusion.fuse(hidden,summary,temporal,mask,'C1')-hidden
        jacobian=torch.autograd.grad(residual.sum(),hidden)[0]
        self.assertTrue(torch.equal(jacobian,torch.zeros_like(jacobian)))

    def test_all_retained_groups_train_jointly(self):
        _,net,_=self.pair();opt=Optimizer(net,protocol('C1','mimic'));opt.schedule(.1)
        before={n:p.detach().clone() for n,p in net.named_parameters()}
        net.train();opt.zero_grad();net.loss(net(self.fixture.batch),torch.tensor(self.fixture.y)).backward()
        report=gradients(net)
        self.assertEqual(set(report),{'trajectory','fusion','numerical','input_members','backbone'})
        opt.step()
        for prefix in ('base.net.','base.numerical.','base.input_members','fusion.trajectory.','fusion.'):
            self.assertTrue(any(not torch.equal(p,before[n]) for n,p in net.named_parameters() if n.startswith(prefix)),prefix)
        self.assertTrue(all(p.requires_grad for p in net.parameters()))


if __name__=='__main__':unittest.main()
