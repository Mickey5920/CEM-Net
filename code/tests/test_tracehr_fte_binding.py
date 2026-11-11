import hashlib
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from scripts.tracehr_fte_binding_common import *
from models.tracehr_fte_binding import UntiedNative

class DictLinear(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.randn(7,2))
    def forward(self, batch):return dict(x_cont=batch['x_cont']@self.weight)

class Encoding(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('numeric_indices',torch.tensor([1]))
    def forward(self,x):return torch.cat((x[:,1:2].repeat(1,3)*torch.tensor([.3,.7,1.4]),x),1)

class Contracts(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(sum(len(matrix(db)) for db in ('mimic','eicu')),12)
        for db in ('mimic','eicu'):
            self.assertEqual({s for v,e,s in matrix(db)},{48,49,50})
            with self.assertRaises(AssertionError):protocol(VARIANT,db,51)
            with self.assertRaises(AssertionError):protocol(PM,db,48)

    def test_recipe(self):
        for db in ('mimic','eicu'):
            for v,e,s in matrix(db):
                old=prior_protocol(PM if db=='mimic' else 'C5_noT_noA',db,s);new=protocol(v,db,s)
                self.assertLessEqual({k for k in set(old)|set(new) if old.get(k)!=new.get(k)},PROTOCOL_FIELDS)
                self.assertEqual(new['training_objective'],'0.5*member_CE(z0)')
                self.assertEqual(new['training_microbatch'],128)
                self.assertTrue(all(new[k]=='z0' for k in ('main_view','checkpoint_selection_view','stopping_view')))
                self.assertFalse(new['tuning'])

    def test_immutable_sources_and_training_algorithm(self):
        for n,h in expected()['parent_manifest'].items():
            self.assertEqual(hashlib.sha256((ROOT/n).read_bytes()).hexdigest(),h,n)
        original=(ROOT/'scripts/tracehr_noa_validation_runtime.py').read_text()
        modified=(ROOT/'scripts/tracehr_fte_binding_runtime.py').read_text()
        original=original.replace("edge['retained_random_tensors_match_noA']","edge['retained_random_tensors_except_expanded_factors_match_PM']")
        original=original.replace('same_seed_initialization_checked_against_archived_noA=True','same_seed_initialization_checked_against_archived_PM=True')
        self.assertEqual(original,modified)

    def test_controls(self):
        for db in ('mimic','eicu'):
            controls=expected()['databases'][db]['controls']
            for v,e,s in matrix(db):
                row=controls[f'{PM}_t{e}_s{s}']
                self.assertEqual((row['variant'],row['endpoint'],row['seed']),(PM,e,s))

    def test_initial_function_gradient_and_independent_update(self):
        old=nn.Module();old.numerical=Encoding();old.net=DictLinear();old.static=nn.Identity()
        old.register_buffer('feature_map',torch.tensor([1,1,1,0,1,2,3]))
        old.input_members=nn.Parameter(torch.randn(8,4))
        old.hp=SimpleNamespace();old.info={};old.members=8;old.centered=False
        x=torch.randn(11,4)
        def tied():return old.net(dict(x_cont=old.input_members[:,None].index_select(2,old.feature_map)*old.numerical(x)[None]))['x_cont'].transpose(0,1)
        before=tied();g=torch.autograd.grad(before.square().mean(),old.input_members)[0]
        new=UntiedNative(old);after=new(x)
        self.assertTrue(torch.equal(before,after))
        from models.tracehr_noa_main_confirmation import NoTrajectoryFunctional
        owner=SimpleNamespace(base=new,auxiliary_enabled=False)
        main,aux=NoTrajectoryFunctional.path_losses(owner,(after,None),torch.zeros(len(x),dtype=torch.long))
        self.assertIsNone(aux);self.assertTrue(torch.isfinite(main))
        with self.assertRaises(ValueError):NoTrajectoryFunctional.path_losses(owner,after,torch.zeros(len(x),dtype=torch.long))
        self.assertEqual(new.input_members.numel()-old.input_members.numel(),24)
        gu=torch.autograd.grad(after.square().mean(),new.input_members)[0]
        folded=torch.zeros_like(g).scatter_add_(1,new.feature_map[None].expand(8,-1),gu)
        torch.testing.assert_close(g,folded)
        opt=torch.optim.SGD([new.input_members],lr=.001)
        opt.zero_grad();new(x).square().mean().backward();opt.step()
        self.assertGreater(coordinate_divergence(SimpleNamespace(base=new))['unequal_coordinates'],0)
        self.assertIs(old.net,new.net);self.assertTrue(all(p.requires_grad for p in new.parameters()))

if __name__=='__main__':unittest.main()
