import unittest
import torch
from pathlib import Path
from scripts.tracehr_r_member_common import MATRIX,protocol,PROTOCOL_FIELDS
from scripts.tracehr_not_stability_common import protocol as old_protocol
from models.tracehr_r_member import member_summary

class MemberMapping(unittest.TestCase):
    def test_authorized_matrix(self):
        self.assertEqual(len(MATRIX),6);self.assertEqual(len(set(MATRIX)),6)
        self.assertEqual({s for _,_,s in MATRIX},{43,44,45})
        self.assertEqual({e for _,e,_ in MATRIX},{0,1})
    def test_unchanged_recipe(self):
        for v,e,s in MATRIX:
            a=protocol(v,'mimic',s);b=old_protocol('C5_noT_noA','mimic',s)
            self.assertLessEqual({k for k in set(a)|set(b) if a.get(k)!=b.get(k)},PROTOCOL_FIELDS)
            self.assertEqual(a['main_view'],a['checkpoint_selection_view']);self.assertEqual(a['main_view'],'z0')
            self.assertEqual(a['training_objective'],'0.5*member_CE(z0)')
    def test_original_runtime_import(self):
        text=(Path(__file__).resolve().parents[1]/'scripts/train_tracehr_r_member.py').read_text()
        self.assertIn('from scripts import tracehr_noa_ablation_runtime as runner',text)
        self.assertNotIn('torch.optim.',text)
    def test_coordinate_mapping(self):
        phi=7;m=torch.arange(8*(phi+832),dtype=torch.float64).reshape(8,-1)
        s=torch.ones(2,32,26,dtype=torch.float64);z=member_summary(s,m,phi)
        for j in range(8):
            for v in range(32):
                self.assertTrue(torch.equal(z[j,0,v,:11],m[j,phi+11*v:phi+11*(v+1)]))
                self.assertTrue(torch.equal(z[j,1,v,11:],m[j,phi+352+15*v:phi+352+15*(v+1)]))
    def test_identity_and_gradient_locality(self):
        phi=9;m=torch.ones(8,phi+832,requires_grad=True);s=torch.arange(2*32*26,dtype=torch.float32).reshape(2,32,26)
        z=member_summary(s,m,phi)
        self.assertTrue(torch.equal(z,s[None].expand(8,-1,-1,-1)))
        z[3].sum().backward()
        self.assertEqual(int(torch.count_nonzero(m.grad[:,:phi])),0)
        self.assertEqual(int(torch.count_nonzero(m.grad[[0,1,2,4,5,6,7]])),0)
        self.assertGreater(float(m.grad[3,phi:].abs().sum()),0)

if __name__=='__main__':unittest.main()
