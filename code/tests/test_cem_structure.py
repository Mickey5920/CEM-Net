import unittest
import numpy as np
import torch
from models.cem_evidence_structure import EvidenceChannels,VARIANTS


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.nv=11;self.phi=4*self.nv+12
        g=torch.Generator().manual_seed(207)
        self.raw=torch.randn(3,self.phi+832,generator=g)
        self.batch=dict(raw=self.raw,event_ids=torch.tensor([[0,2,5,11],[1,3,11,11],[11,11,11,11]]),
            event_mask=torch.tensor([[1,1,1,0],[1,1,0,0],[0,0,0,0]],dtype=torch.bool),
            time_counts=torch.randint(0,3,(3,8,self.nv),generator=g).float())

    def test_empty_masks_finite_and_all_parameters_connected(self):
        for v in VARIANTS:
            m=EvidenceChannels(self.nv,12,v,48).eval()
            # Cover empty event and laboratory records without deleting their parameters.
            self.batch['raw'][:,self.phi:self.phi+352:11]=0
            output=m(self.raw,self.batch,self.phi)
            self.assertEqual(tuple(output.shape),(8,3,256));self.assertTrue(torch.isfinite(output).all())
            output.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()),v)

    def test_set_order_and_padding_invariance(self):
        for v in VARIANTS:
            m=EvidenceChannels(self.nv,12,v,48).eval()
            reference=m(self.raw,self.batch,self.phi)
            changed=dict(self.batch)
            order=torch.tensor([2,0,3,1]);changed['event_ids']=changed['event_ids'][:,order];changed['event_mask']=changed['event_mask'][:,order]
            torch.testing.assert_close(reference,m(self.raw,changed,self.phi),rtol=1e-5,atol=1e-6)
            changed['event_ids']=torch.cat((changed['event_ids'],torch.full((3,5),11)),1)
            changed['event_mask']=torch.cat((changed['event_mask'],torch.zeros((3,5),dtype=torch.bool)),1)
            torch.testing.assert_close(reference,m(self.raw,changed,self.phi),rtol=1e-5,atol=1e-6)

    def test_common_initialization_and_fusion_position_control(self):
        first=EvidenceChannels(self.nv,12,'S1',48).state_dict()
        other={v:EvidenceChannels(self.nv,12,v,48).state_dict() for v in VARIANTS[1:]}
        for state in other.values():
            for n,value in first.items():self.assertTrue(torch.equal(value,state[n]),n)
        self.assertEqual(set(other['S2']),set(other['S5']))
        for n,value in other['S2'].items():self.assertTrue(torch.equal(value,other['S5'][n]),n)

    def test_source_channels_change_outputs(self):
        for v,key in [('S3','raw'),('S4','time_counts')]:
            m=EvidenceChannels(self.nv,12,v,48).eval()
            initial=m(self.raw,self.batch,self.phi)
            changed=dict(self.batch)
            if v=='S3':
                raw=self.raw.clone();raw[:,self.phi:]=0;changed['raw']=raw
                value=m(raw,changed,self.phi)
            else:
                changed[key]=self.batch[key].flip(1)
                value=m(self.raw,changed,self.phi)
            self.assertGreater(float((initial-value).abs().max()),1e-6,v)

    def test_fixed_time_boundaries_conserve_counts(self):
        times=np.array([0.,2.999,3.,20.999,21.,23.999,24.]);types=np.array([0,1,0,1,0,1,0])
        cells=np.minimum((times/3).astype(np.int64),7)*2+types
        matrix=np.bincount(cells,minlength=16).reshape(8,2)
        np.testing.assert_array_equal(matrix.sum(0),np.bincount(types,minlength=2))
        self.assertEqual(matrix[7].sum(),3)


if __name__=='__main__':unittest.main()
