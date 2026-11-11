import copy
import os
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
deps=Path(os.environ['TRACEHR_TABULAR_DEPS']);sys.path.insert(0,str(deps))
for wheel in deps.glob('*.whl'):sys.path.insert(0,str(wheel))
import numpy as np
import torch
from models.tracehr_backbone_refine import TypedNative
from models.tracehr_centered_native import CenteredNative, FullNative
from models.tracehr_unified_next import parameter_hash
from scripts.tracehr_centered_native_common import protocol, stopping, native_summary_statistics, VARIANTS, old_baseline
from scripts.tracehr_backbone_refine_runtime import Optimizer
from scripts.tracehr_multiview_common import read


class CenteredTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        rng=np.random.RandomState(7)
        cls.raw=rng.normal(size=(64,12)).astype(np.float32)
        cls.raw[:,10:]=(cls.raw[:,10:]>0).astype(np.float32)
        cls.y=np.tile([0,1],32).astype(np.int64)
        cls.prep=dict(center=np.zeros(12,np.float32),factor=np.ones(12,np.float32),
                      binary=np.arange(12)>=10,numeric_indices=np.arange(10))
        cls.config=read(old_baseline(Path(os.environ['TRACEHR_PROJECT']),'mimic','B0')/'effective_config_task0.json')

    def test_centering_preserves_initial_weights_and_reference_mean(self):
        old=TypedNative(self.raw,self.y,self.prep,self.config,8,42,64)
        new=CenteredNative(self.raw,self.y,self.prep,self.config,42,True,64)
        ordinary=CenteredNative(self.raw,self.y,self.prep,self.config,42,False,64)
        self.assertEqual(parameter_hash(old),parameter_hash(new))
        self.assertEqual(parameter_hash(old),parameter_hash(ordinary))
        old.eval();new.eval();ordinary.eval();x=torch.tensor(self.raw)
        with torch.no_grad():
            self.assertTrue(torch.equal(old(x),ordinary(x)))
            z=new.numerical(x)[None];r=new.input_members[:,None,new.feature_map]
            transformed=new.encoded_center+r*(z-new.encoded_center)
            self.assertLess(float((transformed.mean(1)-z.mean(1)).abs().max()),1e-6)
            self.assertGreater(float(transformed.std(0).mean()),.01)
            self.assertGreater(float((old(x)-new(x)).abs().max()),1e-5)

    def check_updates(self,model,variant):
        opt=Optimizer(model,protocol(variant));opt.schedule(.1)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        for _ in range(2):
            model.train();opt.zero_grad()
            model.loss(model(torch.tensor(self.raw)),torch.tensor(self.y)).backward()
            for name,p in model.named_parameters():
                self.assertTrue(p.requires_grad and p.grad is not None,name)
                self.assertTrue(torch.isfinite(p.grad).all(),name)
            opt.step()
        for name,p in model.named_parameters():self.assertFalse(torch.equal(p,before[name]),name)
        state=copy.deepcopy(opt.state_dict());before=copy.deepcopy(opt.values())
        opt.schedule(71);opt.load_state_dict(state)
        self.assertEqual(opt.values(),before)

    def test_all_centered_learnable_parameters_update_and_clock_restores(self):
        self.check_updates(CenteredNative(self.raw,self.y,self.prep,self.config,42,True,64),'E1')

    def test_scaling_matches_official_factories_and_preserves_other_columns(self):
        sys.modules['lightning']=None
        from pytabkit.models.nn_models.pipeline import MedianCenterFactory, RobustScaleFactory
        from pytabkit.models.data.data import DictDataset, TensorInfo
        raw=self.raw.copy();raw[:,9]=0;raw[0,9]=.0001
        center,factor,indices=native_summary_statistics(raw,self.prep,6)
        self.assertTrue(np.array_equal(indices,np.arange(6,10)))
        self.assertTrue(np.array_equal(center[:6],self.prep['center'][:6]))
        self.assertTrue(np.array_equal(factor[10:],self.prep['factor'][10:]))
        x=torch.tensor(raw[:,indices])
        ds=DictDataset(dict(x_cont=x),dict(x_cont=TensorInfo(feat_shape=[len(indices)])))
        median=MedianCenterFactory()._fit(ds)
        centered=median(ds.tensors)
        scaled=RobustScaleFactory()._fit(DictDataset(centered,ds.tensor_infos))(centered)['x_cont']
        ours=(x-torch.tensor(center[indices]))*torch.tensor(factor[indices])
        self.assertTrue(torch.equal(ours,scaled))

    def test_full_native_capture_no_training_and_joint_updates(self):
        raw=dict(train=self.raw,dev=self.raw[:16]);labels=dict(train=self.y[:,None],dev=self.y[:16,None])
        tmp=Path(os.environ['TMPDIR'])/'native_unit_fixture'
        model=FullNative(raw,labels,0,tmp,42,device='cpu')
        self.assertEqual(model.info['original_optimizer_updates'],0)
        self.assertEqual(model.info['initial_function_wrapper_max_logit_drift'],0.)
        self.assertEqual(model.members,1)
        # Compare the custom optimizer's first update with the official native optimizer
        # on exactly the same gradients, parameter grouping and schedule progress.
        from pytabkit.models.optim.optimizers import get_opt_class
        from pytabkit.models.training.scheduling import LearnerProgress
        from types import SimpleNamespace
        reference=copy.deepcopy(model)
        ours=Optimizer(model,protocol('E4'));ours.schedule(.1)
        params=[dict(params=[p],lr=.01) for p in reference.parameters()]
        original=get_opt_class(reference.info['config'].get('opt','adam'))(params,reference.hp)
        progress=LearnerProgress();progress.max_epochs=256;progress.epoch_float=.1
        reference.hp.update_hypers(SimpleNamespace(progress=progress))
        for net,opt in [(model,ours),(reference,original)]:
            net.train();net.zero_grad(set_to_none=True);torch.manual_seed(112)
            net.loss(net(torch.tensor(self.raw)),torch.tensor(self.y)).backward();opt.step()
        drift=max(float((p-dict(reference.named_parameters())[n]).abs().max()) for n,p in model.named_parameters())
        self.assertLessEqual(drift,1e-6)
        self.check_updates(model,'E4')

    def test_boundary_rule_and_exact_authorized_matrix(self):
        self.assertEqual(VARIANTS,('E1','E2','E3','E4'))
        with self.assertRaises(AssertionError):protocol('E5')
        rows=[dict(epoch=i+1,dev_ap=.5) for i in range(256)]
        rows[17]['dev_ap']=.51
        for n in (1,17,18,51,53,119,121,255):self.assertFalse(stopping(rows[:n],protocol('E1'))['stop'])
        self.assertTrue(stopping(rows[:52],protocol('E1'))['stop'])
        rows[43]['dev_ap']=.52
        self.assertFalse(stopping(rows[:52],protocol('E1'))['stop'])
        self.assertTrue(stopping(rows[:120],protocol('E1'))['stop'])
        self.assertTrue(stopping(rows,protocol('E1'))['stop'])


if __name__=='__main__':unittest.main()
