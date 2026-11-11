"""Author backbones and a documented clinical-input adapter; no learned caches."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BASE = Path(__file__).resolve().parents[1] / 'work/cem_sequence_comparators_20261102/vendor'


def module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def ft_module():
    return module('cem_author_ft', BASE/'ft/package/rtdl_revisiting_models.py')


def sdpa(self, x_q, x_kv):
    b, n, _ = x_q.shape
    def shape(x):
        return x.reshape(b, x.shape[1], self._n_heads, -1).transpose(1, 2)
    q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
    if self.key_compression is not None:
        k=self.key_compression(k.transpose(1,2)).transpose(1,2)
        v=(self.key_compression if self.value_compression is None else self.value_compression)(v.transpose(1,2)).transpose(1,2)
    # The author KV projection is transposed: its last stride is 128, whereas
    # fused CUDA attention requires stride 1. Copy layout only, preserving values.
    q,k,v=(shape(t).contiguous() for t in (q,k,v))
    # Author KV projection, then the same scaled dot-product attention definition.
    # CUDA FP32 uses the memory-efficient kernel, not the quadratic math fallback.
    if q.is_cuda:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        context = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
    else:
        from contextlib import nullcontext
        context = nullcontext()
    with context:
        y = F.scaled_dot_product_attention(q, k, v,
            dropout_p=self.dropout.p if self.training and self.dropout is not None else 0.)
    y = y.transpose(1, 2).reshape(b, n, -1)
    return y if self.W_out is None else self.W_out(y)


def efficient_attention(model):
    for m in model.modules():
        if isinstance(m, ft_module().MultiheadAttention):
            m.forward = types.MethodType(sdpa, m)


class ClinicalAdapter(nn.Module):
    def __init__(self, num_types):
        super().__init__()
        self.events = nn.EmbeddingBag(num_types, 128, mode='sum', include_last_offset=True)
        nn.init.normal_(self.events.weight, std=.02)

    def forward(self, features, codes, counts, offsets):
        events = self.events(codes, offsets, per_sample_weights=counts)
        return torch.cat((events.reshape(len(features), 24, 128), features), dim=-1)


def same(x, weight, dilation=1, groups=1):
    amount = dilation * (weight.shape[-1] - 1)
    return F.conv1d(F.pad(x, (amount//2, amount-amount//2)), weight,
                    dilation=dilation, groups=groups)


class KerasBN(nn.Module):
    """Keras defaults: population moments, epsilon .001, moving momentum .99."""
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))
        self.register_buffer('mean', torch.zeros(width))
        self.register_buffer('variance', torch.ones(width))

    def forward(self, x):
        if self.training:
            variance, mean = torch.var_mean(x, dim=(0, 2), correction=0)
            with torch.no_grad():
                self.mean.mul_(.99).add_(mean, alpha=.01)
                self.variance.mul_(.99).add_(variance, alpha=.01)
        else:
            mean, variance = self.mean, self.variance
        return (x-mean[None,:,None]) * torch.rsqrt(variance[None,:,None]+.001) * self.weight[None,:,None] + self.bias[None,:,None]


class Separable(nn.Module):
    def __init__(self, inp, out, kernel, dilation=1):
        super().__init__()
        self.depthwise = nn.Parameter(torch.empty(inp, 1, kernel))
        self.pointwise = nn.Parameter(torch.empty(out, inp, 1))
        # Keras Conv initializer computes fans from (kernel, in_channels, multiplier).
        nn.init.uniform_(self.depthwise, -math.sqrt(6/(kernel*(inp+1))), math.sqrt(6/(kernel*(inp+1))))
        nn.init.xavier_uniform_(self.pointwise)
        self.dilation = dilation

    def forward(self, x):
        return F.conv1d(same(x, self.depthwise, self.dilation, x.shape[1]), self.pointwise)


def handcrafted_kernels():
    values=[]
    for phase in (0, 1):
        for k in (2,4,8,16,32,64):
            a=np.ones(k, dtype=np.float32);a[np.arange(k)%2==phase] *= -1
            values.append(a)
    for k in (4,8,16,32,64):
        a=np.linspace(0,1,k//4+1)[1:]**2
        values.append(np.concatenate((-a,-a[::-1],2*a,2*a[::-1],-a,-a[::-1])).astype('float32'))
    return values


class Handcrafted(nn.Module):
    def __init__(self, inp):
        super().__init__()
        for i, a in enumerate(handcrafted_kernels()):
            self.register_buffer('filter_'+str(i), torch.tensor(a)[None,None,:].repeat(inp,1,1))

    def forward(self, x):
        return torch.cat([same(x, w, groups=x.shape[1]) for w in self.buffers()], dim=1).relu()


class LITEMV(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.branches = nn.ModuleList([Separable(inp,32,k) for k in (40,20,10)])
        self.handcrafted = Handcrafted(inp)
        width = 96+17*inp
        self.bn0 = KerasBN(width)
        self.layer1 = Separable(width,32,20,2);self.bn1=KerasBN(32)
        self.layer2 = Separable(32,32,10,4);self.bn2=KerasBN(32)
        self.head = nn.Linear(32,2)
        nn.init.xavier_uniform_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = torch.cat([m(x) for m in self.branches]+[self.handcrafted(x)],dim=1)
        x = self.bn0(x).relu()
        x = self.bn1(self.layer1(x)).relu()
        x = self.bn2(self.layer2(x)).relu()
        return self.head(x.mean(-1))


class KerasAdam(torch.optim.Optimizer):
    """Adam's Keras epsilon-hat placement; no weight decay."""
    def __init__(self, params, lr=.001, eps=1e-7):
        super().__init__(params, dict(lr=lr, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:continue
                state=self.state[p]
                if not state:
                    state.update(step=0,m=torch.zeros_like(p),v=torch.zeros_like(p))
                state['step']+=1;t=state['step']
                state['m'].lerp_(p.grad,.1)
                state['v'].mul_(.999).addcmul_(p.grad,p.grad,value=.001)
                lr=group['lr']*math.sqrt(1-.999**t)/(1-.9**t)
                p.addcdiv_(state['m'],state['v'].sqrt().add_(group['eps']),value=-lr)


def modern(inp):
    base=BASE/'modern/ModernTCN-classification'
    # Only qualify import locations; author source bytes remain intact.
    if 'layers' not in sys.modules:
        package=types.ModuleType('layers');package.__path__=[];sys.modules['layers']=package
    module('layers.RevIN',base/'layers/RevIN.py')
    module('models.ModernTCN_Layer',base/'models/ModernTCN_Layer.py')
    author=module('cem_author_modern',base/'models/ModernTCN.py')
    cfg=types.SimpleNamespace(task_name='classification',stem_ratio=6,downsample_ratio=2,ffn_ratio=1,
        num_blocks=[1,1],large_size=[13,13],small_size=[5,5],dims=[32,64],dw_dims=[32,64],
        enc_in=inp,small_kernel_merged=False,dropout=.1,head_dropout=0.,use_multi_scale=False,
        revin=True,affine=True,subtract_last=False,freq='h',seq_len=24,individual=False,
        pred_len=1,kernel_size=25,patch_size=8,patch_stride=4,class_dropout=0.,num_class=2,decomposition=False)
    m=author.Model(cfg)
    # Forecast head/RevIN are constructed but never referenced by classification.
    # Preserve author parameters and report the unused subset explicitly.
    return m


class Comparator(nn.Module):
    def __init__(self, variant, num_types, input_dim):
        super().__init__();self.variant=variant
        if variant=='FTTransformer':
            self.backbone=ft_module().FTTransformer(n_cont_features=input_dim,cat_cardinalities=[],d_out=1,
                n_blocks=3,d_block=192,attention_n_heads=8,attention_dropout=.2,ffn_d_hidden=None,
                ffn_d_hidden_multiplier=4/3,ffn_dropout=.1,residual_dropout=0.,
                linformer_kv_compression_ratio=math.nextafter(128/(input_dim+1),math.inf),
                linformer_kv_compression_sharing='key-value')
            assert all(m.key_compression.out_features==128 for m in self.backbone.modules() if isinstance(m,ft_module().MultiheadAttention))
            efficient_attention(self.backbone)
        else:
            self.adapter=ClinicalAdapter(num_types)
            if variant=='TCN':
                author=module('cem_author_tcn',BASE/'tcn/TCN/tcn.py')
                self.backbone=author.TemporalConvNet(224,[128]*4,kernel_size=4,dropout=.2)
                self.head=nn.Linear(128,1)
            elif variant=='ModernTCN':self.backbone=modern(224)
            elif variant=='LITEMV':self.backbone=LITEMV(224)
            else:raise ValueError(variant)

    def forward(self, *batch):
        if self.variant=='FTTransformer':return self.backbone(batch[0],None).squeeze(-1)
        x=self.adapter(*batch)
        if self.variant=='ModernTCN':z=self.backbone(x,None,None,None)
        elif self.variant=='TCN':return self.head(self.backbone(x.transpose(1,2))[:,:,-1]).squeeze(-1)
        else:z=self.backbone(x.transpose(1,2))
        # Binary CE for two logits is identically BCE(z1-z0).
        return z[:,1]-z[:,0]

    def allowed_unconnected(self):
        if self.variant!='ModernTCN':return ()
        return ('backbone.model.head.', 'backbone.model.revin_layer.')


def make_model(variant, seed, data):
    from scripts.train_tracehr_unified import seed_all
    seed_all(seed)
    return Comparator(variant,data.num_types,data.input_dim)


def optimizer(model, spec):
    if spec['optimizer']=='KerasAdam':return KerasAdam(model.parameters(),spec['lr'],spec['eps'])
    groups=model.backbone.make_parameter_groups() if model.variant=='FTTransformer' else model.parameters()
    cls=torch.optim.Adam if spec['optimizer']=='Adam' else torch.optim.AdamW
    return cls(groups,lr=spec['lr'],eps=spec['eps'],weight_decay=spec['weight_decay'])


def cpu_checks():
    torch.set_num_threads(2);torch.manual_seed(19)
    author=ft_module()
    original=author.FTTransformer(n_cont_features=255,cat_cardinalities=[],d_out=1,
        n_blocks=2,d_block=16,attention_n_heads=4,attention_dropout=0.,ffn_d_hidden=None,
        ffn_d_hidden_multiplier=4/3,ffn_dropout=0.,residual_dropout=0.,
        linformer_kv_compression_ratio=.5,linformer_kv_compression_sharing='key-value')
    altered=copy.deepcopy(original);efficient_attention(altered)
    x=torch.randn(3,255)
    a=original(x,None);b=altered(x,None);torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
    a.sum().backward();b.sum().backward()
    for p,q in zip(original.parameters(),altered.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=3e-5)
    # Independent cross-correlation reference covers even kernels, dilation and grouping.
    x=torch.randn(2,3,9,dtype=torch.float64,requires_grad=True)
    w=torch.randn(3,1,4,dtype=torch.float64,requires_grad=True)
    actual=same(x,w,2,3)
    ref=torch.stack([sum((F.pad(x,(3,3))[:,:,j+2*k]*w[:,0,k]) for k in range(4)) for j in range(9)],dim=-1)
    torch.testing.assert_close(actual,ref,atol=1e-12,rtol=1e-12)
    ag=torch.autograd.grad(actual.square().sum(),(x,w),retain_graph=True)
    rg=torch.autograd.grad(ref.square().sum(),(x,w))
    for a,b in zip(ag,rg):torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
    bn=KerasBN(3).double();v=torch.randn(2,3,9,dtype=torch.float64)
    mean=v.numpy().mean((0,2));var=v.numpy().var((0,2));result=bn(v)
    np.testing.assert_allclose(result.detach().numpy(),(v.numpy()-mean[None,:,None])/np.sqrt(var[None,:,None]+.001),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(bn.variance.numpy(),.99+.01*var)
    p=nn.Parameter(torch.tensor([.4,-.3],dtype=torch.float64));op=KerasAdam([p]);npvalue=p.detach().numpy().copy();m=np.zeros(2);vv=np.zeros(2)
    for t in range(1,4):
        g=np.array([.2*t,-.1]);p.grad=torch.tensor(g);op.step();m=.9*m+.1*g;vv=.999*vv+.001*g*g
        npvalue-=.001*np.sqrt(1-.999**t)/(1-.9**t)*m/(np.sqrt(vv)+1e-7)
    np.testing.assert_allclose(p.detach().numpy(),npvalue,atol=1e-14,rtol=1e-14)
    assert len(handcrafted_kernels())==17
    lite=LITEMV(3);assert not any(n.startswith('handcrafted') for n,_ in lite.named_parameters())
    assert torch.isfinite(lite(torch.zeros(2,3,24))).all()
    # Check full backbones, adapter empty histories and a genuine backward path.
    for name in ('TCN','ModernTCN','LITEMV'):
        model=Comparator(name,11,7)
        data=(torch.randn(2,24,96),torch.tensor([0,1,2]),torch.tensor([1.,2.,1.]),torch.tensor([0,2,3]+[3]*46))
        z=model(*data);assert z.shape==(2,) and torch.isfinite(z).all();z.square().sum().backward()
        disconnected=[n for n,p in model.named_parameters() if p.grad is None]
        assert all(n.startswith(model.allowed_unconnected()) for n in disconnected),(name,disconnected)
    return ['FT_author_KV128_SDPA_forward_and_gradient','same_padding_dilated_depthwise_reference_gradient',
            'Keras_population_batchnorm_and_moving_moments','Keras_Adam_epsilon_hat_reference',
            'LITEMV_17_frozen_filters','four_backbone_and_adapter_paths',
            'LITEMV_source_translation_no_TensorFlow_runtime_parity_claim']
