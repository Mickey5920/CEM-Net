"""Train-only activation/displacement diagnostics, never used in the loss."""
import numpy as np
import torch


@torch.no_grad()
def diagnostics(model, data, initial, seed=42):
    model.eval()
    indices = np.random.RandomState(seed).permutation(len(data.raw['train']))[:1024]
    first_bias = next(m for m in model.net.modules() if type(m).__name__=='BiasLayer' and m.bias.requires_grad)
    sums, squares, counts = [], [], []
    encoded_sum = None

    def capture(module, inputs, output):
        x = output['x_cont'].double()
        sums.append(x.sum(-2));squares.append(x.square().sum(-2));counts.append(x.shape[-2])

    handle = first_bias.register_forward_hook(capture)
    try:
        for start in range(0,len(indices),128):
            x,_ = data.batch('train',indices[start:start+128])
            model(x)
            if hasattr(model,'encoded_center'):
                part = model.numerical(x).double().sum(0)
                encoded_sum = part if encoded_sum is None else encoded_sum+part
    finally:
        handle.remove()
    mean = sum(sums)/sum(counts)
    var = (sum(squares)/sum(counts)-mean.square()).clamp_min(0)
    displacement = {n:float((p-initial[n]).norm()/(initial[n].norm()+1e-12)) for n,p in model.named_parameters()}
    result = dict(fit_split='train',probe_rows=len(indices),dropout=False,
                  first_postbias_mean_RMS=float(mean.square().mean().sqrt()),
                  first_postbias_std_median=float(var.sqrt().median()),parameter_relative_L2_displacement=displacement)
    if encoded_sum is not None:
        delta = encoded_sum/sum(counts)-model.encoded_center
        result.update(encoded_center_drift_RMS=float(delta.square().mean().sqrt()),
                      encoded_center_relative_L2_drift=float(delta.norm()/(model.encoded_center.double().norm()+1e-12)))
    return result
