"""F0--F5: native-entry control and five isolated E2 recipe changes."""
import copy
import os
import shutil
from pathlib import Path

from scripts.tracehr_centered_native_common import Data, stopping, protocol as e_protocol, model_for as e_model
from scripts.tracehr_multiview_common import atomic, read, sha

RUN = 'tracehr_execution_refine_20261014'
VARIANTS = STAGE1 = tuple('F'+str(i) for i in range(6))
STAGE2 = ()


def protocol(variant, selected=None):
    assert variant in VARIANTS
    s = e_protocol('E4' if variant == 'F0' else 'E2')
    s.update(variant=variant, structure=variant, native_execution=variant == 'F0',
             frequency_lr_factor=.03 if variant == 'F2' else .1,
             scale_lr_factor=2. if variant == 'F3' else 6.,
             description={
                 'F0': 'full native K1 entry with original sampler, RNG, loss, full batch and clock',
                 'F1': 'E2 native-parameter base LR .04 -> .02; adapter LR unchanged',
                 'F2': 'E2 frequency-only LR factor .1 -> .03',
                 'F3': 'E2 post-encoding ScaleLayer-only LR factor 6 -> 2',
                 'F4': 'E2 input adapter LR .0003 -> .001',
                 'F5': 'E2 dropout base .15 -> .25; same flat_cos schedule'}[variant])
    if variant == 'F0':
        s.update(training_microbatch=256, microbatch_policy='native complete 256 batch, drop_last=True')
    if variant == 'F1': s['lr'] = .02
    if variant == 'F4': s['adapter_lr'] = .001
    if variant == 'F5': s['dropout'] = .25
    return s


def prepare(project, database, out):
    assert database == 'mimic'
    previous = project/'runs/tracehr_centered_native_20261014/mimic/v1/preparation'
    record = read(previous/'completion.json')
    assert record['complete'] and record['preparation_complete'] and not record['test_loaded']
    from scripts.tracehr_unified_common import original_pointer
    pointer, folder, _ = original_pointer(project, database)
    assert pointer == record['input_pointer']
    for name, artifact in record['input_artifacts'].items():
        assert name.startswith(('train_', 'dev_'))
        atomic(out/'progress.json', dict(status='checking_input', file=name))
        assert sha(folder/name) == artifact['sha256']
    for name,key in [('preprocessing.npz','preprocessing_sha256'), ('native_configs.json','native_configs_sha256'),
                     ('native_summary_scaling.npz','native_summary_scaling_sha256')]:
        assert sha(previous/name) == record[key]
        shutil.copyfile(previous/name, out/name)
    record.update(reused_preparation=str(previous), previous_manifest_sha256=sha(previous/'completion.json'),
                  protocols={v:protocol(v) for v in VARIANTS}, F6_F7_started=False,
                  raw_cache_duplicated=False, wait_for_eicu=False)
    atomic(out/'completion.json', record)
    atomic(out/'progress.json', dict(status='complete'))


def configure_e2(model, spec):
    """Change runtime hyperparameters only, after common E2 random initialization."""
    before = {n:dict(p.hyper_factors) for n,p in model.named_parameters() if hasattr(p,'hyper_factors')}
    model.hp.config['lr'] = spec['lr']
    if 'lr' in model.hp.hyper_base_values:
        assert set(model.hp.hyper_base_values['lr']) == {''}
        model.hp.hyper_base_values['lr'][''] = spec['lr']
    assert set(model.hp.hyper_base_values['p_drop']) == {''}
    model.hp.hyper_base_values['p_drop'][''] = spec['dropout']
    model.hp.config['p_drop'] = spec['dropout']
    model.info['config'].update(lr=spec['lr'], p_drop=spec['dropout'])
    model.numerical.frequency.hyper_factors = dict(model.numerical.frequency.hyper_factors, lr=spec['frequency_lr_factor'])
    scales = [m for m in model.net.modules() if type(m).__name__=='ScaleLayer']
    assert len(scales)==1
    for p in scales[0].parameters():
        assert p.requires_grad and p.hyper_factors['lr']==6.
        p.hyper_factors = dict(p.hyper_factors, lr=spec['scale_lr_factor'])
    after = {n:dict(p.hyper_factors) for n,p in model.named_parameters() if hasattr(p,'hyper_factors')}
    model.info['runtime_recipe_change'] = dict(variant=spec['variant'], before=before, after=after,
                                              initial_weights_unchanged=True)


def model_for(data, preparation, variant, endpoint, seed, lr_multiplier=1., selected=None, tmp_folder=None):
    assert variant != 'F0' and lr_multiplier == 1.
    from models.tracehr_unified_next import parameter_hash
    model = e_model(data, preparation, 'E2', endpoint, seed, tmp_folder=tmp_folder)
    initial = parameter_hash(model)
    reference = Path(os.environ['TRACEHR_PROJECT'])/'runs/tracehr_centered_native_20261014/mimic/v1/screen/jobs'/f'E2_t{endpoint}_s42/run_manifest.json'
    assert seed == 42 and initial == read(reference)['initial_weights_sha256']
    configure_e2(model, protocol(variant))
    assert parameter_hash(model) == initial
    model.info['E2_initialization_match'] = dict(complete=True, initial_hash=initial, path=str(reference), sha256=sha(reference))
    return model
