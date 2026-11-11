"""Six authorized MIMIC fits; the C5 recipe and fixed z0 rules are unchanged."""
from scripts.tracehr_auxiliary_isolation_common import protocol as c5_protocol
from scripts.tracehr_context_cut_common import model_for as c1_model
from scripts.tracehr_structure_common import Data, stopping
from models.tracehr_c5_ablation import VARIANTS

RUN='tracehr_c5_ablation_20261017'
PROTOCOL_CHANGES={'variant','structure','description','ablated_component','ablation_implementation'}
IMPLEMENTATIONS={
 'C5_noT':('trajectory','remove added trajectory encoder/attention/trajectory combine columns; keep shared variable identity, auxiliary head and both losses'),
 'C5_noP':('learned_numeric_periodic_encoding','remove learned periodic encoder; zero-pad periodic slots, retain normalized raw values and native input width'),
 'C5_noM':('input_member_modulation','remove learned input modulation; identity input map, keep 8 members and member-averaged main CE'),
}


def protocol(variant,database):
    assert variant in VARIANTS and database=='mimic'
    original=c5_protocol('C5',database)
    component,description=IMPLEMENTATIONS[variant]
    result=dict(original,variant=variant,structure=variant,description='C5 single-component ablation: '+description,
                ablated_component=component,ablation_implementation=description)
    assert {k for k in set(original)|set(result) if original.get(k)!=result.get(k)}==PROTOCOL_CHANGES
    return result


def model_for(data,preparation,variant,endpoint,seed,lr_multiplier=1.,selected=None,tmp_folder=None):
    assert variant in VARIANTS and seed==42 and lr_multiplier==1.
    from models.tracehr_c5_ablation import C5AblationNetwork
    return C5AblationNetwork(c1_model(data,preparation,'C1',endpoint,seed),variant,seed)
