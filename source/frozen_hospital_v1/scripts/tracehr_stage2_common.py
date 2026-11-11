"""Fixed scope and original-entry command construction; no scientific changes."""
from pathlib import Path
import json
RUN='tracehr_not_stability_stage2_20261018'
NOA_MATRIX=tuple(('C5_noT_noA',e,s) for s in (45,46,47) for e in (0,1))
FAMILIES=('tabm','lightgbm','xgboost')
ROOT=Path(__file__).resolve().parents[1]
def expected():
    return json.loads((ROOT/'work'/RUN/'expected.json').read_text())
def baseline_command(family,seed,output,python):
    assert family in FAMILIES and seed in (45,46,47)
    command=expected()['templates'][family]['command']
    i=command.index('-B');cmd=[python,*command[i:]]
    cmd[cmd.index('--output')+1]=str(output)
    if '--seed' in cmd:cmd[cmd.index('--seed')+1]=str(seed)
    else:cmd+=['--seed',str(seed)]
    assert 'timeout' not in cmd and '--pilot' not in cmd and '--probe-full' not in cmd
    return cmd
