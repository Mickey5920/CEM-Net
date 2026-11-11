"""Run only CPU statistics on archived predictions and retrieve aggregate results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.launch_ema_cross import HOSTS, ssh

RUN = 'cem_evidence_completion_20261025'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', choices=('mimiciii', 'mimic', 'eicu'), required=True)
    parser.add_argument('--inspect', action='store_true')
    a = parser.parse_args()
    out = Path(os.environ['PROJECT_RUN_DIR']).resolve(strict=True)
    assert out == (ROOT/'runs'/RUN/'nre_cmm_interaction_v1'/a.database).resolve()
    helper = ROOT/'work'/RUN/'nre_cmm_interaction_remote.py'
    current = ROOT/'runs'/RUN/(a.database+'_results_v1')/'verified-results.json'
    old = (ROOT/'runs/mimiciii_ema_pm_20261024/results_v1/verified-results.json' if a.database == 'mimiciii'
           else ROOT/'runs/tracehr_ema_pm_cross_20261025'/(a.database+'_results_v1')/'verified-results.json')
    host = 'eicu' if a.database == 'eicu' else 'mimic'
    d = dict(project=HOSTS[host][1], database=a.database, inspect=a.inspect,
             current_sha256=sha(current), old_sha256=sha(old), analysis_sha256=sha(helper))
    source = helper.read_text(encoding='utf-8')
    compile(source, str(helper), 'exec')
    code = 'import os,json\nfrom pathlib import Path\nd='+repr(d)+'\np=Path(d["project"])\n'
    code += "os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',TMPDIR=str(p/'work'/"+repr(RUN)+"/'tmp'),TEMP=str(p/'work'/"+repr(RUN)+"/'tmp'),XDG_CACHE_HOME=str(p/'cache'/"+repr(RUN)+"))\n"
    code += source+'\nprint(json.dumps(analyze(d),ensure_ascii=False))\n'
    value = ssh(host, code, timeout=7200)
    if a.inspect:
        print(json.dumps(value, ensure_ascii=False))
        return
    assert value['result']['complete'] and not value['result']['patient_records_exported']
    text = json.dumps(value['result'], ensure_ascii=False, indent=2)
    assert hashlib.sha256(text.encode()).hexdigest() == value['receipt']['result_sha256']
    for name, content in (('interaction.json', text), ('receipt.json', json.dumps(value['receipt'], ensure_ascii=False, indent=2))):
        dest = out/name
        if dest.exists():
            assert dest.read_text(encoding='utf-8') == content
            if dest.read_bytes() != content.encode('utf-8'):
                dest.write_bytes(content.encode('utf-8'))
        else:
            dest.write_bytes(content.encode('utf-8'))
    print(json.dumps(dict(database=a.database, complete=True, reused=value['reused'], results=value['result']['results'], verification=value['result']['verification']), ensure_ascii=False))


if __name__ == '__main__':
    main()
