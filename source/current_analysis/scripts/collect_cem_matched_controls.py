"""Use the existing collector for the completed 60-fit matched-control matrix."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import cem_replacement_statistics as prior
sys.modules['prior_utilities'] = prior
from scripts import cem_matched_statistics as analysis
from scripts import collect_cem_replacements as collector
from scripts.launch_ema_cross import HOSTS, ssh
RUN = analysis.RUN


def bundle():
    mapping = {'analysis.py': 'cem_matched_statistics.py', 'prior_utilities.py': 'cem_replacement_statistics.py',
               'prior_statistics.py': 'cem_icde72_statistics.py', 'prior_significance.py': 'cem_icde72_significance.py'}
    files = {name: (ROOT/'scripts'/relative).read_bytes() for name, relative in mapping.items()}
    for name, content in files.items(): compile(content, name, 'exec')
    manifests = {db: prior.read(ROOT/'runs'/RUN/f'local_{db}_v4/build-receipt.json')['source_manifest_sha256'] for db in analysis.DATABASES}
    files['training_manifests.json'] = json.dumps(manifests, indent=2).encode()
    return files, {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}


def host_readiness(host):
    payload = dict(project=HOSTS[host][1], run=RUN, host=host)
    code = r'''
import json,hashlib
from pathlib import Path
d=PAYLOAD;root=Path(d['project'])/'runs'/d['run']/('host_'+d['host']);s=json.loads((root/'state.json').read_text())
exitpath=root/'scheduler_v4_exit.json';rc=json.loads(exitpath.read_text()) if exitpath.exists() else None
groups={g['database']:dict(status=g['status'],train=sum(j['action']=='train' and j['status']=='completed' for j in g['jobs'].values()),evaluate=sum(j['action']=='evaluate' and j['status']=='completed' for j in g['jobs'].values())) for g in s['groups'].values()}
expected={'eicu'} if d['host']=='eicu' else {'mimiciii','mimic'}
ready=set(groups)==expected and rc is not None and rc['returncode']==0 and rc['child_pid']==s['pid'] and s['status']=='complete' and all(g['status']=='complete' and g['train']==g['evaluate']==20 for g in groups.values())
print(json.dumps(dict(ready=ready,groups=groups,exit_path=str(exitpath),returncode=rc['returncode'] if rc else None,exit_sha256=hashlib.sha256(exitpath.read_bytes()).hexdigest() if rc else None)))
'''
    return ssh(host, code.replace('PAYLOAD', repr(payload), 1), 55)


def aggregate_local(out):
    results, inputs, draws = [], {}, {}
    for db in analysis.DATABASES:
        folder = ROOT/'runs'/RUN/(db+'_results_v1'); receipt = prior.read(folder/'collection-receipt.json')
        assert receipt['exit']['returncode'] == 0
        for name, expected in receipt['receipt']['files'].items():
            path = folder/name; assert prior.sha(path) == expected; inputs[str(path)] = expected
        result = prior.read(folder/'verified-results.json'); results.append(result)
        with np.load(folder/'paired_draws.npz', allow_pickle=False) as saved:
            assert set(saved.files) == {f'{variant}:{endpoint}' for variant in analysis.VARIANTS for endpoint in (0, 1)}
            for name in saved.files: draws[f'{db}:{name}'] = saved[name]
    ns = dict(np=np); prior.functions(ROOT/'scripts/cem_icde72_significance.py', ('holm', 'centered_p'), ns)
    combined = analysis.aggregate(results, ns); combined['inputs'] = inputs
    for row in combined['paired']:
        sample = draws[f"{row['database']}:{row['baseline']}:{row['endpoint']}"]
        np.testing.assert_array_equal(np.quantile(sample, [.025, .975]), row['ci95'])
        np.testing.assert_array_equal(np.quantile(sample, [.05/24, 1-.05/24]), row['adjusted_ci'])
        assert ns['centered_p'](sample, row['mean']) == row['p_bootstrap_centered']
    if (out/'completion.json').exists():
        assert prior.read(out/'verified-results.json') == combined
        return dict(complete=True, reused=True)
    prior.save(out/'verified-results.json', combined)
    with (out/'paired_draws.npz').open('xb') as f: np.savez_compressed(f, **draws)
    prior.save(out/'completion.json', dict(complete=True, files={name: prior.sha(out/name) for name in collector.ALLOWED}))
    return dict(complete=True, verified_new_fits=60, verified_fixed_tests=60, comparisons=12, families=1, seeds=5)


collector.RUN = RUN; collector.analysis = analysis; collector.bundle = bundle
collector.host_readiness = host_readiness; collector.aggregate_local = aggregate_local
collector.ALLOWED = ('verified-results.json', 'paired_draws.npz')
collector.REMOTE = collector.REMOTE.replace("{'verified-results.json','RESULTS.md'}", "{'verified-results.json','paired_draws.npz'}")
collector.REMOTE = collector.REMOTE.replace("host/'start_exit.json'", "host/'scheduler_v4_exit.json'")
collector.REMOTE = collector.REMOTE.replace('All 36 fits/tests', 'All 60 fits/tests').replace("'replacement_statistics_'", "'matched_statistics_'")

if __name__ == '__main__':
    collector.main()
