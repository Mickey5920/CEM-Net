"""Freeze, launch and collect the approved same-device matched-control timing."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.launch_ema_cross import HOSTS, ssh
from scripts.tracehr_multiview_common import atomic, read, sha
RUN = 'cem_matched_controls_20261105'
DATABASES = ('mimiciii', 'mimic', 'eicu')

REMOTE = r'''
import base64,fcntl,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
d=PAYLOAD;p=Path(d['project']).resolve(strict=True)
source=p/'work'/d['run']/('resources_'+d['db']+'_v1')
base=p/'runs'/d['run']/(d['db']+'_inference_'+d['execution']);out=base/'measurement'
temp=p/'work'/d['run']/'tmp_resources';cache=p/'cache'/d['run']
frozen=p/'work'/d['run']/d['db']/'v4'
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
if d['mode'] in ('check','start'):
 for path in (source,base,temp,cache):
  path.mkdir(parents=True,exist_ok=True);assert path.resolve().is_relative_to(p)
 assert shutil.disk_usage(p).free>2*2**30
 for name,encoded in d['files'].items():
  content=base64.b64decode(encoded);path=source/name
  assert path.resolve().is_relative_to(source.resolve()) and hashlib.sha256(content).hexdigest()==d['manifest'][name]
  assert not path.exists() or path.read_bytes()==content,'Frozen resource source changed'
  if not path.exists():
   with path.open('xb') as stream:stream.write(content)
  compile(content,str(path),'exec')
 manifest=source/'source_manifest.json'
 if manifest.exists():assert read(manifest)==d['manifest']
 else:
  with manifest.open('x') as stream:json.dump(d['manifest'],stream,indent=2)
 result=dict(checked=True,source_manifest_sha256=sha(manifest),training_launched=False,execution=d['execution'])
 if d['mode']=='start':
  host=p/'runs'/d['run']/('host_'+d['host']);state=read(host/'state.json');exitrecord=read(host/'scheduler_v4_exit.json')
  assert state['status']=='complete' and exitrecord['returncode']==0 and exitrecord['child_pid']==state['pid']
  assert all(g['status']=='complete' for g in state['groups'].values())
  if d['host']=='mimic':
   other=p/'runs'/d['run']/(('mimic' if d['db']=='mimiciii' else 'mimiciii')+'_inference_'+d['execution'])
   if (other/'attempt.json').exists():assert read(other/'exit.json')['returncode']==0 and read(other/'measurement/completion.json')['complete']
  assert not (base/'attempt.json').exists() and not out.exists(),'Existing attempt; inspect before retry'
  previous=p/'runs'/d['run']/(d['db']+'_inference_v1')
  if d['execution']=='v2' and (previous/'attempt.json').exists():
   assert read(previous/'exit.json')['returncode']!=0 and not (previous/'measurement').exists()
   result['prior_failed_execution']=dict(exit_path=str(previous/'exit.json'),exit_sha256=sha(previous/'exit.json'),measurement_started=False)
  deps=p/'work/tracehr_tabular_20261004/deps'
  env=dict(os.environ,TRACEHR_PROJECT=str(p),TRACEHR_DATABASE=d['db'],TRACEHR_TABULAR_DEPS=str(deps),
   PYTHONPATH=os.pathsep.join(map(str,[frozen,deps,*deps.glob('*.whl'),p/'work/cem_evidence_completion_20261025/test_runtime_v1/deps'])),
   OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
   PYTHONNOUSERSITE='1',NUMPY_MADVISE_HUGEPAGE='0',TMPDIR=str(temp),TEMP=str(temp),TMP=str(temp),XDG_CACHE_HOME=str(cache),
   TORCH_HOME=str(cache/'torch'),HF_HOME=str(cache/'huggingface'),MPLCONFIGDIR=str(cache/'matplotlib'),CUDA_CACHE_PATH=str(cache/'cuda'),CUBLAS_WORKSPACE_CONFIG=':4096:8')
  sys.path.insert(0,str(frozen))
  from scripts.cem_evidence_host_queue import inventory
  eligible=[x['index'] for x in inventory(p) if x['index'] in ([2,3] if d['host']=='eicu' else [0,1,2,3]) and x['fully_idle'] and x['free_mib']>=8192]
  assert eligible,'No idle authorized GPU; retry start after it becomes available'
  gpu=eligible[0]
  wrapper=frozen/'scripts/tracehr_recovery_attempt.py'
  assert sha(wrapper)==read(frozen/'source_manifest.json')['scripts/tracehr_recovery_attempt.py']
  command=[sys.executable,'-B',str(source/'benchmark.py'),'--project',str(p),'--source',str(frozen),'--output',str(out),'--database',d['db'],'--gpu',str(gpu)]
  spec=dict(command=command,cwd=str(source),receipt=str(base/'exit.json'),job_id='matched_resources_'+d['db'],source_manifest_sha256=sha(manifest))
  with (base/'attempt.json').open('x') as stream:json.dump(spec,stream,indent=2)
  with (base/'benchmark.log').open('x') as log:
   child=subprocess.Popen([sys.executable,'-B',str(wrapper),'--spec',str(base/'attempt.json')],cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  result.update(started=True,wrapper_pid=child.pid,gpu=gpu)
  with (base/'launch.json').open('x') as stream:json.dump(result,stream,indent=2)
elif d['mode']=='status':
 result={name:read(base/name) for name in ('launch.json','exit.json') if (base/name).exists()}
 result.update({name:read(out/name) for name in ('progress.json','completion.json') if (out/name).exists()})
 if (base/'exit.json').exists() and read(base/'exit.json')['returncode']!=0:result['log_tail']=(base/'benchmark.log').read_text(errors='replace')[-5000:]
else:
 exitrecord=read(base/'exit.json');done=read(out/'completion.json')
 assert exitrecord['returncode']==0 and done['complete'] and set(done['files'])=={'verified-results.json'}
 assert sha(out/'verified-results.json')==done['files']['verified-results.json']
 result=dict(receipt=done,exit=exitrecord,content=base64.b64encode((out/'verified-results.json').read_bytes()).decode())
print(json.dumps(result))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('check', 'start', 'status', 'results'), required=True)
    parser.add_argument('--execution', choices=('v1', 'v2'), default='v2')
    parser.add_argument('--database', choices=DATABASES, required=True); args = parser.parse_args()
    out = Path(os.environ['PROJECT_RUN_DIR']).resolve(strict=True)
    assert out == ROOT/'runs'/RUN/(args.database+'_resources_v1')
    host = 'eicu' if args.database == 'eicu' else 'mimic'
    payload = dict(project=HOSTS[host][1], run=RUN, db=args.database, host=host, mode=args.mode, execution=args.execution)
    if args.mode in ('check', 'start'):
        files = {'benchmark.py': (ROOT/'scripts/benchmark_cem_matched_controls.py').read_bytes(),
                 'reference_inference.py': (ROOT/'scripts/cem_icde72_frozen_inference.py').read_bytes()}
        for name, content in files.items(): compile(content, name, 'exec')
        payload.update(files={n: base64.b64encode(v).decode() for n, v in files.items()},
                       manifest={n: hashlib.sha256(v).hexdigest() for n, v in files.items()})
    result = ssh(host, REMOTE.replace('PAYLOAD', repr(payload), 1), 55)
    if args.mode == 'results':
        content = base64.b64decode(result.pop('content'))
        assert hashlib.sha256(content).hexdigest() == result['receipt']['files']['verified-results.json']
        path = out/'verified-results.json'
        if path.exists(): assert path.read_bytes() == content
        else: path.write_bytes(content)
        atomic(out/'collection-receipt.json', result)
        result = dict(complete=True, database=args.database)
    else: atomic(out/(args.mode+'-receipt.json'), result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
