"""Rebuild paper displays from aggregate results, without clinical data or fitting."""
import argparse
import ast
import base64
import csv
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import sys

BUNDLE=Path(__file__).resolve().parent
RUN='cem_icde_evidence_20261028'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_five_seed(exact_signed_rank):
    """Verify the extension without predictions, resampling or model fitting."""
    import numpy as np
    folder=BUNDLE/'aggregate/runs/cem_five_seed_20261102'
    combined=read(folder/'paired_results_v1/verified-results.json')
    done=read(folder/'paired_results_v1/completion.json')
    assert done['complete'] and combined['complete']
    for name,digest in done['files'].items():assert sha(folder/'paired_results_v1'/name)==digest
    auto=read(folder/'auto_statistics_v1/completion.json')
    assert auto['complete'] and auto['statistics_complete']
    assert auto['aggregate_sha256']==sha(folder/'paired_results_v1/verified-results.json')
    assert read(folder/'auto_statistics_v1/exit.json')['returncode']==0
    assert combined['verified_new_fits']==combined['verified_fixed_tests']==30
    assert combined['seeds']==[48,49,50,51,52] and combined['replicates']==20000
    assert combined['old_three_seed_results_unchanged'] and not combined['existing_AP_caches_recomputed']
    assert combined['calibration_fits']==0 and len(combined['rows'])==150 and len(combined['paired'])==18
    manifests={'mimiciii':'c52119023f8334294f748be65caa5247e3954a591e38be548537eb1e8acade1e',
               'mimic':'0928a51b3f29c12968c9e976a12c6a8c038cea88c62eeb31323f94d6149f249c',
               'eicu':'dfb2222873f9ff3f3ad346be7790507a80d1cf0e93f16205b038ceffed5dff73'}
    joined=[]
    for db,digest in manifests.items():
        result_dir=folder/(db+'_results_v1')
        receipt=read(result_dir/'collection-receipt.json')
        assert receipt['exit']['returncode']==0 and receipt['receipt']['complete']
        for name,h in receipt['receipt']['files'].items():assert sha(result_dir/name)==h
        result=read(result_dir/'verified-results.json')
        count={'mimiciii':12,'mimic':8,'eicu':10}[db]
        assert result['complete'] and result['original_method_retained']
        assert result['source_manifest_sha256']==receipt['exit']['source_manifest_sha256']
        training_manifests={v for k,v in result['artifacts'].items()
                            if k.endswith(f'/work/cem_five_seed_20261102/{db}/v3/source_manifest.json')}
        assert training_manifests=={digest}
        assert result['verified_new_fits']==result['verified_fixed_tests']==len(result['fit_audits'])==count
        assert result['old_three_seed_results_unchanged'] and not result['existing_AP_caches_recomputed']
        assert result['calibration_fits']==0
        for job in result['fit_audits']:
            assert job['fit_exit']==job['test_exit']==job['test_updates']==0
            assert job['epochs']>=job['best_epoch']>0
        snapshot=BUNDLE/'source/frozen_five_seed_v3'/db
        assert sha(snapshot/'source_manifest.json')==digest
        assert read(snapshot/'build-receipt.json')['source_manifest_sha256']==digest
        sources=read(snapshot/'source_bundle.json');mapping=read(snapshot/'source_manifest.json')
        assert sources.keys()==mapping.keys() and len(mapping)==442
        for name,h in mapping.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            assert Path(name).suffix not in ('.npz','.npy','.pt','.pth','.parquet','.pkl','.joblib')
            assert hashlib.sha256(base64.b64decode(sources[name])).hexdigest()==h,name
        prior=read(BUNDLE/'aggregate/runs'/RUN/(db+'_results_v1/verified-results.json'))
        for row in result['rows']:
            if row['seed'] in (48,49,50):
                old=prior['summaries'][f"random:{row['split']}:{row['method']}:{row['endpoint']}"]
                for metric in ('AP','AUROC','Brier','log_loss'):
                    assert abs(row[metric]-old[metric]['values'][row['seed']-48])<1e-12
        joined.extend(result['rows'])
    assert combined['rows']==joined
    scores={(r['database'],r['split'],r['method'],r['endpoint'],r['seed']):r['AP'] for r in joined}
    assert len(scores)==150
    contrasts={('mimiciii',0,'TabM_PLE'),('mimiciii',0,'TabM_mini_PLE'),
               ('mimiciii',1,'TabM-mini'),('mimiciii',1,'LightGBM_full'),
               ('mimic',0,'TabM_PLE'),('mimic',1,'TabM_PLE'),
               ('eicu',0,'TabM_mini_PLE'),('eicu',1,'TabM_mini_PLE'),('eicu',1,'TabM_PLE')}
    for split in ('dev','test'):
        rows=[r for r in combined['paired'] if r['split']==split]
        assert len(rows)==9 and {(r['database'],r['endpoint'],r['baseline']) for r in rows}==contrasts
        for r in rows:
            d=[scores[r['database'],split,'CEM-Net',r['endpoint'],s]-scores[r['database'],split,r['baseline'],r['endpoint'],s] for s in (48,49,50,51,52)]
            np.testing.assert_allclose(d,r['values'],rtol=0,atol=1e-12)
            np.testing.assert_allclose([np.mean(d),np.std(d,ddof=1)],[r['mean'],r['sample_SD']],rtol=0,atol=1e-12)
            assert r['candidate']=='CEM-Net' and r['family_count']==54
            assert 1/20001<=r['p_bootstrap_centered']<=1
            assert r['p_bonferroni']==min(1.,54*r['p_bootstrap_centered'])
            assert r['wilcoxon']==exact_signed_rank(d)
            assert r['adjusted_ci'][0]<=r['ci95'][0]<=r['ci95'][1]<=r['adjusted_ci'][1]
    return dict(five_seed_new_fits=30,five_seed_fixed_tests=30,five_seed_paired_comparisons=18,
                five_seed_source_snapshots_verified=3,five_seed_old_scores_unchanged=True)


def verify_core_five_seed():
    """Check saved aggregates and executed sources without fitting or resampling."""
    import collections
    import statistics as st
    folder=BUNDLE/'aggregate/runs/cem_core_five_seed_20261107'
    combined=read(folder/'paired_results_v1/verified-results.json')
    assert sha(folder/'paired_results_v1/verified-results.json')=='c911fa18764f0a75b972bfef05d86b0b1e50f4845386c249d20585487d2e56b0'
    assert combined['complete'] and combined['seeds']==[48,49,50,51,52]
    assert (combined['verified_cells'],combined['new_fits'],combined['reused_saved_fit'])==(204,203,1)
    assert not combined['original_results_modified']
    frozen=BUNDLE/'source/frozen_core_five_seed_v1'
    source_hashes=set()
    for manifest in sorted(frozen.glob('local_*/source_manifest.json')):
        mapping=read(manifest);bundle=manifest.parent/'source_bundle.json'
        sources=read(bundle) if bundle.exists() else None
        if sources is not None:assert sources.keys()==mapping.keys()
        for name,digest in mapping.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            assert Path(name).suffix not in ('.npz','.npy','.pt','.pth','.parquet','.pkl','.joblib')
            payload=base64.b64decode(sources[name]) if sources is not None else (manifest.parent/name).read_bytes()
            assert hashlib.sha256(payload).hexdigest()==digest,name
        source_hashes.add(sha(manifest))
    joined=[];audits=[];replayed=0
    for db,count in [('mimiciii',62),('mimic',62),('eicu',80)]:
        scope=folder/(db+'_results_v1');result=read(scope/'verified-results.json')
        receipt=read(scope/'collection-receipt.json')
        # The collector serializes the Linux JSON on Windows (CRLF); the receipt
        # hashes the remote LF bytes. Preserve both files and check that exact form.
        remote_bytes=(scope/'verified-results.json').read_bytes().replace(b'\r\n',b'\n')
        assert receipt['complete'] and receipt['result_sha256']==hashlib.sha256(remote_bytes).hexdigest()
        assert result['complete'] and result['verified_new_cells']==len(result['fit_audits'])==count
        assert not result['pending_cells'] and not result['missing_cells']
        assert result['old_three_seed_predictions_unchanged']
        assert not any(result[k] for k in ('model_modified','original_results_modified','patient_records_exported'))
        assert result['source_manifest_sha256'] in source_hashes
        replay=result['original_replay']
        assert replay['complete'] and replay['maximum_adjusted_interval_error']==0
        assert replay['original_point_differences_and_centered_p_reproduced']
        replayed+=replay['original_contrasts_reproduced']
        for job in result['fit_audits']:
            assert job['fit_exit']==job['test_exit']==job['test_updates']==0
            assert job['epochs']>=job['best_epoch']>0
            assert job['source_manifest_sha256'] in source_hashes
        joined.extend(result['rows']);audits.extend(result['fit_audits'])
    assert combined['rows']==joined and len(joined)==1200 and len(combined['summaries'])==240
    assert replayed==108 and sum(a['reused_fit'] for a in audits)==1
    metrics={(r['database'],r['arm'],r['split'],r['method'],r['endpoint'],r['seed']):r for r in joined}
    assert len(metrics)==1200
    for row in combined['summaries']:
        key=tuple(row[k] for k in ('database','arm','split','method','endpoint'))
        for scope,seeds in [('three_seed',[48,49,50]),('five_seed',[48,49,50,51,52])]:
            for metric,value in row[scope].items():
                values=[metrics[(*key,s)][metric] for s in seeds]
                assert values==value['values']
                assert abs(st.mean(values)-value['mean'])<1e-12
                assert abs(st.stdev(values)-value['sample_SD'])<1e-12
    families=collections.Counter(row['family'] for row in combined['paired'])
    assert dict(families)==combined['family_counts'] and sum(families.values())==108
    for row in combined['paired']:
        assert row['family_count']==families[row['family']]
        for scope,seeds in [('three_seed',[48,49,50]),('five_seed',[48,49,50,51,52])]:
            value=row[scope]
            expected=[sum(coef*metrics[row['database'],row['arm'],row['split'],method,row['endpoint'],s]['AP']
                          for method,coef in row['coefficients'].items()) for s in seeds]
            assert value['seeds']==seeds
            assert all(abs(a-b)<1e-12 for a,b in zip(expected,value['values']))
            assert abs(st.mean(expected)-value['mean'])<1e-12
            assert abs(st.stdev(expected)-value['sample_SD'])<1e-12
            assert value['adjusted_ci'][0]<=value['ci95'][0]<=value['ci95'][1]<=value['adjusted_ci'][1]
            assert value['p_bonferroni']==min(1.,row['family_count']*value['p_bootstrap_centered'])
    for scope in ('three_seed','five_seed'):
        for family,n in families.items():
            rows=sorted((r[scope] for r in combined['paired'] if r['family']==family),key=lambda r:r['p_bootstrap_centered'])
            running=0.
            for i,row in enumerate(rows):
                running=max(running,min(1.,(n-i)*row['p_bootstrap_centered']))
                assert abs(running-row['p_holm_full_family'])<1e-12
    return dict(core_five_seed=dict(added_cells=204,new_fits=203,reused_saved_fit=1,
                metric_rows=1200,summaries=240,paired_contrasts=108,families=11,
                source_snapshots_verified=len(source_hashes),original_replay_receipts_verified=replayed,
                five_seed_means_and_SD_reconstructed=True,Holm_values_reconstructed=True,
                intervals_verified_against_saved_aggregate_hash=True,
                fitting_executed=False,bootstrap_resampled=False))


def verify_secondary_metrics():
    name='cem_los_secondary_metrics_20261110'
    frozen=BUNDLE/'source/frozen_secondary_metrics_v1'
    for version in ('source_v1','source_v2'):
        folder=frozen/version
        for file,digest in read(folder/'source_manifest.json').items():
            assert sha(folder/file)==digest
    for db in ('mimiciii','mimic','eicu'):
        result=read(BUNDLE/'aggregate/runs'/name/db/'verified-results.json')
        assert result['source_sha256']==sha(frozen/'source_v2/cem_secondary_metrics.py')
        receipt=read(BUNDLE/'aggregate/runs'/name/db/'analyze_exit.json')
        assert receipt['returncode']==0 and receipt['source_manifest_sha256']==sha(frozen/'source_v2/source_manifest.json')
    return load('cem_secondary_verification',BUNDLE/'source/report_cem_secondary_metrics.py').verify(BUNDLE/'aggregate')


def verify():
    manifest=read(BUNDLE/'manifest.json')
    for rel,digest in manifest['files'].items():
        path=(BUNDLE/rel).resolve()
        assert path.is_relative_to(BUNDLE) and sha(path)==digest,rel
    snapshot=BUNDLE/'source/frozen_hospital_v1'
    assert sha(snapshot/'source_manifest.json')=='65a75589cf731b90a036298bc49210eb93872c89f1487381eb772f3079b0c884'
    for rel,digest in read(snapshot/'source_manifest.json').items():
        path=(snapshot/rel).resolve()
        assert path.is_relative_to(snapshot) and sha(path)==digest,rel
    fixed=[read(BUNDLE/'aggregate/runs'/RUN/(db+'_results_v1/verified-results.json')) for db in ('mimiciii','mimic','eicu')]
    assert sum(x['verified_new_fits'] for x in fixed)==72
    assert sum(x['verified_fixed_tests'] for x in fixed)==72
    assert sum(x['calibration_fits'] for x in fixed)==36
    for result in fixed:
        assert result['complete'] and result['original_method_retained'] and not result['main_method_switched']
        for job in result['worker_audits']:
            assert job['fit_exit']==job['test_exit']==job['test_updates']==0
    stats=read(BUNDLE/'aggregate/runs'/RUN/'statistical_tests_v1/verified-results.json')
    assert stats['complete'] and len(stats['rows'])==744
    inference=[read(BUNDLE/'aggregate/runs'/RUN/('local_'+db+'_inference_v2/verified-results.json')) for db in ('mimiciii','mimic','eicu')]
    assert all(x['complete'] for x in inference) and sum(x['fixed_stress_cases'] for x in inference)==144
    influence=read(BUNDLE/'aggregate/runs'/RUN/'hospital_influence_v1/verified-results.json')
    assert influence['complete'] and len(influence['rows'])==140
    assert not any(influence[k] for k in ('training_executed','checkpoints_changed','bootstrap_resampled','selection_changed','primary_intervals_changed','patient_records_exported'))
    import gzip,io
    import numpy as np
    loss_rows=[]
    for db in ('mimiciii','mimic','eicu'):
        folder=BUNDLE/'aggregate/runs'/RUN/'probability_intervals_v2'/db
        loss=read(folder/'verified-results.json')
        assert loss['complete'] and loss['replicates']==20000
        assert loss['training_runs']==loss['calibration_refits']==loss['AP_recomputations']==0
        assert loss['original_AP_caches_unchanged'] and not loss['seed_probability_pooling']
        assert loss['primary_sha256']==sha(BUNDLE/'aggregate/runs'/RUN/(db+'_results_v1/verified-results.json'))
        assert loss['secondary_sha256']==sha(BUNDLE/'aggregate/runs'/RUN/(db+'_secondary_v1/verified-results.json'))
        assert loss['source_sha256']==sha(BUNDLE/'source/current_analysis/scripts/cem_probability_intervals.py')
        for split in ('dev','test'):
            content=gzip.decompress((folder/(split+'_loss_differences.csv.gz')).read_bytes())
            draw=np.loadtxt(io.BytesIO(content),delimiter=',',skiprows=1)
            rows=[r for r in loss['rows'] if r['split']==split]
            assert draw.shape==(20000,8) and len(rows)==8
            assert content.decode().splitlines()[0]==','.join(str(r['endpoint'])+':'+r['form']+':'+r['metric'] for r in rows)
            for i,row in enumerate(rows):
                assert row['family']==24 and row['negative_favors_CEM']
                np.testing.assert_allclose(np.quantile(draw[:,i],[.025,.975]),row['ci95'],rtol=0,atol=1e-14)
                np.testing.assert_allclose(np.quantile(draw[:,i],[.05/48,1-.05/48]),row['adjusted_ci'],rtol=0,atol=1e-14)
                v=np.asarray(row['method_values']['CEM-Net'])-row['method_values'][row['baseline']]
                np.testing.assert_allclose(v,row['seed_differences'],rtol=0,atol=1e-12)
                np.testing.assert_allclose([np.mean(v),np.std(v,ddof=1)],[row['mean'],row['sample_SD']],rtol=0,atol=1e-12)
            loss_rows.extend(rows)
    assert len(loss_rows)==48 and all(sum(r['split']==s for r in loss_rows)==24 for s in ('dev','test'))
    added_run='cem_baseline_replacement_20261030'
    new_root=BUNDLE/'aggregate/runs'/added_run
    source_hashes={'mimiciii':'2cfcf36c30b8e471b692869e30f11c06872cd3ce9e3c90d2eeef6fcfa90cfe91',
                   'mimic':'d262f5b6ed2a24e25c55eb9511d0cf79803f06fc4ffb043afaefff700e665d36',
                   'eicu':'71ea9072a2a27be3501b28f38470c5360e96347e97594b23a6ad25e3721510eb'}
    new_results=[]
    for db,digest in source_hashes.items():
        folder=new_root/(db+'_results_v1')
        receipt=read(folder/'collection-receipt.json')
        assert receipt['exit']['returncode']==0 and receipt['receipt']['complete']
        for name,value in receipt['receipt']['files'].items():assert sha(folder/name)==value
        result=read(folder/'verified-results.json')
        assert result['complete'] and result['verified_new_fits']==result['verified_fixed_tests']==12
        assert result['replicates']==20000 and result['calibration_fits']==0
        assert not result['existing_AP_caches_recomputed']
        assert result['old_families_unchanged'] and result['secondary_selected_baselines_unchanged']
        assert result['prior_result_sha256']==sha(BUNDLE/'aggregate/runs'/RUN/(db+'_results_v1/verified-results.json'))
        assert len(result['fit_audits'])==12
        for job in result['fit_audits']:
            assert job['fit_exit']==job['test_exit']==job['test_updates']==0
            assert job['source_manifest_sha256']==digest
        snapshot=BUNDLE/'source/frozen_replacements_v1'/db
        assert sha(snapshot/'source_manifest.json')==digest
        assert read(snapshot/'build-receipt.json')['source_manifest_sha256']==digest
        sources=read(snapshot/'source_bundle.json');source_manifest=read(snapshot/'source_manifest.json')
        assert sources.keys()==source_manifest.keys()
        for name,value in source_manifest.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            assert Path(name).suffix not in ('.npz','.npy','.pt','.pth','.parquet','.pkl','.joblib')
            assert hashlib.sha256(base64.b64decode(sources[name])).hexdigest()==value,name
        new_results.append(result)
    combined=read(new_root/'paired_results_v1/verified-results.json')
    completed=read(new_root/'paired_results_v1/completion.json')
    assert completed['complete']
    for name,value in completed['files'].items():assert sha(new_root/'paired_results_v1'/name)==value
    assert combined['complete'] and combined['verified_new_fits']==combined['verified_fixed_tests']==36
    assert len(combined['paired'])==48 and len(combined['rows'])==144
    assert combined['rows']==[row for result in new_results for row in result['rows']]
    namespace={'np':np,'itertools':itertools}
    source=BUNDLE/'source/current_analysis/scripts/cem_icde72_significance.py'
    definitions=[node for node in ast.parse(source.read_text(encoding='utf8')).body
                 if isinstance(node,ast.FunctionDef) and node.name in ('holm','exact_signed_rank')]
    assert len(definitions)==2
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(source),'exec'),namespace)
    scores={(r['database'],r['split'],r['method'],r['endpoint'],r['seed']):r['AP'] for r in combined['rows']}
    for split in ('dev','test'):
        for candidate in ('CEM-Net','EMA_NRE_MLP'):
            family=[r for r in combined['paired'] if r['split']==split and r['candidate']==candidate]
            assert len(family)==12 and {(r['database'],r['baseline'],r['endpoint']) for r in family}=={
                (db,m,e) for db in source_hashes for m in ('XGBoost_TD','STraTS_supervised') for e in (0,1)}
            expected_holm=namespace['holm']([r['p_bootstrap_centered'] for r in family])
            for r,hp in zip(family,expected_holm):
                d=[scores[r['database'],split,candidate,r['endpoint'],s]-scores[r['database'],split,r['baseline'],r['endpoint'],s] for s in (48,49,50)]
                np.testing.assert_allclose(d,r['values'],rtol=0,atol=1e-12)
                np.testing.assert_allclose([np.mean(d),np.std(d,ddof=1)],[r['mean'],r['sample_SD']],rtol=0,atol=1e-12)
                assert r['family_count']==r['holm_family_count']==12
                assert r['p_bonferroni']==min(1.,12*r['p_bootstrap_centered'])
                assert r['p_holm_full_family']==hp and 1/20001<=r['p_bootstrap_centered']<=1
                assert r['wilcoxon']==namespace['exact_signed_rank'](d)
    five_seed_checks=verify_five_seed(namespace['exact_signed_rank'])
    manuscript=(BUNDLE/'paper_seed/TRACEHR_manuscript.tex').read_text(encoding='utf8')
    author_state='blank' if r'\author{}' in manuscript else 'present'
    assert manifest['author_information_in_manuscript']==author_state
    index=read(BUNDLE/'recipes/run_index.json')
    traced=0
    for host in ('mimic','eicu'):
        original={(r['database'],r['cell']):r for r in index[host]['original_CEM_fits']}
        records=index[host]['budget_provenance']['records']
        assert len(records)==len(original)
        assert len({(r['database'],r['seed'],r['endpoint']) for r in records})==len(records)
        for row in records:
            ref=original[(row['database'],f"{row['seed']}:{row['endpoint']}")]
            assert row['current_manifest_sha256']==ref['manifest_sha256']
            assert row['current_source_manifest_sha256']==ref['settings']['source_manifest_sha256']
            assert row['predecessor_epochs']==row['inherited_epoch_budget']==row['current_completed_epochs']==ref['settings']['epoch_budget']
            assert row['training_started_at']>=row['formal_released_at']
            assert row['predecessor_stopping_status']['stop']
            if row['database']=='eicu':
                assert row['inherited_epoch_budget']==256 and not row['predecessor_stopping_status']['early_stopping_enabled']
            else:
                assert row['boundary_replay'][-1]['stop'] and all(not r['stop'] for r in row['boundary_replay'][:-1])
            traced+=1
    assert traced==18
    diagnostic=read(BUNDLE/'verification/grud_existing_log_audit.json')
    records=diagnostic['records']
    assert len(records)==18 and len({(r['database'],r['seed'],r['endpoint']) for r in records})==18
    for r in records:
        history=r['history'];done=r['completion'];pred=r['prediction_summary']
        assert r['database'] in ('mimiciii','mimic','eicu') and r['seed'] in (48,49,50) and r['endpoint'] in (0,1)
        assert r['exit_record']['returncode']==0 and not r['log_error_lines']
        assert done['from_scratch'] and done['frozen_learnable_parameters']==0 and done['optimizer_updates']>0
        assert done['checkpoint_replay']['complete'] and done['checkpoint_replay']['max_probability_drift']==0
        assert done['epochs']==len(history) and [x['epoch'] for x in history]==list(range(1,len(history)+1))
        assert abs(history[done['best_epoch']-1]['dev_ap']-done['dev_ap'])<1e-12
        assert abs(max(x['dev_ap'] for x in history)-done['dev_ap'])<=1e-9
        assert np.isfinite([[x['loss'],x['dev_ap']] for x in history]).all()
        assert pred['count']>0 and 0<=pred['minimum']<=pred['mean']<=pred['maximum']<=1
        assert np.isfinite(pred['std']) and pred['std']>=0
        assert r['protocol']['patience']==10 and done['epochs']-done['best_epoch']>=10
        assert len(r['source_hashes'])==3 and len(r['file_hashes'])==6
        assert all(len(h)==64 and set(h)<=set('0123456789abcdef') for h in [*r['source_hashes'].values(),*r['file_hashes'].values()])
    return dict(verified_files=len(manifest['files']),fits=72,fixed_tests=72,fixed_stress_cases=144,
                calibrators=36,statistical_comparisons=744,hospital_paired_leaveouts=140,budget_provenance_records=traced,
                probability_loss_intervals=48,probability_intervals_reconstructed_from_aggregate_draws=True,
                grud_existing_log_records=len(records),grud_diagnostic_source='saved histories and prediction summaries; no rerun',
                additional_fits=36,additional_fixed_tests=36,additional_paired_comparisons=48,
                additional_families=4,additional_family_size=12,additional_calibrators=0,
                additional_source_snapshots_verified=3,**five_seed_checks,
                historical_cnn_snapshot=verify_cnn_comparison(),
                **verify_completed_sequence(namespace['exact_signed_rank']),
                **load('cem_matched_verification',BUNDLE/'source/verify_cem_matched_bundle.py').verify(BUNDLE),
                **load('cem_aggregation_verification',BUNDLE/'source/verify_cem_matched_bundle.py').verify_aggregation(BUNDLE),
                **load('cem_targeted_verification',BUNDLE/'source/verify_cem_matched_bundle.py').verify_targeted(BUNDLE),
                **verify_core_five_seed(),**verify_secondary_metrics())


def verify_cnn_comparison():
    """Check the completed two-CNN scope, with pending inference explicit."""
    import statistics as st
    folder=BUNDLE/'aggregate/runs/cem_sequence_comparators_20261102'
    subset=folder/'tcn_moderntcn_results_v1';done=read(subset/'completion.json');r=read(subset/'verified-results.json')
    assert done['complete'] and r['complete'] and sha(subset/'verified-results.json')==done['files']['verified-results.json']
    assert r['verified_new_fits']==r['verified_fixed_tests']==len(r['fit_audits'])==60
    assert r['models']==['TCN','ModernTCN'] and r['seeds']==[48,49,50,51,52]
    assert not r['patient_records_exported'] and not r['new_sampling'] and not r['existing_AP_caches_recomputed']
    assert r['original_method_retained'] and r['calibration_fits']==0
    expected={(db,m,s,e) for db in ('mimiciii','mimic','eicu') for m in r['models'] for s in r['seeds'] for e in (0,1)}
    assert {(a['database'],a['id'].split('_t')[0],int(a['id'].split('_s')[1]),int(a['id'].split('_t')[1].split('_s')[0])) for a in r['fit_audits']}==expected
    for a in r['fit_audits']:assert a['fit_exit']==a['test_exit']==a['test_updates']==0 and a['epochs']>=a['best_epoch']>0
    lookup={(z['database'],z['split'],z['method'],z['endpoint'],z['seed']):z for z in r['rows']}
    assert len(lookup)==len(r['rows'])==180
    for key,metrics in r['summaries'].items():
        db,split,method,e=key.split(':');e=int(e)
        for metric,summary in metrics.items():
            values=[lookup[db,split,method,e,s][metric] for s in r['seeds']]
            assert values==summary['values']
            assert abs(st.mean(values)-summary['mean'])<1e-12 and abs(st.stdev(values)-summary['sample_SD'])<1e-12
    for rel,digest in r['source_results'].items():assert sha(BUNDLE/'aggregate'/rel)==digest
    for db in ('mimiciii','mimic','eicu'):
        old=read(BUNDLE/'aggregate/runs/cem_five_seed_20261102'/(db+'_results_v1/verified-results.json'))
        for row in old['rows']:
            if row['method']=='CEM-Net':
                stored=lookup[db,row['split'],row['method'],row['endpoint'],row['seed']]
                for metric in ('AP','AUROC','Brier','log_loss'):assert stored[metric]==row[metric]
        src=BUNDLE/'source/frozen_sequence_v3'/db;mapping=read(src/'source_manifest.json');sources=read(src/'source_bundle.json')
        build=read(src/'build-receipt.json')
        assert sha(src/'source_manifest.json')==build['source_manifest_sha256'] and sources.keys()==mapping.keys()
        for name,h in mapping.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            assert hashlib.sha256(base64.b64decode(sources[name])).hexdigest()==h
        for model in r['models']:assert r['recipes'][db+':'+model]==build['contract']['recipes'][model]
    assert len(r['paired'])==8*len(r['paired_complete_databases'])
    assert set(r['pending_paired_databases'])=={'mimiciii','mimic','eicu'}-set(r['paired_complete_databases'])
    for row in r['paired']:
        db,split,e,m=row['database'],row['split'],row['endpoint'],row['baseline']
        values=[lookup[db,split,'CEM-Net',e,s]['AP']-lookup[db,split,m,e,s]['AP'] for s in r['seeds']]
        assert all(abs(a-b)<1e-12 for a,b in zip(values,row['values']))
        assert abs(st.mean(values)-row['mean'])<1e-12 and abs(st.stdev(values)-row['sample_SD'])<1e-12
        assert row['family_count']==30 and 1/20001<=row['p_bootstrap_centered']<=1
        assert row['p_bonferroni']==min(1.,30*row['p_bootstrap_centered'])
        assert row['adjusted_ci'][0]<=row['ci95'][0]<=row['ci95'][1]<=row['adjusted_ci'][1]
        full=read(folder/(db+'_cnn_results_v1/verified-results.json'))
        assert row in full['paired'] and full['replicates']==20000 and full['complete']
    return dict(cnn_completed_fits=60,cnn_fixed_tests=60,cnn_rows=180,cnn_paired_databases=r['paired_complete_databases'],
                cnn_pending_paired_databases=r['pending_paired_databases'],cnn_family_allowance=30,cnn_source_snapshots_verified=3)


def verify_completed_sequence(exact_signed_rank):
    """Verify complete families and frozen executions without patient data."""
    import statistics as st
    specs=[('cem_sequence_comparators_20261102','cnn_final_results_v1',('TCN','ModernTCN','LITEMV'),90,30,'frozen_sequence_v3','auto_cnn_quiet_v1'),
           ('cem_medformer_20261103','paired_results_v1',('Medformer',),30,6,'frozen_medformer_v1','auto_statistics_quiet_v1')]
    combined={};reports={}
    for run,final,models,fits,family,source,monitor in specs:
        folder=BUNDLE/'aggregate/runs'/run;path=folder/final
        done=read(path/'completion.json');result=read(path/'verified-results.json')
        assert done['complete'] and result['complete']
        for name,digest in done['files'].items():assert sha(path/name)==digest
        auto=read(folder/monitor/'completion.json')
        assert auto['complete'] and auto['statistics_complete'] and auto['aggregate_sha256']==sha(path/'verified-results.json')
        assert read(folder/monitor/'exit.json')['returncode']==read(folder/monitor/'launcher_exit.json')['returncode']==0
        for name,digest in read(folder/monitor/'attempt.json')['source_hashes'].items():
            assert sha(BUNDLE/'source/current_analysis/scripts'/name.split('/')[-1])==digest
        policy=read(folder/monitor/'runtime_policy.json')
        assert policy['poll_seconds']==600 and policy['all_other_control_flow_unchanged']
        for name,key in [('resume.py','wrapper_sha256'),('sitecustomize.py','hook_sha256'),('handoff.json','handoff_sha256')]:
            assert sha(BUNDLE/'source/monitor_quiet_v1'/name)==policy[key]
        assert result['verified_new_fits']==result['verified_fixed_tests']==len(result['fit_audits'])==fits
        assert result['seeds']==[48,49,50,51,52] and result['replicates']==20000 and result['family_count']==family
        assert result['old_results_unchanged'] and not result['existing_AP_caches_recomputed'] and result['calibration_fits']==0
        for item in [result]+([read(folder/'cnn_paired_results_v1/verified-results.json')] if family==30 else []):
            for raw,digest in item['inputs'].items():
                relative='runs/'+raw.replace('\\','/').split('/runs/',1)[1]
                assert sha(BUNDLE/'aggregate'/relative)==digest
        if family==30:
            assert result['retired_methods']==['FTTransformer'] and result['deferred_methods']==['TVNet']
            assert result['pending_family_methods']==[] and all(x['returncode']==0 for x in result['host_exits'].values())
        expected={(db,m,s,e) for db in ('mimiciii','mimic','eicu') for m in models for s in result['seeds'] for e in (0,1)}
        assert {(a['database'],a['id'].split('_t')[0],int(a['id'].split('_s')[1]),int(a['id'].split('_t')[1].split('_s')[0])) for a in result['fit_audits']}==expected
        for a in result['fit_audits']:
            assert a['fit_exit']==a['test_exit']==a['test_updates']==0 and a['epochs']>=a['best_epoch']>0
            method=a['id'].split('_t')[0]
            assert a['epochs']<={'TCN':50,'ModernTCN':100,'LITEMV':1500,'Medformer':100}[method]
            if method=='LITEMV':assert a['epochs']==1500
        lookup={(r['database'],r['split'],r['method'],r['endpoint'],r['seed']):r for r in result['rows']}
        expected_rows={(db,split,m,e,s) for db in ('mimiciii','mimic','eicu') for split in ('dev','test') for m in (*models,'CEM-Net') for e in (0,1) for s in result['seeds']}
        assert set(lookup)==expected_rows and len(lookup)==len(result['rows'])
        for key,row in lookup.items():
            if key in combined:assert combined[key]==row
            combined[key]=row
        for key,metrics in result['summaries'].items():
            db,split,method,e=key.split(':');e=int(e)
            for metric,summary in metrics.items():
                values=[lookup[db,split,method,e,s][metric] for s in result['seeds']]
                assert values==summary['values']
                assert abs(st.mean(values)-summary['mean'])<1e-12 and abs(st.stdev(values)-summary['sample_SD'])<1e-12
        for db in ('mimiciii','mimic','eicu'):
            collected=folder/(db+('_cnn_results_v1' if family==30 else '_results_v1'))
            receipt=read(collected/'collection-receipt.json');local=read(collected/'verified-results.json')
            assert receipt['exit']['returncode']==0 and receipt['receipt']['complete'] and local['complete']
            for name,digest in receipt['receipt']['files'].items():assert sha(collected/name)==digest
            assert local['verified_new_fits']==local['verified_fixed_tests']==fits//3
            assert local['source_manifest_sha256']==receipt['exit']['source_manifest_sha256']
            for row in local['rows']:assert lookup[db,row['split'],row['method'],row['endpoint'],row['seed']]==row
            old=read(BUNDLE/'aggregate/runs/cem_five_seed_20261102'/(db+'_results_v1/verified-results.json'))
            for row in old['rows']:
                if row['method']=='CEM-Net':
                    for metric in ('AP','AUROC','Brier','log_loss'):
                        assert lookup[db,row['split'],'CEM-Net',row['endpoint'],row['seed']][metric]==row[metric]
            src=BUNDLE/'source'/source/db;mapping=read(src/'source_manifest.json');sources=read(src/'source_bundle.json');build=read(src/'build-receipt.json')
            assert sha(src/'source_manifest.json')==build['source_manifest_sha256'] and sources.keys()==mapping.keys()
            for name,digest in mapping.items():
                assert not Path(name).is_absolute() and '..' not in Path(name).parts
                assert hashlib.sha256(base64.b64decode(sources[name])).hexdigest()==digest
            if family==6:
                recipe=build['contract']['recipes']['Medformer']
                assert recipe['batch']==recipe['micro']==recipe['eval_batch']==128 and recipe['epochs']==100 and recipe['patience']==10
        for split in ('dev','test'):
            rows=[r for r in result['paired'] if r['split']==split]
            assert len(rows)==len(models)*6
            assert {(r['database'],r['endpoint'],r['baseline']) for r in rows}=={(db,e,m) for db in ('mimiciii','mimic','eicu') for e in (0,1) for m in models}
            assert all(r['candidate']=='CEM-Net' for r in rows)
            pvalues=[r['p_bootstrap_centered'] for r in rows]+[1.]*(family-len(rows))
            adjusted=[0.]*family;previous=0.
            for rank,index in enumerate(sorted(range(family),key=lambda i:pvalues[i])):
                previous=max(previous,min(1.,(family-rank)*pvalues[index]));adjusted[index]=previous
            for i,row in enumerate(rows):
                db,e,m=row['database'],row['endpoint'],row['baseline']
                values=[lookup[db,split,'CEM-Net',e,s]['AP']-lookup[db,split,m,e,s]['AP'] for s in result['seeds']]
                assert all(abs(a-b)<1e-12 for a,b in zip(values,row['values']))
                assert abs(st.mean(values)-row['mean'])<1e-12 and abs(st.stdev(values)-row['sample_SD'])<1e-12
                assert row['family_count']==family and 1/20001<=row['p_bootstrap_centered']<=1
                assert row['p_bonferroni']==min(1.,family*row['p_bootstrap_centered']) and abs(row['p_holm']-adjusted[i])<1e-12
                assert row['wilcoxon']==exact_signed_rank(row['values'])
                assert row['adjusted_ci'][0]<=row['ci95'][0]<=row['ci95'][1]<=row['adjusted_ci'][1]
                source_result=read(folder/(db+('_cnn_results_v1' if family==30 else '_results_v1'))/'verified-results.json')
                original=next(r for r in source_result['paired'] if (r['split'],r['baseline'],r['endpoint'])==(split,m,e))
                assert all(original[k]==v for k,v in row.items() if k!='p_holm')
        reports['cnn' if family==30 else 'medformer']=dict(fits=fits,fixed_tests=fits,comparisons=len(result['paired']),family_allowance=family,source_snapshots=3)
    assert len(combined)==300
    readiness=read(BUNDLE/'aggregate/runs/cem_medformer_20261103/mimiciii_results_v1/start-receipt.json')['global_readiness']
    assert all(h['ready'] for h in readiness.values())
    assert sha(BUNDLE/'source/medformer_early_statistics_v1/collect.py')=='024ac0451a1a4336312181f6f85b54ed75c0d98174aeb895dfed77c9b1207045'
    return dict(cnn_completed_fits=90,cnn_fixed_tests=90,medformer_completed_fits=30,medformer_fixed_tests=30,
                sequence_rows=300,sequence_families=reports,cnn_pending_paired_databases=[],medformer_pending_paired_databases=[],
                bootstrap_resampled=False,old_CEM_scores_unchanged=True)


def synthetic_interface():
    import numpy as np
    src=BUNDLE/'source/current_analysis/scripts'
    event=load('cem_repro_events',src/'tracehr_exact_statistics.py')
    values=load('cem_repro_values',src/'tracehr_value_features.py')
    types=np.array([0,1,0]);times=np.array([1.,12.,23.]);sources=np.array([0,1,0])
    full=event.EventStatistics(3,('a','b'),24).add(types,times,sources).features()
    left=event.EventStatistics(3,('a','b'),24).add(types[:1],times[:1],sources[:1])
    right=event.EventStatistics(3,('a','b'),24).add(types[1:],times[1:],sources[1:])
    np.testing.assert_allclose(left.merge(right).features(),full,rtol=0,atol=1e-7)
    assert len(full)==4*3+2*2+4
    np.testing.assert_allclose(full[:3],np.log1p([2,1,0]),rtol=0,atol=1e-7)
    try:event.EventStatistics(3,('a','b'),24).add([0],[25.],[0])
    except ValueError:pass
    else:raise AssertionError('Post-landmark event was accepted')
    np.testing.assert_array_equal(values.simple_summary([],[],[]),np.zeros(11,np.float32))
    np.testing.assert_array_equal(values.duration_summary([],[]),np.zeros(15,np.float32))
    np.testing.assert_allclose(values.duration_summary([0.],[2.]).reshape(3,5),np.tile([1.,2.,4.,1.,0.],(3,1)),rtol=0,atol=1e-7)
    assert not np.array_equal(values.duration_summary([0.],[2.]),values.duration_summary([20.],[2.]))
    try:values.duration_summary([1.,1.],[0.,1.])
    except AssertionError:pass
    else:raise AssertionError('Unaggregated duplicate time accepted')
    empty=event.EventStatistics(3,('a','b'),24).features()
    assert empty[-4]==np.float32(1/512) and empty[-1]==1
    return ['event_schema_and_counts','partition_merge','post_landmark_rejected',
            'missing_lab_indicator_and_slots','closed_form_duration','same_latest_different_duration',
            'duplicate_times_require_upstream_aggregation','documented_empty_record_floor']


def render(output):
    assert not output.exists(),'Use a new output directory; existing paper/results are never overwritten'
    assert not BUNDLE.is_relative_to(output) and not output.is_relative_to(BUNDLE/'aggregate')
    output.mkdir(parents=True)
    os.environ['MPLCONFIGDIR']=str(output/'cache/matplotlib')
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    checks=verify()
    checks['synthetic_interface']=synthetic_interface()
    shutil.copytree(BUNDLE/'aggregate/runs',output/'runs')
    paper=output/'paper/ICDE2027'
    shutil.copytree(BUNDLE/'paper_seed',paper)
    renderer=load('cem_repro_renderer',BUNDLE/'source/paper_integration.py')
    renderer.ROOT=output;renderer.PAPER=paper;renderer.TABLES=paper/'tables/cem_icde72_20261029'
    renderer.primary();renderer.secondary();renderer.response_and_influence()
    matched=load('cem_matched_renderer',BUNDLE/'source/report_cem_matched_controls.py')
    matched.render(output,paper/'tables/cem_icde72_20261029')
    secondary=load('cem_secondary_renderer',BUNDLE/'source/report_cem_secondary_metrics.py')
    secondary.render(output,paper/'tables/cem_icde72_20261029')
    return check_render(output,checks)


def check_render(output,checks=None):
    if checks is None:
        checks=verify();checks['synthetic_interface']=synthetic_interface()
    paper=output/'paper/ICDE2027'
    names=[]
    for p in (paper/'tables/cem_icde72_20261029').glob('*.tex'):
        original=BUNDLE/'paper_seed/tables/cem_icde72_20261029'/p.name
        assert p.read_bytes()==original.read_bytes(),p.name
        names.append(p.name)
    assert set(names)=={'calibration.tex','cost.tex','development.tex','evidence.tex','hospital.tex',
                        'hospital_pairs.tex','lgbm_recipe.tex','paired.tex','pipeline.tex','primary.tex','review_budget.tex','cnn_comparison.tex',
                        'matched_controls.tex','matched_cost.tex','aggregation_stress.tex','targeted_aggregation.tex','secondary_metrics.tex'}
    for filename in ('interaction.tex','context.tex','ablation.tex'):
        relative=Path('tables/cem_generality_20261028')/filename
        assert (paper/relative).read_bytes()==(BUNDLE/'paper_seed'/relative).read_bytes()
        names.append('cem_generality_20261028/'+filename)
    supplied=[]
    for relative in (Path('tables/cem_review_20261025/cohort.tex'),):
        assert (paper/relative).read_bytes()==(BUNDLE/'paper_seed'/relative).read_bytes()
        supplied.append(relative.as_posix())
    import numpy as np
    from PIL import Image
    figure_names=('fixed_stress','clinical_subgroups','probability_quality','observation_response','hospital_component_influence',
                  'coverage_sensitivity','probability_losses')
    for name in figure_names:
        np.testing.assert_array_equal(
            np.array(Image.open(paper/f'figures/cem_icde72_20261029/{name}.png')),
            np.array(Image.open(BUNDLE/f'paper_seed/figures/cem_icde72_20261029/{name}.png')))
    parameter_renderer=load('cem_parameter_counts',BUNDLE/'source/paper_integration.py')
    parameter_renderer.ROOT=output
    composition=parameter_renderer.parameter_composition()
    checks.update(parameter_composition=composition,
                  tables_byte_identical=sorted(names),supplied_tables_byte_identical=supplied,
                  figures_pixel_identical=list(figure_names),
                  interaction_bootstrap_draws={'NRE_x_CMM':4000,'raw_x_encoded_modulation':4000},
                  activation_control_retained_in_aggregate=True,
                  training_executed=False,clinical_records_loaded=False,bootstrap_resampled=False,
                  source_archive_import_test='Only the two pure evidence-interface modules; not an end-to-end training replay')
    rows=[]
    for db in ('mimiciii','mimic','eicu'):
        rows += [dict(database=db,**row) for row in read(output/'runs'/RUN/(db+'_results_v1/verified-results.json'))['rows']]
    with (output/'per_seed_metrics.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['database','arm','split','method','seed','endpoint','AP','AUROC','Brier','log_loss'])
        writer.writeheader();writer.writerows(rows)
    added=read(output/'runs/cem_baseline_replacement_20261030/paired_results_v1/verified-results.json')
    with (output/'per_seed_metrics.csv').open('a',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['database','arm','split','method','seed','endpoint','AP','AUROC','Brier','log_loss'])
        writer.writerows(dict(arm='random',**row) for row in added['rows'] if row['method'] in ('XGBoost_TD','STraTS_supervised'))
    sequence=parameter_renderer.sequence_results()
    with (output/'sequence_five_seed_metrics.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['database','split','method','seed','endpoint','AP','AUROC','Brier','log_loss'])
        writer.writeheader();writer.writerows(sequence['rows'])
    summary=[]
    for key,metrics in sequence['summaries'].items():
        db,split,method,endpoint=key.split(':')
        for metric,value in metrics.items():
            summary.append(dict(database=db,split=split,method=method,endpoint=int(endpoint),metric=metric,mean=value['mean'],sample_SD=value['sample_SD']))
    with (output/'sequence_five_seed_summary.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    paired=[]
    for row in sequence['paired']:
        record={k:row[k] for k in ('database','split','candidate','baseline','endpoint','mean','sample_SD','family_count','p_bootstrap_centered','p_bonferroni','p_holm')}
        record.update({f'seed{s}':v for s,v in zip(range(48,53),row['values'])})
        record.update(low95=row['ci95'][0],high95=row['ci95'][1],low_adjusted=row['adjusted_ci'][0],high_adjusted=row['adjusted_ci'][1],wilcoxon_two_sided=row['wilcoxon']['p_two_sided'])
        paired.append(record)
    with (output/'sequence_paired_statistics.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    (output/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
    return checks


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['verify','render'])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.mode=='render':
        if args.output is None:parser.error('--output is required for render')
        result=render(args.output.resolve())
    else:result=verify()
    print(json.dumps(result,indent=2))
