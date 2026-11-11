"""Render the authorized ICDE evidence from verified aggregate artifacts only."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics as st
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = 'cem_icde_evidence_20261028'
PAPER = ROOT/'paper/ICDE2027'
TABLES = PAPER/'tables/cem_icde72_20261029'
DBS = ('mimiciii', 'mimic', 'eicu')
NAMES = dict(mimiciii='MIMIC-III', mimic='MIMIC-IV', eicu='eICU')
LABELS = {'TabM_mini_PLE':'TabM-mini+PLE','TabM_PLE':'TabM+PLE',
          'LightGBM_full':'LightGBM','GRU_D':'GRU-D','mTAND_Enc':'mTAND encoder',
          'XGBoost_TD':'XGBoost-TD','STraTS_supervised':r'STraTS (ss$^{-}$)',
          'EMA_NRE_MLP':'CEM-Net (ReLU)', 'CEM_hospital':'CEM-Net',
          'TabM_mini_PLE_hospital':'TabM-mini+PLE','LightGBM_hospital':'LightGBM'}
SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
READ = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
CNN_RUN = 'cem_sequence_comparators_20261102'
CNN_MODELS = ('TCN', 'ModernTCN')
CNN_SNAPSHOT = 'tcn_moderntcn_results_v1'
MED_RUN = 'cem_medformer_20261103'
CORE_RUN = 'cem_core_five_seed_20261107'
CORE_SHA = 'c911fa18764f0a75b972bfef05d86b0b1e50f4845386c249d20585487d2e56b0'


def core_results():
    """Read completed five-seed aggregates; preserve all three-seed records."""
    path = ROOT/'runs'/CORE_RUN/'paired_results_v1/verified-results.json'
    assert SHA(path) == CORE_SHA
    result = READ(path)
    assert result['complete'] and result['seeds'] == [48,49,50,51,52]
    assert (result['verified_cells'], result['new_fits'], result['reused_saved_fit']) == (204,203,1)
    assert len(result['summaries']) == 240 and len(result['paired']) == 108
    assert not result['original_results_modified']
    for row in result['summaries']:
        for metric, value in row['five_seed'].items():
            assert value['values'][:3] == row['three_seed'][metric]['values']
            assert abs(st.mean(value['values'])-value['mean']) < 1e-12
            assert abs(st.stdev(value['values'])-value['sample_SD']) < 1e-12
    return result


def core_pair(result, db, endpoint, family, name):
    rows = [r for r in result['paired'] if
            (r['database'],r['endpoint'],r['family'],r['name']) == (db,endpoint,family,name)]
    assert len(rows) == 1
    return rows[0]['five_seed']


def ablation_table(result):
    summaries = {(r['database'],r['method'],r['endpoint']):r['five_seed']['AP']
                 for r in result['summaries'] if r['arm']=='random' and r['split']=='test'}
    rows = []
    for method, title in [('CEM_noEMA','w/o EMA'),('EMA_noP','w/o learned responses'),
                         ('EMA_noM','w/o member modulation'),
                         ('EMA_no_NRE_CMM','w/o both components'),
                         ('EMA_no_duration','w/o duration'),('CEM-Net','CEM-Net')]:
        rows.append(title+' & '+' & '.join('$'+ms(summaries[db,method,e]).replace(r'\,','')+'$'
                    for e in (1,0) for db in DBS)+r' \\')
    gains=[]
    for e in (1,0):
        for db in DBS:
            v=core_pair(result,db,e,'response_modulation_simple','modulation_with_responses')
            gains.append(r'\shortstack[r]{'+f"${v['mean']:+.4f}$"+r'\\'+
                         f"$[{v['adjusted_ci'][0]:.4f},{v['adjusted_ci'][1]:.4f}]$"+'}')
    rows += [r'\midrule',r'Modulation gain & '+' & '.join(gains)+r' \\']
    for title,family,name in [(r'Response $\times$ modulation','response_modulation_interaction','interaction'),
                              (r'Direct $\times$ encoded','raw_encoded_interaction','raw-by-encoded modulation')]:
        cells=[]
        for e in (1,0):
            for db in DBS:
                if family=='raw_encoded_interaction' and db!='eicu':
                    cells.append('---');continue
                v=core_pair(result,db,e,family,name)
                cells.append(r'\shortstack[r]{'+f"${v['mean']:+.4f}$"+r'\\'+
                             f"$[{v['adjusted_ci'][0]:.4f},{v['adjusted_ci'][1]:.4f}]$"+'}')
        rows.append(title+' & '+' & '.join(cells)+r' \\')
    text = [r'\begin{table*}[tp]',r'\centering\small',r'\setlength{\tabcolsep}{1.6pt}',
            r'\caption{Five-seed component ablations: AP mean $\pm$ sample SD. Both components denotes responses and modulation. Retained-parameter initialization, sampling and endpoint budgets match the full model. Modulation gain retains responses. The last three rows give patient-cluster intervals from 4,000 draws: 99.7917\% for 24 simple effects, 99.1667\% for six response--modulation interactions, and 97.5\% for two direct--encoded interactions.}',
            r'\label{tab:ablation}\label{tab:interaction}',r'\begin{tabular}{lrrrrrr}',r'\toprule',
            r'& \multicolumn{3}{c}{Prolonged LOS} & \multicolumn{3}{c}{Mortality} \\',
            r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
            r'Method & MIMIC-III & MIMIC-IV & eICU & MIMIC-III & MIMIC-IV & eICU \\',r'\midrule',
            *rows,r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (PAPER/'tables/cem_generality_20261028/ablation.tex').write_text('\n'.join(text)+'\n',encoding='utf8')


def sequence_results():
    """Read completed, separately corrected families; never start a writer."""
    specifications=[(CNN_RUN,'cnn_final_results_v1',('TCN','ModernTCN','LITEMV'),90,30)]
    if (ROOT/'runs'/MED_RUN/'paired_results_v1/completion.json').exists():
        specifications.append((MED_RUN,'paired_results_v1',('Medformer',),30,6))
    merged=dict(models=[],rows=[],paired=[],summaries={},families={})
    unique={}
    for run,folder,models,fits,family in specifications:
        path=ROOT/'runs'/run/folder;done=READ(path/'completion.json');result=READ(path/'verified-results.json')
        assert done['complete'] and result['complete']
        for name,digest in done['files'].items():assert SHA(path/name)==digest
        assert result['verified_new_fits']==result['verified_fixed_tests']==fits
        assert result['seeds']==[48,49,50,51,52] and result['replicates']==20000
        assert result['family_count']==family and result['old_results_unchanged']
        assert not result['existing_AP_caches_recomputed'] and result['calibration_fits']==0
        assert len(result['paired'])==len(models)*12 and all(r['p_holm'] is not None for r in result['paired'])
        assert len(result['fit_audits'])==fits
        assert all(a['fit_exit']==a['test_exit']==a['test_updates']==0 for a in result['fit_audits'])
        merged['models'].extend(models);merged['paired'].extend(result['paired']);merged['families'][run]=family
        for row in result['rows']:
            key=(row['database'],row['split'],row['method'],row['endpoint'],row['seed'])
            if key in unique:assert unique[key]==row
            unique[key]=row
        for key,summary in result['summaries'].items():
            if key in merged['summaries']:assert merged['summaries'][key]==summary
            merged['summaries'][key]=summary
    merged['rows']=list(unique.values())
    assert len(unique)==60*(len(merged['models'])+1)
    for db in DBS:
        old=READ(ROOT/'runs/cem_five_seed_20261102'/(db+'_results_v1/verified-results.json'))
        for row in old['rows']:
            if row['method']!='CEM-Net':continue
            saved=unique[db,row['split'],'CEM-Net',row['endpoint'],row['seed']]
            for metric in ('AP','AUROC','Brier','log_loss'):assert saved[metric]==row[metric]
    return merged



def label(m): return LABELS.get(m,m)
def ms(r): return f"{r['mean']:.4f}\\,\\pm\\,{r['sample_SD']:.4f}"
def ci(r):
    edge=lambda x: f'{x:+.6f}' if 0<abs(x)<.00005 else f'{x:+.4f}'
    return f"${r['mean']:+.4f}\\ [{edge(r['adjusted_ci'][0])},{edge(r['adjusted_ci'][1])}]$"
def table(name, caption, ident, headers, rows, columns=None, wide=True, output_dir=None):
    env='table*' if wide else 'table'
    size=r'\centering\small' if name in ('paired','lgbm_recipe','interaction','hospital') else r'\centering\footnotesize'
    text=[f'\\begin{{{env}}}[tp]',size,r'\setlength{\tabcolsep}{3pt}',
          '\\caption{'+caption+'}', '\\label{tab:'+ident+'}',
          '\\begin{tabular}{'+(columns or 'l'+'r'*(len(headers)-1))+'}',r'\toprule',
          ' & '.join(headers)+r' \\',r'\midrule']
    text+=[' & '.join(row)+r' \\' for row in rows]
    text += [r'\bottomrule',r'\end{tabular}',f'\\end{{{env}}}']
    ((output_dir or TABLES)/(name+'.tex')).write_text('\n'.join(text)+'\n',encoding='utf-8')

def primary():
    fixed={db:READ(ROOT/'runs'/RUN/(db+'_results_v1')/'verified-results.json') for db in DBS}
    hashes={db:SHA(ROOT/'runs'/RUN/(db+'_results_v1')/'verified-results.json') for db in DBS}
    assert hashes==dict(mimiciii='3868b55d69cadafb5593a461ac24a90e7775623db2767d73da969fc7d083a308',
                       mimic='a718fac0278648004ad9aead05aba3047c6b6cd4ede206da8c2cff68f187dc8f',
                       eicu='9d2a99ad4c57a4e2842b30d3222265fd6e736f84edf68edbae51978bd946ba2d')
    assert sum(x['verified_new_fits'] for x in fixed.values())==72
    assert sum(x['verified_fixed_tests'] for x in fixed.values())==72
    assert sum(x['calibration_fits'] for x in fixed.values())==36
    for db,x in fixed.items():
        assert x['complete'] and not x['main_method_switched']
        for a in x['worker_audits']:
            assert a['fit_exit']==a['test_exit']==a['test_updates']==0
        for r in x['summaries'].values():
            for v in r.values():
                assert len(v['values'])==3
                assert abs(st.mean(v['values'])-v['mean'])<1e-12
                assert abs(st.stdev(v['values'])-v['sample_SD'])<1e-12
    TABLES.mkdir(parents=True,exist_ok=True)
    added_root=ROOT/'runs/cem_baseline_replacement_20261030'
    added=READ(added_root/'paired_results_v1/verified-results.json')
    added_done=READ(added_root/'paired_results_v1/completion.json')
    assert added_done['complete'] and added['complete'] and added['all_databases']
    assert SHA(added_root/'paired_results_v1/verified-results.json')==added_done['files']['verified-results.json']
    assert added['verified_new_fits']==added['verified_fixed_tests']==36 and len(added['paired'])==48
    # Validate the historical aggregates before rendering the completed five-seed pool.
    display={db:dict(fixed[db]['summaries']) for db in DBS}
    for db in DBS:
        new=READ(added_root/(db+'_results_v1')/'verified-results.json')
        assert new['complete'] and new['prior_result_sha256']==hashes[db]
        assert not new['existing_AP_caches_recomputed'] and new['calibration_fits']==0
        for key,value in new['summaries'].items():
            target='random:'+key
            if target in display[db]:
                assert abs(display[db][target]['AP']['mean']-value['AP']['mean'])<1e-12
            else:display[db][target]=value
    core=core_results()
    display={db:{} for db in DBS}
    for row in core['summaries']:
        method='TabM-mini' if row['method']=='TabM_mini' else row['method']
        display[row['database']][f"{row['arm']}:{row['split']}:{method}:{row['endpoint']}"]=row['five_seed']
    cnn=sequence_results()
    for db in DBS:
        for split in ('dev','test'):
            for e in (0,1):
                assert display[db][f'random:{split}:CEM-Net:{e}']==cnn['summaries'][f'{db}:{split}:CEM-Net:{e}']
    methods=['SeFT','mTAND_Enc','STraTS_supervised','RealMLP','TabM','TabM-mini','TabM_mini_PLE','TabM_PLE','LightGBM_full','XGBoost_TD','CEM-Net']
    for split,name in [('test','primary'),('dev','development')]:
        old=(PAPER/'tables/cem_generality_20261028'/(name+'.tex')).read_text()
        # The paper's paired comparison uses the contemporary pool. Historical
        # records remain unchanged in their existing table and the full report.
        if split=='test':
            prefix=old[:old.index(r'\multicolumn{7}{l}{\textit{Historical')]
        else:
            prefix=old[:old.index('RealMLP &')]
        caption=('Five-seed predictive AP (mean $\\pm$ sample SD; seeds 48--52). Bold/underlined means rank all models in each column. Hourly-model second lines give CEM-Net-minus-comparator AP intervals: 99.8333\\% for CNNs and 99.1667\\% for Medformer.' if split=='test'
                 else r'Five-seed validation AP for the comparison pool in Table~\ref{tab:primary}.')
        start=prefix.index(r'\caption{');end=prefix.index('\n',start)
        prefix=prefix[:start]+'\\caption{'+caption+'}'+prefix[end:]
        if split=='test':
            prefix=prefix.replace(r'\centering\footnotesize',r'\centering\fontsize{8.5}{10.2}\selectfont')
            prefix=prefix.replace(r'\centering\small',r'\centering\fontsize{8.5}{10.2}\selectfont')
            prefix=re.sub(r'\\setlength\{\\tabcolsep\}\{[^}]+\}',r'\\setlength{\\tabcolsep}{1.4pt}',prefix)
            prefix=prefix.replace(r'& \multicolumn{2}{c}{MIMIC-III} & \multicolumn{2}{c}{MIMIC-IV} & \multicolumn{2}{c}{eICU}',
                                  r'& \multicolumn{3}{c}{Prolonged LOS} & \multicolumn{3}{c}{Mortality}')
            prefix=prefix.replace(r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}')
            prefix=prefix.replace('Method & Mortality & LOS & Mortality & LOS & Mortality & LOS',
                                  'Method & MIMIC-III & MIMIC-IV & eICU & MIMIC-III & MIMIC-IV & eICU')
        rows=[]
        for m in methods:
            if split=='test' and m in ('SeFT','RealMLP'):
                heading='Observation-level comparators: values, times and masks' if m=='SeFT' else 'Common clinical evidence profile: model-specific preprocessing and training'
                if m=='RealMLP': rows.append(r'\midrule')
                rows.append(r'\multicolumn{7}{l}{\textit{'+heading+r'}} \\')
            cells=[]
            for db in DBS:
                for e in (0,1):
                    k=f'random:{split}:{m}:{e}'
                    v=display[db][k]['AP']; formatted=ms(v).replace(r'\,','') if split=='test' else ms(v)
                    best=sorted([display[db][f'random:{split}:{x}:{e}']['AP']['mean'] for x in methods]+
                                [cnn['summaries'][f'{db}:{split}:{x}:{e}']['AP']['mean'] for x in cnn['models']],reverse=True)
                    if v['mean']==best[0]: formatted=r'\mathbf{'+formatted+'}'
                    elif v['mean']==best[1]: formatted=r'\underline{'+formatted+'}'
                    cells.append('$'+formatted+'$')
            if split=='test':cells=[cells[i] for i in (1,3,5,0,2,4)]
            rows.append(label(m)+' & '+' & '.join(cells)+r' \\')
        rows += [r'\midrule',r'\multicolumn{7}{l}{\textit{Hourly sequence models}} \\']
        for m in cnn['models']:
            cells=[]
            for db in DBS:
                for e in (0,1):
                    v=cnn['summaries'][f'{db}:{split}:{m}:{e}']['AP'];formatted=ms(v).replace(r'\,','') if split=='test' else ms(v)
                    means=sorted([display[db][f'random:{split}:{x}:{e}']['AP']['mean'] for x in methods]+
                                 [cnn['summaries'][f'{db}:{split}:{n}:{e}']['AP']['mean'] for n in cnn['models']],reverse=True)
                    if v['mean']==means[0]:formatted=r'\mathbf{'+formatted+'}'
                    elif v['mean']==means[1]:formatted=r'\underline{'+formatted+'}'
                    cell='$'+formatted+'$'
                    if m!='CEM-Net':
                        pair=next(r for r in cnn['paired'] if (r['database'],r['split'],r['baseline'],r['endpoint'])==(db,split,m,e))
                        interval=f"$[{pair['adjusted_ci'][0]:+.4f},{pair['adjusted_ci'][1]:+.4f}]$"
                        cell=r'\shortstack[r]{'+cell+r'\\'+interval+'}'
                    cells.append(cell)
            if split=='test':cells=[cells[i] for i in (1,3,5,0,2,4)]
            rows.append(m+' & '+' & '.join(cells)+r' \\')
        (TABLES/(name+'.tex')).write_text(prefix+'\n'.join(rows)+'\n'+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'+r'\end{table*}'+'\n',encoding='utf-8')
    cnn_comparison(cnn)
    # The displayed table uses held-out scores. Development scores remain a
    # separate artifact; the underlying results and selection rules are fixed.
    testfile=ROOT/'runs'/RUN/'statistical_tests_v1/verified-results.json'
    testrows=READ(testfile)['rows'] if testfile.exists() else None
    five_root=ROOT/'runs/cem_five_seed_20261102/paired_results_v1'
    five=READ(five_root/'verified-results.json')
    five_done=READ(five_root/'completion.json')
    assert five_done['complete'] and five['complete']
    assert SHA(five_root/'verified-results.json')==five_done['files']['verified-results.json']
    assert five['verified_new_fits']==five['verified_fixed_tests']==30
    assert five['seeds']==[48,49,50,51,52] and five['replicates']==20000
    assert five['old_three_seed_results_unchanged'] and not five['existing_AP_caches_recomputed']
    assert five['calibration_fits']==0 and len(five['paired'])==18
    paired=[];factor=[]
    for db in DBS:
        for e in (0,1):
            ref=fixed[db]['strongest']['dev'][str(e)]['validation_selected']
            paired.append([r'\multicolumn{3}{l}{\textit{'+NAMES[db]+(' / Mortality' if e==0 else ' / LOS')+r'}}'])
            refs=[ref]
            leader=fixed[db]['strongest']['test'][str(e)]['mean_strongest']
            if leader!=ref:refs.append(leader)
            for rname in refs:
                p=next(x for x in five['paired'] if (x['database'],x['split'],x['endpoint'],x['baseline'])==(db,'test',e,rname))
                assert p['candidate']=='CEM-Net' and p['family_count']==54 and len(p['values'])==5
                assert abs(st.mean(p['values'])-p['mean'])<1e-12
                paired.append([label(rname)+(r'$\dagger$' if rname!=ref else ''),ci(p),f"{p['p_bonferroni']:.3g}"])
            row=[NAMES[db]+(' / Mort.' if e==0 else ' / LOS')]
            for s in ('test',):
                v={name:core_pair(core,db,e,'evidence_original_recipe',name)
                   for name in ('CEM_full_minus_basic','baseline_full_minus_basic','interaction')}
                row += [f"${v['CEM_full_minus_basic']['mean']:+.4f}$",f"${v['baseline_full_minus_basic']['mean']:+.4f}$",ci(v['interaction'])]
            factor.append(row)
    table('paired',r'Five-seed AP differences: CEM-Net minus baseline. References come from the three-seed comparison: unmarked, validation-selected; $\dagger$, additional observed-mean leader. Brackets show 99.9074\% patient-cluster intervals; $p_{\mathrm{adj}}$ uses Bonferroni multiplicity 54.','paired',
          ['Reference',r'$\Delta$AP [interval]',r'$p_{\mathrm{adj}}$'],paired,'lrr',wide=False)
    # Additional test-mean leaders are descriptive references from the same
    # frozen 54-comparison family, never replacements for validation selection.
    table('evidence','Added temporal evidence with five seeds: mean AP changes for CEM-Net ($\\Delta C$), baseline ($\\Delta B$), and interaction $I_E=\\Delta C-\\Delta B$. Brackets show 99.7222\\% patient-cluster intervals (18 contrasts). Baselines are TabM-mini for MIMIC-III LOS, TabM-mini+PLE for eICU, and TabM+PLE otherwise.','evidence',
          ['Task',r'$\Delta C$',r'$\Delta B$','$I_E$ [interval]'],factor,'@{}lrrr@{}',wide=False)
    hp=[]
    for m in ('TabM_mini_PLE_hospital','LightGBM_hospital'):
        for e in (0,1):
            hp.append([label(m),'Mortality' if e==0 else 'LOS']+[ci(fixed['eicu']['paired'][s]['hospital'][m][str(e)]) for s in ('dev','test')])
    table('hospital_pairs','Hospital-held-out CEM-Net minus each comparator, with 98.75\\% patient-connected hospital-component intervals for four contrasts per partition.','hospital-pairs',
          ['Comparator','Endpoint','Validation difference [interval]','Test difference [interval]'],hp,'llrr')
    # The hospital question concerns comparative performance under held-out
    # hospitals. Absolute AP/SD remains in the unchanged aggregate records.
    hospital_table(core)
    ablation_table(core)
    # Display labels change; the response-by-modulation analysis retains its frozen keys and draws.
    # Activation-control records remain in the full aggregate artifact.
    interaction_rows=[[r'\multicolumn{2}{l}{\textit{Responses $\times$ member modulation: 99.1667\% intervals}}']]
    mechanism_hashes=dict(mimiciii='1b10927a33e4ac024c792fe5e21f130f1ca558858bba1b7615f525c22e9b816c',
                          mimic='87cccc8fe8efde6700eb8887616c60dd625716ee66581aaf5b41f68534621f67',
                          eicu='6a15ffdb1d47195771d18e78db0839f20efc5869fd651f5bc667cac7400347cd')
    for db in DBS:
        mechanism_path=ROOT/'runs/cem_evidence_completion_20261025/nre_cmm_interaction_v1'/db/'interaction.json'
        assert SHA(mechanism_path)==mechanism_hashes[db]
        mechanism=READ(mechanism_path)
        assert mechanism['complete'] and mechanism['bootstrap']['replicates']==4000
        assert fixed[db]['bootstrap']['replicates']==20000
        for e in (0,1):
            for split in ('test',):
                n=core_pair(core,db,e,'response_modulation_interaction','interaction')
                r=fixed[db]['paired'][split]['MLP_minus_CEM'][str(e)]
                old_r=mechanism['MLP_minus_CEM'][split][str(e)]
                assert r['family_count']==6
                assert abs(r['mean']-old_r['mean'])<1e-12 and r['positive_seeds']==old_r['above_zero']
                interaction_rows.append([NAMES[db]+(' / Mort.' if e==0 else ' / LOS'),ci(n)])
    interaction_rows.append([r'\multicolumn{2}{l}{\textit{Direct $\times$ encoded modulation: 97.5\% intervals}}'])
    for e in (0,1):
        r=core_pair(core,'eicu',e,'raw_encoded_interaction','raw-by-encoded modulation')
        interaction_rows.append(['eICU'+(' / Mort.' if e==0 else ' / LOS'),
                                 ci(r)])
    table('interaction',r'Five-seed component and path interactions in AP. Intervals use 4,000 patient-cluster draws with Bonferroni correction within each panel.',
          'interaction',['Task',r'Interaction [interval]'],
          interaction_rows,'lr',wide=False,output_dir=PAPER/'tables/cem_generality_20261028')
    context=READ(ROOT/'runs/cem_observation_context_20261027/eicu_results_v1/verified-results.json')
    assert context['complete'] and context['fixed_tests_zero_updates'] and not context['main_method_changed']
    context_table(context)
    cal=[]
    for db in DBS:
        for e in (0,1):
            ref=fixed[db]['strongest']['dev'][str(e)]['validation_selected']
            values={}
            for m in ('CEM-Net',ref):
                rs=[x for x in fixed[db]['calibration'] if x['split']=='test' and x['method']==m and x['endpoint']==e]
                assert len(rs)==3
                values[m]={k:f"{st.mean(x['raw'][k] for x in rs):.4f}$\\to${st.mean(x['Platt'][k] for x in rs):.4f}" for k in ('Brier','log_loss','ECE15')}
                values[m]['AUROC']=f"{st.mean(x['raw']['AUROC'] for x in rs):.4f}"
            c=values['CEM-Net'];b=values[ref]
            cal.append([NAMES[db]+(' / Mort.' if e==0 else ' / LOS'),c['AUROC']+'/'+b['AUROC'],c['Brier'],c['log_loss'],c['ECE15'],b['ECE15']])
    table('calibration','Test probability quality: descriptive means. C/B denote CEM-Net / validation-selected baseline; arrows show raw $\\to$ validation-fitted Platt calibration. Lower Brier, log loss and ECE are better; AUROC is raw.','calibration',
          ['Database / endpoint','AUROC C/B','C: Brier','C: log loss','C: ECE15','B: ECE15'],cal,'lrrrrr')
    table('lgbm_recipe','Baseline architectures and training recipes. Checkpoints maximize validation AP; patience is in epochs or boosting rounds.','lgbm-recipe',
          ['Methods','Architecture and input transformation','Optimization, budget and stopping'],[
          ['RealMLP','Pre-tuned full classifier; complete evidence profile.',
           'Batch 256; 256 epochs, no early stopping.'],
          ['TabM / mini; with / without PLE',r'Three width-256 layers, 32 members, dropout 0.2; training-quantile transform. PLE: version B, dimension 16, no activation, up to 48 bins; merge duplicate edges; unit interval for constants.',
           r'AdamW, learning rate $5\times10^{-4}$, decay $10^{-3}$; effective batch 256 (PLE microbatch 32); cap 256, patience 16.'],
          ['SeFT',r'Input/output: three width-32 layers. Attention embedding: two width-64 layers; four heads, latent/dot-product width 128. Time embedding 4, maximum scale 100, attention dropout 0.1.',
           r'Adam $10^{-3}$, no decay; batch/microbatch 64/4; cap 300, patience 30. Halve rate after 15 stale epochs; cooldown 15, floor $10^{-5}$; unweighted BCE.'],
          ['mTAND encoder',r'Supervised: hidden 256, learned time embedding 128, one head, 128 reference queries. Sparse and reference dense implementations match in outputs and gradients.',
           r'Adam $10^{-4}$, no decay; batch/microbatch 256/4; cap/patience 200; two-class cross-entropy.'],
          [r'STraTS (ss$^{-}$)',r'Author supervised variant: hidden 64, two layers, 16 heads; dropout/attention dropout 0.2. Full observations; extra demographic coordinate zero; no pretraining.',
           r'Author AdamW $5\times10^{-4}$, $\epsilon=10^{-6}$, no decay, clip 0.3; batch/microbatch 16/1; cap 50, patience 10; train-ratio-weighted BCE.'],
          ['LightGBM',r'4.6.0, LGBM-TD: 50 leaves, unlimited depth, 255 bins, minimum leaf rows 40, Hessian $10^{-7}$, split gain 0. Zero is observed.',
           r'BCE; rate 0.04, row fraction 0.75, feature fraction 1, $L_1/L_2=0$, no class weights; cap 1,000, patience 300; four threads, deterministic column-wise layout.'],
          ['XGBoost-TD',r'3.2.0, CPU histogram trees: depth 6, 256 bins, minimum child weight $5\times10^{-6}$. Zero is observed.',
           r'BCE; rate 0.08, row fraction 0.65, column fraction per level 0.9, $L_1/L_2=0$; cap 1,000, patience 300; four threads.']],
          r'@{}>{\raggedright\arraybackslash}p{.12\textwidth}>{\raggedright\arraybackslash}p{.42\textwidth}>{\raggedright\arraybackslash}p{.40\textwidth}@{}')
    assert hashes=={db:SHA(ROOT/'runs'/RUN/(db+'_results_v1')/'verified-results.json') for db in DBS}
    print(json.dumps(dict(core_seeds=core['seeds'],verified_added_cells=core['verified_cells'],core_summaries=len(core['summaries']),core_contrasts=len(core['paired']),inputs_unchanged=True,bootstrap_resampled=False)))


def cnn_comparison(cnn):
    """Summarize complete families; every interval remains in Table V."""
    text=[]
    cnn_pairs=[r for r in cnn['paired'] if r['split']=='test' and r['baseline'] in ('TCN','ModernTCN','LITEMV')]
    med_pairs=[r for r in cnn['paired'] if r['split']=='test' and r['baseline']=='Medformer']
    if len(cnn_pairs)==18 and len(med_pairs)==6 and all(r['adjusted_ci'][0]>0 for r in cnn_pairs+med_pairs):
        los_cnn=[r for r in cnn_pairs if r['endpoint']==1]
        los_med=[r for r in med_pairs if r['endpoint']==1]
        cnn_range=f"{min(r['mean'] for r in los_cnn)*100:.2f}--{max(r['mean'] for r in los_cnn)*100:.2f}"
        med_range=f"{min(r['mean'] for r in los_med)*100:.2f}--{max(r['mean'] for r in los_med)*100:.2f}"
        text.append('Across the three LOS tasks, CEM-Net exceeds TCN, ModernTCN and LITEMV '+
                    f'by {cnn_range} AP percentage points and Medformer by {med_range} points '+
                    r'(Table~\ref{tab:primary}). All corresponding family-adjusted intervals are positive '+
                    f"(Holm $p\\leq {max(r['p_holm'] for r in los_cnn):.4f}$ for the CNNs and "+
                    f"$p\\leq {max(r['p_holm'] for r in los_med):.4f}$ for Medformer). "+
                    'The corresponding mortality intervals are also positive in all three databases.')
    else:
        for ps,name in [(cnn_pairs,'the three CNNs'),(med_pairs,'Medformer')]:
            if ps:
                text.append(r'Table~\ref{tab:primary} reports the complete five-seed comparisons with '+name+
                            r'; task-level intervals retain both supported and unresolved contrasts.')
    if cnn['models'] and all(
        (cnn['summaries'][f'{db}:test:CEM-Net:{e}'][metric]['mean']-
         cnn['summaries'][f'{db}:test:{model}:{e}'][metric]['mean'])*direction>0
        for db in DBS for e in (0,1) for model in cnn['models']
        for metric,direction in [('AUROC',1),('Brier',-1)]
    ):
        text.append('Mean AUROC and Brier score also favor CEM-Net throughout these comparisons.')
    metric_names={'AUROC':'AUROC','Brier':'Brier score','log_loss':'log loss'}
    for db in DBS:
        for e in (0,1):
            for metric,name in metric_names.items():
                value=cnn['summaries'][f'{db}:test:CEM-Net:{e}'][metric]['mean']
                adverse=[]
                for model in cnn['models']:
                    baseline=cnn['summaries'][f'{db}:test:{model}:{e}'][metric]['mean']
                    if (value-baseline)*(1 if metric=='AUROC' else -1)<0:adverse.append(f'{model} ({baseline:.4f})')
                if adverse:
                    comparison=', '.join(adverse[:-1])+' and '+adverse[-1] if len(adverse)>1 else adverse[0]
                    text.append(f"For {NAMES[db]} {'mortality' if e==0 else 'LOS'}, however, mean {name} is "+
                                ('higher' if metric=='AUROC' else 'lower')+' for '+comparison+
                                f' than for CEM-Net ({value:.4f}).')
    (TABLES/'cnn_comparison.tex').write_text(' '.join(text)+'\n',encoding='utf8')


def parameter_composition():
    """Count the frozen architecture algebraically; never instantiate or fit it.

    d/q come from the completed input inspections and TypedNative metadata.
    Each sum is checked against all six saved per-database model counts.
    NRE: frequency q*16, phase q*16, weight q*16*3, bias q*3.
    CMM: UntiedNative.input_members, 8*(d+3*q).
    Native trunk: one front scale, four weight/bias layers, three pSELU vectors.
    """
    rows={}
    for db,d,q in (('mimiciii',13672,10417),('mimic',12508,9545),('eicu',18250,13844)):
        D=d+3*q
        parts=dict(input_projection=256*D,NRE=83*q,CMM=8*D,
                   remaining_network=D+2*256**2+256*2+(3*256+2)+3*256)
        total=sum(parts.values())
        result=READ(ROOT/'runs'/RUN/('local_'+db+'_inference_v2')/'verified-results.json')
        counts=[r['timing']['parameters'] for r in result['records'] if r['job']['method']=='CEM-Net']
        assert result['complete'] and len(counts)==6 and all(n==total for n in counts)
        rows[db]=dict(d=d,q=q,expanded_width=D,components=parts,total=total,
                      shares={k:v/total for k,v in parts.items()},verified_saved_counts=len(counts))
    return rows


def hospital_table(core):
    rows=[]
    for m in ('TabM_mini_PLE_hospital','LightGBM_hospital'):
        cells=[label(m)]
        for e in (0,1):
            value=ci(core_pair(core,'eicu',e,'hospital','CEM_hospital - '+m))
            point,interval=value[1:-1].split(r'\ ',1)
            cells.append(r'\shortstack[r]{'+'$'+point+'$'+r'\\'+'$'+interval+'$'+'}')
        rows.append(cells)
    table('hospital','Five-seed hospital-held-out AP differences on eICU: CEM-Net minus comparator. Brackets show 98.75\\% intervals from patient-connected hospital components (four contrasts).','hospital',
          ['Comparator','Mortality','LOS'],rows,'lrr',wide=False)


def context_table(context):
    rows=[]
    for split in ('test',):
        for variant,name in [('EMA_OC_local','Local'),('EMA_OC_global','Global'),('EMA_OC_mismatch','Mismatched')]:
            rows.append([name]+['$'+ms(context[split]['summary'][str(e)][variant])+'$' for e in (0,1)])
    table('context',r'Observation-context gating on eICU (AP); all variants have equal gate capacity.',
          'context',['Context','Mortality','LOS'],rows,'lrr',wide=False,output_dir=PAPER/'tables/cem_generality_20261028')


def review_budget_table(data):
    # Retain one absolute retrieval measure and the paired clinical increment;
    # precision and the comparator's recall remain in the aggregate results.
    budgets=[]
    for i,b in enumerate((.05,.1)):
        budgets.append([(r'\midrule'+'\n' if i else '')+r'\multicolumn{3}{l}{\textit{'+str(int(b*100))+r'\% review budget}}'])
        for db in DBS:
            for e in (0,1):
                row=[NAMES[db]+(' / Mort.' if e==0 else ' / LOS')]
                rr={r['metric']:r for r in data[db]['review_budget_pairs'] if r['split']=='test' and r['endpoint']==e and r['budget']==b}
                row.append(f"{100*rr['recall']['methods']['CEM-Net']['mean']:.1f}")
                r=rr['true_positives_per_1000']['CEM_minus_baseline']
                row.append(f"${r['mean']:+.2f}\\ [{r['adjusted_ci'][0]:+.2f},{r['adjusted_ci'][1]:+.2f}]$")
                budgets.append(row)
    table('review_budget','Retrieval under fixed review budgets. Recall (\\%) is for CEM-Net; $\\Delta$TP is CEM-Net minus the validation-selected baseline in positives retrieved per 1,000 stays. Brackets show 99.5833\\% patient-cluster intervals (12 contrasts).','review-budget',
          ['Database / endpoint','Recall',r'$\Delta$TP [interval]'],budgets,'lrr',wide=False)


def stress_figure(data, fixed):
    """Render the existing paired change estimates; no scoring or resampling."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    assert Path(font_manager.findfont('Times New Roman', fallback_to_default=False)).exists()
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.titlesize':9,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    conditions=('lab_drop25','lab_drop50','lab_delay2h','lab_delay6h')
    all_ends=[]
    for db in DBS:
        assert data[db]['complete'] and data[db]['bootstrap_replicates']==20000
        assert len(data[db]['stress'])==8
        for r in data[db]['stress']:
            ref=fixed[db]['strongest']['dev'][str(r['endpoint'])]['validation_selected']
            expected=np.asarray(r['methods']['CEM-Net']['change']['values'])-np.asarray(r['methods'][ref]['change']['values'])
            pair=r['relative_degradation']
            np.testing.assert_allclose(expected,pair['values'],rtol=0,atol=1e-15)
            np.testing.assert_allclose(np.mean(expected),pair['mean'],rtol=0,atol=1e-15)
            assert pair['family']==24
            all_ends.extend(np.asarray(pair['adjusted_ci'])*100)
    bound=np.ceil(max(abs(np.asarray(all_ends)))*2)/2
    fig,axes=plt.subplots(2,3,figsize=(7.0,3.6),sharex=True,sharey=True)
    for col,db in enumerate(DBS):
        for e in (0,1):
            ax=axes[e,col]
            rows={r['condition']:r['relative_degradation'] for r in data[db]['stress'] if r['endpoint']==e}
            means=np.array([rows[c]['mean'] for c in conditions])*100
            ends=np.array([rows[c]['adjusted_ci'] for c in conditions])*100
            y=np.arange(4)
            ax.errorbar(means,y,xerr=np.array([means-ends[:,0],ends[:,1]-means]),
                        fmt='o',color='#245b8b',markersize=3.5,capsize=2,linewidth=.85)
            ax.axvline(0,color='#555555',linestyle='--',linewidth=.7)
            ax.grid(axis='x',color='#dddddd',linewidth=.45)
            ref=fixed[db]['strongest']['dev'][str(e)]['validation_selected']
            ax.set_title(NAMES[db]+' / '+('Mortality' if e==0 else 'LOS')+'\nvs. '+label(ref),pad=4)
            ax.set_yticks(y,['Drop 25%','Drop 50%','Delay 2 h','Delay 6 h'])
            ax.set_ylim(3.5,-.5);ax.set_xlim(-bound,bound)
            ax.tick_params(axis='y',length=0)
    fig.supxlabel('Difference in AP change (percentage points); right favors CEM-Net',fontsize=9,y=.01)
    fig.tight_layout(pad=.65,h_pad=.8,w_pad=.8,rect=(0,.05,1,1))
    figdir=PAPER/'figures/cem_icde72_20261029';figdir.mkdir(exist_ok=True)
    fig.savefig(figdir/'fixed_stress.pdf',bbox_inches='tight')
    fig.savefig(figdir/'fixed_stress.png',dpi=300,bbox_inches='tight');plt.close(fig)


def response_and_influence():
    """Two complementary views of frozen aggregates; no patient data or scoring."""
    import json
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    from matplotlib.lines import Line2D
    assert Path(font_manager.findfont('Times New Roman',fallback_to_default=False)).exists()
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.titlesize':9,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    data={db:json.loads((ROOT/'runs'/RUN/(db+'_secondary_v1/verified-results.json')).read_text()) for db in DBS}
    fixed={db:json.loads((ROOT/'runs'/RUN/(db+'_results_v1/verified-results.json')).read_text()) for db in DBS}
    folder=PAPER/'figures/cem_icde72_20261029';folder.mkdir(exist_ok=True)
    colors={'CEM-Net':'#0072B2','baseline':'#D55E00'}
    fig=plt.figure(figsize=(7.15,3.0))
    grid=fig.add_gridspec(2,3,left=.065,right=.99,bottom=.16,top=.91,hspace=.90,wspace=.28)
    bounds=[]
    for e in (0,1):
        ends=[v for db in DBS for r in data[db]['stress'] if r['endpoint']==e
              for m in r['methods'].values() for v in m['change_ci']['adjusted_ci']]
        bounds.append((min(ends)*100-.5,max(0,max(ends)*100)+.5))
    for col,db in enumerate(DBS):
        assert data[db]['complete'] and data[db]['bootstrap_replicates']==20000
        for e in (0,1):
            sub=grid[e,col].subgridspec(1,2,wspace=.18)
            rows={r['condition']:r for r in data[db]['stress'] if r['endpoint']==e}
            ref=fixed[db]['strongest']['dev'][str(e)]['validation_selected']
            axes=[]
            for k,(conds,x,xlabel) in enumerate(((('lab_drop25','lab_drop50'),[0,25,50],'Removed (%)'),
                                                 (('lab_delay2h','lab_delay6h'),[0,2,6],'Delay (h)'))):
                ax=fig.add_subplot(sub[0,k]);axes.append(ax)
                for model,color,marker in ((ref,colors['baseline'],'s'),('CEM-Net',colors['CEM-Net'],'o')):
                    yy=[0]+[rows[c]['methods'][model]['change']['mean']*100 for c in conds]
                    ci=np.array([[0,0]]+[rows[c]['methods'][model]['change_ci']['adjusted_ci'] for c in conds])*100
                    for c in conds:
                        m=rows[c]['methods'][model]
                        np.testing.assert_allclose(np.asarray(m['perturbed']['values'])-m['clean']['values'],m['change']['values'],rtol=0,atol=1e-14)
                        assert m['change_ci']['family']==24
                    ax.fill_between(x,ci[:,0],ci[:,1],color=color,alpha=.13,linewidth=0)
                    ax.plot(x,yy,marker=marker,markersize=3,linewidth=1,color=color)
                ax.axhline(0,color='#666666',linewidth=.5,zorder=0)
                ax.set_ylim(bounds[e]);ax.set_xticks(x);ax.set_xlabel(xlabel,labelpad=1,fontsize=8)
                ax.tick_params(axis='both',labelsize=7,pad=2,length=2)
                ax.grid(axis='y',color='#dddddd',linewidth=.35)
                if col==0 and k==0:ax.set_ylabel('AP change (pp)',fontsize=8,labelpad=2)
                else:ax.tick_params(axis='y',labelleft=False)
            pos=grid[e,col].get_position(fig)
            title=NAMES[db]+' / '+('Mortality' if e==0 else 'LOS')+'\nvs. '+label(ref)
            fig.text((pos.x0+pos.x1)/2,pos.y1+.022,title,ha='center',va='bottom',fontsize=9,linespacing=1.05)
    fig.legend(handles=[Line2D([],[],color=colors['baseline'],marker='s',markersize=3,label='Baseline'),
                        Line2D([],[],color=colors['CEM-Net'],marker='o',markersize=3,label='CEM-Net')],
               loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.53,-.008),fontsize=8)
    fig.savefig(folder/'observation_response.pdf',bbox_inches='tight')
    fig.savefig(folder/'observation_response.png',dpi=300,bbox_inches='tight');plt.close(fig)

    h=json.loads((ROOT/'runs'/RUN/'hospital_influence_v1/verified-results.json').read_text())
    assert h['complete'] and h['partitions']['test']['components']==22
    sizes={c['component']:c['stay_fraction']*100 for c in h['partitions']['test']['component_details']}
    fig,axes=plt.subplots(1,2,figsize=(3.45,2.05),sharex=True)
    for e,ax in enumerate(axes):
        for method,color,marker in (('TabM_mini_PLE_hospital','#D55E00','o'),('LightGBM_hospital','#666666','^')):
            rows=[r for r in h['rows'] if (r['split'],r['endpoint'],r['baseline'])==('test',e,method)]
            assert len(rows)==22
            x=[sizes[r['omitted_component']] for r in rows];y=[r['influence']*100 for r in rows]
            ax.scatter(x,y,s=13,color=color,marker=marker,alpha=.8,linewidths=.25,edgecolors='white',zorder=3)
        ax.axhline(0,color='#444444',linestyle='--',linewidth=.55)
        ax.set_xscale('log');ax.set_xlim(.01,24);ax.set_xticks([.01,.1,1,10],['0.01','0.1','1','10'])
        ax.set_title('Mortality' if e==0 else 'LOS',pad=4)
        ax.grid(axis='y',color='#dddddd',linewidth=.35)
        ax.tick_params(labelsize=7,pad=2,length=2)
        ax.set_xlabel('Component share (%)',fontsize=7.5,labelpad=2)
    axes[0].set_ylabel('Influence on AP gap (pp)',fontsize=8,labelpad=2)
    fig.legend(handles=[Line2D([],[],color='#D55E00',marker='o',linestyle='none',markersize=3,label='vs. TabM-mini+PLE'),
                        Line2D([],[],color='#666666',marker='^',linestyle='none',markersize=3,label='vs. LightGBM')],
               loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.52,-.015),fontsize=7)
    fig.tight_layout(pad=.35,w_pad=.45,rect=(0,.13,1,1))
    fig.savefig(folder/'hospital_component_influence.pdf',bbox_inches='tight')
    fig.savefig(folder/'hospital_component_influence.png',dpi=300,bbox_inches='tight');plt.close(fig)
    print(json.dumps(dict(new_figures=['observation_response','hospital_component_influence'],fixed_aggregates_only=True)))


def subgroup_figure(data,compact=False):
    """Plot the existing subgroup contrasts with their pointwise intervals."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    assert Path(font_manager.findfont('Times New Roman',fallback_to_default=False)).exists()
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.titlesize':9,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    names=('coverage_Q1','coverage_Q2','coverage_Q3','coverage_Q4','age_18-64','age_65+','sex_F','sex_M','LOS_excluding_early_death')
    display=['Coverage Q1','Coverage Q2','Coverage Q3','Coverage Q4','Age 18–64','Age 65+','Female','Male','Exclude early deaths']
    if compact:
        names=names[:4]+names[-1:]
        display=display[:4]+display[-1:]
    groups={(db,e):{r['subset']:r for r in data[db]['subgroup_AP_pairs'] if r['split']=='test' and r['endpoint']==e} for db in DBS for e in (0,1)}
    bound=np.ceil(max(abs(v)*100 for pairs in groups.values() for n,r in pairs.items() if n in names for v in r['ci95']))
    fig,axes=plt.subplots(2,3,figsize=(7,2.7 if compact else 3.5),sharex=True,sharey='row')
    for col,db in enumerate(DBS):
        for e in (0,1):
            ax=axes[e,col];pairs=groups[db,e]
            for i,n in enumerate(names):
                if n not in pairs:continue
                r=pairs[n];v=100*r['mean'];lo,hi=np.asarray(r['ci95'])*100
                ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color='#0072B2',markersize=3.2,capsize=1.7,linewidth=.8)
            ax.axvline(0,color='#666666',linestyle='--',linewidth=.65)
            ax.grid(axis='x',color='#dddddd',linewidth=.4)
            visible=display if e==1 else display[:-1]
            ax.set_yticks(range(len(visible)),visible)
            ax.set_ylim(len(visible)-.4,-.6);ax.set_xlim(-bound,bound)
            ax.tick_params(axis='y',length=0)
            refs={r['CEM_minus'] for r in pairs.values()}
            assert len(refs)==1
            ref=next(iter(refs))
            ax.set_title(NAMES[db]+' / '+('Mortality' if e==0 else 'LOS')+'\nvs. '+LABELS.get(ref,ref),pad=4)
    fig.supxlabel('CEM-Net − baseline: AP difference (percentage points)',fontsize=9,y=.01)
    fig.tight_layout(pad=.6,h_pad=.5,w_pad=.7,rect=(0,.04,1,1))
    figdir=PAPER/'figures/cem_icde72_20261029';figdir.mkdir(exist_ok=True)
    name='coverage_sensitivity' if compact else 'clinical_subgroups'
    fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(figdir/(name+'.png'),dpi=300,bbox_inches='tight');plt.close(fig)


def probability_figure(fixed,compact=False):
    """Render descriptive AUROC/ECE and existing paired loss intervals."""
    import hashlib
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.titlesize':9,'pdf.fonttype':42,'ps.fonttype':42})
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1 if compact else 2,2,figsize=(7,2.2 if compact else 4.1),sharey=True,squeeze=False)
    tasks=[(db,e) for db in DBS for e in (0,1)]
    references={(db,e):fixed[db]['strongest']['dev'][str(e)]['validation_selected'] for db,e in tasks}
    labels=[NAMES[d]+(' / Mortality' if e==0 else ' / LOS')+'\nvs. '+LABELS.get(references[d,e],references[d,e]) for d,e in tasks]
    intervals={}
    for db in DBS:
        path=ROOT/'runs'/RUN/(db+'_results_v1/verified-results.json')
        saved=READ(ROOT/'runs'/RUN/'probability_intervals_v2'/db/'verified-results.json')
        assert saved['complete'] and saved['replicates']==20000
        assert saved['primary_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        assert saved['training_runs']==saved['calibration_refits']==saved['AP_recomputations']==0
        rows=[r for r in saved['rows'] if r['split']=='test']
        assert len(rows)==8
        for r in rows:
            assert r['baseline']==references[db,r['endpoint']] and r['family']==24 and r['negative_favors_CEM']
            assert not r['validation_resubstitution']
            intervals[db,r['endpoint'],r['form'],r['metric']]=r
    for ax,(metric,title) in zip([] if compact else axes[0],[('AUROC','(a) Raw AUROC (higher is better)'),('ECE15','(b) ECE (lower is better)')]):
        for y,(db,e) in enumerate(tasks):
            ref=references[db,e]
            for j,(model,color) in enumerate(((ref,'#D55E00'),('CEM-Net','#0072B2'))):
                rows=[r for r in fixed[db]['calibration'] if r['split']=='test' and r['method']==model and r['endpoint']==e]
                assert len(rows)==3
                raw=st.mean(r['raw'][metric] for r in rows);cal=st.mean(r['Platt'][metric] for r in rows)
                pos=y+(.12 if j else -.12)
                if metric=='AUROC':
                    ax.plot(raw,pos,'o',markerfacecolor='white',markeredgecolor=color,markersize=3.7)
                else:
                    ax.plot([raw,cal],[pos,pos],color=color,linewidth=1)
                    ax.plot(raw,pos,'o',markerfacecolor='white',markeredgecolor=color,markersize=4)
                    ax.plot(cal,pos,'o',color=color,markersize=3.7)
        ax.set_title(title,pad=6)
        if metric=='ECE15':ax.set_xlim(left=0)
    loss_titles=[('Brier',r'(a) Brier difference ($\times 10^{-3}$)'),('log_loss',r'(b) Log-loss difference ($\times 10^{-3}$)')] if compact else [('Brier',r'(c) Brier difference ($\times 10^{-3}$)'),('log_loss',r'(d) Log-loss difference ($\times 10^{-3}$)')]
    for ax,(metric,title) in zip(axes[-1],loss_titles):
        for y,(db,e) in enumerate(tasks):
            for offset,form in ((-.15,'raw'),(.15,'Platt')):
                row=intervals[db,e,form,metric]
                value=1000*row['mean'];lo,hi=1000*np.asarray(row['adjusted_ci'])
                assert lo<=value<=hi
                ax.errorbar(value,y+offset,xerr=[[value-lo],[hi-value]],fmt='o',color='#444444',
                            markerfacecolor='white' if form=='raw' else '#444444',markersize=3.6,capsize=2,linewidth=.85)
        ax.axvline(0,color='#777777',linewidth=.65,linestyle='--')
        ax.set_title(title,pad=6)
    for ax in axes.flat:
        ax.set_yticks(range(6),labels);ax.set_ylim(5.6,-.6)
        ax.grid(axis='x',color='#dddddd',linewidth=.45);ax.tick_params(axis='y',length=0,labelsize=7.2)
        ax.tick_params(axis='x',labelsize=7.5)
    handles=[Line2D([],[],color='#D55E00',marker='o',linestyle='none',label='Baseline named at left'),Line2D([],[],color='#0072B2',marker='o',linestyle='none',label='CEM-Net'),Line2D([],[],color='#444444',marker='o',markerfacecolor='white',linestyle='none',label='Raw'),Line2D([],[],color='#444444',marker='o',linestyle='none',label='Platt')]
    fig.legend(handles=handles[-2:] if compact else handles,loc='lower center',ncol=2 if compact else 4,frameon=False,fontsize=8,bbox_to_anchor=(.5,0))
    fig.tight_layout(pad=.6,h_pad=1.5,w_pad=1.5,rect=(0,.12 if compact else .065,1,1))
    folder=PAPER/'figures/cem_icde72_20261029';folder.mkdir(exist_ok=True)
    name='probability_losses' if compact else 'probability_quality'
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(folder/(name+'.png'),dpi=300,bbox_inches='tight');plt.close(fig)


def cost_table(inference,fixed):
    rows=[]
    for db in DBS:
        references={e:fixed[db]['strongest']['dev'][str(e)]['validation_selected'] for e in (0,1)}
        groups=[(tuple(e for e in (0,1) if references[e]==ref),ref) for ref in dict.fromkeys(references.values())]
        for endpoints,ref in groups:
            title=NAMES[db]+(' / '+('Mortality' if endpoints[0]==0 else 'LOS') if len(endpoints)==1 else ' / both endpoints')
            rows.append([r'\multicolumn{5}{l}{\textit{'+title+r'}}'])
            for model in (ref,'CEM-Net'):
                rs=[r for r in inference[db]['records'] if r['job']['method']==model and r['job']['endpoint'] in endpoints]
                assert len(rs)==3*len(endpoints)
                params={r['timing']['parameters'] for r in rs};assert len(params)==1
                ts=[next(v for v in r['timing']['rows'] if v['batch']==1) for r in rs]
                mem=[next(v for v in r['timing']['rows'] if v['batch']==32)['peak_allocated_mib'] for r in rs]
                timing=[f"{st.median(t[scope]['median_ms'] for t in ts):.2f}" for scope in ('resident_model','dense_transform_transfer_model')]
                rows.append([label(model),f'{next(iter(params))/1e6:.2f}',*timing,f'{st.median(mem):.0f}'])
    table('cost',r'Parameters (millions), batch-32 allocated memory (MiB) and batch-1 latency (ms): resident forward versus prediction with preprocessing/transfers. Checkpoint medians use the same GPU per database and validation-selected baselines; source-table processing is excluded.',
          'cost',['Method','Params','Resident','With prep.','Memory'],rows,'lrrrr',wide=False)


def secondary():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    assert Path(font_manager.findfont('Times New Roman',fallback_to_default=False)).exists()
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.titlesize':10,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    inputs={}
    def load(folder):
        p=ROOT/'runs'/RUN/folder/'verified-results.json';inputs[p]=SHA(p)
        v=READ(p);assert v['complete'];return v
    data={db:load(db+'_secondary_v1') for db in DBS}
    inference={db:load('local_'+db+'_inference_v2') for db in DBS}
    fixed={db:load(db+'_results_v1') for db in DBS}
    for db in DBS:
        assert data[db]['primary_sha256']==inputs[ROOT/'runs'/RUN/(db+'_results_v1')/'verified-results.json']
        assert data[db]['stress_sha256']==inputs[ROOT/'runs'/RUN/('local_'+db+'_inference_v2')/'verified-results.json']
        assert data[db]['bootstrap_replicates']==20000 and data[db]['calibration_refits']==0
        assert inference[db]['fixed_stress_cases']==48 and inference[db]['zero_updates']
    figdir=PAPER/'figures/cem_icde72_20261029';figdir.mkdir(exist_ok=True)
    stress_figure(data,fixed)

    subgroup_figure(data)
    probability_figure(fixed)
    subgroup_figure(data,compact=True)
    probability_figure(fixed,compact=True)

    review_budget_table(data)
    cost_table(inference,fixed)
    pipelines=[]
    for db in DBS:
        c=data[db]['pipeline_cost'];assert c['complete'] and all(r['raw_reconstruction_exact'] for r in c['rows'])
        cpu={'mimiciii':'Xeon 8468','mimic':'i7-14700KF','eicu':'i7-14700KF'}[db]
        pipelines.append([NAMES[db],cpu]+[f"{r['median_seconds']:.2f}" for r in c['rows']])
    table('pipeline','Feature-pipeline median seconds, three runs per size and two CPU threads. CPUs/runtimes differ; OS caches were not flushed.','pipeline',
          ['Database','CPU','$n=128$','$512$','$2{,}048$'],pipelines,'llrrr',wide=False)
    assert all(SHA(p)==digest for p,digest in inputs.items())
    print(json.dumps(dict(secondary_tables=3,figures=5,fixed_stress_cases=144,existing_results_unchanged=True)))



