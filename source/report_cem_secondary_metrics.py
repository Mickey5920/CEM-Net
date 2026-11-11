"""Render and verify the LOS-focused secondary metrics from aggregate draws."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

RUN='cem_los_secondary_metrics_20261110'
DBS=('mimiciii','mimic','eicu')
NAMES=dict(mimiciii='MIMIC-III',mimic='MIMIC-IV',eicu='eICU')


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(root):
    base=Path(root)/'runs'/RUN
    result=read(base/'paired_results_v1/verified-results.json')
    assert result['complete'] and result['replicates']==20000 and result['family_count']==24
    assert result['seeds']==[48,49,50,51,52] and len(result['rows'])==90 and len(result['paired'])==24
    assert result['model_fits']==result['model_inference']==result['calibrator_fits']==0
    pvalues=[]
    for db in DBS:
        path=base/db/'verified-results.json';assert sha(path)==result['source_results'][db]
        data=read(path);assert data['complete'] and data['original_results_unchanged'] and not data['patient_records_exported']
        assert sha(base/db/'paired_draws.npz')==data['paired_draws_sha256']
        draws=dict(np.load(base/db/'paired_draws.npz',allow_pickle=False))
        for row in data['paired']:
            e,ref,metric=row['endpoint'],row['baseline'],row['metric']
            combined=next(r for r in result['paired'] if (r['database'],r['endpoint'],r['baseline'],r['metric'])==(db,e,ref,metric))
            for key,value in row.items():assert combined[key]==value
            scores={method:sorted([r for r in data['rows'] if r['method']==method and r['endpoint']==e],key=lambda r:r['seed']) for method in ('CEM-Net',ref)}
            values=np.array([c[metric]-b[metric] for c,b in zip(scores['CEM-Net'],scores[ref])])
            np.testing.assert_array_equal(values,row['values'])
            assert abs(values.mean()-row['mean'])<1e-14 and abs(values.std(ddof=1)-row['sample_SD'])<1e-14
            x=draws[f't{e}_{ref}_{metric}'];assert x.shape==(20000,) and np.isfinite(x).all()
            np.testing.assert_array_equal(np.quantile(x,[.05/48,1-.05/48]),row['adjusted_ci'])
            np.testing.assert_array_equal(np.quantile(x,[.025,.975]),row['ci95'])
            p=(1+np.count_nonzero(np.abs(x-row['mean'])>=abs(row['mean'])))/20001
            assert p==row['p_bootstrap_centered'] and min(1.,24*p)==row['p_bonferroni']
            pvalues.append((combined,p))
    running=0.
    for rank,(row,p) in enumerate(sorted(pvalues,key=lambda pair:pair[1])):
        running=max(running,(24-rank)*p)
        assert row['p_holm']==min(1.,running)
    return dict(secondary_metric_contrasts=24,secondary_metric_seeds=5,secondary_metric_draws=20000,
                secondary_metric_intervals_and_Holm_reconstructed=True,secondary_metric_new_fits=0)


def render(root,output):
    root,output=Path(root),Path(output)
    checks=verify(root)
    result=read(root/'runs'/RUN/'paired_results_v1/verified-results.json')
    lines=[r'\begin{table*}[tp]',r'\centering\small',r'\setlength{\tabcolsep}{5pt}',
           r'\caption{Discrimination and probability quality.}',
           r'\label{tab:secondary-metrics}',r'\begin{tabular}{llrrrr}',r'\toprule',
           r'Database & Ref. & $\Delta$AUROC [interval] & $p_H$ & $\Delta$Brier [interval] & $p_H$ \\']
    def cell(row):
        edge=lambda x:f'{x:+.6f}' if 0<abs(x)<.00005 else f'{x:+.4f}'
        lo,hi=row['adjusted_ci']
        return rf"${edge(row['mean'])}\ [{edge(lo)},{edge(hi)}]$"
    for e,label in ((1,'LOS'),(0,'Mortality')):
        lines += [r'\midrule',r'\multicolumn{6}{l}{\textit{'+('Prolonged LOS' if e else 'Mortality')+r'}} \\']
        for db in DBS:
            selected=result['selected_baselines'][db][str(e)]
            for ref,tag in ((selected,'S'),('matched_tabm','M')):
                rows={r['metric']:r for r in result['paired'] if (r['database'],r['endpoint'],r['baseline'])==(db,e,ref)}
                lines.append(' & '.join([NAMES[db] if tag=='S' else '',tag,cell(rows['AUROC']),f"{rows['AUROC']['p_holm']:.3g}",
                                        cell(rows['Brier']),f"{rows['Brier']['p_holm']:.3g}"])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',
              r'\par\smallskip{\footnotesize\raggedright CEM-Net minus reference: positive favors CEM-Net for AUROC, negative for Brier. Brackets: 99.7917\% patient-cluster intervals; Holm $p_H$ adjusts 24 contrasts. S: validation-selected TabM-mini for MIMIC-III LOS, TabM-mini+PLE for eICU, TabM+PLE otherwise. M: matched TabM-mini.\par}',
              r'\end{table*}','']
    output.mkdir(parents=True,exist_ok=True)
    (output/'secondary_metrics.tex').write_text('\n'.join(lines),encoding='utf8')
    return checks


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    print(json.dumps(render(args.project,args.output) if args.output else verify(args.project)))
