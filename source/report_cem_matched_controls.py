"""Render the matched-control displays from verified aggregate results only."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np

RUN = 'cem_matched_controls_20261105'
AGGREGATION_RUN = 'cem_aggregation_stress_20261106'
TARGETED_RUN = 'cem_targeted_evidence_20261106'
DATABASES = ('mimiciii', 'mimic', 'eicu')
NAMES = dict(mimiciii='MIMIC-III', mimic='MIMIC-IV', eicu='eICU')


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_results(root):
    folder = Path(root)/'runs'/RUN/'paired_results_v1'
    done = read(folder/'completion.json')
    assert done['complete']
    for name, digest in done['files'].items(): assert sha(folder/name) == digest
    result = read(folder/'verified-results.json')
    assert result['complete'] and result['verified_new_fits'] == result['verified_fixed_tests'] == 60
    assert result['family_count'] == 12 and len(result['paired']) == 12
    assert result['seeds'] == [48, 49, 50, 51, 52]
    return result


def ap_cell(result, db, method, endpoint):
    value = result['summaries'][f'{db}:test:{method}:{endpoint}']['AP']
    return rf"${value['mean']:.4f}\pm{value['sample_SD']:.4f}$"


def contrast_cell(result, db, method, endpoint):
    row = next(r for r in result['paired'] if (r['database'], r['baseline'], r['endpoint']) == (db, method, endpoint))
    lo, hi = row['adjusted_ci']; p = row['p_holm']
    # The existing readable exporter flattens this two-line form exactly.
    return (r'\shortstack[r]{' + rf"${row['mean']:+.4f};\ p_H={p:.3g}$"
            + r'\\' + rf'$[{lo:+.4f},{hi:+.4f}]$' + '}')


def comparison_table(result):
    lines = [r'\begin{table}[!tp]', r'\centering\small', r'\setlength{\tabcolsep}{2pt}',
             r'\caption{Matched predictor comparison.}',
             r'\label{tab:matched-controls}', r'\begin{tabular}{lrrr}', r'\toprule',
             r'Task & Matched AP & $\Delta$AP [interval] & $p_H$ \\', r'\midrule']
    for endpoint, label in ((1, 'LOS'), (0, 'Mortality')):
        for db in DATABASES:
            row=next(r for r in result['paired'] if (r['database'],r['baseline'],r['endpoint'])==(db,'matched_tabm',endpoint))
            cells = [dict(mimiciii='III',mimic='IV',eicu='eICU')[db]+'/'+('M' if endpoint==0 else 'L'),
                     ap_cell(result, db, 'matched_tabm', endpoint),targeted_inline(row),f"{row['p_holm']:.3g}"]
            lines.append(' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule', r'\end{tabular}',
        r'\par\smallskip{\footnotesize\raggedright Matched AP: TabM-mini mean $\pm$ sample SD. Differences: CEM-Net minus matched predictor. III/IV: MIMIC; L/M: LOS/mortality. Brackets: 99.5833\% patient-cluster intervals; Holm $p_H$ uses 12 contrasts, including windows in Table~\ref{tab:targeted-aggregation}.\par}',
        r'\end{table}', ''])


def resource_summary(root):
    summaries = {}
    for db in DATABASES:
        folder = Path(root)/'runs'/RUN/(db+'_resources_v1')
        receipt = read(folder/'collection-receipt.json'); result = read(folder/'verified-results.json')
        assert receipt['exit']['returncode'] == 0 and result['complete']
        assert sha(folder/'verified-results.json') == receipt['receipt']['files']['verified-results.json']
        assert result['original_checkpoints_unchanged'] and result['optimizer_updates'] == 0
        assert len(result['rows']) == 20
        for variant in ('reference', 'matched_tabm'):
            rows = [r for r in result['rows'] if r['variant'] == variant]
            assert {(r['endpoint'], r['seed']) for r in rows} == {(e, s) for e in (0, 1) for s in range(48, 53)}
            assert all(r['replay_max_drift'] <= 1e-5 for r in rows)
            parameters = {r['timing']['parameters'] for r in rows}; assert len(parameters) == 1
            cell = dict(parameters=parameters.pop(), checkpoints=len(rows), batches={})
            for batch in (1, 8, 32):
                batch_rows = [next(x for x in r['timing']['rows'] if x['batch'] == batch) for r in rows]
                cell['batches'][str(batch)] = dict(
                    resident_ms=float(np.median([x['resident_model']['median_ms'] for x in batch_rows])),
                    transfer_ms=float(np.median([x['dense_transform_transfer_model']['median_ms'] for x in batch_rows])),
                    allocated_mib=float(np.median([x['peak_allocated_mib'] for x in batch_rows])))
            summaries[db+':'+variant] = cell
    return summaries


def resource_table(summaries):
    lines = [r'\begin{table}[tp]', r'\centering\footnotesize', r'\setlength{\tabcolsep}{3pt}',
             r'\caption{CEM-Net / matched TabM-mini costs: medians over ten checkpoints per method (five seeds, two outcomes), with units and batches as in Table~\ref{tab:cost}. Both share input transformation and transfer.}',
             r'\label{tab:matched-cost}', r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Database & Params & Resident & With prep. & Memory \\', r'\midrule']
    for db in DATABASES:
        a, b = [summaries[db+':'+v] for v in ('reference', 'matched_tabm')]
        fields = [NAMES[db], f"{a['parameters']/1e6:.2f}/{b['parameters']/1e6:.2f}"]
        fields += [f"{a['batches']['1'][key]:.2f}/{b['batches']['1'][key]:.2f}" for key in ('resident_ms', 'transfer_ms')]
        fields += [f"{a['batches']['32']['allocated_mib']:.0f}/{b['batches']['32']['allocated_mib']:.0f}"]
        lines.append(' & '.join(fields)+r' \\')
    return '\n'.join(lines+[r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


def combined_resource_table(original, summaries):
    """Keep both measured cost scopes in one display, without merging values."""
    begin, end = '% original-cost-rows\n', '% end-original-cost-rows\n'
    if begin in original:
        original_rows = original.split(begin, 1)[1].split(end, 1)[0].rstrip()
    else:
        original_rows = original.split(r'\midrule', 1)[1].split(r'\bottomrule', 1)[0].strip()
    matched_rows = []
    for db in DATABASES:
        matched_rows.append(r'\multicolumn{5}{l}{\textit{'+NAMES[db]+r' / both endpoints}} \\')
        for variant, label in (('matched_tabm', 'TabM-mini'), ('reference', 'CEM-Net')):
            value = summaries[db+':'+variant]
            fields = [label, f"{value['parameters']/1e6:.2f}"]
            fields += [f"{value['batches']['1'][key]:.2f}" for key in ('resident_ms', 'transfer_ms')]
            fields.append(f"{value['batches']['32']['allocated_mib']:.0f}")
            matched_rows.append(' & '.join(fields)+r' \\')
    lines = [r'\begin{table*}[tp]', r'\centering\small', r'\setlength{\tabcolsep}{2.5pt}',
             r'\caption{Model size and inference cost.}',
             r'\label{tab:cost}', r'\label{tab:matched-cost}',
             r'\begin{minipage}[t]{0.49\textwidth}', r'\centering', r'\begin{tabular}[t]{@{}lrrrr@{}}', r'\toprule',
             r'\multicolumn{5}{l}{\textit{(a) Original configurations}} \\',
             r'Method & Params & \multicolumn{2}{c}{Latency (ms)} & Memory \\',
             r'& (M) & Resident & With prep. & (MiB) \\', r'\midrule',
             begin.rstrip(), original_rows, end.rstrip(), r'\bottomrule', r'\end{tabular}', r'\end{minipage}\hfill',
             r'\begin{minipage}[t]{0.49\textwidth}', r'\centering', r'\begin{tabular}[t]{@{}lrrrr@{}}', r'\toprule',
             r'\multicolumn{5}{l}{\textit{(b) Matched configurations}} \\',
             r'Method & Params & \multicolumn{2}{c}{Latency (ms)} & Memory \\',
             r'& (M) & Resident & With prep. & (MiB) \\', r'\midrule', *matched_rows,
             r'\bottomrule', r'\end{tabular}', r'\end{minipage}',
             r'\par\smallskip{\footnotesize\raggedright Checkpoint medians on the same GPU within each database. (a) Validation-selected baselines. (b) Ten checkpoints per method across both outcomes, with shared preprocessing and transfer. Latency: batch 1; allocated memory: batch 32. With prep. includes preprocessing and transfers, excluding source-table construction.\par}',
             r'\end{table*}', '']
    return '\n'.join(lines)


def aggregation_results(root):
    folder = Path(root)/'runs'/AGGREGATION_RUN/'paired_results_v1'
    receipt = read(folder/'completion.json')
    assert receipt['complete']
    for name, digest in receipt['files'].items(): assert sha(folder/name) == digest
    result = read(folder/'verified-results.json')
    assert result['complete'] and result['training_runs'] == 0
    assert result['new_inferences'] == 168 and result['reused_inferences'] == 72
    assert result['protocol']['seeds'] == [48, 49, 50, 51, 52]
    assert len(result['rows']) == 24 and len(result['paired']) == 48
    return result


def aggregation_table(result):
    lines = [r'\begin{table*}[!t]', r'\centering\footnotesize', r'\renewcommand{\arraystretch}{1.0}', r'\setlength{\tabcolsep}{3pt}',
             r'\caption{Duration ($D$) versus window ($W$) summaries: five-seed perturbed AP (percent, mean $\pm$ sample SD). With $p/c$ denoting perturbed/clean inputs, $R=(A_{Dp}-A_{Dc})-(A_{Wp}-A_{Wc})$ and $G=A_{Dp}-A_{Wp}$ are in percentage points; positive favors duration. Brackets show 99.7917\% patient-cluster intervals. Holm $p_H$ adjusts 24 contrasts separately for $R$ and $G$. Clean AP appears in Table~\ref{tab:matched-controls}.}',
             r'\label{tab:aggregation-stress}', r'\begin{tabular}{llrrrrrr}', r'\toprule',
             r'Task & Perturbation & Duration AP & Window AP & $R$ [interval] & $p_H$ & $G$ [interval] & $p_H$ \\', r'\midrule']
    conditions = [('lab_drop25', r'Drop 25\%'), ('lab_drop50', r'Drop 50\%'),
                  ('lab_delay2h', 'Delay 2h'), ('lab_delay6h', 'Delay 6h')]
    for db in DATABASES:
        for endpoint, outcome in ((0, 'Mort.'), (1, 'LOS')):
            if db != DATABASES[0] or endpoint:
                lines.append(r'\midrule')
            for index, (condition, label) in enumerate(conditions):
                cell = next(x for x in result['rows'] if
                            (x['database'], x['endpoint'], x['condition']) == (db, endpoint, condition))
                values = [(NAMES[db]+' / '+outcome) if index == 0 else '', label]
                for method in ('duration', 'window'):
                    ap = cell['methods'][method]['perturbed']
                    values.append(rf"${100*ap['mean']:.2f}\pm{100*ap['sample_SD']:.2f}$")
                for estimand in ('degradation', 'absolute'):
                    row = next(x for x in result['paired'] if
                               (x['database'], x['endpoint'], x['condition'], x['estimand']) ==
                               (db, endpoint, condition, estimand))
                    low, high = row['adjusted_ci']
                    values += [rf"${100*row['mean']:+.2f}\ [{100*low:+.2f},{100*high:+.2f}]$",
                               f"{row['p_holm']:.3g}"]
                lines.append(' & '.join(values)+r' \\')
    return '\n'.join(lines+[r'\bottomrule', r'\end{tabular}', r'\end{table*}', ''])


def targeted_results(root):
    folder = Path(root)/'runs'/TARGETED_RUN/'paired_results_v1'
    for name, digest in read(folder/'completion.json')['files'].items():
        assert sha(folder/name) == digest
    result = read(folder/'verified-results.json')
    assert result['complete'] and result['new_fits'] == 51 and len(result['paired']) == 51
    assert result['protocol']['seeds'] == [48, 49, 50, 51, 52]
    return result


def targeted_cell(row, percent=False):
    scale, places = (100, 2) if percent else (1, 4)
    lo, hi = row['adjusted_ci']
    return (r'\shortstack[r]{' + rf"${scale*row['mean']:+.{places}f};\ p_H={row['p_holm']:.3g}$"
            + r'\\' + rf'$[{scale*lo:+.{places}f},{scale*hi:+.{places}f}]$' + '}')


def targeted_inline(row, percent=False):
    scale, places = (100, 2) if percent else (1, 4)
    lo, hi = row['adjusted_ci']
    return rf"${scale*row['mean']:+.{places}f}\ [{scale*lo:+.{places}f},{scale*hi:+.{places}f}]$"


def targeted_evidence_table(root, result):
    core_path = Path(root)/'runs/cem_core_five_seed_20261107/paired_results_v1/verified-results.json'
    assert hashlib.sha256(core_path.read_bytes()).hexdigest() == 'c911fa18764f0a75b972bfef05d86b0b1e50f4845386c249d20585487d2e56b0'
    core = read(core_path)
    assert core['complete'] and core['seeds'] == [48,49,50,51,52]
    lines = [r'\begin{table*}[!tbp]', r'\centering\small', r'\setlength{\tabcolsep}{1.6pt}',
             r'\caption{Added temporal evidence with five seeds: AP gains for CEM-Net ($\Delta C$), comparator ($\Delta B$), and interaction $I_E=\Delta C-\Delta B$. (a) Original recipes; 99.7222\% patient-cluster intervals and Holm $p_H$ over 18 contrasts. Comparators: TabM-mini for MIMIC-III LOS, TabM-mini+PLE for eICU, TabM+PLE otherwise. (b) Matched TabM-mini; 99.4444\% intervals and Holm $p_H$ over nine contrasts.}',
             r'\label{tab:evidence}', r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r'Task & $\Delta C$ [interval] & $p_H$ & $\Delta B$ [interval] & $p_H$ & $I_E$ [interval] & $p_H$ \\', r'\midrule',
             r'\multicolumn{7}{l}{(a) Original recipes: mean gains and interaction intervals} \\']
    edge = lambda x: f'{x:+.6f}' if 0 < abs(x) < .00005 else f'{x:+.4f}'
    for e, label in ((1, 'LOS'), (0, 'Mort.')):
        for db in DATABASES:
            v = {r['name']:r['five_seed'] for r in core['paired'] if
                 (r['database'],r['endpoint'],r['family']) == (db,e,'evidence_original_recipe')}
            i = v['interaction']
            cells = [NAMES[db]+' / '+label, rf"${v['CEM_full_minus_basic']['mean']:+.4f}$", '',
                     rf"${v['baseline_full_minus_basic']['mean']:+.4f}$", '',
                     rf"${i['mean']:+.4f}\ [{edge(i['adjusted_ci'][0])},{edge(i['adjusted_ci'][1])}]$", f"{i['p_holm_full_family']:.3g}"]
            lines.append(' & '.join(cells)+r' \\')
    lines += [r'\midrule', r'\multicolumn{7}{l}{(b) Matched predictor: gain and interaction intervals} \\']
    for db in DATABASES:
        e = 0 if db == 'eicu' else 1
        cells = [NAMES[db]+(' / Mort.' if e == 0 else ' / LOS')]
        for name in ('CEM_full_minus_basic', 'matched_full_minus_basic', 'evidence_interaction'):
            row = next(r for r in result['paired'] if (r['database'], r['name']) == (db, name))
            cells += [targeted_inline(row), f"{row['p_holm']:.3g}"]
        lines.append(' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule', r'\end{tabular}', r'\end{table*}', ''])


def targeted_aggregation_table(result, matched):
    lines = [r'\begin{table*}[!t]', r'\centering\small', r'\setlength{\tabcolsep}{4pt}',
             r'\caption{Temporal aggregation comparisons.}',
             r'\label{tab:targeted-aggregation}', r'\begin{tabular}{@{}lrrr@{}}', r'\toprule']
    for method, label in (('window',r'(a) Window summaries ($W$)'),
                          ('observation',r'(b) Observation-point summaries ($O$)')):
        if method=='observation':lines.append(r'\midrule')
        lines.append(r'\textit{'+label+r'} & Control AP & CEM-Net $-$ control [interval] & $p_H$ \\')
        for e in (1,0):
            for db in DATABASES:
                if method=='window':
                    row=next(r for r in matched['paired'] if
                             (r['database'],r['baseline'],r['endpoint'])==(db,'window_summary',e))
                    ap=ap_cell(matched,db,'window_summary',e)
                else:
                    row=next(r for r in result['paired'] if
                             (r['database'],r['endpoint'],r['name'])==(db,e,'duration_minus_observation'))
                    value=result['summaries'][f'{db}:observation:{e}']['AP']
                    ap=rf"${value['mean']:.4f}\pm{value['sample_SD']:.4f}$"
                lo,hi=row['adjusted_ci']
                cells=[NAMES[db]+' / '+('LOS' if e else 'Mortality'),ap,
                       rf"${row['mean']:+.4f}\ [{lo:+.4f},{hi:+.4f}]$",f"{row['p_holm']:.3g}"]
                lines.append(' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule', r'\end{tabular}',
        r'\par\smallskip{\footnotesize\raggedright AP mean $\pm$ sample SD. Differences compare duration ($D$, CEM-Net) with windows ($W$) or observation-point summaries ($O$). $D-W$: 99.5833\% intervals and 12-way Holm $p_H$; $D-O$: 99.1667\% intervals and six-way Holm $p_H$.\par}',
        r'\end{table*}', ''])


def targeted_cost_table(root, result):
    lines = [r'\begin{table*}[!t]', r'\centering\small', r'\setlength{\tabcolsep}{2pt}',
             r'\caption{Construction medians: seconds at 128/512/2,048 stays, three repetitions, two CPU threads. Full pipeline: MIMIC-III Xeon 8468; IV/eICU i7-14700KF. Matched laboratory block: III/IV Xeon 8468; eICU EPYC 7313, including read, admission, merge, normalization, sort and aggregation. $D/O/W$: duration/observation/window. RSS: peak process MiB for $D/O/W$ at 2,048 stays. OS caches remain for the full pipeline; the matched block uses warm-cache fresh processes.}',
             r'\label{tab:pipeline}', r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Database & Full pipeline & Lab: duration & Lab: observation & Lab: window & RSS ($D/O/W$) \\', r'\midrule']
    for db in DATABASES:
        old = read(Path(root)/'runs/cem_icde_evidence_20261028'/(db+'_secondary_v1/verified-results.json'))['pipeline_cost']
        cells = [NAMES[db], '/'.join(f"{r['median_seconds']:.2f}" for r in old['rows'])]
        rss = []
        for method, label in (('duration', '$D$'), ('observation', '$O$'), ('window', '$W$')):
            rows = [next(r for r in result['costs'] if (r['database'], r['method'], r['stays']) == (db, method, size))
                    for size in (128, 512, 2048)]
            cells.append('/'.join(f"{r['end_to_end_seconds']:.2f}" for r in rows))
            rss.append(f"{rows[-1]['peak_rss_mib']:.0f}")
        cells.append('/'.join(rss))
        lines.append(' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule', r'\end{tabular}', r'\end{table*}', ''])


def render_targeted(root, output):
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    assert output.is_relative_to(root) and root.drive.upper() != 'C:'
    result = targeted_results(root)
    output.mkdir(parents=True, exist_ok=True)
    for name, content in {'evidence.tex': targeted_evidence_table(root, result),
                          'targeted_aggregation.tex': targeted_aggregation_table(result,load_results(root)),
                          'pipeline.tex': targeted_cost_table(root, result)}.items():
        (output/name).write_text(content, encoding='utf8')
    render_visual_summaries(root, output.parents[1], result)
    return dict(targeted_contrasts=51, construction_measurements=81, new_experiments=0)


def render_visual_summaries(root, paper, result):
    """Plot saved contrasts and measured costs; no fitting or resampling."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory

    folder = Path(paper)/'figures/cem_icde72_20261029'
    folder.mkdir(parents=True, exist_ok=True)
    style = {'font.family':'Times New Roman', 'font.size':8.5,
             'axes.labelsize':8.5, 'axes.titlesize':9,
             'pdf.fonttype':42, 'ps.fonttype':42,
             'axes.spines.top':False, 'axes.spines.right':False}
    blue, orange, grey = '#0072B2', '#D55E00', '#50565C'
    core = read(Path(root)/'runs/cem_core_five_seed_20261107/paired_results_v1/verified-results.json')
    with plt.rc_context(style):
        fig = plt.figure(figsize=(7.16, 2.60))
        left = fig.add_axes([.273,.17,.177,.67])
        right = fig.add_axes([.713,.17,.240,.67])
        fig.text(.005,.965,'(a) Original recipes: difference in temporal-evidence gains',
                 fontsize=9, fontweight='bold', va='top')
        fig.text(.525,.965,'(b) Matched predictor: gains and their difference',
                 fontsize=9, fontweight='bold', va='top')
        fig.text(.162,.867,r'$\Delta C$',ha='center')
        fig.text(.230,.867,r'$\Delta B$',ha='center')
        fig.text(.485,.867,r'$p_H$',ha='right')
        fig.text(.995,.867,r'$p_H$',ha='right')
        rows = [(db,e) for e in (1,0) for db in DATABASES]
        trans = blended_transform_factory(fig.transFigure,left.transData)
        for idx,(db,e) in enumerate(rows):
            y = 5-idx
            vals = {r['name']:r['five_seed'] for r in core['paired'] if
                    (r['database'],r['endpoint'],r['family'])==(db,e,'evidence_original_recipe')}
            r = vals['interaction']; lo,hi=r['adjusted_ci']
            if idx<3:left.axhspan(y-.48,y+.48,color=blue,alpha=.045,lw=0,zorder=0)
            left.errorbar(r['mean'],y,xerr=[[r['mean']-lo],[hi-r['mean']]],
                          fmt='D',color=grey,markersize=3.4,capsize=2,lw=1.15)
            left.annotate(f"{r['mean']:+.4f}",(r['mean'],y),xytext=(0,4.5),
                          textcoords='offset points',ha='center',va='bottom',fontsize=8)
            fig.text(.005,y,NAMES[db]+(' / LOS' if e else ' / Mort.'),transform=trans,va='center')
            fig.text(.162,y,f"{vals['CEM_full_minus_basic']['mean']:+.4f}",transform=trans,ha='center',va='center',color=blue)
            fig.text(.230,y,f"{vals['baseline_full_minus_basic']['mean']:+.4f}",transform=trans,ha='center',va='center',color=orange)
            fig.text(.485,y,f"{r['p_holm_full_family']:.3g}",transform=trans,ha='right',va='center')
        left.set_ylim(-.60,5.65);left.set_xlim(-.020,.033)
        left.set_xticks([-.02,0,.02])
        left.set_xlabel(r'Gain difference $I_E=\Delta C-\Delta B$',labelpad=3)
        left.axhline(2.5,color='#bfc3c7',lw=.5)
        handles = [Line2D([],[],marker=m,color=c,lw=1,label=l,markersize=3.5)
                   for m,c,l in [('o',blue,'CEM-Net'),('s',orange,'Matched'),('D',grey,'Difference')]]
        fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.525,.930),ncol=3,
                   frameon=False,fontsize=7.9,columnspacing=.7,handlelength=1.0,handletextpad=.3)
        trans = blended_transform_factory(fig.transFigure,right.transData)
        specs = [('CEM_full_minus_basic',blue,'o'),('matched_full_minus_basic',orange,'s'),
                 ('evidence_interaction',grey,'D')]
        for idx,db in enumerate(DATABASES):
            center = 5-2*idx
            if idx<2:right.axhspan(center-.83,center+.83,color=blue,alpha=.045,lw=0,zorder=0)
            fig.text(.525,center,NAMES[db]+('\nLOS' if db!='eicu' else '\nMortality'),
                     transform=trans,va='center',linespacing=1.1)
            for offset,(name,color,marker) in zip((.55,0,-.55),specs):
                r = next(r for r in result['paired'] if (r['database'],r['name'])==(db,name))
                y=center+offset;lo,hi=r['adjusted_ci']
                right.errorbar(r['mean'],y,xerr=[[r['mean']-lo],[hi-r['mean']]],
                               fmt=marker,color=color,markersize=3.4,capsize=2,lw=1.15)
                right.annotate(f"{r['mean']:+.4f}",(r['mean'],y),xytext=(0,3.4),
                               textcoords='offset points',ha='center',va='bottom',fontsize=8,color=color)
                fig.text(.995,y,f"{r['p_holm']:.3g}",transform=trans,ha='right',va='center',fontsize=8)
        right.set_ylim(-.1,6.15);right.set_xlim(-.019,.023)
        right.set_xticks([-.01,0,.01,.02]);right.set_xlabel('AP gain / gain difference',labelpad=3)
        for ax in (left,right):
            ax.set_yticks([]);ax.spines['left'].set_visible(False)
            ax.axvline(0,color='#999999',lw=.7,ls='--',zorder=0)
            ax.grid(axis='x',color='#e5e7e9',lw=.4);ax.set_axisbelow(True)
            ax.tick_params(axis='x',labelsize=8,pad=2,length=2)
        for suffix in ('pdf','png'):
            fig.savefig(folder/('evidence_gain_summary.'+suffix),dpi=300)
        plt.close(fig)

        full=[];lab=[];rss=[]
        for db in DATABASES:
            old=read(Path(root)/'runs/cem_icde_evidence_20261028'/(db+'_secondary_v1/verified-results.json'))['pipeline_cost']
            assert old['complete'] and all(r['raw_reconstruction_exact'] for r in old['rows'])
            full.append([r['median_seconds'] for r in old['rows']])
            for method in ('duration','observation','window'):
                rows=[next(r for r in result['costs'] if (r['database'],r['method'],r['stays'])==(db,method,n))
                      for n in (128,512,2048)]
                lab.append([r['end_to_end_seconds'] for r in rows]);rss.append(rows[-1]['peak_rss_mib'])
        cmap=LinearSegmentedColormap.from_list('cem_time',['#f5f9fb','#d1e5ef','#7db8d3','#1f668c'])
        cmap.set_bad('#f1f2f3')
        fig=plt.figure(figsize=(7.16,1.50))
        for idx,db in enumerate(DATABASES):
            left=.055+idx/3
            ax=fig.add_axes([left,.245,.218,.515])
            values=np.array([full[idx],*lab[idx*3:idx*3+3]])
            colored=np.ma.array(values,mask=np.zeros_like(values,dtype=bool));colored.mask[0]=True
            ax.imshow(colored,aspect='auto',cmap=cmap,norm=Normalize(0,6))
            ax.set_xticks(range(3),['128','512','2,048']);ax.xaxis.tick_top()
            ax.set_yticks(range(4),['Full','D','O','W']);ax.tick_params(length=0,pad=3)
            fig.text(left-.035,.965,f'({chr(97+idx)}) {NAMES[db]}',fontsize=9,fontweight='bold',va='top')
            fig.text(left+.102,.935,'Stays',ha='center',va='top',fontsize=8)
            for y in range(4):
                for x in range(3):
                    ax.text(x,y,f'{values[y,x]:.2f}',ha='center',va='center',
                            color='white' if y>0 and values[y,x]/6>.62 else '#26343c',fontsize=8.5)
            ax.set_xticks(np.arange(-.5,3,1),minor=True)
            ax.set_yticks(np.arange(-.5,4,1),minor=True)
            ax.grid(which='minor',color='white',linewidth=1.3)
            ax.tick_params(which='minor',length=0)
            for spine in ax.spines.values():spine.set_visible(False)
            ax.axhline(.5,color='white',lw=3)
            trans=blended_transform_factory(fig.transFigure,ax.transData)
            fig.text(left+.247,.84,'RSS',ha='center',fontsize=8)
            fig.text(left+.247,.775,'(MiB)',ha='center',fontsize=7.5)
            fig.text(left+.247,0,'-',transform=trans,ha='center',va='center',color='#666666')
            for y,value in enumerate(rss[idx*3:idx*3+3],1):
                fig.text(left+.247,y,f'{value:.0f}',transform=trans,ha='center',va='center')
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,6),cmap=cmap),
                        cax=fig.add_axes([.52,.105,.27,.045]),orientation='horizontal',ticks=[0,3,6])
        cb.ax.tick_params(labelsize=7.5,length=2,pad=1);cb.outline.set_visible(False)
        fig.text(.49,.13,'Matched laboratory block: median seconds',ha='right',va='center',fontsize=8)
        for suffix in ('pdf','png'):
            fig.savefig(folder/('construction_summary.'+suffix),dpi=300)
        plt.close(fig)


def render(root, output):
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    assert output.is_relative_to(root) and root.drive.upper() != 'C:'
    result = load_results(root); resources = resource_summary(root)
    output.mkdir(parents=True, exist_ok=True)
    aggregation = aggregation_results(root)
    for name, content in {'matched_controls.tex': comparison_table(result), 'matched_cost.tex': resource_table(resources),
                          'aggregation_stress.tex': aggregation_table(aggregation)}.items():
        (output/name).write_text(content, encoding='utf8')
    cost = output/'cost.tex'
    if cost.exists():
        cost.write_text(combined_resource_table(cost.read_text(encoding='utf8'), resources), encoding='utf8')
    render_targeted(root, output)
    return dict(comparisons=12, seeds=5, resource_checkpoints=60, new_fits=60,
                aggregation_conditions=24, aggregation_contrasts=48, aggregation_training_runs=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--targeted-only', action='store_true'); args = parser.parse_args()
    print(json.dumps((render_targeted if args.targeted_only else render)(args.project, args.output)))
