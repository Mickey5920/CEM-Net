"""Render the matched-control displays from verified aggregate results only."""
import argparse
import hashlib
import json
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
             r'\caption{Matched TabM-mini: five-seed Test AP (mean $\pm$ sample SD) and CEM-Net-minus-control differences. III/IV: MIMIC; L/M: LOS/mortality. CEM-Net AP is in Table~\ref{tab:primary}. Brackets show 99.5833\% patient-cluster intervals; Holm $p_H$ retains the 12-contrast family including windows in Table~\ref{tab:targeted-aggregation}.}',
             r'\label{tab:matched-controls}', r'\begin{tabular}{lrrr}', r'\toprule',
             r'Task & Matched AP & $\Delta$AP [interval] & $p_H$ \\', r'\midrule']
    for endpoint, label in ((1, 'LOS'), (0, 'Mortality')):
        for db in DATABASES:
            row=next(r for r in result['paired'] if (r['database'],r['baseline'],r['endpoint'])==(db,'matched_tabm',endpoint))
            cells = [dict(mimiciii='III',mimic='IV',eicu='eICU')[db]+'/'+('M' if endpoint==0 else 'L'),
                     ap_cell(result, db, 'matched_tabm', endpoint),targeted_inline(row),f"{row['p_holm']:.3g}"]
            lines.append(' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


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
        for variant, label in (('matched_tabm', 'Matched TabM-mini'), ('reference', 'CEM-Net')):
            value = summaries[db+':'+variant]
            fields = [label, f"{value['parameters']/1e6:.2f}"]
            fields += [f"{value['batches']['1'][key]:.2f}" for key in ('resident_ms', 'transfer_ms')]
            fields.append(f"{value['batches']['32']['allocated_mib']:.0f}")
            matched_rows.append(' & '.join(fields)+r' \\')
    lines = [r'\begin{table*}[!t]', r'\centering\small', r'\setlength{\tabcolsep}{2.5pt}',
             r'\caption{Predictor costs: parameters (millions), batch-32 allocated memory (MiB), and batch-1 latency (ms), resident forward versus preprocessing/transfers plus prediction. Source-table processing is excluded. (a) Original configurations: checkpoint medians, same GPU per database, validation-selected baselines. (b) Separate matched measurements for CEM-Net and matched TabM-mini: ten checkpoints per method (five seeds, two outcomes), with shared input transformation and transfer.}',
             r'\label{tab:cost}', r'\label{tab:matched-cost}',
             r'\begin{minipage}[t]{0.49\textwidth}', r'\centering', r'\begin{tabular}[t]{lrrrr}', r'\toprule',
             r'\multicolumn{5}{l}{\textit{(a) Original configurations}} \\',
             r'Method & Params & Resident & With prep. & Memory \\', r'\midrule',
             begin.rstrip(), original_rows, end.rstrip(), r'\bottomrule', r'\end{tabular}', r'\end{minipage}\hfill',
             r'\begin{minipage}[t]{0.49\textwidth}', r'\centering', r'\begin{tabular}[t]{lrrrr}', r'\toprule',
             r'\multicolumn{5}{l}{\textit{(b) Matched configurations}} \\',
             r'Method & Params & Resident & With prep. & Memory \\', r'\midrule', *matched_rows,
             r'\bottomrule', r'\end{tabular}', r'\end{minipage}', r'\end{table*}', '']
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
    lines = [r'\begin{table*}[!t]', r'\centering\small', r'\setlength{\tabcolsep}{2.5pt}',
             r'\caption{Five-seed aggregation controls replacing duration ($D$) with windows ($W$) or observation-point summaries ($O$). AP: mean $\pm$ sample SD; differences: CEM-Net minus control. $D-W$ retains 99.5833\% intervals and 12-way Holm $p_H$; $D-O$ uses 99.1667\% intervals and six-way Holm $p_H$.}',
             r'\label{tab:targeted-aggregation}', r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Task & AP ($W$) & $D-W$ [interval] & AP ($O$) & $D-O$ [interval] \\', r'\midrule']
    def find(db, e, name):
        return next(r for r in result['paired'] if (r['database'], r['endpoint'], r['name']) == (db, e, name))
    for e, label in ((1, 'LOS'), (0, 'Mort.')):
        for db in DATABASES:
            ap = result['summaries'][f'{db}:observation:{e}']['AP']
            row = find(db, e, 'duration_minus_observation')
            cells = [NAMES[db]+' / '+label,ap_cell(matched,db,'window_summary',e),contrast_cell(matched,db,'window_summary',e),
                     rf"${ap['mean']:.4f}\pm{ap['sample_SD']:.4f}$",targeted_cell(row)]
            lines.append(' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule', r'\end{tabular}', r'\end{table*}', ''])


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
    return dict(targeted_contrasts=51, construction_measurements=81, new_experiments=0)


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
