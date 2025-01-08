from collections import defaultdict
import numpy as np

from scipy.stats import mannwhitneyu
from src.reports.tables import latex
from src.constants import METHODS, INDIST_EVAL_METRICS_NAMES


MT = "mlp"  # MODEL_TYPE


def stat_worse(pop1, pop2, high_best):  # TODO: this needs to be fixed!
    """Population 2 is staticially above or below population 1."""
    plevel = 5e-2
    alt = "greater" if high_best else "less"
    return mannwhitneyu(pop1, pop2, alternative=alt).pvalue < plevel


def pp_metric(ps, use_adjust, precision, worse):
    def pretty_print(avg, std, worse):
        return latex.eq(
            latex.state_better_same(
                latex.perf_mean(avg, std, precision, use_adjust), not worse
            )
        )
    avgs = np.mean(ps, 1)
    stds = np.std(ps, 1)

    return " & ".join(map(pretty_print, avgs, stds, [worse]))


def get_perfs(mt, datasets, methods, logs, base_dir, metrics):
    ps = {}
    use_adjust = {m: False for m in metrics}
    for dataset in datasets:
        if dataset not in ps:
            ps[dataset] = {}
        for method in methods:
            if method not in ps[dataset]:
                ps[dataset][method] = defaultdict(list)
            for metric in metrics:
                p = logs[mt][method][dataset][metric]
                if len(p) > 0 and isinstance(p[0], str) and "npy" in p[0]:
                    p = np.concatenate([np.load(base_dir / loc) for loc in p])
                else:
                    p = np.array(p)
                ps[dataset][method][metric].append(p)
                use_adjust[metric] = np.any(p.mean() < 0) or use_adjust[metric]
    worse = {}
    for dataset in datasets:
        if dataset not in worse:
            worse[dataset] = {}
        for metric in metrics:

            metric_worse = get_worse([ps[dataset][method][metric] for method in methods], high_best(metric))
            if metric not in worse[dataset]:
                    worse[dataset][metric] = {method: is_worse for  method, is_worse in zip(methods, metric_worse)}

    return ps, use_adjust, worse

def get_worse(ps, high_best):
    avgs = np.array([np.mean(p) for p in ps])

    bidx = np.argmax(avgs) if high_best else np.argmin(avgs)
    worse = np.array([stat_worse(np.array(ps[bidx]).flatten(), np.array(ps[i]).flatten(), high_best) for i in range(len(ps))])
    return worse

def high_best(metric):
    high = {'conf', 'acc'}
    low = {'nll', 'brier', 'ece', 'mci'} 
    return (metric in high) and (metric not in low)

def pp_perfs(logs, base_dir, metrics, dataset):
    methods = list(sorted(logs[MT].keys(), key=lambda m: METHODS.index(m)))

    datasets = sorted(logs[MT][methods[0]])

    perfs, use_adjust, worse = get_perfs(MT, datasets, methods, logs, base_dir, metrics)

    rows = []
    for method in methods:
        row = [f"  {method.upper()}"]
        for metric in metrics:
            row.append(
                pp_metric(perfs[dataset][method][metric], False, 3, worse[dataset][metric][method])
            )  
        rows.append(" & ".join(row) + r"\\")
    perf_res = "\n".join(rows)

    return perf_res


def latex_header(dataset, methods, metrics):
    n = len(metrics)
    pp_metrics = map(pp_indist_metric, metrics)

    header = (
        f"\\begin{{tabular}}{{l{'c'*n}}}\n"
        "  \\toprule \n"
        f"  \\multicolumn{{{n + 1}}}{{c}}{{{dataset.upper()}}}" + r" \\" + "\n"
        f"  Method & {' & '.join(pp_metrics)}" + r" \\"
    )

    return header


def pp_indist_metric(metric):
    assert metric in INDIST_EVAL_METRICS_NAMES, f"Unknown metric {metric}."
    match metric:  # Fix capitalization
        case "conf" | "acc" | "brier":
            pp = metric.capitalize()
        case "nll" | "ece" | "mce":
            pp = metric.upper()

    match metric:
        case "conf" | "acc":
            return pp + " ($\\uparrow$)"
        case "brier" | "nll" | "ece" | "mce":
            return pp + " ($\\downarrow$)"


def _pprint_mnist_latex(logger, metrics):
    root = logger.root
    logs = logger.get_logs()

    methods = list(sorted(logs[MT].keys(), key=lambda m: METHODS.index(m)))

    datasets = sorted(logs[MT][methods[0]])

    header = latex_header(datasets[0], methods, metrics)

    perf = pp_perfs(logs, root, metrics, datasets[0])

    footer = " \\bottomrule\n" "\\end{tabular}"

    return "\n".join((header, perf, footer))

def pprint_mnist_latex(l1_logger, l2_logger, metrics):

    tables = ('----1 layered BNN----',
    _pprint_mnist_latex(l1_logger, metrics),
    '----2 layered BNN----',
    _pprint_mnist_latex(l2_logger, metrics))

    return "\n".join(tables)
