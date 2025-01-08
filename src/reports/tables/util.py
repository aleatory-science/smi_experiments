from collections import defaultdict
import numpy as np

from scipy.stats import mannwhitneyu
from src.reports.tables import latex

from pathlib import Path

METHOD_ORDER = ["SMI", "SVGD", "ASVGD", "OVI", "MAP", "NUTS"]


TAB_DIR = Path(__file__).parent / "tables"
TAB_DIR.mkdir(exist_ok=True)


def stat_worse(pop1, pop2, high_best):
    """Population 2 is staticially above or below population 1."""
    plevel = 5e-2
    alt = "greater" if high_best else "less"
    return mannwhitneyu(pop1, pop2, alternative=alt).pvalue < plevel


def pp_metric(ps, use_adjust, precision, high_best):
    def pretty_print(avg, std, worse, worse_vi):
        return latex.eq(
            latex.state_better_same_vi(
            latex.state_better_same(
                latex.perf_mean(avg, std, precision, use_adjust), not worse
            ), not worse_vi
            )
        )

    avgs = np.mean(ps, 1)
    stds = np.std(ps, 1)

    bidx = np.argmax(avgs) if high_best else np.argmin(avgs)
    worse = np.array([stat_worse(ps[bidx], ps[i], high_best) for i in range(len(ps))])

    n = avgs.shape[0]
    bidx = np.argmax(avgs[:-1]) if high_best else np.argmin(avgs[:-1])
    worse_vi = np.array([stat_worse(ps[bidx], ps[i], high_best) or i == (n-1)for i in range(len(ps))])

    return " & ".join(map(pretty_print, avgs, stds, worse, worse_vi))


def get_perfs(mt, datasets, methods, logs, metrics):
    ps = {}
    use_adjust = {m: False for m in metrics}
    for dataset in datasets:
        if dataset not in ps:
            ps[dataset] = defaultdict(list)
        for metric in metrics:
            for method in methods:
                p = np.array(
                    logs[mt][method][
                        "none" if method in ["ovi", "map", "nuts"] else "rbf"
                    ][dataset][metric]
                )

                ps[dataset][metric].append(p)
                use_adjust[metric] = np.any(p.mean() < 0) or use_adjust[metric]
    return ps, use_adjust
