"""This program build tables for the smi recovery experiement."""

import numpy as np
from collections import defaultdict

from src.reports.tables.util import stat_worse
from src.reports.tables import latex


def svgd_recovery_fn(fail, lppd, num_particles):
    """Determine the last run without error."""
    if fail:
        # GPU OOM trying to double the number of particles
        num_particles = num_particles[lppd > -float("inf")]

    # Experiment stops when lppd(svgd) >= lppd(smi)
    return num_particles[-1], len(num_particles)


def svgd_inf_time_fn(fail, inf_time):
    """Determine the last run without error."""
    if fail:
        # GPU OOM trying to double the number of particles
        inf_time = inf_time[inf_time < float("inf")]

    # Experiment stops when lppd(svgd) >= lppd(smi)
    return inf_time[-1], len(inf_time)


def pp_bound_latex(fail):
    if fail:
        return "> "
    return ""


def pprint_lppd_recover_latex(logger):
    header = r"""\begin{tabular}{lcccc}
    \toprule
    \multicolumn{5}{c}{Low-dimensional BNN}\\
    Region & $\textrm{R}$ & SMI Better?  & $\textrm{S}$ & SMI Better ?  \\
    \midrule"""
    res = defaultdict(list)

    k = "rbf"
    n = "num_particles"
    i = "inf_time"

    logs = logger.get_logs()

    def load(loc):
        np.load(logger.base_dir / loc)

    def pp_rec(fs, rs):
        return latex.eq(
            pp_bound_latex(np.any(fs)) + latex.perf_median(np.median(rs), 0)
        )

    def pp_sp(fs, ss):
        return latex.eq(
            pp_bound_latex(np.any(fs)) + latex.perf_mean(ss.mean(), ss.std(), 1, False)
        )

    for mt in ["low", "high"]:
        for dataset in logs[mt]["svgd"][k]:
            # Determine the recovery point where lppd(svgd(n), dataset) >= lppd(smi(c), dataset).
            # n is number of svgd particles and c is a fixed number of smi particles defined in
            # experiments/1d_syntehic_lppd/smi_recovery.
            svgd_fails = np.array(logs[mt]["svgd"][k][dataset]["fail"])
            svgd_lppd_locs = logs[mt]["svgd"][k][dataset]["lppd"]

            svgd_n_locs = logs[mt]["svgd"][k][dataset][n]

            recovery, rec_ids = zip(
                *[
                    svgd_recovery_fn(fail, load(lppd_loc), load(n_loc))
                    for fail, lppd_loc, n_loc in zip(
                        svgd_fails, svgd_lppd_locs, svgd_n_locs
                    )
                ]
            )
            recovery = np.array(recovery)
            assert (
                recovery.mean() > 0
            ), "SVGD with negative number of particles not possible."

            smi_n = np.array(logs[mt]["smi"][k][dataset][n])
            assert smi_n.std().item() == 0.0, "Number of SMI particles is not constant."

            smi_n_best = latex.eq(
                latex.better(stat_worse(smi_n, recovery, high_best=False))
            )

            # Determine the inference speedup at recovery point given by time(svgd) / time(smi).
            svgd_inf_times, it_ids = zip(
                *[
                    svgd_inf_time_fn(fail, load(i_loc))
                    for fail, i_loc in zip(svgd_fails, logs[mt]["svgd"][k][dataset][i])
                ]
            )

            svgd_inf_times = np.array(svgd_inf_times)
            assert np.all(
                np.array(rec_ids) == np.array(it_ids)
            ), "Recovery point and SVGD inference time disagree."

            smi_inf_time = np.array(logs[mt]["smi"][k][dataset][i])
            assert all(smi_inf_time > 0), "SMI inference times are not all > 0."

            speedup = svgd_inf_times / smi_inf_time.mean()

            smi_sp_best = latex.eq(latex.better(speedup.mean() >= 1))

            res[mt].append(
                "    "
                + " & ".join(
                    [dataset.capitalize(), pp_rec(svgd_fails, recovery), smi_n_best]
                    + [pp_sp(svgd_fails, speedup), smi_sp_best]
                )
                + r"\\"
            )

    low_res = "\n".join(res["low"])

    seperate = r"""    \midrule
    \multicolumn{5}{c}{High-dimensional BNN}\\
    \midrule"""

    high_res = "\n".join(res["high"])

    footer = r"""    \bottomrule
\end{tabular} """

    return "\n".join((header, low_res, seperate, high_res, footer))


if __name__ == "__main__":
    from src.logger import ExpLogger

    lppd_logger = ExpLogger("lppd_recover")
    lppd_logger.load_latest_logs()

    print(pprint_lppd_recover_latex(lppd_logger))
