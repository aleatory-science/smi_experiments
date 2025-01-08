from src.reports.tables.util import METHOD_ORDER, get_perfs, pp_metric
import dill

MT = "bnn"  # MODEL_TYPE


def pprint_split_latex(logs):
    methods = list(sorted(logs[MT].keys(), key=lambda m: METHOD_ORDER.index(m.upper())))

    datasets = sorted(logs[MT][methods[0]]["none"])
    metrics = ["nll", "rmse"]

    perfs, use_adjust = get_perfs(MT, datasets, methods, logs, metrics)

    rows = []
    for dataset in datasets:
        row = [f'    {dataset.split('-')[0].capitalize()}']
        for metric in metrics:
            row.append(pp_metric(perfs[dataset][metric], use_adjust, 1, high_best=False))
        rows.append(" & ".join(row) + r"\\")
    perf_res = "\n".join(rows)

    return perf_res


def pprint_uci_bench_latex(std_logger, gap_logger):
    std_logs = std_logger.get_logs()
    gap_logs = gap_logger.get_logs()

    methods = list(
        sorted(std_logs[MT].keys(), key=lambda m: METHOD_ORDER.index(m.upper()))
    )
    pp_methods = " & ".join(map(str.upper, methods))

    std_header = (
        r"""\begin{tabular}{lccccc|c|ccccc|c}
    \toprule
    \multicolumn{13}{c}{Standard UCI }\\
    &  \multicolumn{6}{c|}{NLL ($\downarrow$)} & \multicolumn{6}{c}{RMSE ($\downarrow$)} \\
    \cmidrule(lr){2-7} \cmidrule(lr){8-13} \\ 
    Dataset &"""
        + pp_methods
        + " & "
        + pp_methods
        + r"""\\
    \midrule"""
    )

    std_perf = pprint_split_latex(std_logs)

    gap_header = r"""    \midrule
    \multicolumn{11}{c}{Gap10 UCI}\\
    \midrule"""

    gap_perf = pprint_split_latex(gap_logs)

    footer = r"""    \bottomrule
\end{tabular}"""

    return "\n".join((std_header, std_perf, gap_header, gap_perf, footer))


def get_hparams(mt, datasets, methods, logs, root):
    hparams = {}
    for dataset in datasets:
        if dataset not in hparams:
            hparams[dataset] = {}
        for method in methods:
            loc = logs[MT][method]["none" if method in ["ovi", "map"] else "rbf"][
                dataset
            ]["artifact"][0]
            with open(root / loc, "rb") as f:
                hparams[dataset][method] = dill.load(f)["hyper_params"]
    return hparams


def pprint_uci_hparams(map_logger, rest_logger):
    map_logs = map_logger.get_logs()
    rest_logs = rest_logger.get_logs()

    methods = list(
        sorted(map_logs[MT].keys(), key=lambda m: METHOD_ORDER.index(m.upper()))
    )

    datasets = sorted(map_logs[MT][methods[0]]["none"])
    hparams = get_hparams(MT, datasets, methods, map_logs, map_logger.root)

    methods = list(
        sorted(rest_logs[MT].keys(), key=lambda m: METHOD_ORDER.index(m.upper()))
    )
    datasets = sorted(rest_logs[MT][methods[0]]["none"])
    rest_hparams = get_hparams(MT, datasets, methods, rest_logs, rest_logger.base_dir)
    for dataset, hs in rest_hparams.items():
        hparams[dataset].update(hs)

    rows = []
    for dataset, methods in sorted(hparams.items(), key=lambda item: item[0]):
        print(sorted(list(methods.keys())))
        lrs = [
            f'${hs['lr']:.0e}$'
            for method, hs in sorted(methods.items(), key=lambda item: item[0])
        ]

        rows.append(" & ".join([dataset.split("-")[0].capitalize()] + lrs) + r"\\")

    return "\n".join(rows)


if __name__ == "__main__":
    from src.logger import ExpLogger

    print("UCI Standard")
    map_logger = ExpLogger("uci_std")
    map_logger.load_latest_logs()
    rest_logger = ExpLogger("uci_std")
    rest_logger.load_logs("2024-08-28 13:46:40.875670")
    map_logger.merge(rest_logger)
    print(pprint_uci_bench_latex(map_logger))
    print("-" * 125)

    print("UCI Gap results")
    map_logger = ExpLogger("uci_gap")
    map_logger.load_latest_logs()
    rest_logger = ExpLogger("uci_gap")
    rest_logger.load_logs("2024-08-29 07:58:36.740033")
    map_logger.merge(rest_logger)
    print(pprint_uci_bench_latex(map_logger))
