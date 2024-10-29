from src.reports.plots import sine_wave_hdi, all_hdi, sine_wave_data, variance_plot
from src.reports.tables import pprint_uci_bench_latex, pprint_lppd_recover_latex

from src.logger import ExpLogger

import argparse


def make_plot(args):
    match args.plot:
        case "1d-reg":
            logger = ExpLogger("lppd_1d")
            article_ts = "2024-09-10 14:58:42.515864"
        case "collapse":
            logger = ExpLogger("var")
            article_ts = "2024-09-30 12:29:00.384211"

    if args.plot != "data":
        match args.log:
            case "latest":
                logger.load_latest_logs()
            case "article":
                logger.load_logs(article_ts)
            case _:
                logger.load_logs(args.log)

    match args.plot:
        case "data":
            sine_wave_data(args.format)
        case "1d-reg":
            all_hdi(logger, args.format)
            sine_wave_hdi(logger, args.format)
        case "collapse":
            variance_plot(logger, add_std=False)


def make_table(args):
    match args.table:
        case "recover":
            loggers = [ExpLogger("lppd_recover")]
            aritcle_tss = ["2024-08-15 12:41:14.732724"]
        case "uci":
            loggers = [
                [ExpLogger("uci_std"), ExpLogger("uci_std")],
                [ExpLogger("uci_gap"), ExpLogger("uci_gap")],
            ]
            # Map was run in separate run
            aritcle_tss = [
                ["2024-09-30 16:26:50.652433", "2024-08-28 13:46:40.875670"],
                ["2024-09-26 14:44:27.705930", "2024-08-29 07:58:36.740033"],
            ]

    for logger, ats in zip(loggers, aritcle_tss):
        match args.log:
            case "latest":
                if args.table == "recover":
                    logger.load_latest_logs()
                else:
                    logger[0].load_latest_logs()
            case "article":
                if args.table == "recover":
                    logger.load_logs(ats)
                else:
                    logger[0].load_logs(ats[0])
                    logger[1].load_logs(ats[1])
                    logger[0].merge(logger[1])
            case _:
                assert (
                    args.table != "uci"
                ), "UCI printer only support latest and article"
                loggers.load_logs(args.log)

    if args.table == "uci":
        loggers = [ls[0] for ls in loggers]

    match args.table:
        case "recover":
            latex = pprint_lppd_recover_latex(*loggers)
        case "uci":
            latex = pprint_uci_bench_latex(*loggers)

    print(latex)


def plot_parser(parser):
    parser.prog = "Plotter"
    parser.add_argument("plot", choices=["collapse", "1d-reg", "data"], 
                        help="Using `collapse` builds fig. 2, `1d-reg` builds fig. 3 and 5, and `data` builds fig. 4")
    parser.add_argument(
        "log", type=str, default="latest", choices=["article", "latest", "timestamp"], 
        help="Using `article` builds figs from downloaded logs (see README)," 
             "`latest` builds figs using the latest timestamp,"
             "`timestamp` uses specific timestamp directory name in logs."
    )
    parser.add_argument("format", type=str, default="png")
    parser.set_defaults(func=make_plot)


def table_parser(parser):
    parser.add_argument("table", choices=["uci", "recover"],
                        help="Using `uci` builds table 3 and `recover` builds table 2.")
    parser.add_argument(
        "log", type=str, default="latest", choices=["article", "latest", "timestamp"],
        help="Using `article` builds figs from downloaded logs (see README)," 
             "`latest` builds figs using the latest timestamp,"
             "`timestamp` uses a timestamp directory name from logs."
    )
    parser.set_defaults(func=make_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Visualize results",
    )

    subparsers = parser.add_subparsers()
    plot_parser(subparsers.add_parser("plot", help='Build figs from the article.'))
    table_parser(subparsers.add_parser("table", help='Build tables from the article.'))

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
