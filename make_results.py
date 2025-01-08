from src.reports.plots import sine_wave_hdi, sine_wave_data, variance_plot, postshape_plot, location_plot, all_hdi
from src.reports.tables import pprint_uci_bench_latex


import argparse
from src.reports.tables import pprint_mnist_latex
from src.constants import INDIST_EVAL_METRICS_NAMES


def make_plot(args):
    match args.plot:
        case "1d-reg":
            from src.logger.reg_logger import ExpLogger

            logger = ExpLogger("lppd_1d")
            article_ts = "2024-09-10 14:58:42.515864"
        case "collapse":
            from src.logger.reg_logger import ExpLogger
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
            postshape_plot(logger, add_std=False)
            location_plot(logger, add_std=False)

def load_log(logger, log_type, ats):
    match log_type:
        case "latest":
            logger.load_latest_logs()
        case "article":
            logger.load_logs(ats)
        case _:
            logger.load_logs(args.log)
    return logger


def make_table(args):
    match args.table:
        case "mnist":
            from src.logger.class_logger import ExpLogger
            l1_at = "2024-11-21 11:49:32.833326"  # 1 layer MLP
            l2_at = "2024-11-20 18:30:56.884221"  # 2 layer mlp
            ats = [l1_at, l2_at]
            loggers = [load_log(ExpLogger("mnist"), args.log, at) for at in ats]

        case "uci":
            from src.logger.reg_logger import ExpLogger
            assert args.log in ['latest', 'article'], f"timestamp not support for UCI"

            aritcle_tss = [ ["2024-09-30 16:26:50.652433", "2024-08-28 13:46:40.875670"],
                ["2024-09-26 14:44:27.705930", "2024-08-29 07:58:36.740033"],
            ]
            loggers = []

            for ats, name in zip(aritcle_tss, ['uci_std', 'uci_gap']):
                l1 = load_log(ExpLogger(name), args.log, ats[0])
                l2 = load_log(ExpLogger(name), args.log, ats[1])
                l1.merge(l2)
                loggers.append(l1)


    match args.table:
        case "mnist":
            latex = pprint_mnist_latex(*loggers, INDIST_EVAL_METRICS_NAMES)
        case "uci":
            latex = pprint_uci_bench_latex(*loggers)

    print(latex)


def plot_parser(parser):
    parser.prog = "Plotter"
    parser.add_argument("plot", choices=["1d-reg", 'collapse'], help="...")
    parser.add_argument(
        "log",
        nargs="?",
        type=str,
        default="latest",
        choices=["article", "latest", "timestamp"],
        help="Using `article` builds figs from downloaded logs (see README),"
        "`latest` builds figs using the latest timestamp,"
        "`timestamp` uses specific timestamp directory name in logs.",
    )
    parser.add_argument("format", nargs="?", type=str, default="png")
    parser.set_defaults(func=make_plot)


def table_parser(parser):
    parser.add_argument(
        "table",
        choices=[
            "mnist",
            "uci",
        ],  # help=["Build table in E.3.1 of GGN article.", "..."]
    )
    parser.add_argument(
        "log",
        nargs="?",
        type=str,
        default="latest",
        # choices=["article", "latest", "timestamp"],
        help="Using `article` builds figs from downloaded logs (see README),"
        "`latest` builds figs using the latest timestamp,"
        "`timestamp` uses a timestamp directory name from logs.",
    )
    parser.set_defaults(func=make_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Visualize results",
    )

    subparsers = parser.add_subparsers()
    plot_parser(subparsers.add_parser("plot", help="Build figs from the article."))
    table_parser(subparsers.add_parser("table", help="Build tables from the article."))

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
