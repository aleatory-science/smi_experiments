import argparse

EXPS = ["collapse", "uci", "recover", "1d-reg"]
HELPS = ["Run the Gaussian variance estimation experiment.", "Run the UCI gap and standard split benchmarks.", "Run the SVGD recovery point experiment.", "Run the 1D regression with synthetic data regression experiment."]


def build_subparser(name, parser):
    assert name in EXPS, f"Unkown experiment {name}"

    match name:
        case "collapse":
            from experiments.variance_collapse import build_argparse
        case "uci":
            from experiments.uci_benchmark import build_argparse
        case "recover":
            from experiments.synthetic.smi_recovery import build_argparse
        case "1d-reg":
            from experiments.synthetic.in_between_uncertainty import build_argparse

    build_argparse(parser)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Run Experiments",
    )
    subparsers = parser.add_subparsers()
    for exp, help_str in zip(EXPS, HELPS):
        build_subparser(exp, subparsers.add_parser(exp, help=help_str))

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
