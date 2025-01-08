import argparse
from src.constants import HELP_STRS, EXPERIMENTS


def build_subparser(name, parser):
    assert name in EXPERIMENTS, f"Unkown experiment {name}"

    match name:
        case "mnist":
            from experiments.img_classifier import build_argparse
        case "collapse":
            from experiments.variance_collapse import build_argparse
        case "uci":
            from experiments.uci_benchmark import build_argparse
        case "recover":
            from experiments.synthetic.smi_recovery import build_argparse
        case "1d-reg":
            from experiments.synthetic.in_between_uncertainty import build_argparse
        case _:
            raise Exception(f"Unknown experiment {name}")

    build_argparse(parser)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Run Experiments",
    )
    subparsers = parser.add_subparsers()
    for exp, help_str in zip(EXPERIMENTS, HELP_STRS):
        build_subparser(exp, subparsers.add_parser(exp, help=help_str))

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
