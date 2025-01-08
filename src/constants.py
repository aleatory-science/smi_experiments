from src.metrics import nll_fn, acc_fn, brier_fn, ece_fn, mce_fn, conf_fn

EXPERIMENTS = [
    # Classification experiments
    "mnist",
    # Regression experiments
    "collapse",
    "uci",
    "1d-reg",
    # Unsupervised experiements
    "recover",
]

# Note: HELP_STRS order must be the same as EXPERIMENTS
HELP_STRS = [
    "Run and evalaute MNIST",
    "Run the UCI gap and standard split benchmarks.",
    "Run the SVGD recovery point experiment.",
    "Run the 1D regression with synthetic data regression experiment.",
    "Run the Gaussian variance estimation experiment.",
]

METHODS = sorted(["nuts", "map", "asvgd", "svgd", "smi", "ovi"])

# Classification metrics
INDIST_EVAL_METRICS = {
    "nll": nll_fn,
    "acc": acc_fn,
    "brier": brier_fn,
    "ece": ece_fn,
    "mce": mce_fn,
    "conf": conf_fn,
}
reparam_article_order = ["conf", "nll", "acc", "brier", "ece", "mce"]
INDIST_EVAL_METRICS_NAMES = sorted(
    tuple(INDIST_EVAL_METRICS.keys()), key=lambda k: reparam_article_order.index(k)
)
