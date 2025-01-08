r"""This program evaluates RMSE and NLL on the UCI benchmark with standard train/test splits from [1] and 10% gap constructed splits [5].

The UCI datasets are:
 - Boston housing,
 - Concrete,
 - Kin8nm,
 - Naval propulsion plant,
 - Protein tertiary structure,
 - Wine quality red,
 - Yacht.


Hyperparams are chosen using grid search at an order of magnitude granularity.
The search is performed by splitting the first UCI standard split of the train in each dataset into 80%-20% (train-validation).

The hyperparameters

 - LR in (5*10^-1,.., 5*10^-6)

The methods include:
 - Stein Variational Gradient Descent (SVGD) [2]
 - Annealing SVGD (ASVGD) [3]
 - Stein Mixture Infernce (SMI) [Ours]
 - Ordinary VI (OVI) [4]

SVGD, ASVGD and SMI use early stopping criteria to select the number of steps dynamically.

All string identifiers are lowercase.



### Refs (MLA):
    [1] Hernández-Lobato, José Miguel, and Ryan Adams. "Probabilistic backpropagation for scalable learning of Bayesian neural networks."
        International conference on machine learning. PMLR, 2015.
    [2] Liu, Qiang, and Dilin Wang. "Stein variational gradient descent: A general purpose Bayesian inference algorithm."
        Advances in neural information processing systems 29 (2016).
    [3] D'Angelo, Francesco, and Vincent Fortuin. "Annealed stein variational gradient descent." arXiv preprint arXiv:2101.09815 (2021).
    [4] Ranganath, Rajesh, Sean Gerrish, and David Blei. "Black box variational inference." Artificial intelligence and statistics. PMLR, 2014.
    [5] Foong, Andrew YK, et al. "'In-Between'Uncertainty in Bayesian Neural Networks." arXiv preprint arXiv:1906.11537 (2019).

"""

# DATASET
from datasets.uci_reg.load_uci import (
    load_datasets,
    load_standard_splits,
    load_modified_gap_splits,
)

# METHODS
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam

from numpyro.infer import NUTS, MCMC

from src.methods import SVGD, ASVGD
from numpyro.contrib.einstein import SteinVI, RBFKernel
from numpyro.infer.autoguide import AutoNormal, AutoDelta
from numpyro.infer.initialization import init_to_uniform

# UCI BNN
from src.models.uci_bnn import model, LATENT, OUT

# GRID-SEARCH
from src.model_select import grid_search, OnlineEarlyStop
from sklearn.model_selection import train_test_split

from src.metrics import (
    rmse_fn,
    nll_reg_fn,
    smi_sampler,
    svgd_sampler,
    svi_sampler,
    mcmc_sampler,
)
from src.preprocess import normalize

from src.constants import METHODS

from src.logger.reg_logger import ExpLogger

from time import time
from functools import partial
from jax import numpy as jnp
from numpyro.handlers import seed
from numpyro import prng_key, set_platform
from numpyro.diagnostics import summary
from experiments.util import setup_logger

UCI = load_datasets()
STD_SPLITS = load_standard_splits(
    use_validation=False
)  # uci_key -> [(tr_idx^(i), te_idx^(i)) for i in range(n(uci_key))]
GAP_SPLITS = load_modified_gap_splits(
    use_validation=False, test_size=1.0 / 10.0
)  # uci_key -> [(tr_idx^(i), te_idx^(i)) for i in range(n(uci_key))]

RNG_SEED = 10

ELBO_DRAWS = 100
NUM_PARTICLES = 5
PERIOD = 35  # early stopping
MAX_STEPS = 60_000
NUM_DRAWS = 5_000
SUBSAMPLE_SIZE = 100
MCMC_WARMUP = 500
MCMC_DRAWS = 3000

# HYPERPARAM SEARCH
PARAM_SPACE = {"lr": [5 * 10 ** (-i) for i in range(1, 7)]}
VAL_SPLT = 0.1

EXP_CONFIG = {
    "elbo_draws": ELBO_DRAWS,
    "num_particles": NUM_PARTICLES,
    "period": PERIOD,
    "max_steps": MAX_STEPS,
    "num_draws": NUM_DRAWS,
    "subsample_size": SUBSAMPLE_SIZE,
    "param_space": PARAM_SPACE,
    "validation_size": VAL_SPLT,
    "mcmc_warmup": MCMC_WARMUP,
    "mcmc_draws": MCMC_DRAWS,
}


def set_exp_config(config):
    # assert all(
    #     map(lambda k: k in config, EXP_CONFIG.keys())
    # ), "Experiment config missing hyperparameters. Use --logs new or different timestamp."
    global \
        ELBO_DRAWS, \
        NUM_PARTICLES, \
        PERIOD, \
        MAX_STEPS, \
        NUM_DRAWS, \
        SUBSAMPLE_SIZE, \
        PARAM_SPACE, \
        VAL_SPLT, \
        MCMC_WARMUP, \
        MCMC_DRAWS

    ELBO_DRAWS = config["elbo_draws"]
    NUM_PARTICLES = config["num_particles"]
    PERIOD = config["period"]
    MAX_STEPS = config["max_steps"]
    NUM_DRAWS = config["num_draws"]
    SUBSAMPLE_SIZE = config["subsample_size"]
    PARAM_SPACE = config["param_space"]
    VAL_SPLT = (config["validation_size"],)
    MCMC_WARMUP = config["mcmc_warmup"]
    MCMC_DRAWS = config["mcmc_draws"]


def setup_engine(method, kernel, lr):
    assert method in [
        "nuts",
        "asvgd",
        "svgd",
        "ovi",
        "smi",
        "map",
    ], f"Unknown method {method}."
    assert kernel in ["rbf", "none"], f"Unknown kernel {kernel}."

    match kernel:
        case "rbf":
            k = RBFKernel()
        case "none":
            assert method in [
                "ovi",
                "nuts",
                "map",
            ], "Only NUTS, OVI and MAP does not use a kernel"

    match method:
        case "smi" | "ovi":
            m = model
            g = AutoNormal(m, init_loc_fn=partial(init_to_uniform, radius=0.1))
        case "asvgd" | "svgd" | "nuts":
            m = model
        case "map":
            m = model
            g = AutoDelta(m, init_loc_fn=partial(init_to_uniform, radius=0.1))

    match method:
        case "smi":
            e = SteinVI(
                m,
                g,
                Adam(lr),
                k,
                num_stein_particles=NUM_PARTICLES,
                num_elbo_particles=ELBO_DRAWS,
            )
        case "svgd":
            e = SVGD(
                m,
                Adam(lr),
                k,
                num_stein_particles=NUM_PARTICLES,
                guide_kwargs={"init_loc_fn": partial(init_to_uniform, radius=0.1)},
            )
        case "asvgd":
            e = ASVGD(
                m,
                Adam(lr),
                k,
                num_stein_particles=NUM_PARTICLES,
                guide_kwargs={"init_loc_fn": partial(init_to_uniform, radius=0.1)},
            )
        case "ovi" | "map":
            e = SVI(m, g, Adam(lr), Trace_ELBO())
        case "nuts":
            mcmc_kernel = NUTS(m)
            e = MCMC(
                mcmc_kernel,
                num_samples=MCMC_DRAWS,
                num_warmup=MCMC_WARMUP,
                progress_bar=True,
                jit_model_args=True,
            )
            e.model = m

    match method:
        case "smi" | "svgd" | "asvgd":

            def run(key, *args, **kwargs):
                OnlineEarlyStop(e, PERIOD).run(
                    key,
                    MAX_STEPS,
                    *args,
                    **kwargs,
                    progress_bar=False,
                    subsample=SUBSAMPLE_SIZE,
                )
        case "ovi" | "map":

            def run(key, *args, **kwargs):
                return e.run(
                    key,
                    MAX_STEPS,
                    *args,
                    **kwargs,
                    progress_bar=False,
                    subsample=SUBSAMPLE_SIZE,
                )
        case "nuts":

            def run(key, *args, **kwargs):
                return e.run(
                    key,
                    *args,
                    **kwargs,
                    subsample=None,
                )

    return e, run


def setup_measure(method):
    """Setup root mean squared error (RMSE), negative log-likelihood (NLL) and time measure."""
    assert method in [
        "nuts",
        "map",
        "svgd",
        "asvgd",
        "smi",
        "ovi",
    ], f"Unknown method {method}."

    match method:
        case "svgd" | "asvgd":
            post_sampler = svgd_sampler
            batch_ndims = 2
            draws = NUM_DRAWS
        case "smi":
            post_sampler = smi_sampler
            batch_ndims = 1
            draws = NUM_DRAWS
        case "ovi" | "map":
            post_sampler = svi_sampler
            batch_ndims = 1
            draws = NUM_DRAWS
        case "nuts":
            post_sampler = mcmc_sampler
            batch_ndims = 1
            draws = MCMC_DRAWS

    def time_measure(fn, *args, **kwargs):
        st = time()
        res = fn(*args, **kwargs)
        time_taken = time() - st
        assert time_taken >= 0.0, "Time not positive"
        return res, time_taken

    def rmse_measure(key, dataset, engine, inf_results):
        """This method assumes x is normalized."""
        nonlocal post_sampler, batch_ndims, draws

        x, y = dataset
        check_test_normed(x)

        post = post_sampler(engine, inf_results, draws, OUT)(key, x, None, None)
        return rmse_fn(post, y, batch_ndims).item()

    def nll_measure(key, dataset, engine, inf_results):
        """This method assumes x is normalized."""
        nonlocal post_sampler, batch_ndims, draws

        x, y = dataset
        check_test_normed(x)

        post = post_sampler(engine, inf_results, draws, LATENT)(key, x, None, None)

        return nll_reg_fn(post, model, x, y, batch_ndims).item()

    return time_measure, rmse_measure, nll_measure


def check_test_normed(x):
    # x (from test) is centered with respect to train so x.mean(1) isn't
    # necessarily close to 0.
    assert jnp.abs(jnp.mean(x)).item() < 1.0, "Test x far from centered train x."
    # x (from test) is standardized with respect to train so x.std(1) isn't
    # necessarily close to 1.
    assert jnp.std(x).item() < 2.0, "Test x std large compared to train x."


def search_hparams(inf_key, eval_key, setup_engine, eval_measure, xtr, ytr):
    xtr, xval, ytr, yval = train_test_split(
        xtr, ytr, test_size=VAL_SPLT, random_state=RNG_SEED
    )

    xtr, mtr, std_tr = normalize(xtr, None, None)

    xval, _, _ = normalize(xval, mtr, std_tr)
    check_test_normed(xval)

    best_params = grid_search(
        inf_key=inf_key,
        eval_key=eval_key,
        setup_engine=setup_engine,
        eval_fn=eval_measure,
        tr_args=(),
        tr_kwargs={"x": xtr, "y": ytr},
        val_args=(xval, yval),
        search_space=PARAM_SPACE,
    )

    return best_params


def run_exps(rng_seed, method, kernel, split):
    time_measure, rmse_measure, nll_measure = setup_measure(method)

    match split:
        case "std":
            splits = STD_SPLITS
        case "gap":
            splits = GAP_SPLITS

    with seed(rng_seed=rng_seed):
        for dataset, (x, y) in sorted(UCI.items(), key=lambda item: item[0]):
            print(method, dataset)
            for i, idx in enumerate(splits[dataset]):
                xtr, ytr = x[idx["tr"]], y[idx["tr"]]
                if i == 0:
                    # x is not normalized because search hparams
                    # splits into train and validation set. Normalizing before
                    # splitting would break the IID assumption of xval.

                    if method != "nuts":
                        hparams = search_hparams(
                            inf_key=prng_key(),
                            eval_key=prng_key(),
                            setup_engine=lambda **kwargs: setup_engine(
                                method, kernel, **kwargs
                            ),
                            eval_measure=rmse_measure,
                            xtr=xtr,
                            ytr=ytr,
                        )
                    else:
                        hparams = {"lr": None}

                xtr, mtr, std_tr = normalize(xtr, None, None)
                e, run = setup_engine(method, kernel, **hparams)

                inf, inf_time = time_measure(run, prng_key(), x=xtr, y=ytr)

                xte, yte = x[idx["te"]], y[idx["te"]]
                xte, _, _ = normalize(xte, mtr, std_tr)

                rmse, rmse_sample_time = time_measure(
                    rmse_measure, prng_key(), (xte, yte), e, inf
                )
                nll, nll_sample_time = time_measure(
                    nll_measure, prng_key(), (xte, yte), e, inf
                )

                result = {
                    "rmse": rmse,
                    "rmse_sample_time": rmse_sample_time,
                    "nll": nll,
                    "nll_sample_time": nll_sample_time,
                    "inf_time": inf_time,
                    "losses": inf.losses.tolist() if method != "nuts" else None,
                }

                if method == "nuts":
                    e.print_summary()
                artifact = {
                    "model": e.model if method != "nuts" else None,
                    "guide": e.guide if method != "nuts" else None,
                    "params": inf.params if method != "nuts" else None,
                    "hyper_params": hparams,
                    "post_draws": e.get_samples(group_by_chain=True)
                    if method == "nuts"
                    else {},
                    "summary": summary(e.get_samples(group_by_chain=True))
                    if method == "nuts"
                    else {},
                }

                yield dataset, result, artifact


def setup_experiment(method):
    assert method in METHODS
    match method:
        case "ovi" | "de" | "map" | "nuts":
            kernel = "none"
            yield method, kernel
        case "svgd" | "smi" | "asvgd":
            kernel = "rbf"
            yield method, kernel


def main(args):
    match args.split:
        case "std":
            logger = ExpLogger("uci_std")
        case "gap":
            logger = ExpLogger("uci_gap")
        case _:
            raise UserWarning(f"Unknown split {args.split}")
            return

    setup_logger(logger, args.logs, EXP_CONFIG, set_exp_config)

    set_platform(args.device)

    model_type = "bnn"
    for method, kernel in setup_experiment(args.method):
        for dataset, res, art in run_exps(
            rng_seed=RNG_SEED, method=method, kernel=kernel, split=args.split
        ):
            logger.write_entry(model_type, method, kernel, dataset, **res)
            logger.write_artifact(model_type, method, kernel, dataset, **art)
            logger.save_logs()


def build_argparse(parser):
    parser.prog = "UCI Benchmark"
    parser.description = "Run the UCI benchmark."
    parser.add_argument(
        "split",
        choices=["gap", "std"],
        type=str,
        help="Train and evaluate on Gap10 or Standard UCI dataset.",
    )
    parser.add_argument("method", nargs="?", default="map", choices=METHODS)
    parser.add_argument("logs", nargs="?", default="new", choices=["new", "latest"])
    parser.add_argument("device", nargs="?", default="gpu", choices=["cpu", "gpu"])
    parser.set_defaults(func=main)
