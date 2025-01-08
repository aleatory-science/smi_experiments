"""This program measures the log point-wise predictive density (lppd) of

 - SVGD,
 - SMI w/ MF normal guide and RBF kernel,
 - ordinary VI (ovi) and
 - Async SVGD.

Each lppd measurement is repeated REPEATS times using NUM_DRAWS draws.

Measurements are performed on the datasets:
 - IN,
 - BETWEEN,
 - ENTIRE,
 - VIS.

The methods are trained on the IN dataset.

The experiment is based on the 1D visualization from [1].

Inference is done with:

   - Adam(1e-3) optimizer,
   - 15k steps for SVGD, SMI, ASVGD,
   - 50k steps for OVI,
   - 5 particles,
   - 100 ELBO draws.

All string identifiers are lowercase.

Refs. (MLA)

    [1] Foong, Andrew YK, et al. "'In-Between'Uncertainty in Bayesian Neural Networks." arXiv preprint arXiv:1906.11537 (2019).

https://web.mit.edu/daveg/Text/poetry/Manifest:MFLF
"""

# Data
from src.sine_wave_data import BETWEEN, ENTIRE, IN, VIS

# Measures
from src.metrics import lppd_fn, smi_sampler, svgd_sampler, svi_sampler, mcmc_sampler
from time import time

# Models
from src.models.reg_1d_bnn import high_bnn, low_bnn, LATENT, OUT

# Methods
from numpyro.contrib.einstein import SteinVI, RBFKernel
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer import NUTS, MCMC
from numpyro.infer.autoguide import AutoNormal
from src.methods import ASVGD, SVGD

from src.logger.reg_logger import ExpLogger

from numpyro.handlers import seed
from numpyro.optim import Adam
from numpyro.diagnostics import summary
from numpyro import prng_key, set_platform
from tqdm import tqdm
from experiments.util import setup_logger
from functools import partial
from src.constants import METHODS


# Experimental Setup
LR = 1e-3
NUM_PARTICLES = 5
STEPS = 15_000
OVI_STEPS = 50_000
ELBO_DRAWS = 100
REPEATS = 10
NUM_DRAWS = 5000
DATASETS = {"in": IN, "between": BETWEEN, "entire": ENTIRE, "vis": VIS}
RNG_SEED = 10
MCMC_WARMUP = 5
MCMC_DRAWS = 50

EXP_CONFIG = {
    "lr": LR,
    "ovi_steps": OVI_STEPS,
    "steps": STEPS,
    "repeats": REPEATS,
    "elbo_draws": ELBO_DRAWS,
    "rng_seed": RNG_SEED,
    "num_particles": NUM_PARTICLES,
    "mcmc_warmup": MCMC_WARMUP,
    "mcmc_draws": MCMC_DRAWS,
}


def set_exp_config(config):
    pass


def setup_engine(model_type, method, kernel):
    assert model_type in ["high", "low"], f"Unknown model type {model_type}"
    assert method in METHODS, f"Unknown method {method}"
    assert kernel in ["rbf", "ppk", "none"], f"Unknown kernel {kernel}"

    match model_type:
        case "high":
            m = high_bnn
        case "low":
            m = low_bnn

    match method:
        case "smi" | "ovi":
            g = AutoNormal(m)

    match kernel:
        case "rbf":
            k = RBFKernel()
        case "none":
            assert method in ["ovi", "nuts"], "Only OVI and NUTS does not use a kernel"

    match method:
        case "smi":
            e = SteinVI(
                m,
                g,
                Adam(LR),
                k,
                num_stein_particles=NUM_PARTICLES,
                num_elbo_particles=ELBO_DRAWS,
            )
            run = lambda rng_key, x, y: e.run(
                rng_key,
                STEPS,
                x,
                y,
                progress_bar=False,
            )
        case "asvgd":
            e = ASVGD(m, Adam(LR), k, num_stein_particles=NUM_PARTICLES)

            run = lambda rng_key, x, y: e.run(
                rng_key,
                STEPS,
                x,
                y,
                progress_bar=False,
            )
            run = partial(e.run, progress_bar=False)
        case "svgd":
            e = SVGD(m, Adam(LR), k, num_stein_particles=NUM_PARTICLES)
            run = lambda rng_key, x, y: e.run(
                rng_key,
                STEPS,
                x,
                y,
                progress_bar=False,
            )
        case "ovi":
            e = SVI(m, g, Adam(LR), Trace_ELBO())
            run = lambda rng_key, x, y: e.run(
                rng_key,
                OVI_STEPS,
                x,
                y,
                progress_bar=False,
            )
        case "nuts":
            mcmc_kernel = NUTS(m)
            e = MCMC(
                mcmc_kernel,
                num_samples=MCMC_DRAWS,
                num_warmup=MCMC_WARMUP,
                progress_bar=True,
                jit_model_args=True,
            )
            run = lambda rng_key, x, y: e.run(rng_key, x, y)
            e.model = m
    return e, run


def setup_measure(method):
    """Setup log point-wise predictive density, y samples and time measurement."""
    assert method in ["asvgd", "smi", "ovi", "svgd", "nuts"], f"Unknown method {method}"

    match method:
        case "svgd" | "asvgd":
            post_sampler = svgd_sampler
            nbatch_dim = 2
        case "smi":
            post_sampler = smi_sampler
            nbatch_dim = 1
        case "ovi":
            post_sampler = svi_sampler
            nbatch_dim = 1
        case "nuts":
            post_sampler = mcmc_sampler
            nbatch_dim = 1

    def time_measure(fn, *args, **kwargs):
        st = time()
        res = fn(*args, **kwargs)
        time_taken = time() - st
        assert time_taken >= 0.0, "Time not positive"
        return res, time_taken

    def lppd_measure(key, dataset, engine, inf_results):
        nonlocal post_sampler

        x, y = dataset.eval
        post = post_sampler(engine, inf_results, NUM_DRAWS, LATENT)(key, x)

        return lppd_fn(post, engine.model, x, y, nbatch_dim).item()

    def y_measure(key, dataset, engine, inf_results):
        nonlocal post_sampler

        x, _ = dataset.eval
        post_ys = post_sampler(engine, inf_results, NUM_DRAWS, OUT)(key, x)
        return post_ys["y_loc"].tolist(), post_ys["y"].tolist()

    return time_measure, lppd_measure, y_measure


def run_exps(rng_seed, model_type, method, kernel):
    """Runs repeated experiments for specific model_type, method and kernel. Generates results."""
    time_measure, lpd_measure, y_measure = setup_measure(method)

    with seed(rng_seed=rng_seed):  # Setup random number generator.
        for r in range(REPEATS):
            e, run = setup_engine(model_type=model_type, method=method, kernel=kernel)

            x, y = IN.tr
            inf, inf_time = time_measure(
                run,
                prng_key(),
                x,
                y,
            )

            for data_name, data in DATASETS.items():
                lppd, lppd_sample_time = time_measure(
                    lpd_measure, prng_key(), data, e, inf
                )
                y_loc, y_like = y_measure(prng_key(), data, e, inf)

                artifact = {
                    "model": e.model if method != "nuts" else None,
                    "guide": e.guide if method != "nuts" else None,
                    "params": inf.params if method != "nuts" else None,
                    "hyper_params": {},
                    "post_draws": e.get_samples(group_by_chain=True)
                    if method == "nuts"
                    else {},
                    "summary": summary(e.get_samples(group_by_chain=True))
                    if method == "nuts"
                    else {},
                }

                result = {
                    "lppd": lppd,
                    "inf_time": inf_time,
                    "lppd_sample_time": lppd_sample_time,
                    "y_loc": y_loc,
                    "y_like": y_like,
                    "losses": inf.losses.tolist() if method != "nuts" else [],
                }
                yield data_name, result, artifact


def setup_experiment(method):
    """Combinations of methods, model type and kernel to be tested."""
    for model_type in ["high", "low"]:
        match method:
            case "ovi" | "nuts":
                yield model_type, method, "none"
            case "svgd" | "asvgd" | "smi":
                yield model_type, method, "rbf"


def main(args):
    logger = ExpLogger("lppd_1d")
    setup_logger(logger, args.logs, EXP_CONFIG, set_exp_config)

    set_platform(args.device)

    for mt, m, k in setup_experiment(args.method):
        for d, r, a in tqdm(
            run_exps(rng_seed=RNG_SEED, model_type=mt, method=m, kernel=k),
            total=REPEATS,
        ):
            logger.write_entry(mt, m, k, d, **r)
            logger.write_artifact(mt, m, k, d, **a)
            logger.save_logs()


def build_argparse(parser):
    parser.prog = "1D Regression"
    parser.description = "Run the 1D regression experiment"
    parser.add_argument("method", nargs="?", default="map", choices=METHODS)
    parser.add_argument("logs", nargs="?", default="new", choices=["new", "latest"])
    parser.add_argument("device", nargs="?", default="gpu", choices=["cpu", "gpu"])
    parser.set_defaults(func=main)
