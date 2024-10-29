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
from src.metrics import lppd_fn, smi_sampler, svgd_sampler, ovi_sampler
from time import time

# Models
from src.models.bnn_1d import high_bnn, low_bnn, LATENT, OUT

# Methods
from numpyro.contrib.einstein import SteinVI, RBFKernel
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from src.methods import ASVGD, SVGD

from src.logger import ExpLogger

from numpyro.handlers import seed
from numpyro.optim import Adam
from numpyro import prng_key, set_platform
from tqdm import tqdm
from experiments.util import total_exps

# Make sure you have a GPU available!
set_platform("gpu")

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

EXP_CONFIG = {
    "lr": LR,
    "ovi_steps": OVI_STEPS,
    "steps": STEPS,
    "repeats": REPEATS,
    "elbo_draws": ELBO_DRAWS,
    "rng_seed": RNG_SEED,
    "num_particles": NUM_PARTICLES,
}


def setup_engine(model_type, method, kernel):
    assert model_type in ["high", "low"], f"Unknown model type {model_type}"
    assert method in ["svgd", "smi", "ovi", "asvgd"], f"Unknown method {method}"
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
            assert method == "ovi", "Only OVI does not use a kernel"

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
        case "asvgd":
            e = ASVGD(m, Adam(LR), k, num_stein_particles=NUM_PARTICLES)
        case "svgd":
            e = SVGD(m, Adam(LR), k, num_stein_particles=NUM_PARTICLES)
        case "ovi":
            e = SVI(m, g, Adam(LR), Trace_ELBO())

    return e


def setup_measure(method):
    """Setup log point-wise predictive density, y samples and time measurement."""
    assert method in ["asvgd", "smi", "ovi", "svgd"], f"Unknown method {method}"

    match method:
        case "svgd" | "asvgd":
            post_sampler = svgd_sampler
            nbatch_dim = 2
        case "smi":
            post_sampler = smi_sampler
            nbatch_dim = 1
        case "ovi":
            post_sampler = ovi_sampler
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
            e = setup_engine(model_type=model_type, method=method, kernel=kernel)

            x, y = IN.tr
            inf, inf_time = time_measure(
                e.run,
                prng_key(),
                STEPS if method != "ovi" else OVI_STEPS,
                x,
                y,
                progress_bar=False,
            )

            for data_name, data in DATASETS.items():
                lppd, lppd_sample_time = time_measure(
                    lpd_measure, prng_key(), data, e, inf
                )
                y_loc, y_like = y_measure(prng_key(), data, e, inf)

                artifact = {
                    "model": e.model,
                    "guide": e.guide,
                    "params": inf.params,
                    "hyper_params": {},
                }
                result = {
                    "lppd": lppd,
                    "inf_time": inf_time,
                    "lppd_sample_time": lppd_sample_time,
                    "y_loc": y_loc,
                    "y_like": y_like,
                    "losses": inf.losses.tolist(),
                }
                yield data_name, result, artifact


def setup_experiment():
    """Combinations of methods, model type and kernel to be tested."""
    for model_type in ["high", "low"]:
        for method in ["smi", "asvgd", "smi", "svgd", "ovi"]:
            match method:
                case "ovi":
                    yield model_type, method, "none"
                case "svgd" | "asvgd":
                    yield model_type, method, "rbf"
                case "smi":
                    for kernel in ["rbf"]:
                        yield model_type, method, kernel


def main(args):
    logger = ExpLogger("lppd_1d")
    logger.write_exp_config(**EXP_CONFIG)
    logger.save_logs()

    for mt, m, k in tqdm(setup_experiment(), total=total_exps(setup_experiment)):
        for d, r, a in run_exps(rng_seed=RNG_SEED, model_type=mt, method=m, kernel=k):
            logger.write_entry(mt, m, k, d, **r)
            logger.write_artifact(mt, m, k, d, **a)
            logger.save_logs()


def build_argparse(parser):
    parser.prog = "1D Regression"
    parser.description = "Run the 1D regression experiment"
    parser.set_defaults(func=main)
