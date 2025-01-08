"""Experimental setup from E.3 of [1p22].

### Evaluations
    Measurement is done using 50 posterior samples.

    Evaluation includes: See [metrics](src/metrics.py) and [constants](src/constants.py)

        Confidence (Conf)
        Negative log likelihood (NLL)
        Accuracy (Acc)
        Brier score (brier)[2]
        Expected Calibration Error (ECE) [3]
        Max Calibration Error (MCE) [1]
        Inference time (Time)

### Methods
    MAP
    SVGD (RBFKernel)
    SMI (RBFKernel)

### Hyperparams
    SVI Learning rate (SVI_LR): 1e-3  (i.e., OVI and MAP)
    SVI Optimizer: Adam (i.e., OVI and MAP)
    SteinVI Optimizer: Adagrad (i.e., SVGD and SMI)
    SteinVI Learning rate (STEIN_LR): 7e-1  (i.e., SVGD and SMI)
    SteinVI Particles: 5 (i.e., SVGD and SMI)
    SteinVI ELBO Particles: 55 (i.e., SVGD and SMI)
    Posterior draws for evaluation (PDRAWS): 50
    Epochs: 100


### REFERENCES (MLA)

  1. Roy, Hrittik, et al. "Reparameterization invariance in approximate Bayesian inference." arXiv preprint arXiv:2406.03334 (2024).
  2. Brier, Glenn W. "Verification of forecasts expressed in terms of probability." Monthly weather review 78.1 (1950): 1-3.
  3. Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht. "Obtaining well calibrated probabilities using bayesian binning."
     Proceedings of the AAAI conference on artificial intelligence. Vol. 29. No. 1. 2015.
"""

from numpyro.infer import SVI, Trace_ELBO
from numpyro.contrib.einstein import ASVGD, SVGD, RBFKernel, SteinVI
from numpyro.optim import Adam, Adagrad
from numpyro.infer.autoguide import AutoDelta, AutoNormal

from time import time
from numpyro.handlers import seed
from numpyro import prng_key, set_platform

from src.models.mlp import model, OUT
from datasets.ood_detection.load_ood_detect import test_data
from src.engine_run import setup_run

from src.metrics import svi_sampler, smi_sampler, svgd_sampler

from src.logger.class_logger import ExpLogger
from src.constants import METHODS, INDIST_EVAL_METRICS
from experiments.util import setup_logger

import os

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
)

SVI_LR = 1e-3  # MAP learning rate
EPOCHS = 100
STEIN_LR = 8e-1  # SVGD, SMI learning rate  7e-1 best for 2 layer MLP
STEIN_PARTICLES = 5
STEIN_ELBO_PARTICLES = 55
PDRAWS = 50  # number of posterior draws
RNG_SEED = 142
BATCH_SIZE = 128

EXP_CONFIG = {
    # MAP hparams
    "map_lr": SVI_LR,
    # Stein hparams
    "stein_lr": STEIN_LR,
    "stein_particles": STEIN_PARTICLES,
    "stein_elbo_particles": STEIN_ELBO_PARTICLES,
    # Shared hparams
    "post_draws": PDRAWS,
    "rng_seed": RNG_SEED,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
}


def set_exp_config(config):
    assert all(
        map(lambda k: k in config, EXP_CONFIG.keys())
    ), "Experiment config missing hyperparameters. Use --logs new or different timestamp."
    global \
        SVI_LR, \
        EPOCHS, \
        STEIN_LR, \
        STEIN_PARTICLES, \
        STEIN_ELBO_PARTICLES, \
        BATCH_SIZE, \
        PDRAWS, \
        RNG_SEED

    # MAP Hparams
    SVI_LR = config["map_lr"]

    # Stein Hparams
    STEIN_LR = config["stein_lr"]
    STEIN_PARTICLES = config["stein_particles"]
    STEIN_ELBO_PARTICLES = config["stein_elbo_particles"]

    # Shared Hparams
    EPOCHS = config["epochs"]
    PDRAWS = config["post_draws"]
    RNG_SEED = config["rng_seed"]
    BATCH_SIZE = config["batch_size"]


def time_eval(fn, *args, **kwargs):
    st = time()
    res = fn(*args, **kwargs)
    time_taken = time() - st
    assert time_taken >= 0.0, "Time not positive"
    return res, time_taken


def setup_eval(method, draws):
    assert method in METHODS, f"Unknown method {method}"

    match method:
        case "map" | "ovi":
            sampler = svi_sampler
            bndims = 0

        case "svgd" | "asvgd":
            sampler = svgd_sampler
            bndims = 1

        case "smi":
            sampler = smi_sampler
            bndims = 0

    def eval(key, x, y, n, e, params):
        ps = sampler(e, params, draws, OUT)(key, x, y=None, n=n, batch_size=None)
        return {
            metric: fn(ps, e.model, x, y, n, bndims)
            for metric, fn in INDIST_EVAL_METRICS.items()
        }

    return eval


def setup_method(method, dataset):
    assert method in METHODS, f"Unknown method {method}"
    m = model

    match method:
        # Stein Methods
        case "svgd":
            e = SVGD(
                m, Adagrad(STEIN_LR), RBFKernel(), num_stein_particles=STEIN_PARTICLES
            )
        case "asvgd":
            e = ASVGD(
                m, Adagrad(STEIN_LR), RBFKernel(), num_stein_particles=STEIN_PARTICLES
            )

        case "smi":
            g = AutoNormal(m)
            e = SteinVI(
                m,
                g,
                Adagrad(STEIN_LR),
                RBFKernel(),
                num_stein_particles=STEIN_PARTICLES,
                num_elbo_particles=STEIN_ELBO_PARTICLES,
            )
        # SVI Methods
        case "map":
            g = AutoDelta(m) 
            e = SVI(m, g, Adam(SVI_LR), Trace_ELBO())
        case "ovi":
            g = AutoNormal(m)
            e = SVI(m, g, Adam(SVI_LR), Trace_ELBO())

    run = setup_run(e, dataset)
    return e, run


def run_exps(method, dataset):
    set_platform("gpu")
    with seed(rng_seed=RNG_SEED):
        eval_fn = setup_eval(method, PDRAWS)
        e, run = setup_method(method, dataset)

        inf, time_taken = time_eval(
            run, prng_key(), EPOCHS, BATCH_SIZE, init_state=None
        )

        xte, yte, n = test_data(dataset)
        eval_res = {
            "inf_time": time_taken,
            **eval_fn(prng_key(), xte, yte, n, e, inf),
        }

    artifact = {
        "model": e.model,
        "guide": e.guide,
        "params": inf.params,
        "hyper_params": {},
        "post_draws": {},
        "summary": {},
    }

    return eval_res, artifact


def main(args):
    logger = ExpLogger("mnist")
    setup_logger(logger, args.logs, EXP_CONFIG, set_exp_config)

    model_type = "mlp"

    res, art = run_exps(
        method=args.method,
        dataset="mnist",
    )

    logger.write_entry(model_type, args.method, "mnist", **res)
    logger.write_artifact(model_type, args.method, "mnist", **art)

    logger.save_logs()


def build_argparse(parser):
    parser.prog = "MNIST"
    parser.description = "Infer and evaluate on MNIST"
    parser.add_argument("method", nargs="?", default="map", choices=METHODS)
    parser.add_argument("logs", nargs="?", default="new", choices=["new", "latest"])

    parser.set_defaults(func=main)
