"""This program measures the number of particles SVGD requires to achieve the same log point-wise predictive density (lppd) as SMI with five particles.

Measurements are performed on the datasets:
 - IN,
 - BETWEEN,
 - ENTIRE.

The SMI and SVGD are trained on the IN dataset.

SVGD starts from 1 particle and increases by a factor of two for each additional run.

We disregard the first run of SVGD and SMI to ignore compile time.

Inference is done with:

   - Adam(1e-3) optimizer,
   - 15k steps
   - 5 SMI particles,
   - 1024 MAX_SVGD_PARTICLES
   - 100 ELBO draws.

All string identifiers are lowercase.
"""

from collections import defaultdict

# Data
from src.sine_wave_data import BETWEEN, ENTIRE, IN

# Measures
from src.metrics import lppd_fn, smi_sampler, svgd_sampler
from time import time

# Methods
from numpyro.contrib.einstein import SteinVI, RBFKernel
from numpyro.contrib.einstein import SVGD
from numpyro.infer.autoguide import AutoNormal

# Models
from src.models.reg_1d_bnn import high_bnn, low_bnn, LATENT

from src.logger.reg_logger import ExpLogger
from experiments.util import total_exps
import tqdm

from numpyro.handlers import seed
from numpyro.optim import Adam
from numpyro import prng_key

LR = 1e-3
RNG_SEED = 10
REPEATS = 10
NUM_SMI_PARTICLES = 5
MAX_SVGD_PARTICLES = 1024
STEPS = 15_000
ELBO_DRAWS = 100
NUM_DRAWS = 5_000
DATASETS = {"in": IN, "between": BETWEEN, "entire": ENTIRE}


def setup_engine(model_type, method, num_svgd_particles):
    assert model_type in ["high", "low"], f"Unknown model type {model_type}"
    assert method in ["svgd", "smi"], f"Unknown method {method}"

    match model_type:
        case "high":
            m = high_bnn
        case "low":
            m = low_bnn

    match method:
        case "smi":
            g = AutoNormal(m)

    match method:
        case "smi":
            e = SteinVI(
                m,
                g,
                Adam(LR),
                RBFKernel(),
                num_stein_particles=NUM_SMI_PARTICLES,
                num_elbo_particles=ELBO_DRAWS,
            )
        case "svgd":
            e = SVGD(m, Adam(LR), RBFKernel(), num_stein_particles=num_svgd_particles)
    return e


def setup_measure(method):
    """Setup log point-wise predictive density and time measure."""
    match method:
        case "svgd" | "asvgd":
            post_sampler = svgd_sampler
            nbatch_dim = 2
        case "smi":
            post_sampler = smi_sampler
            nbatch_dim = 1

    def time_measure(fn, *args, **kwargs):
        st = time()
        res = fn(*args, **kwargs)
        time_taken = time() - st
        assert time_taken >= 0.0, "Time not positive"
        return res, time_taken

    def lppd_measure(key, dataset, engine, inf_results):
        nonlocal post_sampler, nbatch_dim

        x, y = dataset.eval
        post = post_sampler(engine, inf_results, NUM_DRAWS, LATENT)(key, x)

        return lppd_fn(post, engine.model, x, y, nbatch_dim).item()

    return time_measure, lppd_measure


def setup_svgd_runner(model_type, time_measure, lppd_measure, x, y):
    assert model_type in ["high", "low"], "Unknown model type"

    def setup_svgd(num_particles):
        return setup_engine(model_type, "svgd", num_svgd_particles=num_particles)

    def run_inf(key, svgd):
        return time_measure(svgd.run, key, STEPS, x, y, progress_bar=False)

    def run(inf_key, meas_key, num_particles, data):
        nonlocal setup_svgd
        nonlocal run_inf

        fail = False

        try:
            e = setup_svgd(num_particles)
            inf, inf_time = run_inf(inf_key, e)
            lppd, lppd_sample_time = time_measure(lppd_measure, meas_key, data, e, inf)
        except Exception():
            inf_time = float("inf")
            lppd_sample_time = float("inf")
            lppd = -float("inf")
            fail = True

        check_measures("svgd", inf_time, lppd_sample_time)

        return {
            "lppd": lppd,
            "inf_time": inf_time,
            "lppd_sample_time": lppd_sample_time,
            "fail": fail,
            "num_particles": num_particles,
        }

    return run


def check_measures(method, inf_time, sample_time):  # if only we had refinement types
    assert inf_time >= 0, f"{method} inference timing {inf_time}<=0"
    assert sample_time >= 0, f"{method} sample time timing {sample_time}<=0"


def burnin(key, model_type, method, num_particles, x, y):
    """Compile and transfer the method to GPU"""
    e = setup_engine(model_type, method, num_svgd_particles=num_particles)
    e.run(key, 1, x, y, progress_bar=False)


def run_exps(rng_seed, model_type):
    time_measure, lppd_measure = setup_measure("svgd")

    x, y = IN.tr
    run_svgd = setup_svgd_runner(model_type, time_measure, lppd_measure, x, y)

    time_measure, lppd_measure = setup_measure("smi")
    with seed(rng_seed=rng_seed):  # Setup random number generator
        burnin(
            prng_key(), model_type, "smi", 0, x, y
        )  # We do not measure compile and GPU transfer time for inference time
        for r in range(REPEATS):
            # Could be written as another logger but don't need anything beyond defaultdict methods so
            # keeping it simple.

            # Setup for SMI target
            smi = setup_engine(model_type, "smi", num_svgd_particles=0)
            smi_inf, smi_inf_time = time_measure(
                smi.run, prng_key(), STEPS, x, y, progress_bar=False
            )

            for data_name, data in DATASETS.items():
                results = {"smi": {}, "svgd": defaultdict(list)}
                results["smi"]["inf_time"] = smi_inf_time

                # 1. Measure SMI lppd on dataset to get lppd to beat for SVGD
                smi_lppd, smi_lppd_sample_time = time_measure(
                    lppd_measure, prng_key(), data, smi, smi_inf
                )
                results["smi"]["lppd_sample_time"] = smi_lppd_sample_time
                results["smi"]["lppd"] = smi_lppd
                results["smi"]["num_particles"] = NUM_SMI_PARTICLES

                check_measures("smi", smi_inf_time, smi_lppd_sample_time)

                num_svgd_particles = 1
                svgd_lppd = -float("inf")
                svgd_fail = False

                while (
                    not svgd_fail
                    and svgd_lppd <= smi_lppd
                    and num_svgd_particles <= MAX_SVGD_PARTICLES
                ):
                    # We do not measure compile and GPU transfer time for inference time
                    try:
                        burnin(prng_key(), model_type, "svgd", num_svgd_particles, x, y)
                    except Exception():
                        pass
                    for res_name, res in run_svgd(
                        inf_key=prng_key(),
                        meas_key=prng_key(),
                        num_particles=num_svgd_particles,
                        data=data,
                    ).items():
                        match res_name:
                            case "fail":
                                svgd_fail = res
                                results["svgd"][res_name] = res
                            case "lppd":
                                svgd_lppd = res
                                results["svgd"][res_name].append(res)
                            case _:
                                results["svgd"][res_name].append(res)

                    num_svgd_particles = 2 * num_svgd_particles

                yield data_name, results


def setup_experiment():
    for model_type in ["high", "low"]:
        yield model_type


def main(args):
    logger = ExpLogger("lppd_recover")

    for mt in tqdm(setup_experiment(), total=total_exps(setup_experiment)):
        for dataset, res in run_exps(rng_seed=RNG_SEED, model_type=mt):
            for m, mres in res.items():
                logger.write_entry(mt, m, "rbf", dataset, **mres)
                logger.save_logs()


def build_argparse(parser):
    parser.prog = "SMI Recovery"
    parser.description = "Run the SMI recovery experiment."
    parser.set_defaults(func=main)
