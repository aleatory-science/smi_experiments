"""Determine the dimension averaged variance when the model is an MVN with increasing dimensionality of
    - SVGD
    - ASVGD
    - SMI with mean-field normal guide [Ours]
All methods use the RBF kernel.

We test all combinations of the following:
    - Model dimemension in [1, 2, 4, 8, 10, 20, 40, 60, 80, 100]
    - Repulsion temperature (alpha in the article) in [0.001, 0.1, 1, 10, 100]
    - Uniform particle initial location a radius of [0.2, 2, 20] units from the origin

Inference is done with:
    - Adam(5e-2) optimizer for ASVGD and SVGD,
    - Adagrad(5e-2) optimizer for SMI,
        - Adagrad produces a marginally better variance estimation than Adam in the 1-particle case.
        - However, it takes far more steps.
    - 60k steps,
    - 20 particles for ASVGD and SVGD,
    - 1 and 20 particle(s) for SMI,
    - 100 SMI ELBO draws,
"""

from numpyro import sample
from numpyro.distributions import MultivariateNormal


# Methods
from numpyro.contrib.einstein import SteinVI, RBFKernel, LinearKernel
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer import init_to_uniform
from src.methods import ASVGD, SVGD

from numpyro.optim import Adagrad, Adam

from src.logger.reg_logger import ExpLogger

from jax import numpy as jnp
import numpy as np

from time import time
from numpyro.handlers import seed
from numpyro import prng_key
from functools import partial


LR = 0.05
MAX_STEPS = 60_000
ELBO_DRAWS = 100
RNG_SEED = 10
MODEL_DIMENSIONS = [1, 2, 4, 8, 10, 20, 40, 60, 80, 100]
REPULSION_TEMP = [0.001, 0.1, 1.0, 10.0, 100.0]
NUM_SVGD_PARTICLES = 20  # same for ASVGD
NUM_SMI_PARTICLES = [1, NUM_SVGD_PARTICLES]
INIT_RADIUS = [0.2, 2, 20]


EXP_CONFIG = {
    "lr": LR,
    "max_steps": MAX_STEPS,
    "elbo_draws": ELBO_DRAWS,
    "rng_seed": RNG_SEED,
    "model_dimensions": MODEL_DIMENSIONS,
    "repulsion_temp": REPULSION_TEMP,
    "num_svgd_particles": NUM_SVGD_PARTICLES,
    "num_smi_particles": NUM_SMI_PARTICLES,
    "init_radius": INIT_RADIUS,
}


def setup_model(dim):
    def model():
        sample("x", MultivariateNormal(covariance_matrix=jnp.eye(dim)))

    return model


def setup_engine(method, kernel, mdim, r, n, rt):
    assert method in ["svgd", "smi", "asvgd"], f"Unknown method {method}"
    assert kernel in ["rbf", "linear"], f"Unknown kernel {kernel}"

    match kernel:
        case "rbf":
            k = RBFKernel()
        case "linear":
            k = LinearKernel()

    m = setup_model(mdim)

    match method:
        case "smi":
            g = AutoNormal(m, init_loc_fn=partial(init_to_uniform, radius=r))
        case "asvgd" | "svgd":
            gkwargs = {"init_loc_fn": partial(init_to_uniform, radius=r)}

    match method:
        case "smi":
            e = SteinVI(
                m,
                g,
                Adagrad(LR),
                k,
                num_stein_particles=n,
                num_elbo_particles=ELBO_DRAWS,
                repulsion_temperature=rt,
            )
        case "svgd":
            e = SVGD(
                m,
                Adam(LR),
                k,
                num_stein_particles=n,
                guide_kwargs=gkwargs,
                repulsion_temperature=rt,
            )
        case "asvgd":
            e = ASVGD(
                m,
                Adam(LR),
                k,
                num_stein_particles=n,
                guide_kwargs=gkwargs,
                repulsion_temperature=rt,
            )

    def run(key):
        return e.run(key, MAX_STEPS, progress_bar=False)

    return e, run


def setup_measure(method):
    assert method in ["asvgd", "smi", "svgd"], f"Unknown method {method}"

    def time_measure(fn, *args, **kwargs):
        st = time()
        res = fn(*args, **kwargs)
        time_taken = time() - st
        assert time_taken >= 0.0, "Time not positive"
        return res, time_taken

    def dim_loc_measure(inf_results):
        loc = np.array(inf_results.params["x_auto_loc"].mean(0))
        return loc.tolist()

    def dim_var_measure(inf_results):
        nonlocal method

        loc = inf_results.params["x_auto_loc"]

        match method:
            case "smi":
                scale = inf_results.params["x_auto_scale"]
                print(scale.shape)

                # Compute gaussian mixture variance
                var = np.array(
                    (scale**2).mean(0) + (loc**2).mean(0) - (loc.mean(0) ** 2)
                )

            case "asvgd" | "svgd":
                var = np.var(loc, axis=0)

        return var.tolist()

    return time_measure, dim_loc_measure, dim_var_measure


def run_exps(rng_seed, method, kernel, num_particles, repulsion_temp):
    """Runs experiments for specific model_type, method and kernel. Generates results."""
    time_measure, loc_measure, var_measure = setup_measure(method)

    dataset = "none"

    with seed(rng_seed=rng_seed):  # Setup random number generator.
        for mdim in MODEL_DIMENSIONS:
            for r in INIT_RADIUS:
                e, run = setup_engine(
                    method=method,
                    kernel=kernel,
                    mdim=mdim,
                    n=num_particles,
                    r=r,
                    rt=repulsion_temp,
                )
                inf, inf_time = time_measure(run, prng_key())

                var = var_measure(inf)
                loc = loc_measure(inf)
                artifact = {
                    "model": e.model,
                    "guide": e.guide,
                    "params": inf.params,
                    "hyper_params": {
                        "num_particles": num_particles,
                        "repulsion_temperature": repulsion_temp,
                        "init_radius": r,
                    },
                    "post_draws": {},
                }
                result = {
                    "var": var,
                    "inf_time": inf_time,
                    "loc": loc,
                    "model_dim": mdim,
                    "losses": inf.losses.tolist(),
                }

                yield dataset, result, artifact


def setup_experiment():
    kernel = "rbf"
    for method in ["smi", "svgd", "asvgd"]:
        match method:
            case "smi":
                for n in NUM_SMI_PARTICLES:
                    for rt in REPULSION_TEMP:
                        yield method, kernel, n, rt
            case "svgd" | "asvgd":
                for rt in REPULSION_TEMP:
                    yield method, kernel, NUM_SVGD_PARTICLES, rt


def main(args):
    logger = ExpLogger("var")

    logger.write_exp_config(**EXP_CONFIG)
    logger.save_logs()

    model_type = "bnn"
    for method, kernel, n, rt in setup_experiment():
        for dataset, res, art in run_exps(
            rng_seed=RNG_SEED,
            method=method,
            kernel=kernel,
            num_particles=n,
            repulsion_temp=rt,
        ):
            logger.write_entry(model_type, method, kernel, dataset, **res)
            logger.write_artifact(model_type, method, kernel, dataset, **art)
            logger.save_logs()


def build_argparse(parser):
    parser.prog = "Variance Collapse"
    parser.description = "Run the variance collapse experiment"
    parser.set_defaults(func=main)
