"""TODO: merge with mlp"""

from jax import numpy as jnp

from numpyro import deterministic, plate, sample
from numpyro.distributions import Normal
from src.sine_wave_data import NOISE_LEVEL


LATENT = ["prec", "nn_w1", "nn_b1", "nn_w2", "nn_b2", "nn_w3", "nn_b3"]
OUT = ["y_loc", "y"]


def low_bnn(x, y=None):
    bnn(x=x, y=y, hdim=5, subsample=None)


def high_bnn(x, y=None):
    bnn(x=x, y=y, hdim=100, subsample=None)


def bnn(x, y, hdim, subsample):
    """BNN described in appendix D of [1]

    **References:**
        1. *Understanding the Variance Collapse of SVGD in High Dimensions*
           Jimmy Ba, Murat A. Erdogdu, Marzyeh Ghassemi, Shengyang Sun, Taiji Suzuki, Denny Wu, Tianzong Zhang

    """
    w1 = sample(
        "nn_w1",
        Normal(0.0, 1.0).expand(
            (
                x.shape[1] if len(x.shape) > 1 else 1,
                hdim,
            )
        ),  # This will not work with ProductKernel, must be mean-field approximation! (it's a meanfield kernel?)
    )  # prior on l1 weights

    b1 = sample("nn_b1", Normal(0.0, 1.0).expand((hdim,)))  # prior on output bias term

    w2 = sample("nn_w2", Normal(0.0, 1.0).expand((hdim, hdim)))  # prior on l1 weights

    b2 = sample("nn_b2", Normal(0.0, 1.0).expand((hdim,)))  # prior on output bias term

    w3 = sample("nn_w3", Normal(0.0, 1.0).expand((hdim,)))  # prior on output weights

    b3 = sample("nn_b3", Normal(0.0, 1.0))  # prior on output bias term

    with plate(
        "data",
        x.shape[0],
        subsample_size=subsample if subsample is not None else x.shape[0],
    ) as idx:
        x_batch = x[idx] if len(x.shape) > 1 else x
        y_batch = y[idx] if y is not None and len(y.shape) > 0 else y

        # 2 hidden layer with tanh activation
        loc_y = deterministic(
            "y_loc", jnp.tanh(jnp.tanh(x_batch @ w1 + b1) @ w2 + b2) @ w3 + b3
        )

        sample(
            "y",
            Normal(loc_y, NOISE_LEVEL),
            obs=y_batch,
        )
