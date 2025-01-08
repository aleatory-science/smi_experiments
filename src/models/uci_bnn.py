"""TODO: merge with mlp"""

from jax import numpy as jnp, nn

from numpyro import deterministic, plate, sample
from numpyro.distributions import Normal, Gamma


LATENT = [
    "prec",
    "w1",
    "b1",
    "w2",
    "b2",
    "w3",
    "b3",
]
OUT = ["y_loc", "y"]


def model(x, y, subsample):
    """BNN described in appendix D of [1]

    **References:**
        1. Understanding the Variance Collapse of SVGD in High Dimensions
           Jimmy Ba, Murat A. Erdogdu, Marzyeh Ghassemi, Shengyang Sun, Taiji Suzuki, Denny Wu, Tianzong Zhang
    """

    hdim = 50  # Hidden dimension is fixed

    prec = sample("prec", Gamma(1.0, 0.1))

    w1 = sample(
        "w1",
        Normal(0.0, 1).expand(
            (
                x.shape[1] if len(x.shape) > 1 else 1,
                hdim,
            )
        ),
    )  # prior on l1 weights
    b1 = sample("b1", Normal(0.0, 1).expand((hdim,)))  # prior on output bias term

    w2 = sample("w2", Normal(0.0, 1).expand((hdim, hdim)))  # prior on l1 weights
    b2 = sample("b2", Normal(0.0, 1).expand((hdim,)))  # prior on output bias term

    w3 = sample("w3", Normal(0.0, 1).expand((hdim,)))  # prior on output weights

    b3 = sample("b3", Normal(0.0, 1))  # prior on output bias term

    with plate(
        "data",
        x.shape[0],
        subsample_size=subsample if subsample is not None else x.shape[0],
    ) as idx:
        x_batch = x[idx] if len(x.shape) > 1 else x
        y_batch = y[idx] if y is not None and len(y.shape) > 0 else y

        # 1 hidden layer with relu activation
        y_loc = deterministic(
            "y_loc", nn.relu(nn.relu(x_batch @ w1 + b1) @ w2 + b2) @ w3 + b3
        )

        sample(
            "y",
            Normal(y_loc, jnp.sqrt(1 / prec)),
            obs=y_batch,
        )
