"""Implementation of multi-layered perceptron for 10 way image classification."""

import jax.numpy as jnp
from jax import nn as jnn
from flax import linen as nn

from numpyro.contrib.module import random_flax_module
from numpyro import deterministic, plate, sample
from numpyro.distributions import Normal, Categorical

OUT = ["logits", "y"]


def model(x, y, n, batch_size):
    b, m, k, i = x.shape
    assert (batch_size is None and b == n) or batch_size == b
    assert m == k, f"Image is note square ({m},{k})"
    assert i == 1

    # Lift flax NN to a BNN by adding a prior
    bnn = random_flax_module(
        "mlp",
        MLP(),
        Normal(0, 1.0),  # Prior on kernel and bias
        input_shape=(1, m, m, 1),
    )

    with plate("batch", n, subsample_size=batch_size):
        logits = deterministic("logits", bnn(x))
        sample("y", Categorical(logits=logits), obs=y)


class MLP(nn.Module):
    output_dim: int = 10  # denuted o
    hidden_dim: int = 100  # denoted h
    depth = 1

    def afn(self, x):
        return jnn.tanh(x)

    @nn.compact
    def __call__(self, x):
        if jnp.ndim(x) != 2:
            # Flatten images e.g. for MNIST image img[28, 28] |-> img[28**2].
            # Broadcast to each image
            x = x.reshape(x.shape[0], -1)

        n, m = x.shape

        # Input layer: Real(n,m) -> Real(n,h)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = self.afn(x)
        assert x.shape == (n, self.hidden_dim)

        # Hidden layers: Real(n,h) -> Real(n,h)
        for _ in range(self.depth):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = self.afn(x)
        assert x.shape == (n, self.hidden_dim)

        # Output layer: Real(n,h) -> Real(n,o)
        x = nn.Dense(features=self.output_dim)(x)
        assert x.shape == (n, self.output_dim)

        return x
