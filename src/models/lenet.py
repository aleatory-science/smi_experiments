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

    bnn = random_flax_module(
        "lenet",
        LeNet(),
        Normal(0, 1.0),  # Prior on kernel and bias
        input_shape=(1, m, m, 1),
    )

    with plate("batch", n, subsample_size=batch_size):
        logits = deterministic("logits", bnn(x))
        sample("y", Categorical(logits=logits), obs=y)


class LeNet(nn.Module):
    output_dim: int = 10
    activation: str = "tanh"

    def afn(self, x):
        if self.activation == "tanh":
            return jnn.tanh(x)
        if self.activation == "relu":
            return jnn.relu(x)

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=6, kernel_size=(5, 5), strides=(1, 1), padding=((0, 0), (0, 0))
        )(x)
        x = self.afn(x)
        x = nn.max_pool(
            x, window_shape=(2, 2), strides=(2, 2), padding=((0, 0), (0, 0))
        )
        x = nn.Conv(
            features=16, kernel_size=(5, 5), strides=(1, 1), padding=((0, 0), (0, 0))
        )(x)
        x = self.afn(x)
        x = nn.max_pool(
            x, window_shape=(2, 2), strides=(2, 2), padding=((0, 0), (0, 0))
        )
        x = jnp.transpose(x, (0, 3, 1, 2))
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(features=120)(x)
        x = self.afn(x)
        x = nn.Dense(features=84)(x)
        x = self.afn(x)
        x = nn.Dense(features=self.output_dim)(x)

        return x
