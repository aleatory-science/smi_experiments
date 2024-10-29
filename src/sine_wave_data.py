from collections import namedtuple
import numpy as np
from jax import numpy as jnp


DATASET = namedtuple("DATASET", ["tr", "eval"])
DATA_SEED = 52
DOM_C1 = (-1.5, -0.5)
DOM_C2 = (1.3, 1.7)
DOM_DATA = (-2.0, 2.0)
X_LINESPACE = np.linspace(*DOM_DATA, 500)
NOISE_LEVEL = 0.1


def sine_wave_fn(t):
    amp = 1.5
    freq = 1
    phase = 2 / 3 * np.pi
    linear_coef = 3.0
    bias = -1.0
    return amp * np.sin(2 * np.pi * freq * t + phase) + linear_coef * t - bias


def data_wave_fn(t):
    np.random.seed(DATA_SEED)
    return jnp.array(sine_wave_fn(t) + np.random.randn(t.shape[0]) * NOISE_LEVEL)


def sample_clusters(nc1, nc2):
    np.random.seed(DATA_SEED)
    c1x = np.random.uniform(*DOM_C1, size=(nc1,))
    c2x = np.random.uniform(*DOM_C2, size=(nc2,))
    x = np.concatenate([c1x, c2x])
    idx = jnp.argsort(x)
    x = x[idx]
    y = data_wave_fn(x)
    return jnp.array(x).reshape(-1, 1), jnp.array(y)


def sample_uniform_interval(min_val, max_val, n):
    np.random.seed(DATA_SEED)
    x = np.random.uniform(min_val, max_val, size=(n,))
    idx = jnp.argsort(x)
    x = x[idx]
    y = data_wave_fn(x)
    return jnp.array(x).reshape(-1, 1), jnp.array(y)


def sample_uniform_clusters(n):
    np.random.seed(DATA_SEED)
    xs = []
    for _ in range(n):
        if np.random.binomial(2, 0.5):
            xs.append(np.random.uniform(*DOM_C1))
        else:
            xs.append(np.random.uniform(*DOM_C2))
    x = np.array(xs)
    idx = jnp.argsort(x)
    x = x[idx]
    y = data_wave_fn(x)
    return jnp.array(x).reshape(-1, 1), jnp.array(y)


_TR_BALANCED = sample_clusters(20, 20)
_TR_IMBALANCED = sample_clusters(10, 100)
IN = DATASET(_TR_BALANCED, sample_uniform_clusters(20))
BETWEEN = DATASET(_TR_BALANCED, sample_uniform_interval(-0.5, 1.3, 60))
ENTIRE = DATASET(_TR_BALANCED, sample_uniform_interval(*DOM_DATA, 120))
VIS = DATASET(
    _TR_BALANCED, (jnp.array(X_LINESPACE).reshape(-1, 1), data_wave_fn(X_LINESPACE))
)
