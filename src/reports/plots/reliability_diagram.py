from jax import numpy as jnp, nn
from matplotlib import pyplot as plt


def reliability_diagram(logits, targets, bins):
    """implements reliability diagram as outlined in [1] section 2

    choose boundaries to be equal sized

    *** Refs ***
      1. On calibration of Modern NNs"""
    # 1. average over samples
    assert len(logits.shape) == 3  # (samples, data, classes)
    confs = nn.softmax(logits).mean(0)

    # 2. get predictions
    preds = jnp.argmax(confs, axis=-1)
    confs = jnp.amax(confs, axis=-1)

    assert (
        preds.shape == targets.shape == confs.shape
    ), f"{preds.shape} != {targets.shape}"
    assert (
        preds.shape[0] % bins == 0
    ), ""  # TODO: remove simplifying assumption that bins divids number of data points

    accs = preds == targets

    idx = jnp.argsort(confs)

    # makes bins of equal size
    accs = accs[idx].reshape(bins, -1)
    confs = confs[idx].reshape(bins, -1)

    bin_accs = accs.mean(-1)
    bin_confs = confs.mean(-1)

    plt.scatter(
        bin_confs,
        bin_accs,
    )
    plt.xlabel("confidence")

    plt.ylabel("accuracy")
    x = jnp.linspace(0, 1)
    plt.plot(x, x, "k--", label="optimal")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.legend()
    return bin_confs, bin_accs
