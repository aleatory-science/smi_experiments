from numpyro.infer import log_likelihood
from jax import numpy as jnp, nn, vmap, tree
from numpyro.infer.util import Predictive
import numpy as np

from numpyro.contrib.einstein import MixtureGuidePredictive

from functools import reduce
import operator

from jax.scipy.special import logsumexp


# FIXME
def _check_likelihood(logits, y):
    return jnp.log(nn.softmax(logits)[y])


def nll_fn(posterior_samples, model, x, y, n, batch_ndims):
    r"""Compute the negative log likelihood given by
        [- \sum_i \log p(y_i|theta^(s), x_i))]_s~q(theta|D)
    where q(theta|D) is the variational posterior.
    """

    logits = flatten_post_dim(posterior_samples)["logits"]
    # The shape is (number of posterior samples, number of datapoints, number of classes)
    p, n, c = logits.shape

    lls = log_likelihood(
        model,
        posterior_samples,
        x,
        y,  # Note: uses true ys
        n=n,
        batch_size=None,
        batch_ndims=batch_ndims,
    )
    lls = lls["y"].reshape(p, n)
    assert y.shape == (n,)

    # FIXME
    # Check that the first lls agree with manual computation
    # man_ll0 = _man_likelihood(logits[0, 0, :],y[0])
    # assert jnp.allclose(
    #     lls[0, 0], man_ll0, atol=1e-5
    # ), f"log_likelihood ll {lls[0,0]:.5f} not the same as manual computation {man_ll0:.5f}"

    nll = -lls.mean(axis=1)
    return nll.tolist()


def conf_fn(posterior_samples, model, x, y, n, batch_ndims):
    """Compute the confidence given by
        [1/n sum_j p(y_{(s,j)}|theta_j)]_{y_{(s,:)} sim p(y|theta_s)}
    where theta_s sim q(theta|D) and q(theta|D) is the variational posterior.
    """

    logits = flatten_post_dim(posterior_samples)["logits"]
    p, n, c = logits.shape
    lls = log_likelihood(
        model,
        posterior_samples,
        x,
        posterior_samples["y"],  # Note: uses predicted ys
        n=n,
        batch_size=None,
        batch_ndims=batch_ndims,
    )

    lls = lls["y"].reshape(p, n)

    # FIXME
    # Check that the first lls agree with manual computation
    # man_ll0 = _man_likelihood(logits[0, 0, :], posterior_samples["y"][0, 0])
    # assert jnp.allclose(
    #     lls[0, 0], man_ll0, atol=1e-5
    # ), f"log_likelihood ll {lls[0,0]:.5f} not the same as manual computation {man_ll0:.5f}"

    confs = jnp.exp(lls).mean(axis=1)
    return confs.tolist()


def acc_fn(posterior_samples, model, x, y, n, batch_ndims):
    """Compute the accuracy"""
    y_pred = flatten_post_dim(posterior_samples)["y"]
    accs = (y[None] == y_pred).mean(1)
    return accs.tolist()


def ece_fn(posterior_samples, model, x, y, n, batch_ndims):
    """Compute the expected calibration error"""

    logits = flatten_post_dim(posterior_samples)["logits"]
    # The shape is (number of posterior samples, number of datapoints, number of classes)
    p, n, c = logits.shape

    bins = 100

    def bca(logits):
        return binned_conf_acc(logits, y, bins=bins)

    bin_mconfs, bin_maccs = vmap(bca)(logits)

    assert bin_mconfs.shape == (p, bins)
    assert bin_maccs.shape == bin_mconfs.shape

    ece = jnp.abs(bin_mconfs - bin_maccs).mean(axis=1)

    return ece.tolist()


def mce_fn(posterior_samples, model, x, y, n, batch_ndims):
    """Compute the mean calibration error"""
    logits = flatten_post_dim(posterior_samples)["logits"]

    bins = 100
    p, n, c = logits.shape

    y = np.asarray(y, copy=False)
    logits = np.asarray(logits, copy=False)

    def bca(logits):
        return binned_conf_acc(logits, y, bins=bins)

    bin_mconfs, bin_maccs = vmap(bca)(logits)
    assert bin_mconfs.shape == (p, bins)
    assert bin_maccs.shape == bin_mconfs.shape

    mce = jnp.max(jnp.abs(bin_mconfs - bin_maccs), axis=1)

    return mce.tolist()


def brier_fn(posterior_samples, model, x, y, n, batch_ndims):
    """Compute the brier score from [1] eq. 2.

    ### REFERENCES
      1. Brier, Glenn W. "Verification of forecasts expressed in terms of probability."
        Monthly weather review 78.1 (1950): 1-3.
    """

    logits = flatten_post_dim(posterior_samples)["logits"]
    y_probs = nn.softmax(logits, axis=-1)

    assert y_probs.shape == logits.shape
    assert (
        jnp.ndim(y_probs) == 3
    ), "Expected data with shape (pdraws, num datapoints, num categories)"

    d, n, c = y_probs.shape
    assert (n == x.shape[0]) and (
        n == y.shape[0]
    ), "Posterior and test data is inconsitent"

    in_sqrs = ((y_probs - nn.one_hot(y, c)[None]) ** 2).sum(2)
    brier = in_sqrs.mean(1)
    return brier.tolist()


def lppd_fn(posterior_samples, model, xs, ys, batch_ndims):
    r"""Compute Log Pointwise Predictive Density (lppd) defined in eq. 5  of [1:p5].
    The LPPD is given posterior S samples T={theta_s}_s=1^S and N new datapoints D={(x_i, y_i)}_i=1^N, lppd is given by
        lppd(T, D) = \sum_i=1^N log (1/S \sum_s=1^S p(y_i|x_i, T^s))

    Refs.
        [1] Gelman, Andrew, Jessica Hwang, and Aki Vehtari. "Understanding predictive information criteria for Bayesian models."
            Statistics and computing 24 (2014): 997-1016."""

    def lpd(x, y):
        r"""Compute log predictive density of single point given by
        log (1/S \sum_s=1^S p(y|x, T^s))
        """

        lls = log_likelihood(model, posterior_samples, x, y, batch_ndims=batch_ndims)

        assert len(lls) == 1, "lppd is not defined for multiple likelihood sites"
        obs_name = list(lls.keys())[0]

        lls = lls[obs_name]
        num_draws = lls.shape[0]
        return logsumexp(lls) - jnp.log(num_draws)

    # Sum individual contributions
    return vmap(lpd)(xs, ys).sum()


def rmse_fn(posterior_samples, y, batch_ndims):
    """Compute the root mean squared error"""

    y_pred = posterior_samples["y_loc"]
    return jnp.sqrt(((y[None] - y_pred) ** 2).mean())


def nll_reg_fn(posterior_samples, model, x, y, batch_ndims):
    r"""Compute the negative log likelihood given by
    1/S \sum_{s=1}^S (-log \sum_i p(y_i|theta^(s), x_i)).
    """
    lls = log_likelihood(model, posterior_samples, x, y, None, batch_ndims=batch_ndims)[
        "y"
    ]
    return -(logsumexp(lls, axis=0) - jnp.log(lls.shape[0])).mean()


def svi_sampler(engine, inf_results, num_samples, return_sites):
    pred = Predictive(
        model=engine.model,
        guide=engine.guide,
        params=inf_results.params,
        num_samples=num_samples,
        return_sites=return_sites,
    )
    return pred


def smi_sampler(engine, inf_results, num_samples, return_sites):
    pred = MixtureGuidePredictive(
        model=engine.model,
        guide=engine.guide,
        params=inf_results.params,
        num_samples=num_samples,
        return_sites=return_sites,
        guide_sites=engine.guide_sites,
    )
    return pred


def svgd_sampler(engine, inf_results, num_samples, return_sites):
    pred = Predictive(
        model=engine.model,
        guide=engine.guide,
        params=inf_results.params,
        num_samples=1,
        return_sites=return_sites,
        batch_ndims=1,
    )
    return pred


def mcmc_sampler(engine, inf_results, num_samples, return_sites):
    post_samples = engine.get_samples()
    pred = Predictive(
        model=engine.model,
        posterior_samples=post_samples,
        return_sites=return_sites,
    )
    return pred


def binned_conf_acc(logits, targets, bins):
    """Implements binned confidence and accuracy as outlined in [1] section 2.

    ### Refs

        1. Guo, Chuan, et al. "On calibration of modern neural networks."
            International conference on machine learning. PMLR, 2017.
    """
    assert len(logits.shape) == 2  # (data, classes)
    confs = nn.softmax(logits, axis=-1)

    # 2. get predictions
    preds = jnp.argmax(confs, axis=-1)
    confs = jnp.amax(confs, axis=-1)

    assert (
        preds.shape == confs.shape
    ), f"Prediction shape {preds.shape} not same as confidence shape {confs.shape}"
    assert (
        preds.shape == targets.shape
    ), f"Prediction shape {preds.shape} not same as targets shape {confs.shape}"

    accs = preds == targets

    # Sort after confidence to allow hist counts to align with bin assignments
    # f.ex hist([1,2,3,5,6,9], bins=3).counts == (3,2,1)
    # [0] + cumsum(counts) == [0,3,5,6]
    # so [0,3) is in bin 1,
    #    [3,5) is in bin 2
    #    [5,6) is in bin 3
    idx = jnp.argsort(confs)
    confs = confs[idx]
    accs = accs[idx]

    counts, _ = jnp.histogram(confs, bins, range=(0, 1))
    counts = jnp.astype(counts, int)
    assert counts.shape == (
        bins,
    ), f"Counts shape {counts.shape}different than expected {(bins,)}"

    def compute_bins_mean(arr, counts):
        # Counts in [0,len(arr)]
        bins_end = counts.cumsum()
        # Account for first bin(s) being empty
        bins_end = jnp.where(counts > 0, bins_end - 1, bins_end)

        bin_cumsum = arr.cumsum()[bins_end]
        prev_bins_cumsum = jnp.concatenate([jnp.array([0]), bin_cumsum[:-1]])

        bin_sum = bin_cumsum - prev_bins_cumsum
        assert (
            bin_sum.shape == counts.shape
        ), f"Bin sum shape {bin_sum.shape} does not agree with counts {counts.shape}"

        return jnp.where(counts > 0, bin_sum / counts, bin_sum)

    # Compute bin averages
    # bin_maccs = vmap(lambda start, end: accs[start:end].mean())(counts[:-1], counts[1:])
    # bin_mconfs = vmap(lambda start, end: confs[start:end].mean())(
    #     counts[:-1], counts[1:]
    # )

    bin_maccs = compute_bins_mean(accs, counts)
    bin_mconfs = compute_bins_mean(confs, counts)

    assert (
        bin_maccs.shape == (bins,)
    ), f"Binned mean accuracy has wrong size expected:{(bins,)} and got {bin_maccs.shape}"
    assert (
        bin_mconfs.shape == (bins,)
    ), f"Binned mean confidence has wrong size expected {(bins,)} and got {bin_mconfs.shape}"

    return bin_mconfs, bin_maccs


def flatten_post_dim(posterior_samples):
    assert "logits" in posterior_samples, "Model is missing logits deterministic site"
    logits = posterior_samples["logits"]
    if jnp.ndim(logits) == 4:

        def flat_fn(v):
            dims = (0, 1)
            shape = v.shape
            size = reduce(operator.mul, (shape[d] for d in dims), 1)
            return jnp.reshape(v, (size, *shape[len(dims) :]))

        posterior_samples = tree.map(flat_fn, posterior_samples)

    assert (
        jnp.ndim(posterior_samples["logits"]) == 3
    ), f"Unexpected (ndim not 3) logit dimension {logits.shape}."
    return posterior_samples
