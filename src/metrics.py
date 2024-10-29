from numpyro.infer import log_likelihood
from jax.scipy.special import logsumexp
from jax import numpy as jnp, vmap

from numpyro.contrib.einstein import MixtureGuidePredictive
from numpyro.infer import Predictive


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


def nll_fn(posterior_samples, model, x, y, batch_ndims):
    r"""Compute the negative log likelihood given by
    1/S \sum_{s=1}^S (-log \sum_i p(y_i|theta^(s), x_i)).
    """
    lls = log_likelihood(model, posterior_samples, x, y, None, batch_ndims=batch_ndims)[
        "y"
    ]
    return -(logsumexp(lls, axis=0) - jnp.log(lls.shape[0])).mean()


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


def ovi_sampler(engine, inf_results, num_samples, return_sites):
    pred = Predictive(
        model=engine.model,
        guide=engine.guide,
        params=inf_results.params,
        num_samples=num_samples,
        return_sites=return_sites,
    )
    return pred


def normalize(val, mean=None, std=None):
    """Normalize data to zero mean, unit variance"""
    if mean is None and std is None:
        # Only use training data to estimate mean and std.
        std = jnp.std(val, 0, keepdims=True)
        std = jnp.where(std == 0, 1.0, std)
        mean = jnp.mean(val, 0, keepdims=True)
    return (val - mean) / std, mean, std
