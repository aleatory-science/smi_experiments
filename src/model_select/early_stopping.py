"""This program implements early stopping for SteinVI. The UI is experimental and need not reflect the final version."""

from numpyro.contrib.einstein.steinvi import SteinVI


from jax import numpy as jnp
from jax.lax import cond


class OnlineEarlyStop(SteinVI):
    def __init__(self, engine, period):
        self.engine = engine
        self.period = period

    def setup_run(self, rng_key, num_steps, args, init_state, kwargs):
        istep, idiag, icol, iext, iinit = self.engine.setup_run(
            rng_key,
            num_steps,
            args,
            init_state,
            kwargs,
        )

        period = self.period

        decay_target = 0.001  # decay_target = fast_decay ** period

        fast_decay = jnp.exp(jnp.log(decay_target) / float(period))

        slow_period = 10 * period
        slow_decay = jnp.exp(jnp.log(decay_target) / float(slow_period))

        slow_online_avg, fast_online_avg = 0.0, 0.0

        def step(info):
            _, skip, _, _, _ = info

            def skip_fn(*args):
                return args

            def step_fn(t, skip, foa, soa, iinfo):
                iinfo = istep(iinfo)
                tau_fast = jnp.minimum(t, period)

                loss = icol(iinfo)

                foa = (
                    loss / (tau_fast + 1)
                    + (tau_fast / (tau_fast + 1)) * fast_decay * foa
                )

                tau_slow = jnp.minimum(t, slow_period)
                soa = (
                    loss / (tau_slow + 1)
                    + (tau_slow / (tau_slow + 1)) * slow_decay * soa
                )

                skip = jnp.logical_or(skip, foa > 2 * soa)

                return (t + 1, skip, foa, soa, iinfo)

            return cond(skip, skip_fn, step_fn, *info)

        info_init = (0.0, False, fast_online_avg, slow_online_avg, iinit)

        def diagnostic(info):
            _, _, _, _, iinfo = info
            return idiag(iinfo)

        def collect(info):
            __, _, _, _, iinfo = info
            return icol(iinfo)

        def extract_state(info):
            _, _, _, _, iinfo = info
            return iext(iinfo)

        return step, diagnostic, collect, extract_state, info_init

    def get_params(self, state):
        return self.engine.get_params(state)
