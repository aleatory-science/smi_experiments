from functools import partial
from jax import jit, device_get, numpy as jnp
from collections import namedtuple
from datasets.ood_detection.load_ood_detect import train_batches
import tqdm
from src.methods import ASVGD

Result = namedtuple("Result", ["params", "state", "losses"])


def setup_run(engine, dataset):
    def run(rng_key, num_epochs, batch_size, init_state):
        if num_epochs < 1:
            raise ValueError("num_epochs must be a positive integer.")

        if isinstance(engine, ASVGD):
            cyc_fn = ASVGD._cyclical_annealing(
                num_epochs, engine.num_cycles, engine.trans_speed
            )

            @partial(jit, static_argnames=["e", "n", "batch_size"])
            def body_fn(state, x, y, e, n, batch_size):
                t, state = state
                engine.loss_temperature = cyc_fn(t) / float(engine.num_stein_particles)
                state, loss = e.update(state, x, y, n=n, batch_size=batch_size)
                return (t + 1, state), loss

        @partial(jit, static_argnames=["e", "n", "batch_size"])
        def body_fn(state, x, y, e, n, batch_size):
            state, loss = e.update(state, x, y, n=n, batch_size=batch_size)
            return state, loss

        state = init_state
        losses = []

        with tqdm.trange(1, num_epochs + 1) as t:
            for i in t:
                batches, n = train_batches(batch_size, dataset)
                for x, y in batches:
                    if state is None:
                        state = engine.init(
                            rng_key,
                            x,
                            y,
                            n=n,
                            batch_size=batch_size,
                        )
                        loss = engine.evaluate(state, x, y, n=n, batch_size=batch_size)
                        if isinstance(engine, ASVGD):
                            state = (0, state)
                    else:
                        state, loss = body_fn(
                            state, x, y, batch_size=batch_size, e=engine, n=n
                        )

                    losses.append(device_get(loss))

                avg_loss = sum(losses[-n // batch_size :]) / batch_size
                t.set_postfix_str(
                    "init loss: {:.2f}, avg. loss [{}-{}]: {:.2f}".format(
                        losses[0], i - 1, i, avg_loss
                    ),
                    refresh=False,
                )
        losses = jnp.stack(losses)
        if isinstance(engine, ASVGD):
            _, state = state
        return Result(engine.get_params(state), state, losses)

    return run
