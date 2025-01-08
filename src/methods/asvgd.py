from numpyro.contrib.einstein import ASVGD as NPASVGD


class ASVGD(NPASVGD):
    def __init__(
        self,
        model,
        optim,
        kernel_fn,
        num_stein_particles=10,
        num_cycles=10,
        trans_speed=10,
        repulsion_temperature=1.0,
        guide_kwargs={},
        **static_kwargs,
    ):
        super().__init__(
            model,
            optim,
            kernel_fn,
            num_stein_particles,
            num_cycles,
            trans_speed,
            guide_kwargs,
            **static_kwargs,
        )
        self.repulsion_temperature = repulsion_temperature
