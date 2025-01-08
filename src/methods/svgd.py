from numpyro.contrib.einstein import SVGD as NPSVGD


class SVGD(NPSVGD):
    def __init__(
        self,
        model,
        optim,
        kernel_fn,
        num_stein_particles=10,
        repulsion_temperature=1.0,
        guide_kwargs={},
        **static_kwargs,
    ):
        super().__init__(
            model,
            optim,
            kernel_fn,
            num_stein_particles=10,
            guide_kwargs={},
            **static_kwargs,
        )

        self.repulsion_temperature = repulsion_temperature
