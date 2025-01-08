from flax import linen as nn

resnet_kernel_init = nn.initializers.variance_scaling(
    2.0, mode="fan_out", distribution="normal"
)


class ResNetBlockSmall(nn.Module):
    act_fn: callable  # Activation function
    c_out: int  # Output feature size
    subsample: bool = False  # If True, we apply a stride inside F

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.Conv(
            self.c_out,
            kernel_size=(3, 3),
            strides=(1, 1) if not self.subsample else (2, 2),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(x)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(
            self.c_out,
            kernel_size=(3, 3),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(z)
        z = nn.BatchNorm()(z, use_running_average=not train)

        if self.subsample:
            x = nn.Conv(
                self.c_out,
                kernel_size=(1, 1),
                strides=(2, 2),
                kernel_init=resnet_kernel_init,
            )(x)

        x_out = self.act_fn(z + x)
        return x_out


class PreActResNetBlockSmall(ResNetBlockSmall):
    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.BatchNorm()(x, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(
            self.c_out,
            kernel_size=(3, 3),
            strides=(1, 1) if not self.subsample else (2, 2),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(z)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(
            self.c_out,
            kernel_size=(3, 3),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(z)

        if self.subsample:
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)
            x = nn.Conv(
                self.c_out,
                kernel_size=(1, 1),
                strides=(2, 2),
                kernel_init=resnet_kernel_init,
                use_bias=False,
            )(x)

        x_out = z + x
        return x_out


class ResNetSmall(nn.Module):
    num_classes: int
    act_fn: callable
    block_class: nn.Module
    num_blocks: tuple = (3, 3, 3)
    c_hidden: tuple = (16, 32, 64)

    @nn.compact
    def __call__(self, x, train=True):
        # A first convolution on the original image to scale up the channel size
        x = nn.Conv(
            self.c_hidden[0],
            kernel_size=(3, 3),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(x)
        if (
            self.block_class == ResNetBlockSmall
        ):  # If pre-activation block, we do not apply non-linearities yet
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)

        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = bc == 0 and block_idx > 0
                # ResNet block
                x = self.block_class(
                    c_out=self.c_hidden[block_idx],
                    act_fn=self.act_fn,
                    subsample=subsample,
                )(x, train=train)

        # Mapping to classification output
        x = x.mean(axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        return x
