import jax
from flax import linen as nn
from jax import numpy as jnp


class Reshape(nn.Module):
    shape: tuple

    @nn.compact
    def __call__(self, x):
        return x.reshape((x.shape[0],) + self.shape)


class Flatten(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x.reshape(x.shape[0], -1)


class Concatenate(nn.Module):
    axis: int = -1

    @nn.compact
    def __call__(self, *xs):
        return jnp.concatenate(xs, axis=self.axis)


class Conv(nn.Module):
    channels: int
    kernel_size: int
    strides: int = 1
    init_std: float = None

    @nn.compact
    def __call__(self, x):
        if self.init_std is not None:
            init_std = self.init_std
        else:
            init_std = 1.0 / (self.channels * self.kernel_size ** 2)

        return nn.Conv(
            features=self.channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.strides, self.strides),
            padding='SAME',
            use_bias=False,
            dtype=jnp.bfloat16,
            kernel_init=nn.initializers.normal(init_std, dtype=jnp.float32)
        )(x)


class LayerNormF32(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        return x.astype(x.dtype)


class UpSample(nn.Module):
    scale: int = 2
    method: str = 'nearest'

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape
        x = jax.image.resize(x, (b, h * self.scale, w * self.scale, c), method=self.method)
        x = Conv(c, kernel_size=3)(x)
        return x


class DownSample(nn.Module):
    @nn.compact
    def __call__(self, x):
        return Conv(x.shape[-1], kernel_size=3, strides=2)(x)


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x, t=None):
        residual = x

        x = LayerNormF32()(x)
        x = nn.swish(x)
        x = Conv(self.channels, kernel_size=3)(x)

        if t is not None:
            t = t[:, None, None, :]
            t = Conv(self.channels, kernel_size=1, init_std=0.01 / self.channels)(t)
            t = nn.swish(t)
            x = x + t

        x = LayerNormF32()(x)
        x = nn.swish(x)
        x = Conv(self.channels, kernel_size=3, init_std=0.0001 / self.channels)(x)

        if residual.shape[-1] != self.channels:
            residual = Conv(self.channels, kernel_size=1)(residual)

        return x + residual


class AttentionBlock(nn.Module):
    channels: int
    n_heads: int

    @nn.compact
    def __call__(self, x):
        residual = x

        x = LayerNormF32()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.channels,
            dtype=jnp.bfloat16,
            out_kernel_init=nn.initializers.normal(0.2 / self.channels ** 0.5),
            use_bias=False
        )(x)
        x = x + residual

        return x
