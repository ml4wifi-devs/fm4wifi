import jax
import jax.numpy as jnp
from flax import linen as nn

from lai4wifi.flow_matching.layers import AttentionBlock, Conv, DownSample, LayerNormF32, ResidualBlock, UpSample, Concatenate, Reshape


class MLP(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.dim, use_bias=False, dtype=jnp.bfloat16)(x)
        x = nn.silu(x)
        x = nn.Dense(self.dim, use_bias=False, dtype=jnp.bfloat16)(x)
        return x


class TimeEmbedding(nn.Module):
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings_flax.py

    proj_dim: int
    dim: int

    @staticmethod
    def get_sinusoidal_embeddings(timesteps, embedding_dim, min_timescale=1, max_timescale=1.0e4):
        num_timescales = float(embedding_dim // 2)
        log_timescale_increment = jnp.log(max_timescale / min_timescale) / num_timescales
        inv_timescales = min_timescale * jnp.exp(jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment)
        time = jnp.expand_dims(timesteps, 1) * jnp.expand_dims(inv_timescales, 0)
        signal = jnp.concatenate([jnp.cos(time), jnp.sin(time)], axis=1)
        signal = jnp.reshape(signal, [jnp.shape(timesteps)[0], embedding_dim])
        return signal

    @nn.compact
    def __call__(self, t):
        t = self.get_sinusoidal_embeddings(t, self.proj_dim)
        return MLP(self.dim)(t)


class CondAttentionBlock(nn.Module):
    channels: int
    n_heads: int

    @nn.compact
    def __call__(self, x, c):
        b, h, w, _ = x.shape
        c = nn.Dense(x.shape[-1], use_bias=False, dtype=jnp.bfloat16)(c)
        c = c[:, None]
        x = Reshape((-1, x.shape[-1]))(x)
        x = Concatenate(axis=1)(x, c)
        x = AttentionBlock(self.channels, self.n_heads)(x)
        x = x[:, 1:]
        x = Reshape((h, w, x.shape[-1]))(x)
        return x


class UNet(nn.Module):
    channels: int
    channel_multipliers: tuple
    n_resnet_blocks: int
    n_heads: int

    @nn.compact
    def __call__(self, x, cond, t):
        in_channels = x.shape[-1]

        emb_dim = self.channels * self.channel_multipliers[0]
        c = MLP(2 * emb_dim)(cond)
        t = TimeEmbedding(emb_dim, 2 * emb_dim)(t)

        x = Conv(self.channels, kernel_size=3)(x)
        hidden = tuple()

        for i, multiplier in enumerate(self.channel_multipliers):
            channels = self.channels * multiplier

            for _ in range(self.n_resnet_blocks):
                if i == len(self.channel_multipliers) - 2:
                    x = CondAttentionBlock(channels, self.n_heads)(x, c)
                    hidden = hidden + (x,)

                x = ResidualBlock(channels)(x, t)
                hidden = hidden + (x,)

            if i != len(self.channel_multipliers) - 1:
                x = DownSample()(x)

        x = ResidualBlock(channels)(x, t)

        for _ in range(self.n_resnet_blocks):
            x = CondAttentionBlock(channels, self.n_heads)(x, c)
            x = ResidualBlock(channels)(x, t)

        idx = 1

        for i, multiplier in enumerate(self.channel_multipliers[::-1]):
            channels = self.channels * multiplier

            for _ in range(self.n_resnet_blocks):
                if i == 1:
                    x = Concatenate()(hidden[-idx], x)
                    x = CondAttentionBlock(channels, self.n_heads)(x, c)
                    idx += 1

                x = Concatenate()(hidden[-idx], x)
                x = ResidualBlock(channels)(x, t)
                idx += 1

            if i != len(self.channel_multipliers) - 1:
                x = UpSample()(x)
                _, h, w, _ = hidden[-idx].shape
                x = x[:, :h, :w]

        x = LayerNormF32()(x)
        x = nn.swish(x)
        x = Conv(in_channels, kernel_size=1)(x)

        return x


class FMUnet(UNet):
    channels: int = 4
    channel_multipliers: tuple = (2, 3, 4)
    n_resnet_blocks: int = 2
    n_heads: int = 2
    out_shape: tuple = (44, 44, 1)

    def gen(self, cond, n_steps=10):
        def scan_fn(unet, x, t):
            t = jnp.full(cond.shape[0], t)
            v = unet(x, cond, t)
            x = x + v / n_steps
            return x, None

        z = jax.random.normal(self.make_rng('lai'), (cond.shape[0], *self.out_shape))
        scan = nn.scan(scan_fn, variable_broadcast='params')
        x, _ = scan(self, z, jnp.linspace(0.0, 1.0, n_steps, endpoint=False))
        return x
