import jax
import jax.numpy as jnp
from flax import linen as nn


class FeedForwardBlock(nn.Module):
    ff_dim: int
    drop_rate: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, training=True):
        out_dim = x.shape[-1]
        x = nn.Dense(self.ff_dim, dtype=self.dtype, use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.drop_rate)(x, deterministic=not training)
        x = nn.Dense(out_dim, dtype=self.dtype, use_bias=False)(x)
        x = nn.Dropout(self.drop_rate)(x, deterministic=not training)
        return x


class TransformerBlock(nn.Module):
    num_heads: int
    ff_dim: int
    drop_rate: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, mask, training=True):
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=x.shape[-1],
            dtype=self.dtype,
            use_bias=False
        )(x, mask=mask)
        x = x + residual

        residual = x
        x = nn.LayerNorm()(x)
        x = FeedForwardBlock(self.ff_dim, self.drop_rate, self.dtype)(x, training=training)
        x = x + residual

        return x


class Transformer(nn.Module):
    num_layers: int
    num_heads: int
    ff_dim: int
    drop_rate: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, t, mask=None, training=True):
        for block in range(self.num_layers):
            x = TransformerBlock(self.num_heads, self.ff_dim, self.drop_rate, self.dtype)(x, mask, training=training)
            t = nn.Dense(x.shape[-1], dtype=self.dtype, use_bias=False)(t)
            x = x + t

        x = nn.LayerNorm()(x)
        return x


class TimeEmbedding(nn.Module):
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings_flax.py

    ff_dim: int
    latent_size: int
    dtype: jnp.dtype

    @staticmethod
    def get_sinusoidal_embeddings(timesteps, embedding_dim, min_timescale=1, max_timescale=1.0e4):
        num_timescales = float(embedding_dim // 2)
        log_timescale_increment = jnp.log(max_timescale / min_timescale) / num_timescales
        inv_timescales = min_timescale * jnp.exp(jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment)
        time = jnp.expand_dims(timesteps, -1) * jnp.expand_dims(inv_timescales, 0)
        signal = jnp.concatenate([jnp.cos(time), jnp.sin(time)], axis=-1)
        signal = signal.reshape([*timesteps.shape, embedding_dim])
        return signal

    @nn.compact
    def __call__(self, t):
        t = self.get_sinusoidal_embeddings(t, self.latent_size)
        return FeedForwardBlock(self.ff_dim, drop_rate=0., dtype=self.dtype)(t)


class FlowMatching(nn.Module):
    num_layers: int
    num_heads: int
    ff_dim: int
    latent_size: int
    drop_rate: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, t, mask, training=True):
        x = x.reshape(x.shape[0], x.shape[1], -1)
        t = TimeEmbedding(self.ff_dim, x.shape[-1], self.dtype)(t)
        x = Transformer(self.num_layers, self.num_heads, self.ff_dim, self.drop_rate, self.dtype)(x, t, mask, training=training)
        x = nn.Dense(self.latent_size, dtype=self.dtype)(x)
        return x

    def gen(self, context, mask=None, n_steps=10):
        def scan_fn(unet, x, t):
            cx = jnp.concatenate([context, x], axis=2)
            t = jnp.full(x.shape[:2], t)
            v = unet(cx, t, mask, training=False)[..., None, :]
            x = x + v / n_steps
            return x, None

        z = jax.random.normal(self.make_rng('lai'), context.shape[:2] + (1, self.latent_size,), dtype=jnp.float32)
        scan = nn.scan(scan_fn, variable_broadcast='params')
        x, _ = scan(self, z, jnp.linspace(0.0, 1.0, n_steps, endpoint=False))
        return x[..., 0, :]
