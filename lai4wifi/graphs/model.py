from typing import Callable, Sequence

import jax.numpy as jnp
import jraph
import flax.linen as nn

from lai4wifi.graphs.attributes import *


class MLP(nn.Module):
    features: list
    dtype: jnp.dtype
    act_fn: Callable = nn.relu
    out_fn: Callable = None

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.features):
            x = nn.Dense(size, dtype=self.dtype)(x)
            if i < len(self.features) - 1:
                x = self.act_fn(x)

        if self.out_fn is not None:
            x = self.out_fn(x)

        return x


def make_embed_fn(latent_size, dtype):
    @jraph.concatenated_args
    def embed_fn(inputs):
        return nn.Dense(latent_size, dtype=dtype)(inputs)

    return embed_fn


def make_mlp(features, dtype, out_fn=None):
    @jraph.concatenated_args
    def update_fn(inputs):
        return MLP(features, dtype, out_fn=out_fn)(inputs)

    return update_fn


class Encoder(nn.Module):
    # https://arxiv.org/pdf/2107.09422v1
    # https://github.com/google-deepmind/jraph/blob/master/jraph/ogb_examples/train_flax.py

    latent_size: int
    features: Sequence[int]
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, graph):
        embed = jraph.GraphMapFeatures(
            embed_node_fn=make_embed_fn(self.latent_size, self.dtype)
        )
        encode = jraph.GraphNetwork(
            update_edge_fn=None,
            update_node_fn=make_mlp(self.features, self.dtype)
        )

        x = embed(graph)
        z = encode(x)
        return z


class Decoder(nn.Module):
    features: Sequence[int]
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, z):
        decode = jraph.GraphNetwork(
            update_edge_fn=None,
            update_node_fn=make_mlp(self.features, self.dtype, nn.relu)
        )

        x = decode(z)
        nodes = Connection(
            active=nn.Dense(len(IsActive), dtype=self.dtype)(x.nodes),
            mcs=nn.Dense(len(McsValue), dtype=self.dtype)(x.nodes),
            rssi=nn.Dense(1, dtype=self.dtype)(x.nodes),
            selected=nn.Dense(len(IsSelected), dtype=self.dtype)(x.nodes),
            success=nn.Dense(1, dtype=self.dtype)(x.nodes),
            tx_power=nn.Dense(len(TxPowerValue), dtype=self.dtype)(x.nodes),
            type=nn.Dense(len(EdgeType), dtype=self.dtype)(x.nodes)
        )

        return jraph.GraphsTuple(
            nodes=nodes,
            edges=None,
            receivers=x.receivers,
            senders=x.senders,
            globals=None,
            n_node=x.n_node,
            n_edge=x.n_edge
        )


class GnnAutoencoder(nn.Module):
    latent_size: int
    encoder_features: Sequence[int]
    decoder_features: Sequence[int]
    dtype: jnp.dtype

    def setup(self):
        self.encoder = Encoder(self.latent_size, self.encoder_features, self.dtype)
        self.decoder = Decoder(self.decoder_features, self.dtype)

    def __call__(self, graph):
        z = self.encoder(graph)
        x = self.decoder(z)
        return x

    def encode(self, graph):
        return self.encoder(graph)

    def decode(self, z):
        return self.decoder(z)
