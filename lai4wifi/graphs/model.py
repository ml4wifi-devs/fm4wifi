from typing import Callable, Sequence

import jraph
import flax.linen as nn

from lai4wifi.graphs.attributes import Connection


class MLP(nn.Module):
    features: list
    act_fn: Callable = nn.relu
    out_fn: Callable = None

    @nn.compact
    def __call__(self, x):
        for size in self.features[:-1]:
            x = nn.Dense(size)(x)
            x = self.act_fn(x)

        if self.out_fn is not None:
            x = self.out_fn(x)

        return nn.Dense(self.features[-1])(x)


def make_embed_fn(latent_size):
    @jraph.concatenated_args
    def embed_fn(inputs):
        return nn.Dense(latent_size)(inputs)

    return embed_fn


def make_mlp(features, out_fn=None):
    @jraph.concatenated_args
    def update_fn(inputs):
        return MLP(features, out_fn=out_fn)(inputs)

    return update_fn


class Encoder(nn.Module):
    # https://arxiv.org/pdf/2107.09422v1
    # https://github.com/google-deepmind/jraph/blob/master/jraph/ogb_examples/train_flax.py

    latent_size: int
    features: Sequence[int]

    @nn.compact
    def __call__(self, graph):
        embed = jraph.GraphMapFeatures(
            embed_node_fn=make_embed_fn(self.latent_size)
        )
        encode = jraph.GraphNetwork(
            update_edge_fn=None,
            update_node_fn=make_mlp(self.features)
        )

        x = embed(graph)
        z = encode(x)
        return z


class Decoder(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, z):
        decode = jraph.GraphNetwork(
            update_edge_fn=None,
            update_node_fn=make_mlp(self.features, nn.relu)
        )

        x = decode(z)
        nodes = Connection(
            active=nn.Dense(2)(x.nodes),
            mcs=nn.Dense(12)(x.nodes),
            rate=nn.Dense(1)(x.nodes),
            rssi=nn.Dense(1)(x.nodes),
            selected=nn.Dense(2)(x.nodes),
            tx_power=nn.Dense(4)(x.nodes),
            type=nn.Dense(2)(x.nodes)
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
    latent_size: int = 64
    encoder_features: Sequence[int] = (32, 32)
    decoder_features: Sequence[int] = (32, 64)

    def setup(self):
        self.encoder = Encoder(self.latent_size, self.encoder_features)
        self.decoder = Decoder(self.decoder_features)

    def __call__(self, graph):
        z = self.encoder(graph)
        x = self.decoder(z)
        return x

    def encode(self, graph):
        return self.encoder(graph)

    def decode(self, z):
        return self.decoder(z)
