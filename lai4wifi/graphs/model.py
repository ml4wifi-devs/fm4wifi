from typing import Callable

import jraph
import flax.linen as nn


class MLP(nn.Module):
    features: list
    activation_fn: Callable = nn.relu
    activation_final_fn: Callable = None

    @nn.compact
    def __call__(self, x):
        for size in self.features[:-1]:
            x = nn.Dense(size)(x)
            x = self.activation_fn(x)

        if self.activation_final_fn is not None:
            x = self.activation_final_fn(x)

        return nn.Dense(self.features[-1])(x)


class Encoder(nn.Module):
    # https://arxiv.org/pdf/2107.09422v1
    # https://github.com/google-deepmind/jraph/blob/master/jraph/ogb_examples/train_flax.py

    latent_size: int = 64
    features: tuple[int] = (32, 32)

    @nn.compact
    def __call__(self, graph):
        def make_embed_fn(latent_size):
            @jraph.concatenated_args
            def embed_fn(inputs):
                return nn.Dense(latent_size)(inputs)

            return embed_fn

        def make_mlp(features):
            @jraph.concatenated_args
            def update_fn(inputs):
                return MLP(features)(inputs)

            return update_fn

        embed = jraph.GraphMapFeatures(
            embed_node_fn=make_embed_fn(self.latent_size)
        )
        mlp = jraph.GraphNetwork(
            update_edge_fn=None,
            update_node_fn=make_mlp(self.features)
        )
        return mlp(embed(graph))
