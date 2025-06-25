from dataclasses import dataclass
from random import choice
from typing import Optional

import jax
import jax.numpy as jnp
import jraph
import networkx as nx

from lai4wifi.autoencoder.attributes import Connection, IsSelected


@dataclass
class Graph:
    graph: jraph.GraphsTuple
    embedding: Optional[jax.Array]
    pos: jax.Array
    walls: jax.Array
    selected_idx: int
    line_senders: jax.Array
    line_receivers: jax.Array


def nearest_power_of_two(n):
    if not isinstance(n, int):
        n = n.item()
    return 2 ** (n - 1).bit_length()


def make_batch(graph_list, pad=True):
    batch = jraph.batch(graph_list)

    if pad:
        pad_n_node = nearest_power_of_two(jnp.sum(batch.n_node) + 1)
        pad_n_edge = nearest_power_of_two(batch.senders.shape[0])
        pad_n_graph = batch.n_node.shape[0] + 1
        batch = jraph.pad_with_graphs(batch, pad_n_node, pad_n_edge, pad_n_graph)

    return batch


def nx_to_jraph(G, return_misc=False, pos=None, walls=None):
    edges, senders, receivers = zip(*[[jax.tree.map(jnp.atleast_1d, Connection(**data)), u, v] for u, v, data in G.edges(data=True)])
    senders, receivers = senders + receivers, receivers + senders
    edges += edges

    graph = jraph.GraphsTuple(
        nodes=jnp.empty((len(G.nodes), 0), dtype=jnp.float32),
        edges=jax.tree.map(lambda *args: jnp.stack(args), *edges),
        receivers=jnp.asarray(receivers),
        senders=jnp.asarray(senders),
        globals=None,
        n_node=jnp.asarray([len(G.nodes)]),
        n_edge=jnp.asarray([len(edges)])
    )

    if return_misc:
        Gl = nx.line_graph(G)
        line_senders, line_receivers = jnp.where(nx.to_numpy_array(Gl))
        selected_idx = choice([i for i, (_, _, d) in enumerate(G.edges(data=True)) if d['selected'] == IsSelected.TRUE])
        return Graph(graph, None, pos, walls, selected_idx, line_senders, line_receivers)

    return graph


def clear_graph(x):
    return x._replace(nodes=jnp.empty((y.nodes.shape[0], 0), dtype=x.nodes.dtype), globals=None)


def eig_laplacian(senders, receivers, n_nodes, k):
    adj = jnp.zeros((n_nodes, n_nodes), dtype=jnp.float32)
    adj = adj.at[(senders, receivers)].set(1)
    D = jnp.diag(adj.sum(axis=1))
    L = D - adj

    evals, evects = jnp.linalg.eigh(L)

    idx = jnp.argsort(evals)[:k]
    evals, evects = evals[idx].real, evects[:, idx].real
    evals = evals.clip(min=0)

    norm = jnp.linalg.norm(evects, axis=0)
    norm = norm.clip(min=1e-12)
    evects /= norm

    if (N := len(evals)) < k:
        pad_width = [(0, 0)] * (evects.ndim - 1) + [(0, k - N)]
        evects = jnp.pad(evects, pad_width, mode='constant', constant_values=jnp.nan)
        pad_width = [(0, 0)] * (evals.ndim - 1) + [(0, k - N)]
        evals = jnp.pad(evals, pad_width, mode='constant', constant_values=jnp.nan)

    return evals, evects
