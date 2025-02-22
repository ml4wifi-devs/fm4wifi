from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jraph
import networkx as nx

from lai4wifi.graphs.attributes import Connection, EdgeType, IsSelected


@dataclass
class Graph:
    graph: jraph.GraphsTuple
    pos: jax.Array
    walls: jax.Array
    aps: jax.Array
    stas: jax.Array
    idx_to_edge: jax.Array
    idx_to_ap: jax.Array
    idx_to_sta: jax.Array
    selected_idx: int


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
    Gl = nx.line_graph(G)
    nx.set_node_attributes(Gl, dict(map(lambda e: (e[0:2], e[2]), G.edges(data=True))))

    senders, receivers = jnp.where(nx.to_numpy_array(Gl))
    nodes = [jax.tree.map(jnp.atleast_1d, Connection(**data)) for node, data in Gl.nodes(data=True)]
    nodes = jax.tree.map(lambda *args: jnp.stack(args), *nodes)

    graph = jraph.GraphsTuple(
        nodes=nodes,
        edges=None,
        receivers=receivers,
        senders=senders,
        globals=None,
        n_node=jnp.asarray([len(Gl.nodes)]),
        n_edge=jnp.asarray([2 * len(Gl.edges)])
    )

    if return_misc:
        idx_to_edge = jnp.array([(u, v) for (u, v), d in Gl.nodes(data=True) if d['type'] == EdgeType.STA])
        idx_to_ap = jnp.array([u for (u, _), d in Gl.nodes(data=True) if d['type'] == EdgeType.STA])
        idx_to_sta = jnp.array([v for (_, v), d in Gl.nodes(data=True) if d['type'] == EdgeType.STA])
        selected_idx = next(i for i, ((_, _), d) in enumerate(Gl.nodes(data=True)) if d['selected'] == IsSelected.TRUE)
        aps = jnp.unique(idx_to_ap)
        stas = jnp.unique(idx_to_sta)
        return Graph(graph, pos, walls, aps, stas, idx_to_edge, idx_to_ap, idx_to_sta, selected_idx)

    return graph
