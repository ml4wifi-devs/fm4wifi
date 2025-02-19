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
    return 2 ** (n - 1).bit_length()


def pad_batch(graph_list):
    max_n_node = nearest_power_of_two(max([g.n_node[0] for g in graph_list]).item())
    max_n_edge = nearest_power_of_two(max([g.n_edge[0] for g in graph_list]).item())
    return list(map(lambda g: jraph.pad_with_graphs(g, max_n_node, max_n_edge), graph_list))


def make_batch(graph_list):
    return jax.tree.map(lambda *args: jnp.stack(args), *graph_list)


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
        n_edge=jnp.asarray([len(Gl.edges)])
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
