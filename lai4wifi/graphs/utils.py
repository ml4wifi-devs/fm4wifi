import jax
import jax.numpy as jnp
import jraph
import networkx as nx

from lai4wifi.graphs.attributes import Edge


def make_batch(graph_list):
    return jax.tree.map(lambda *args: jnp.stack(args), *graph_list)


def nx_to_jraph(G, pad=True, n_node=64, n_edge=256):
    Gl = nx.line_graph(G)
    nx.set_node_attributes(Gl, dict(map(lambda e: (e[0:2], e[2]), G.edges(data=True))))

    senders, receivers = jnp.where(nx.to_numpy_array(Gl))
    nodes = [jax.tree.map(jnp.atleast_1d, Edge(**data)) for node, data in Gl.nodes(data=True)]
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

    if pad:
        graph = jraph.pad_with_graphs(graph, n_node, n_edge)

    return graph
