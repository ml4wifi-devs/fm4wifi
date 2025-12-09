import jax
import jax.numpy as jnp
import jraph

from lai4wifi.graphs.attributes import Connection


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


def unbatch(batched_graph):
    graphs = jraph.unpad_with_graphs(batched_graph)
    return jraph.unbatch(graphs)


def nx_to_jraph(G):
    edges, senders, receivers = zip(*[[jax.tree.map(jnp.atleast_1d, Connection(**data)), u, v] for u, v, data in G.edges(data=True)])
    senders, receivers = senders + receivers, receivers + senders
    edges += edges

    return jraph.GraphsTuple(
        nodes=jnp.empty((len(G.nodes), 0), dtype=jnp.float64),
        edges=jax.tree.map(lambda *args: jnp.stack(args).astype(jnp.float64), *edges),
        receivers=jnp.asarray(receivers),
        senders=jnp.asarray(senders),
        globals=None,
        n_node=jnp.asarray([len(G.nodes)]),
        n_edge=jnp.asarray([len(edges)])
    )


def clear_graph(x):
    return x._replace(nodes=jnp.empty((x.nodes.shape[0], 0), dtype=x.nodes.dtype), globals=None)
