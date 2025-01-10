import jax.numpy as jnp
import networkx as nx

from lai4wifi.graph import embed_graph, embed_edge, decode_graph
from lai4wifi.model import model
from lai4wifi.scenarios import train_scenarios
from lai4wifi.sim import simulate
from lai4wifi.upper_bound import solver, draw_conf


def forward_gdm(network_graph: nx.Graph, active_connections: list[nx.Edge], selected_connection: nx.Edge, history: list[nx.Graph], conf: nx.Graph) -> nx.Graph:
    """
    Forward pass for the GDM model.

    Graph description:
    - nodes: APs and stations,
    - edges: connections between APs and stations (downlink only),
    - node attributes:
        - 'type' (AP or station),
        - 'tx_power' (transmission power) - optional, only for APs,
        - 'MCS' (modulation and coding scheme) - optional, only for APs,
        - 'RSSI' (received signal strength indicator) - optional, only for stations, only for history,
    - edge attributes:
        - 'is_active' (whether the connection is active).

    Parameters
    ----------
    network_graph : nx.Graph
        The graph representing the whole network.
    active_connections : list[nx.Edge]
        List of active connections (not all connections must be active).
    selected_connection : nx.Edge
        The connection to be selected (sharing AP + designated station).
    history : list[nx.Graph]
        List of previous MAPC configurations for past `k` TXOPs.
    conf : nx.Graph
        The MAPC configuration for this TXOP (noise to be added).

    Returns
    -------
    nx.Graph
        Denoised MAPC configuration for this TXOP.
    """

    emb_graph = embed_graph(network_graph)
    emb_connections = sum([embed_edge(emb_graph, edge) for edge in active_connections])
    emb_selected_connection = embed_edge(emb_graph, selected_connection)
    emb_history = jnp.asarray([embed_graph(graph) for graph in history])

    emb_input = jnp.concatenate([emb_graph, emb_connections, emb_selected_connection, emb_history], axis=-1)

    emb_conf = embed_graph(conf)
    emb_conf_noised = emb_conf + noise()

    emb_output = model(emb_input, emb_conf_noised)
    conf = decode_graph(emb_output)

    return emb_conf, emb_output, conf


def train_gdm():
    # Training parameters
    n_txops_per_scenario = 100
    n_confs_per_scenario = 100
    history_len = 10
    dataset = []

    # Dataset generation
    for network_graph in train_scenarios:
        best_conf = solver(network_graph)
        random_txop_conf = [draw_conf(best_conf) for _ in range(n_txops_per_scenario)]

        for _ in range(n_confs_per_scenario):
            confs = choice(random_txop_conf, history_len + 1)
            history = [simulate(network_graph, conf) for conf in confs[:-1]]
            active_connections = [edge for edge in confs[-1].edges if confs[-1].edges[edge]['is_active']]
            selected_connection = choice(active_connections)
            dataset.append((network_graph, active_connections, selected_connection, history, confs[-1]))

    # Training loop
    for *x, y in dataset:
        true, pred, _ = forward_gdm(*x, y)
        loss = mse(true, pred)
        ...

    # Inference loop
    for *x, _ in dataset:
        z = noise()
        _, _, conf = forward_gdm(*x, z)
        ...
