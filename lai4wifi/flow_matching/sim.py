import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from itertools import combinations

import jax
import jax.numpy as jnp
import networkx as nx
import orbax.checkpoint as ocp
from mapc_research.envs.test_scenarios import EXEMPLARY_SCENARIOS as ALL_SCENARIOS
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from lai4wifi.flow_matching.model import FlowMatching
from lai4wifi.autoencoder.attributes import *
from lai4wifi.autoencoder.dataset import MAX_TX_POWER, RSSI_MEAN, RSSI_STD, RSSI_THR
from lai4wifi.autoencoder.model import GnnAutoencoder
from lai4wifi.autoencoder.utils import nx_to_jraph


def associations_to_nx(scenario):
    distance = jnp.sqrt(jnp.sum((scenario.pos[:, None, :] - scenario.pos[None, ...]) ** 2, axis=-1))
    path_loss = scenario.path_loss_fn(distance, scenario.walls)

    G = nx.Graph()

    for ap, stas in scenario.associations.items():
        for sta in stas:
            if not isinstance(sta, int):
                sta = sta.item()

            rssi = (MAX_TX_POWER - path_loss[ap, sta] - RSSI_MEAN) / RSSI_STD
            G.add_edge(ap, sta, active=IsActive.TRUE, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.NA, success=0.0, tx_power=TxPowerValue.NA, type=EdgeType.STA)

    for ap_1, ap_2 in combinations(scenario.associations.keys(), r=2):
        rssi = (MAX_TX_POWER - path_loss[ap_1, ap_2] - RSSI_MEAN) / RSSI_STD
        if rssi > RSSI_THR:
            G.add_edge(ap_1, ap_2, active=IsActive.NA, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.NA, success=0.0, tx_power=TxPowerValue.NA, type=EdgeType.AP)

    return G


def get_initial_configurations(scenario, aps, stas, n_timesteps, key, encode_fn):
    G = associations_to_nx(scenario)
    graph = nx_to_jraph(G)
    context = [encode_fn(graph)] * n_timesteps

    n_nodes = len(aps) + sum(len(stas) for stas in scenario.associations.values())

    for _ in range(n_timesteps):
        key, ap_key, sta_key, tx_key, scenario_key = jax.random.split(key, 5)
        sharing_ap = jax.random.choice(ap_key, aps).item()
        selected_sta = jax.random.choice(sta_key, stas[sharing_ap]).item()

        G = associations_to_nx(scenario)
        G.edges[sharing_ap, selected_sta]['selected'] = IsSelected.TRUE
        G.edges[sharing_ap, selected_sta]['tx_power'] = TxPowerValue.LVL_1
        G.edges[sharing_ap, selected_sta]['mcs'] = McsValue.MCS_11
        graph = nx_to_jraph(G)

        tx = connections_to_tx(scenario.associations, n_nodes, tx_key, graph, sharing_ap, selected_sta)
        _, _, internals = scenario(scenario_key, *tx, return_internals=True)

        context = get_next_context(G, graph, sharing_ap, selected_sta, context, internals, encode_fn)

    return G, context, graph


def get_network_desc(scenario, sharing_ap, selected_sta, encode_fn):
    G = associations_to_nx(scenario)
    G.edges[sharing_ap, selected_sta]['selected'] = IsSelected.TRUE
    network_desc = encode_fn(nx_to_jraph(G))
    return network_desc, G


def connections_to_tx(associations, n_nodes, key, graph, sharing_ap, selected_sta):
    selected = defaultdict(list)

    for i in range(graph.n_edge[0]):
        ap, sta = graph.senders[i].item(), graph.receivers[i].item()
        if ap in associations and sta in associations[ap] and jnp.argmax(graph.edges.selected[i]) == IsSelected.TRUE.value:
            selected[ap].append(i)

    tx_matrix = jnp.zeros((n_nodes, n_nodes), dtype=int)
    tx_power = jnp.zeros(n_nodes, dtype=int)
    mcs = jnp.zeros(n_nodes, dtype=int)
    selected_for_tx = []

    for ap, sta_indices in selected.items():
        key, subkey = jax.random.split(key)
        selected_sta_idx = jax.random.choice(subkey, len(sta_indices), replace=False).item()
        selected_for_tx.append(sta_indices[selected_sta_idx])

    for i in selected_for_tx:
        ap, sta = graph.senders[i].item(), graph.receivers[i].item()
        tx_matrix = tx_matrix.at[ap, sta].set(1)
        tx_power = tx_power.at[i].set(jnp.argmax(graph.edges.tx_power[i, :-1]))
        mcs = mcs.at[i].set(jnp.argmax(graph.edges.mcs[i, :-1]))

    return tx_matrix, tx_power, mcs


def get_next_context(G, graph, sharing_ap, selected_sta, context, internals, encode_fn):
    selected_idx = 0
    connections = graph.edges
    rssi = jnp.zeros_like(connections.rssi)
    success = jnp.zeros_like(connections.success)

    for i, (ap, sta, data) in enumerate(G.edges(data=True)):
        if ap == sharing_ap and sta == selected_sta:
            selected_idx = i

        if data['type'] == EdgeType.STA and (jnp.argmax(connections.selected[i]) == 1 or (ap == sharing_ap and sta == selected_sta)):
            rssi = rssi.at[i].set((internals.signal_power[ap, sta] - RSSI_MEAN) / RSSI_STD)
            success = success.at[i].set(internals.frames_transmitted[ap] / max(1, internals.ampdu_size[ap]))

    connections = Connection(
        active=connections.active,
        mcs=connections.mcs,
        rssi=rssi,
        selected=connections.selected.at[selected_idx].set(jnp.array(IsSelected.TRUE)),
        success=success,
        tx_power=connections.tx_power,
        type=connections.type
    )
    graph = graph._replace(edges=connections)
    next_conf = encode_fn(graph)

    context.append(next_conf)
    return context[1:]


def run_scenario(scenario, n_reps, n_steps, n_timesteps, seed, gen_fn, encode_fn, decode_fn):
    key = jax.random.PRNGKey(seed)
    runs = []
    actions = []

    aps = jnp.asarray(list(scenario.associations.keys()))
    stas = {ap: jnp.asarray(stas) for ap, stas in scenario.associations.items()}
    n_nodes = len(aps) + sum(len(stas) for stas in scenario.associations.values())

    for _ in trange(n_reps):
        scenario.reset()
        runs.append([])
        actions.append([])

        key, conf_key = jax.random.split(key)
        G, context, graph = get_initial_configurations(scenario, aps, stas, n_timesteps, conf_key, encode_fn)

        for _ in range(n_steps):
            key, ap_key, sta_key, fm_key, tx_key, scenario_key = jax.random.split(key, 6)

            sharing_ap = jax.random.choice(ap_key, aps).item()
            selected_sta = jax.random.choice(sta_key, stas[sharing_ap]).item()
            network_desc, G = get_network_desc(scenario, sharing_ap, selected_sta, encode_fn)
            all = jax.tree.map(lambda *x: jnp.stack(x, axis=1), network_desc, *context)
            x = network_desc._replace(
                edges=all.edges,
                nodes=all.nodes,
                globals=all.globals
            )

            z = gen_fn(x, rngs=fm_key)
            connections = decode_fn(z).edges
            graph = graph._replace(edges=connections)
            tx = connections_to_tx(scenario.associations, n_nodes, key, graph, sharing_ap, selected_sta)

            data_rate, _, internals = scenario(scenario_key, *tx, return_internals=True)
            runs[-1].append(data_rate.item())
            actions[-1].append(scenario.tx_to_action(*tx))

            context = get_next_context(G, graph, sharing_ap, selected_sta, context, internals, encode_fn)

    return runs, actions


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--checkpoint', type=str, required=True)
    args.add_argument('--n_reps', type=int, default=1)
    args.add_argument('--output', type=str, default='all_results.json')
    args.add_argument('--seed', type=int, default=42)
    args = args.parse_args()

    fm_cfg = OmegaConf.load(os.path.join(args.checkpoint, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(args.checkpoint), item_names=['params']) as ckpt_manager:
        fm_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

    fm = FlowMatching(**fm_cfg.model)
    gen_fn = jax.jit(partial(fm.apply, fm_params, method='gen'))

    gnn_cfg = OmegaConf.load(os.path.join(fm_cfg.gnn_path, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(fm_cfg.gnn_path), item_names=['params']) as ckpt_manager:
        gnn_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

    gnn = GnnAutoencoder(**gnn_cfg.model)
    encode_fn = jax.jit(partial(gnn.apply, gnn_params, method='encode'))
    decode_fn = jax.jit(partial(gnn.apply, gnn_params, method='decode'))

    all_results = []

    for scenario in tqdm(ALL_SCENARIOS, desc='Scenarios'):
        runs, actions = run_scenario(
            scenario, args.n_reps, scenario.n_steps, fm_cfg.train.n_timesteps, args.seed, gen_fn, encode_fn, decode_fn
        )
        all_results.append([{
            'runs': runs,
            'actions': actions
        }])

    with open(args.output, 'w') as file:
        json.dump(all_results, file)
