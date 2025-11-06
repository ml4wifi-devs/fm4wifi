import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import json
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import combinations

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import networkx as nx
import orbax.checkpoint as ocp
from mapc_research.envs.scenario_impl import residential_scenario
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from lai4wifi.flow_matching.model import FlowMatching
from lai4wifi.flow_matching.training import get_network_desc
from lai4wifi.graphs.attributes import *
from lai4wifi.graphs.dataset import MAX_TX_POWER, RSSI_MEAN, RSSI_STD, RSSI_THR
from lai4wifi.graphs.model import GnnAutoencoder
from lai4wifi.graphs.utils import nx_to_jraph


def associations_to_nx(scenario):
    distance = np.sqrt(np.sum((scenario.pos[:, None, :] - scenario.pos[None, ...]) ** 2, axis=-1))
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


def get_initial_configurations(scenario, n_timesteps, key, encode_fn):
    G = associations_to_nx(scenario)
    graph = nx_to_jraph(G)
    context = [encode_fn(graph)] * n_timesteps

    ap_sta_idxs = np.where(np.argmax(graph.edges.type, axis=-1) == EdgeType.STA.value)[0]
    n_nodes = len(scenario.associations) + sum(len(stas) for stas in scenario.associations.values())

    for _ in range(n_timesteps):
        key, sel_key, tx_key, scenario_key = jax.random.split(key, 4)

        selected_idx = jax.random.choice(sel_key, ap_sta_idxs).item()
        sharing_ap = graph.senders[selected_idx].item()
        selected_sta = graph.receivers[selected_idx].item()

        G = associations_to_nx(scenario)
        G.edges[sharing_ap, selected_sta]['selected'] = IsSelected.TRUE
        G.edges[sharing_ap, selected_sta]['tx_power'] = TxPowerValue.LVL_1
        G.edges[sharing_ap, selected_sta]['mcs'] = McsValue.MCS_11
        graph = nx_to_jraph(G)

        tx = connections_to_tx(scenario.associations, n_nodes, tx_key, graph)
        _, _, internals = scenario(scenario_key, *tx, return_internals=True)

        context = get_next_context(graph, context, internals, encode_fn)

    return context, graph, ap_sta_idxs


def connections_to_tx(associations, n_nodes, key, graph):
    selected = defaultdict(list)
    connections = graph.edges

    for i in range(graph.n_edge[0]):
        ap, sta = graph.senders[i].item(), graph.receivers[i].item()
        if ap in associations and sta in associations[ap] and np.argmax(connections.selected[i]) == IsSelected.TRUE.value:
            selected[ap].append(i)

    tx_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    tx_power = np.zeros(n_nodes, dtype=int)
    mcs = np.zeros(n_nodes, dtype=int)
    selected_for_tx = []

    for ap, sta_indices in selected.items():
        key, subkey = jax.random.split(key)
        selected_sta_idx = jax.random.choice(subkey, len(sta_indices), replace=False).item()
        selected_for_tx.append(sta_indices[selected_sta_idx])

    for idx in selected_for_tx:
        ap, sta = graph.senders[idx].item(), graph.receivers[idx].item()
        tx_matrix[ap, sta] = 1
        tx_power[ap] = np.argmax(connections.tx_power[idx, :-1])
        mcs[ap] = np.argmax(connections.mcs[idx, :-1])

    return tx_matrix, tx_power, mcs


def tx_to_action(associations, internals, tx_matrix, tx_power, mcs):
    return {
        ap: (sta, internals.average_data_rate[ap].item() / 1e6, tx_power[ap].item(), mcs[ap].item()) 
        for ap in associations.keys() for sta in associations[ap] if tx_matrix[ap, sta]
    }


def get_next_context(graph, context, internals, encode_fn):
    connections = graph.edges
    rssi = np.zeros_like(connections.rssi)
    success = np.zeros_like(connections.success)

    for i in range(graph.n_edge.item()):
        ap, sta = graph.senders[i].item(), graph.receivers[i].item()
        is_sta_ap = np.argmax(connections.type[i]) == EdgeType.STA.value
        rssi_val = (internals.signal_power[ap, sta] - RSSI_MEAN) / RSSI_STD

        if is_sta_ap or rssi_val > RSSI_THR:
            rssi[i] = rssi_val

        if is_sta_ap and np.argmax(connections.selected[i]) == IsSelected.TRUE.value:
            success[i] = internals.frames_transmitted[ap] / max(1, internals.ampdu_size[ap])

    edges = Connection(
        active=connections.active,
        mcs=connections.mcs,
        rssi=rssi,
        selected=connections.selected,
        success=success,
        tx_power=connections.tx_power,
        type=connections.type
    )
    edges = jax.tree.map(lambda x: x.astype(jnp.float64), edges)
    graph = graph._replace(edges=edges)
    next_conf = encode_fn(graph)

    context.append(next_conf)
    return context[1:]


def run_scenario(scenario, n_reps, n_steps, n_timesteps, n_samples_eval, seed, gen_fn, encode_fn, decode_fn):
    key = jax.random.PRNGKey(seed)
    runs = []
    actions = []

    n_nodes = len(scenario.associations) + sum(len(stas) for stas in scenario.associations.values())

    for _ in trange(n_reps):
        scenario.reset()
        runs.append([])
        actions.append([])

        key, conf_key = jax.random.split(key)
        context, graph, ap_sta_idxs = get_initial_configurations(scenario, n_timesteps, conf_key, encode_fn)
        graph_template = deepcopy(graph)

        for _ in range(n_steps):
            key, sel_key, fm_key, tx_key, scenario_key = jax.random.split(key, 5)

            selected_idx = jax.random.choice(sel_key, ap_sta_idxs).item()
            network_desc = get_network_desc(graph_template, selected_idx, encode_fn)
            all = jax.tree.map(lambda *x: jnp.stack(x, axis=1), network_desc, *context)
            x = network_desc._replace(
                edges=all.edges,
                nodes=all.nodes,
                globals=all.globals
            )

            # https://arxiv.org/abs/2412.08296
            x_eval = jraph.batch([x for _ in range(n_samples_eval)])
            z = gen_fn(x_eval, rngs=fm_key)
            best, best_val = None, float('-inf')

            for zt in jraph.unbatch(decode_fn(z)):
                tx_key, subkey = jax.random.split(tx_key)
                tx = connections_to_tx(scenario.associations, n_nodes, subkey, graph)

                scenario_key, subkey = jax.random.split(scenario_key)
                data_rate, _ = scenario(subkey, *tx, return_internals=False)

                if data_rate > best_val:
                    best = tx
                    best_val = data_rate
                    graph = graph._replace(edges=zt.edges)

            data_rate, _, internals = scenario(scenario_key, *best, return_internals=True)
            runs[-1].append(data_rate.item())
            actions[-1].append(tx_to_action(scenario.associations, internals, *best))
            context = get_next_context(graph, context, internals, encode_fn)

    return runs, actions


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--checkpoint', type=str, default='runs/flow_matching_base')
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

    scenarios = [
        residential_scenario(seed=12, n_steps=500, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=30.0)
    ]
    all_results = []

    for scenario in tqdm(scenarios, desc='Scenarios'):
        runs, actions = run_scenario(
            scenario, args.n_reps, scenario.n_steps, fm_cfg.train.n_timesteps, fm_cfg.train.n_eval_samples, args.seed, gen_fn, encode_fn, decode_fn
        )
        all_results.append([{
            'runs': runs,
            'actions': actions
        }])

    with open(args.output, 'w') as file:
        json.dump(all_results, file)
