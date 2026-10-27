import jax
import jax.numpy as jnp
import numpy as np

from lai4wifi.graphs.utils import make_batch
from lai4wifi.pipeline.conversion import connections_to_tx_batch_fast
from lai4wifi.surrogate.utils import surrogate_scores


def iter_edge_batched_ranges(graphs, max_edges_per_batch):
    n_graphs = len(graphs)
    start = 0

    while start < n_graphs:
        end = start
        total_edges = 0

        while end < n_graphs:
            graph_edges = int(np.asarray(graphs[end].n_edge).sum())
            if end > start and total_edges + graph_edges > max_edges_per_batch:
                break
            total_edges += graph_edges
            end += 1

        yield start, end
        start = end


def select_configs(scores, top_k):
    sorted_indices = np.argsort(-scores)[:top_k]
    return [(int(idx), float(scores[idx])) for idx in sorted_indices]


def generate_candidates(key, fm_inputs, cache, gen_fn, decode_fn, batch_size, deterministic):
    z_batches = []
    all_txs = []

    for batch_start, batch_end in iter_edge_batched_ranges(fm_inputs, batch_size):
        key, fm_key = jax.random.split(key)
        batch_graphs = fm_inputs[batch_start:batch_end]
        curr_batch_size = len(batch_graphs)
        tx_keys = []

        for _ in range(curr_batch_size):
            key, tx_key = jax.random.split(key)
            tx_keys.append(tx_key)

        z = gen_fn(make_batch(batch_graphs), rngs=fm_key)
        z_batches.append(z)
        tx_batch, tx_power_batch, mcs_batch = connections_to_tx_batch_fast(
            jnp.asarray(tx_keys), decode_fn(z), cache, curr_batch_size, deterministic
        )
        all_txs.extend(
            (tx_batch[j], tx_power_batch[j], mcs_batch[j])
            for j in range(curr_batch_size)
        )

    return key, z_batches, all_txs


def score_candidates(key, z_batches, all_txs, surrogate_fn, scenario_eval_fn, use_simulator):
    all_scores = []

    if use_simulator:
        with jax.default_device(jax.devices('cpu')[0]):
            for tx in all_txs:
                key, sim_key = jax.random.split(key)
                data_rate, _, _ = scenario_eval_fn(sim_key, *tx, return_internals=True)
                all_scores.append(data_rate.item())
    else:
        for z in z_batches:
            scores = surrogate_scores(surrogate_fn(z))
            all_scores.extend(np.asarray(scores).tolist())

    return key, np.asarray(all_scores, dtype=np.float64)
