import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import json
from argparse import ArgumentParser
from functools import partial

import jax
import jax.numpy as jnp
import jraph
import orbax.checkpoint as ocp
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from lai4wifi.flow_matching.model import FlowMatching
from lai4wifi.flow_matching.utils import TEST_SCENARIOS, connections_to_tx, tx_to_action, tx_to_jraph, random_tx
from lai4wifi.graphs.model import GnnAutoencoder
from lai4wifi.graphs.utils import make_batch, unbatch
from lai4wifi.surrogate.model import SurrogateModel
from lai4wifi.surrogate.utils import select_best_configuration


def select_configs(associations, all_txs, scores, n_samples_eval, step, min_top_configs):
    step_scores = scores[step * n_samples_eval:(step + 1) * n_samples_eval]
    sorted_indices = jnp.argsort(-step_scores)

    all_stas = {sta for stas in associations.values() for sta in stas}
    covered_stas = set()
    selected = []

    for rank, idx in enumerate(sorted_indices):
        flat_idx = step * n_samples_eval + idx.item()
        tx_matrix, *_ = all_txs[flat_idx]
        _, cfg_stas = jnp.where(tx_matrix)
        cfg_stas = set(cfg_stas.tolist())
        new_stas = cfg_stas - covered_stas

        if rank < min_top_configs or (new_stas and covered_stas != all_stas):
            selected.append((flat_idx, step_scores[idx].item()))
            covered_stas |= cfg_stas

        if len(selected) >= min_top_configs and covered_stas >= all_stas:
            break

    return selected


def run_scenario(
        scenario, n_steps, n_timesteps, n_samples_eval, seed, ideal_mcs, deterministic,
        gen_fn, encode_fn, decode_fn, surrogate_fn, batch_size, use_simulator, min_top_configs, n_eval_repeats
):
    key = jax.random.PRNGKey(seed)
    total_contexts = n_steps * n_timesteps
    context_graphs_raw = []

    with jax.default_device(jax.devices('cpu')[0]):
        for _ in trange(total_contexts, desc='Context generation'):
            key, tx_key, sim_key = jax.random.split(key, 3)
            tx = random_tx(tx_key, scenario)
            _, _, internals = scenario(sim_key, *tx, return_internals=True)
            context_graphs_raw.append(tx_to_jraph(scenario, *tx, internals))

    encoded_contexts = []
    
    for i in trange(0, len(context_graphs_raw), batch_size, desc='Encoding contexts'):
        batch = make_batch(context_graphs_raw[i:i + batch_size])
        encoded = encode_fn(batch)
        encoded_contexts += unbatch(encoded)
    
    fm_inputs = []

    for step in range(n_steps):
        ctx_graphs = encoded_contexts[step * n_timesteps:(step + 1) * n_timesteps]
        stacked = jax.tree.map(lambda *x: jnp.stack(x, axis=1), *ctx_graphs)
        context = ctx_graphs[0]._replace(edges=stacked.edges, nodes=stacked.nodes, globals=stacked.globals)
        fm_inputs += [context] * n_samples_eval

    decoded_candidates = []

    for i in trange(0, len(fm_inputs), batch_size, desc='FM generation'):
        key, fm_key = jax.random.split(key)
        batch = make_batch(fm_inputs[i:i + batch_size])
        z = gen_fn(batch, rngs=fm_key)
        decoded = decode_fn(z)
        decoded_candidates += unbatch(decoded)

    all_txs = []

    with jax.default_device(jax.devices('cpu')[0]):
        for zt in tqdm(decoded_candidates, desc='Converting to tx'):
            key, tx_key = jax.random.split(key)
            tx = connections_to_tx(scenario.associations, tx_key, zt, deterministic)
            all_txs.append(tx)

    all_scores = []

    if use_simulator:
        with jax.default_device(jax.devices('cpu')[0]):
            for i, tx in enumerate(tqdm(all_txs, desc='Simulator eval')):
                key, sim_key = jax.random.split(key)
                data_rate, _, _ = scenario(sim_key, *tx, return_internals=True)
                all_scores.append(data_rate.item())
    else:
        surrogate_input_graphs = [
            tx_to_jraph(scenario, *tx) for tx in tqdm(all_txs, desc='Building surrogate inputs')
        ]

        for i in trange(0, len(surrogate_input_graphs), batch_size, desc='Surrogate eval'):
            batch = make_batch(surrogate_input_graphs[i:i + batch_size])
            encoded = encode_fn(batch)
            pred = surrogate_fn(encoded)

            mask = jraph.get_graph_padding_mask(encoded)
            logits, means, scales = jax.tree.map(lambda x: x[mask], pred)
            _, scores = select_best_configuration(logits, means, scales)
            all_scores.extend(scores.tolist())

    scores_array = jnp.array(all_scores)
    all_results = []

    with jax.default_device(jax.devices('cpu')[0]):
        for step in trange(n_steps, desc='Final sim'):
            selected = select_configs(
                scenario.associations, all_txs, scores_array,
                n_samples_eval, step, min_top_configs
            )
            results = []

            for flat_idx, score in selected:
                eval_tx = all_txs[flat_idx]
                repeats = []

                for _ in range(n_eval_repeats):
                    key, scenario_key = jax.random.split(key)

                    if ideal_mcs:
                        tx_mat, tx_power, _ = eval_tx
                        data_rate, _, internals = scenario(scenario_key, tx_mat, tx_power, return_internals=True)
                        eval_tx = (tx_mat, tx_power, internals.mcs)
                    else:
                        data_rate, _, internals = scenario(scenario_key, *eval_tx, return_internals=True)

                    repeats.append({
                        'data_rate': data_rate.item(),
                        'action': tx_to_action(scenario.associations, internals, *eval_tx)
                    })

                results.append({
                    'score': score,
                    'runs': repeats
                })

            all_results.append({'configs': results})

    return all_results


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--checkpoint', type=str, default='runs/fm_medium')
    args.add_argument('--surrogate', type=str, default='../surrogate/runs/sur_medium')
    args.add_argument('--n_samples_eval', type=int, default=1)
    args.add_argument('--batch_size', type=int, default=8)
    args.add_argument('--ideal_mcs', default=False, action='store_true')
    args.add_argument('--deterministic', default=False, action='store_true')
    args.add_argument('--use_simulator', default=False, action='store_true')
    args.add_argument('--output', type=str, default='all_results.json')
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--min_top_configs', type=int, default=1)
    args.add_argument('--n_eval_repeats', type=int, default=10)
    args.add_argument('--n_steps', type=int, default=512)
    args = args.parse_args()

    surrogate_fn = None

    if not args.use_simulator:
        if args.surrogate is None:
            raise ValueError('--surrogate is required when not using --use_simulator')

        s_cfg = OmegaConf.load(os.path.join(args.surrogate, 'config.yaml'))
        with ocp.CheckpointManager(os.path.abspath(args.surrogate), item_names=['params']) as ckpt_manager:
            s_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

        surrogate = SurrogateModel(**s_cfg.model)
        surrogate_fn = jax.jit(partial(surrogate.apply, s_params, training=False))

    fm_cfg = OmegaConf.load(os.path.join(args.checkpoint, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(args.checkpoint), item_names=['ema_params']) as ckpt_manager:
        fm_params = ckpt_manager.restore(None, args=ocp.args.Composite(ema_params=ocp.args.PyTreeRestore())).ema_params

    fm = FlowMatching(**fm_cfg.model)
    gen_fn = jax.jit(partial(fm.apply, fm_params, method='gen'))

    gnn_cfg = OmegaConf.load(os.path.join(fm_cfg.gnn_path, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(fm_cfg.gnn_path), item_names=['params']) as ckpt_manager:
        gnn_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

    gnn = GnnAutoencoder(**gnn_cfg.model)
    encode_fn = jax.jit(partial(gnn.apply, gnn_params, method='encode'))
    decode_fn = jax.jit(partial(gnn.apply, gnn_params, method='decode'))

    all_results = []

    for scenario in tqdm(TEST_SCENARIOS, desc='Scenarios'):
        split_results = []

        for split_scenario, _ in scenario.split_scenario():
            split_results.append([run_scenario(
                split_scenario, args.n_steps, fm_cfg.train.n_timesteps, args.n_samples_eval,
                args.seed, args.ideal_mcs, args.deterministic, gen_fn, encode_fn, decode_fn,
                surrogate_fn, args.batch_size, args.use_simulator, args.min_top_configs, args.n_eval_repeats
            )])

        all_results.append({'splits': split_results})

    with open(args.output, 'w') as file:
        json.dump(all_results, file)
