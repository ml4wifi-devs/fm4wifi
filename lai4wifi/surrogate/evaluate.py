import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import argparse
import json
from functools import partial

import jax
import jax.numpy as jnp
import jraph
import numpy as np
from tqdm import tqdm

from lai4wifi.dataset.utils import load_dataset
from lai4wifi.pipeline.checkpoints import load_surrogate
from lai4wifi.surrogate.utils import RATE_MEAN, RATE_STD, evaluate_model


def effective_components(logits):
    """Entropy-based effective number of mixture components per sample."""
    probs = jax.nn.softmax(logits, axis=-1)
    log_probs = jnp.log(probs + 1e-10)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    return jnp.exp(entropy)


def evaluate_mdn(model, params, val_data):
    all_nll, all_eff_components = [], []
    apply_fn = jax.jit(model.apply, static_argnames=['training'])

    for batch, rates in tqdm(val_data):
        logits, means, scales = apply_fn(params, batch, training=False)
        mask = jraph.get_graph_padding_mask(batch)

        log_probs_pi = jax.nn.log_softmax(logits, axis=-1)
        log_probs_gauss = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(scales) \
            - 0.5 * ((rates[..., None] - means) / scales) ** 2
        nll = -jax.nn.logsumexp(log_probs_pi + log_probs_gauss, axis=-1)
        eff = effective_components(logits)

        all_nll.append(np.asarray(nll[mask]))
        all_eff_components.append(np.asarray(eff[mask]))

    nll_vals = np.concatenate(all_nll)
    eff_vals = np.concatenate(all_eff_components)

    q25 = float(np.percentile(eff_vals, 25))
    q75 = float(np.percentile(eff_vals, 75))
    iqr = q75 - q25
    lower_whisker = float(np.min(eff_vals[eff_vals >= q25 - 1.5 * iqr]))
    upper_whisker = float(np.max(eff_vals[eff_vals <= q75 + 1.5 * iqr]))

    return {
        'n_components': int(model.n_components),
        'NLL': float(np.mean(nll_vals)),
        'eff_components_mean': float(np.mean(eff_vals)),
        'eff_components_median': float(np.median(eff_vals)),
        'eff_components_q25': q25,
        'eff_components_q75': q75,
        'eff_components_lower_whisker': lower_whisker,
        'eff_components_upper_whisker': upper_whisker,
    }


def evaluate_risk_factors(surrogate_fn, val_data, lambdas):
    """Evaluate risk factor lambda on prebatched validation data.

    For each batch of candidates, score them with the surrogate MDN at each lambda,
    select the highest-scoring candidate, and record its true (normalized) data rate.
    """
    results_per_lambda = {l: [] for l in lambdas}
    all_oracle_rates = []
    all_random_rates = []

    for batch, rates in val_data:
        logits, means, scales = surrogate_fn(batch)
        mask = jraph.get_graph_padding_mask(batch)

        probs = jax.nn.softmax(logits, axis=-1)
        expected = jnp.sum(probs * means, axis=-1)
        second_moment = jnp.sum(probs * (scales ** 2 + means ** 2), axis=-1)
        variance = second_moment - expected ** 2
        std_dev = jnp.sqrt(jnp.maximum(variance, 0.0))

        # Mask out padding graphs
        true_rates = rates[mask]
        expected_m = expected[mask]
        std_dev_m = std_dev[mask]

        # Denormalize for reporting
        true_rates_real = true_rates * RATE_STD + RATE_MEAN

        all_oracle_rates.append(float(jnp.max(true_rates_real)))
        all_random_rates.append(float(jnp.mean(true_rates_real)))

        for lam in lambdas:
            scores = expected_m - lam * std_dev_m
            best_idx = int(jnp.argmax(scores))
            results_per_lambda[lam].append(float(true_rates_real[best_idx]))

    return {
        'lambdas': {str(l): float(np.mean(rates)) for l, rates in results_per_lambda.items()},
        'oracle': float(np.mean(all_oracle_rates)),
        'random': float(np.mean(all_random_rates)),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a surrogate checkpoint on prebatched validation data')
    parser.add_argument('mode', choices=['quality', 'mdn', 'risk'],
                        help='quality: prediction metrics, mdn: NLL and effective components, risk: risk factor sweep')
    parser.add_argument('--ckpt-path', type=str, required=True, help='Path to checkpoint directory')
    parser.add_argument('--val-dataset-path', type=str, required=True, help='Path to prebatched val dataset')
    parser.add_argument('--output', type=str, default=None,
                        help='Default: <ckpt-path>/eval_metrics.json (quality), <ckpt-path>/mdn_metrics.json (mdn), '
                             'risk_factor_results.json (risk)')
    args = parser.parse_args()

    model, params = load_surrogate(args.ckpt_path)
    with jax.default_device(jax.devices('cpu')[0]):
        val_data = load_dataset(args.val_dataset_path)

    if args.mode == 'quality':
        metrics = evaluate_model(model, params, val_data)
        output = os.path.join(args.ckpt_path, 'eval_metrics.json')
    elif args.mode == 'mdn':
        metrics = evaluate_mdn(model, params, val_data)
        output = os.path.join(args.ckpt_path, 'mdn_metrics.json')
    else:
        surrogate_fn = jax.jit(partial(model.apply, params, training=False))
        metrics = evaluate_risk_factors(surrogate_fn, val_data, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        output = 'risk_factor_results.json'

    print(json.dumps(metrics, indent=2))
    with open(args.output or output, 'w') as f:
        json.dump(metrics, f, indent=2)
