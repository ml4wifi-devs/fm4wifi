import jax
import jax.numpy as jnp
import jraph
import numpy as np
from tqdm import tqdm

from lai4wifi.graphs.attributes import IsSelected
from lai4wifi.dataset.utils import MCS_VALUES


RATE_MEAN = 657.2
RATE_STD = 462.2


def mdn_loss(logits, means, scales, rates):
    """Mixture Density Network https://research.aston.ac.uk/en/publications/mixture-density-networks"""
    log_probs_pi = jax.nn.log_softmax(logits, axis=-1)
    log_probs_gauss = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(scales) - 0.5 * ((rates[..., None] - means) / scales) ** 2
    log_probs_total = log_probs_pi + log_probs_gauss
    return -jax.nn.logsumexp(log_probs_total, axis=-1)


def select_best_configuration(logits, means, scales, risk_averse=False, risk_factor=0.0):
    probs = jax.nn.softmax(logits, axis=-1)
    expected_value = jnp.sum(probs * means, axis=-1)

    if not risk_averse:
        scores = expected_value
    else:
        second_moment = (probs * (scales ** 2 + means ** 2)).sum(axis=-1)
        variance = second_moment - expected_value ** 2
        std_dev = jnp.sqrt(jnp.maximum(variance, 0.0))
        scores = expected_value - risk_factor * std_dev

    scores = scores[:-1]
    best_idx = jnp.argmax(scores)
    return best_idx, scores


def surrogate_scores(pred):
    if isinstance(pred, tuple):
        _, scores = select_best_configuration(*pred)
        return scores
    return pred[:-1]


def calculate_data_rate(graph):
    mcs = jnp.argmax(graph.edges.mcs, axis=-1)
    selected = jnp.argmax(graph.edges.selected, axis=-1) == IsSelected.TRUE.value
    rates = MCS_VALUES[mcs] * graph.edges.success.squeeze() * selected
    return rates.sum() / 2.0


def evaluate_model(model, params, val_data):
    apply_fn = jax.jit(model.apply, static_argnames=['training'])
    all_preds, all_targets = [], []

    for batch, rates in tqdm(val_data):
        output = apply_fn(params, batch, training=False)
        mask = jraph.get_graph_padding_mask(batch)

        if isinstance(output, tuple):
            logits, means, _ = output
            output = jnp.sum(jax.nn.softmax(logits, axis=-1) * means, axis=-1)

        all_preds.append(np.asarray(output[mask]))
        all_targets.append(np.asarray(rates[mask]))

    preds = np.concatenate(all_preds) * RATE_STD + RATE_MEAN
    targets = np.concatenate(all_targets) * RATE_STD + RATE_MEAN
    errors = preds - targets
    abs_errors = np.abs(errors)

    metrics = {
        'R2': float(1 - np.sum(errors ** 2) / np.sum((targets - np.mean(targets)) ** 2)),
        'RMSE': float(np.sqrt(np.mean(errors ** 2))),
        'MAE': float(np.mean(abs_errors)),
        'Correlation': float(np.corrcoef(preds, targets)[0, 1]),
        'Bias': float(np.mean(errors))
    }

    for p in [90, 95, 99]:
        metrics[f'P{p}'] = float(np.percentile(abs_errors, p))

    metrics['max_error'] = float(np.max(abs_errors))
    return metrics
