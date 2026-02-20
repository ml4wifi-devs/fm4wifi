import jax
import jax.numpy as jnp

from lai4wifi.graphs.attributes import IsSelected
from lai4wifi.dataset.utils import MCS_VALUES


RATE_MEAN = 657.2
RATE_STD = 462.2


def select_best_configuration(logits, means, scales, risk_averse=True, risk_factor=2.5):
    probs = jax.nn.softmax(logits, axis=-1)
    expected_value = jnp.sum(probs * means, axis=-1)
    
    if not risk_averse:
        scores = expected_value
    else:
        second_moment = (probs * (scales ** 2 + means ** 2)).sum(axis=-1)
        variance = second_moment - expected_value ** 2
        std_dev = jnp.sqrt(jnp.maximum(variance, 0.0))
        scores = expected_value - risk_factor * std_dev
        
    best_idx = jnp.argmax(scores)
    return best_idx, scores


def calculate_data_rate(graph):
    mcs = jnp.argmax(graph.edges.mcs, axis=-1)
    selected = jnp.argmax(graph.edges.selected, axis=-1) == IsSelected.TRUE.value
    rates = MCS_VALUES[mcs] * graph.edges.success.squeeze() * selected
    return rates.sum() / 2.0
