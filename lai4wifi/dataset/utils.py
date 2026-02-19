from dataclasses import dataclass

import cloudpickle
import jax
import lz4.frame
import numpy as np
from mapc_sim.constants import DEFAULT_TX_POWER, DATA_RATES
from mapc_research.envs.scenario import Scenario
from mapc_research.envs.scenario_impl import *
from tqdm import tqdm


SCENARIOS = [  # Note! The values are drawn from the interval [a, b) - a is inclusive, b is exclusive!
    (
        toy_scenario_1,
        {'d': (10, 31)}
    ),
    (
        toy_scenario_2,
        {'d_ap': (10, 31), 'd_sta': (1, 11)}
    ),
    (
        small_office_scenario,
        {'d_ap': (10, 21), 'd_sta': (1, 11)}
    ),
    (
        random_scenario,
        {'d_ap': (20, 101), 'n_ap': (2, 11), 'd_sta': (1, 9), 'n_sta_per_ap': (1, 6), 'randomize': (0, 1)}
    ),
    (
        residential_scenario,
        {'x_apartments': (2, 6), 'y_apartments': (2, 3), 'n_sta_per_ap': (1, 5), 'size': (5, 21)}
    ),
    (
        hidden_station_scenario,
        {'d': (21, 51)}
    ),
    (
        flow_in_the_middle_scenario,
        {'d': (1, 31)}
    ),
    (
        dense_point_scenario,
        {'n_ap': (2, 11), 'n_associations': (1, 6)}
    ),
    (
        spatial_reuse_scenario,
        {'d_ap': (10, 21), 'd_sta': (1, 11)}
    ),
    (
        test_scenario,
        {'scale': (10, 31)}
    ),
    (
        indoor_small_bsss_scenario,
        {'grid_layers': (3, 4), 'n_sta_per_ap': (3, 11), 'frequency_reuse': (3, 4), 'bss_radius': (5, 21)}
    ),
]


@dataclass
class TxPair:
    ap: int
    sta: int
    mcs: int
    tx_power: int
    success: float


@dataclass
class Configuration:
    links: list[TxPair]


@dataclass
class DatasetItem:
    scenario: Scenario
    configurations: list
    

N_TX_POWER_LEVELS = 4
TX_POWER_DELTA = 3.0
TX_POWER_LEVELS = np.array([DEFAULT_TX_POWER - i * TX_POWER_DELTA for i in range(N_TX_POWER_LEVELS)])
MCS_VALUES = DATA_RATES[80]  # Using 80 MHz channel width


def rate_to_mcs(rate):
    return (np.abs(MCS_VALUES - rate)).argmin().item()


def tx_power_to_lvl(tx_power):
    return (np.abs(TX_POWER_LEVELS - tx_power)).argmin().item()


def save_dataset(dataset, path):
    with lz4.frame.open(path, 'wb') as f:
        f.write(cloudpickle.dumps(dataset))


def load_dataset(path):
    with lz4.frame.open(path, 'rb') as f:
        return cloudpickle.loads(f.read())


def draw_realizations(key, scenario_fn, param_ranges):
    *param_keys, seed_key = jax.random.split(key, len(param_ranges) + 1)
    params = {p: jax.random.randint(k, (), *v) for (p, v), k in zip(param_ranges.items(), param_keys)}
    seed = jax.random.randint(seed_key, (), 0, 2**30)
    (scenario, _), *_ = scenario_fn(seed=seed, channel_width=80, **params).split_scenario()
    yield DatasetItem(scenario, configurations=[])


def draw_scenarios(n_realizations, key, scenarios):
    n_params = list(map(len, [p for _, p in scenarios]))
    n_params = np.asarray(n_params)
    probs = n_params / n_params.sum()

    key, subkey = jax.random.split(key)
    scenario_idx = jax.random.choice(subkey, len(scenarios), p=probs, shape=(n_realizations,)).tolist()
    selected_scenarios = [scenarios[i] for i in scenario_idx]

    for scenario, param_ranges in tqdm(selected_scenarios, desc='Scenarios'):
        key, subkey = jax.random.split(key)
        yield from draw_realizations(subkey, scenario, param_ranges)


def draw_history(n_configurations, key, dataset, draw_configuration_fn):
    for dataset_item in tqdm(dataset, desc='Configurations'):
        key, subkey = jax.random.split(key)
        dataset_item.configurations = list(draw_configuration_fn(n_configurations, subkey, dataset_item))

    return dataset


def generate_dataset(seed, n_realizations, n_configurations, draw_configuration_fn, save_path):
    key = jax.random.PRNGKey(seed)
    scenarios_key, configurations_key = jax.random.split(key)

    dataset = list(draw_scenarios(n_realizations, scenarios_key, SCENARIOS))
    dataset = draw_history(n_configurations, configurations_key, dataset, draw_configuration_fn)

    save_dataset(dataset, save_path)
