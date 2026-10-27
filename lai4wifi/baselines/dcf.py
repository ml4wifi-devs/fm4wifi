import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

from argparse import ArgumentParser

import jax
import numpy as np
import simpy
from mapc_dcf.channel import TAU, Channel
from mapc_dcf.nodes import AccessPoint
from mapc_dcf.logger import Logger
from tqdm import tqdm

from lai4wifi.envs.scenarios import TEST_SCENARIOS


def flatten_scenarios(scenarios):
    scenarios_flattened = []

    for scenario in scenarios:
        str_repr = scenario.__str__()
        list_of_scenarios = scenario.split_scenario()
        for i, s in enumerate(list_of_scenarios):
            suffix = f"_{chr(ord('a') + i)}" if len(list_of_scenarios) > 1 else ""
            scenarios_flattened.append((s[0], s[1], f"{str_repr}{suffix}"))

    return scenarios_flattened


def run_dcf_single(key, run, scenario, sim_time, logger, spatial_reuse):
    key, key_channel = jax.random.split(key)
    des_env = simpy.Environment()
    channel = Channel(key_channel, spatial_reuse, scenario.channel_width, scenario.pos, scenario.walls)
    aps: dict[int, AccessPoint] = {}

    for ap in scenario.associations:
        key, key_ap = jax.random.split(key)
        clients = np.array(scenario.associations[ap])
        tx_power = scenario.tx_power[ap]
        aps[ap] = AccessPoint(key_ap, ap, scenario.pos, tx_power, clients, channel, des_env, logger)
        aps[ap].start_operation(run)
    
    des_env.run(until=(logger.warmup_length + sim_time))
    logger.dump_acumulators(run)
    del des_env


def run_dcf(scenarios, seed, n_runs, warmup, spatial_reuse):
    scenarios = flatten_scenarios(scenarios)
    key = jax.random.PRNGKey(seed)
    
    job_dir = 'sr_results' if spatial_reuse else 'dcf_results'
    sim_time = 10000 * TAU

    for scenario, _, scenario_name in tqdm(scenarios):
        logger = Logger(sim_time, warmup, os.path.join(job_dir, scenario_name), scenario_name)
        run_dcf_single(key, 1, scenario, sim_time, logger, spatial_reuse)
        logger.shutdown({
            "name": scenario_name,
            "simulation_length": sim_time,
            "warmup_length": warmup,
            "n_runs": n_runs
        })


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--scenarios', type=str, choices=list(TEST_SCENARIOS), default='residential')
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--dcf_n_runs', type=int, default=1)
    args.add_argument('--dcf_warmup', type=float, default=0.5)
    args.add_argument('--spatial_reuse', action='store_true', default=False)
    args.add_argument('--idx', type=int, default=None)
    args = args.parse_args()

    scenarios = TEST_SCENARIOS[args.scenarios]
    if args.idx is not None:
        scenarios = [scenarios[args.idx]]

    run_dcf(scenarios, args.seed, args.dcf_n_runs, args.dcf_warmup, args.spatial_reuse)
