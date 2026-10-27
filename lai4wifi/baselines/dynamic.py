import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import json
from argparse import ArgumentParser

from lai4wifi.baselines.h_mab import run_h_mab_dynamic
from lai4wifi.envs.mobility import add_env_args, create_env


if __name__ == '__main__':
    args = ArgumentParser()
    add_env_args(args)
    args.add_argument('--output', type=str, default='h_mab_dynamic_results.json')
    args.add_argument('--seed', type=int, default=42)
    args = args.parse_args()

    env, n_steps, metadata = create_env(args)
    results = run_h_mab_dynamic(env, n_steps, args.scenario_seed, args.seed)

    if args.env == 'enterprise':
        metadata['topology_change_steps'] = env.topology_change_steps

    with open(args.output, 'w') as file:
        json.dump({
            'scenario': str(env.scenario),
            'n_steps': n_steps,
            'sim_time_s': n_steps * args.sim_dt,
            'scenario_seed': args.scenario_seed,
            'seed': args.seed,
            **metadata,
            'results': results,
        }, file)
