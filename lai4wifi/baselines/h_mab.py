import jax
import numpy as np
from mapc_mab import MapcAgentFactory
from reinforced_lib.agents.mab import UCB
from tqdm import trange

from lai4wifi.pipeline.conversion import tx_to_action


def to_python_dict(d):
    result = {}
    for k, v in d.items():
        if hasattr(k, 'item'):
            k = k.item()
        result[k] = np.asarray(v).tolist()
    return result


def create_agent(associations, seed):
    return MapcAgentFactory(
        associations=to_python_dict(associations),
        agent_type=UCB,
        agent_params_lvl1={'c': 1.5, 'gamma': 0.5},
        agent_params_lvl2={'c': 0.5, 'gamma': 0.5},
        agent_params_lvl3={'c': 0.2, 'gamma': 0.8},
        hierarchical=True,
        seed=seed,
    ).create_mapc_agent()


def run_h_mab(scenario, seed=42, n_steps=2000):
    agent = create_agent(scenario.associations, seed)

    key = jax.random.PRNGKey(seed)
    scenario.reset()
    reward = 0.0
    runs = []
    actions = []

    for _ in trange(n_steps, desc='H-MAB steps'):
        key, scenario_key = jax.random.split(key)
        tx, tx_power = agent.sample(reward)
        data_rate, reward, internals = scenario(scenario_key, tx, tx_power, return_internals=True)
        runs.append(data_rate.item())
        actions.append(tx_to_action(scenario.associations, internals, tx, tx_power, internals.mcs))

    return {
        'runs': [runs],
        'actions': [actions]
    }


def run_h_mab_dynamic(env, n_steps, scenario_seed, seed):
    scenario_key = env.reset(jax.random.PRNGKey(scenario_seed))
    agent = None
    reward = 0.0
    all_results = []

    for step in trange(n_steps, desc='H-MAB dynamic'):
        scenario_key, _, info = env.step(scenario_key, step)

        if step == 0 or info.get('topology_changed', False):
            agent = create_agent(env.scenario.associations, seed)
            reward = 0.0

        scenario_key, eval_key = jax.random.split(scenario_key)

        tx, tx_power = agent.sample(reward)
        data_rate, reward, internals = env.eval_fn(eval_key, tx, tx_power, return_internals=True)
        reward = reward.item()

        all_results.append({
            'step': step,
            **info,
            'data_rate': data_rate.item(),
            'action': tx_to_action(env.scenario.associations, internals, tx, tx_power, internals.mcs),
        })

    return all_results
