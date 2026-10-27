import os
os.environ['JAX_ENABLE_X64'] = 'True'

from itertools import product

import jax
import jax.numpy as jnp
from mapc_research.envs.scenario_impl import StaticScenario, random_scenario, residential_scenario
from mapc_sim.sim import network_data_rate


data_rate_fn = jax.jit(network_data_rate, static_argnames=('channel_width', 'return_internals', 'path_loss_fn'))


def open_enterprise_scenario(
        seed: int,
        n_steps: int,
        x_apartments: int = 2,
        y_apartments: int = 2,
        n_sta_per_ap: int = 4,
        size: float = 10.,
        channel_width: int = None
) -> StaticScenario:
    """
    Grid of x_apartments * y_apartments rooms with APs at the center of each room
    and STAs randomly placed within the room boundaries. No walls between rooms.
    """

    key = jax.random.PRNGKey(seed)
    str_repr = f"open_enterprise_{seed}_{x_apartments}_{y_apartments}_{n_sta_per_ap}_{size}"
    associations, pos = {}, []

    for x, y in product(range(x_apartments), range(y_apartments)):
        ap = len(pos)
        stas = list(range(ap + 1, ap + n_sta_per_ap + 1))
        associations[ap] = stas

        # AP at center of room
        pos.append([(x + 0.5) * size, (y + 0.5) * size])

        # STAs randomly placed within room boundaries
        key, sta_key = jax.random.split(key)
        sta_positions = jax.random.uniform(sta_key, (n_sta_per_ap, 2))
        sta_positions = sta_positions * jnp.array([size, size]) + jnp.array([x * size, y * size])
        pos.extend(sta_positions.tolist())

    return StaticScenario(
        jnp.array(pos), 
        associations, 
        n_steps,
        str_repr=str_repr,
        channel_width=channel_width
    )


def create_scenario_eval_fn(scenario):
    if hasattr(scenario, 'switch_steps'):
        return scenario

    def eval_fn(key, tx, tx_power=None, mcs=None, return_internals=False):
        if tx_power is None:
            tx_power = jnp.zeros_like(scenario.tx_power)

        out = data_rate_fn(
            key,
            tx,
            pos=scenario.pos,
            mcs=mcs,
            tx_power=scenario.tx_power - scenario.tx_power_delta * tx_power,
            sigma=scenario.sigma,
            walls=scenario.walls,
            channel_width=scenario.channel_width,
            return_internals=return_internals,
            path_loss_fn=scenario.path_loss_fn,
        )

        if return_internals:
            thr, *internals = out
        else:
            thr, internals = out, tuple()

        reward = thr / scenario.normalize_reward
        return thr, reward, *internals

    return eval_fn


RESIDENTIAL_SCENARIOS = [
    residential_scenario(seed=100, n_steps=1500, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=10.0, channel_width=80),
    residential_scenario(seed=103, n_steps=3000, x_apartments=2, y_apartments=3, n_sta_per_ap=4, size=10.0, channel_width=80),
    residential_scenario(seed=106, n_steps=20000, x_apartments=3, y_apartments=3, n_sta_per_ap=4, size=10.0, channel_width=80),
    residential_scenario(seed=109, n_steps=200000, x_apartments=3, y_apartments=4, n_sta_per_ap=4, size=10.0, channel_width=80),
    residential_scenario(seed=112, n_steps=2000, x_apartments=4, y_apartments=4, n_sta_per_ap=4, size=10.0, channel_width=80),
    residential_scenario(seed=115, n_steps=2000, x_apartments=4, y_apartments=5, n_sta_per_ap=4, size=10.0, channel_width=80),
    residential_scenario(seed=118, n_steps=2000, x_apartments=5, y_apartments=5, n_sta_per_ap=4, size=10.0, channel_width=80),
    residential_scenario(seed=121, n_steps=2000, x_apartments=5, y_apartments=6, n_sta_per_ap=4, size=10.0, channel_width=80)
]

FREE_SPACE_SCENARIOS = [
    random_scenario(seed=100, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, n_steps=1000, channel_width=80),
    random_scenario(seed=101, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, n_steps=1000, channel_width=80),
    random_scenario(seed=102, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, n_steps=1000, channel_width=80),
    random_scenario(seed=103, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, n_steps=2000, channel_width=80),
    random_scenario(seed=104, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, n_steps=1000, channel_width=80),
    random_scenario(seed=105, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, n_steps=3000, channel_width=80),
    random_scenario(seed=106, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, n_steps=1000, channel_width=80),
    random_scenario(seed=107, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, n_steps=1000, channel_width=80),
    random_scenario(seed=108, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, n_steps=1000, channel_width=80),
    random_scenario(seed=109, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, n_steps=2000, channel_width=80),
    random_scenario(seed=110, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, n_steps=1000, channel_width=80),
    random_scenario(seed=111, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, n_steps=3000, channel_width=80),
    random_scenario(seed=112, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, n_steps=1000, channel_width=80),
    random_scenario(seed=113, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, n_steps=1000, channel_width=80),
    random_scenario(seed=114, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, n_steps=1000, channel_width=80),
    random_scenario(seed=115, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, n_steps=2000, channel_width=80),
    random_scenario(seed=116, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, n_steps=1000, channel_width=80),
    random_scenario(seed=117, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, n_steps=3000, channel_width=80),
    random_scenario(seed=118, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, n_steps=1000, channel_width=80),
    random_scenario(seed=119, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, n_steps=1000, channel_width=80),
    random_scenario(seed=120, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, n_steps=1000, channel_width=80),
    random_scenario(seed=121, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, n_steps=2000, channel_width=80),
    random_scenario(seed=122, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, n_steps=2000, channel_width=80),
    random_scenario(seed=123, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, n_steps=3000, channel_width=80)
]

ENTERPRISE_SCENARIOS = [
    open_enterprise_scenario(seed=101, n_steps=1500, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=20.0, channel_width=80),
    open_enterprise_scenario(seed=104, n_steps=3000, x_apartments=2, y_apartments=3, n_sta_per_ap=4, size=20.0, channel_width=80),
    open_enterprise_scenario(seed=107, n_steps=20000, x_apartments=3, y_apartments=3, n_sta_per_ap=4, size=20.0, channel_width=80),
    open_enterprise_scenario(seed=110, n_steps=200000, x_apartments=3, y_apartments=4, n_sta_per_ap=4, size=20.0, channel_width=80),
    open_enterprise_scenario(seed=113, n_steps=2000, x_apartments=4, y_apartments=4, n_sta_per_ap=4, size=20.0, channel_width=80),
    open_enterprise_scenario(seed=116, n_steps=20000, x_apartments=4, y_apartments=5, n_sta_per_ap=4, size=20.0, channel_width=80),
    open_enterprise_scenario(seed=119, n_steps=200000, x_apartments=5, y_apartments=5, n_sta_per_ap=4, size=20.0, channel_width=80),
    open_enterprise_scenario(seed=122, n_steps=200000, x_apartments=5, y_apartments=6, n_sta_per_ap=4, size=20.0, channel_width=80)
]

TEST_SCENARIOS = {
    'scaling': [RESIDENTIAL_SCENARIOS[0], RESIDENTIAL_SCENARIOS[2], RESIDENTIAL_SCENARIOS[4]],
    'residential': RESIDENTIAL_SCENARIOS,
    'enterprise': ENTERPRISE_SCENARIOS,
    'random': FREE_SPACE_SCENARIOS
}
