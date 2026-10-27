from dataclasses import dataclass
from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from mapc_research.envs.scenario_impl import random_scenario, residential_scenario
from mapc_sim.constants import TAU
from mapc_sim.sim import network_data_rate

from lai4wifi.envs.scenarios import data_rate_fn, open_enterprise_scenario


@dataclass
class MobilityState:
    pos: np.ndarray
    velocity: np.ndarray
    remaining_time: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    mobile_mask: np.ndarray
    speed_min: float
    speed_max: float
    segment_min_s: float
    segment_max_s: float


def room_bounds(x_apartments, y_apartments, n_sta_per_ap, size):
    """Compute per-node bounds based on room assignments in the residential scenario.

    Returns (bounds_min, bounds_max, mobile_mask) arrays.
    Node ordering matches residential_scenario: product(range(x_apartments), range(y_apartments)),
    each room has 1 AP + n_sta_per_ap STAs.
    """
    nodes_per_room = n_sta_per_ap + 1
    n_nodes = x_apartments * y_apartments * nodes_per_room
    bounds_min = np.zeros((n_nodes, 2), dtype=np.float64)
    bounds_max = np.zeros((n_nodes, 2), dtype=np.float64)
    mobile_mask = np.zeros(n_nodes, dtype=bool)

    node_idx = 0
    for x, y in product(range(x_apartments), range(y_apartments)):
        room_min = np.array([x * size, y * size], dtype=np.float64)
        room_max = np.array([(x + 1) * size, (y + 1) * size], dtype=np.float64)

        # AP: not mobile (bounds pinned to initial position, set later)
        bounds_min[node_idx] = room_min
        bounds_max[node_idx] = room_max
        mobile_mask[node_idx] = False
        node_idx += 1

        # STAs: mobile within room
        for _ in range(n_sta_per_ap):
            bounds_min[node_idx] = room_min
            bounds_max[node_idx] = room_max
            mobile_mask[node_idx] = True
            node_idx += 1

    return bounds_min, bounds_max, mobile_mask


def area_bounds(scenario, bounds):
    """Compute per-node bounds and mobile mask for the random scenario.

    APs are pinned (mobile_mask=False), STAs move freely within [0, bounds]^2.
    """
    ap_set = set(scenario.associations.keys())
    n_nodes = len(scenario.pos)
    bounds_min = np.zeros((n_nodes, 2), dtype=np.float64)
    bounds_max = np.full((n_nodes, 2), bounds, dtype=np.float64)
    mobile_mask = np.array([i not in ap_set for i in range(n_nodes)], dtype=bool)
    return bounds_min, bounds_max, mobile_mask


def _sample_velocities(key, n_nodes, speed_min, speed_max):
    key_angle, key_speed = jax.random.split(key)
    angles = np.asarray(jax.random.uniform(key_angle, (n_nodes,), minval=0.0, maxval=2 * jnp.pi))
    speeds = np.asarray(jax.random.uniform(key_speed, (n_nodes,), minval=speed_min, maxval=speed_max))
    return np.stack((np.cos(angles), np.sin(angles)), axis=-1) * speeds[:, None]


def _sample_segment_times(key, n_nodes, segment_min_s, segment_max_s):
    return np.asarray(
        jax.random.uniform(key, (n_nodes,), minval=segment_min_s, maxval=segment_max_s),
        dtype=np.float64,
    )


def create_mobility_state(key, initial_pos, bounds_min, bounds_max, mobile_mask,
                          speed_min, speed_max, segment_min_s, segment_max_s):
    pos = np.clip(np.asarray(initial_pos, dtype=np.float64), bounds_min, bounds_max)
    n_nodes = pos.shape[0]
    key_vel, key_time = jax.random.split(key)

    velocity = _sample_velocities(key_vel, n_nodes, speed_min, speed_max)
    velocity[~mobile_mask] = 0.0

    return MobilityState(
        pos=pos,
        velocity=velocity,
        remaining_time=_sample_segment_times(key_time, n_nodes, segment_min_s, segment_max_s),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        mobile_mask=mobile_mask,
        speed_min=speed_min,
        speed_max=speed_max,
        segment_min_s=segment_min_s,
        segment_max_s=segment_max_s,
    )


def step_mobility_state(key, state: MobilityState, dt):
    if dt <= 0.0:
        raise ValueError('sim_dt must be > 0')

    proposed_pos = state.pos + state.velocity * dt
    clamped_pos = np.clip(proposed_pos, state.bounds_min, state.bounds_max)
    hit_mask = np.any(np.abs(proposed_pos - clamped_pos) > 1e-12, axis=1)

    remaining_time = state.remaining_time - dt
    timeout_mask = remaining_time <= 0.0
    redraw_mask = (hit_mask | timeout_mask) & state.mobile_mask

    if np.any(redraw_mask):
        key_vel, key_time = jax.random.split(key)
        n_nodes = state.pos.shape[0]
        sampled_velocity = _sample_velocities(key_vel, n_nodes, state.speed_min, state.speed_max)
        new_velocity = state.velocity.copy()
        new_velocity[redraw_mask] = sampled_velocity[redraw_mask]
        state.velocity = new_velocity

        sampled_segment_times = _sample_segment_times(
            key_time, n_nodes, state.segment_min_s, state.segment_max_s,
        )
        remaining_time[redraw_mask] = sampled_segment_times[redraw_mask]

    state.remaining_time = remaining_time
    state.pos = clamped_pos


def add_stations(key, state: MobilityState, n_new):
    """Add n_new new stations at random positions within the global bounds."""
    if n_new <= 0:
        return

    key_pos, key_vel, key_time = jax.random.split(key, 3)

    new_pos = np.asarray(
        jax.random.uniform(key_pos, (n_new, 2),
                           minval=jnp.array(state.bounds_min),
                           maxval=jnp.array(state.bounds_max)),
        dtype=np.float64,
    )
    new_velocity = _sample_velocities(key_vel, n_new, state.speed_min, state.speed_max)
    new_remaining = _sample_segment_times(key_time, n_new, state.segment_min_s, state.segment_max_s)
    new_mobile = np.ones(n_new, dtype=bool)

    state.pos = np.concatenate([state.pos, new_pos], axis=0)
    state.velocity = np.concatenate([state.velocity, new_velocity], axis=0)
    state.remaining_time = np.concatenate([state.remaining_time, new_remaining], axis=0)
    state.mobile_mask = np.concatenate([state.mobile_mask, new_mobile], axis=0)


def compute_associations(pos, ap_indices):
    """Assign each non-AP node to its nearest AP based on Euclidean distance."""
    ap_pos = pos[ap_indices]  # (n_aps, 2)
    associations = {ap: [] for ap in ap_indices}

    for node_idx in range(pos.shape[0]):
        if node_idx in ap_indices:
            continue
        dists = np.sum((ap_pos - pos[node_idx]) ** 2, axis=1)
        nearest_ap = ap_indices[np.argmin(dists)]
        associations[nearest_ap].append(node_idx)

    return associations


def update_scenario(scenario, mobility_state: MobilityState):
    """Update scenario pos, associations, walls, and tx_power to match current mobility state."""
    n_nodes = mobility_state.pos.shape[0]
    scenario.pos = jnp.asarray(mobility_state.pos)
    scenario.associations = compute_associations(
        mobility_state.pos, [int(ap) for ap in np.flatnonzero(~mobility_state.mobile_mask)],
    )
    scenario.walls = jnp.zeros((n_nodes, n_nodes))
    scenario.tx_power = jnp.full(n_nodes, scenario.tx_power[0].item())

    scenario.data_rate_fn = partial(
        network_data_rate,
        pos=scenario.pos,
        sigma=scenario.sigma,
        walls=scenario.walls,
        path_loss_fn=scenario.path_loss_fn,
        channel_width=scenario.channel_width,
    )


def create_mobile_eval_fn(scenario, mobility_state: MobilityState):
    def eval_fn(key, tx, tx_power=None, mcs=None, return_internals=False):
        if tx_power is None:
            tx_power = jnp.zeros_like(scenario.tx_power)

        with jax.default_device(jax.devices('cpu')[0]):
            out = data_rate_fn(
                key=key,
                tx=tx,
                pos=jnp.asarray(mobility_state.pos),
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


class MobilityEnv:
    """Stations move within per-node bounds; APs and associations are fixed."""

    def __init__(self, scenario, bounds_min, bounds_max, mobile_mask, sim_dt,
                 speed_min, speed_max, segment_min_s, segment_max_s):
        self.scenario = scenario
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.mobile_mask = mobile_mask
        self.sim_dt = sim_dt
        self.mobility_params = (speed_min, speed_max, segment_min_s, segment_max_s)

    def reset(self, scenario_key):
        initial_pos = np.asarray(self.scenario.pos, dtype=np.float64)
        self.bounds_min[~self.mobile_mask] = initial_pos[~self.mobile_mask]
        self.bounds_max[~self.mobile_mask] = initial_pos[~self.mobile_mask]

        scenario_key, mobility_key = jax.random.split(scenario_key)
        self.state = create_mobility_state(
            mobility_key, initial_pos, self.bounds_min, self.bounds_max, self.mobile_mask, *self.mobility_params,
        )
        self.eval_fn = create_mobile_eval_fn(self.scenario, self.state)
        return scenario_key

    def step(self, scenario_key, step):
        scenario_key, config_key, mobility_key = jax.random.split(scenario_key, 3)
        step_mobility_state(mobility_key, self.state, self.sim_dt)
        return scenario_key, config_key, {}


class EnterpriseEnv:
    """Stations move freely, new stations join periodically, and stations reassociate to the nearest AP."""

    def __init__(self, scenario, ap_indices, bounds_min, bounds_max, sim_dt, speed_min, speed_max,
                 segment_min_s, segment_max_s, spawn_interval_steps, spawn_count, no_mobility):
        self.scenario = scenario
        self.ap_indices = ap_indices
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.sim_dt = sim_dt
        self.mobility_params = (0.0, 0.0, 1.0, 1.0) if no_mobility else (speed_min, speed_max, segment_min_s, segment_max_s)
        self.spawn_interval_steps = spawn_interval_steps
        self.spawn_count = spawn_count
        self.no_mobility = no_mobility

    def reset(self, scenario_key):
        initial_pos = np.asarray(self.scenario.pos, dtype=np.float64)
        mobile_mask = np.ones(initial_pos.shape[0], dtype=bool)
        mobile_mask[self.ap_indices] = False

        scenario_key, mobility_key = jax.random.split(scenario_key)
        self.state = create_mobility_state(
            mobility_key, initial_pos, self.bounds_min, self.bounds_max, mobile_mask, *self.mobility_params,
        )
        update_scenario(self.scenario, self.state)
        self.eval_fn = create_mobile_eval_fn(self.scenario, self.state)

        self.prev_associations = None
        self.last_spawn_step = None
        self.topology_change_steps = []
        return scenario_key

    def step(self, scenario_key, step):
        scenario_key, config_key, mobility_key, spawn_key = jax.random.split(scenario_key, 4)

        if not self.no_mobility:
            step_mobility_state(mobility_key, self.state, self.sim_dt)

        spawned = False
        if self.spawn_count > 0 and self.spawn_interval_steps > 0:
            if self.last_spawn_step is None or (step - self.last_spawn_step) >= self.spawn_interval_steps:
                add_stations(spawn_key, self.state, self.spawn_count)
                self.last_spawn_step = step
                spawned = True

        update_scenario(self.scenario, self.state)

        associations = self.scenario.associations
        reassociated = self.prev_associations is not None and associations != self.prev_associations and not spawned
        topology_changed = self.prev_associations is None or associations != self.prev_associations
        n_stations = int(np.sum(self.state.mobile_mask))

        if topology_changed:
            self.prev_associations = {ap: list(stas) for ap, stas in associations.items()}
            self.topology_change_steps.append({
                'step': step,
                'time_s': step * self.sim_dt,
                'reason': 'init' if step == 0 else ('spawn' if spawned else 'reassociation'),
                'n_stations': n_stations,
            })

        return scenario_key, config_key, {
            'spawned': spawned,
            'reassociated': reassociated,
            'topology_changed': topology_changed,
            'n_stations': n_stations,
            'n_nodes': int(self.state.pos.shape[0]),
        }


def add_env_args(args):
    args.add_argument('--env', type=str, choices=['residential', 'random', 'enterprise'], default='residential')
    args.add_argument('--scenario_seed', type=int, default=42)
    args.add_argument('--sim_time_s', type=float, default=20.0)
    args.add_argument('--n_steps', type=int, default=None)
    args.add_argument('--sim_dt', type=float, default=TAU)
    args.add_argument('--mobility_speed_min', type=float, default=0.3)
    args.add_argument('--mobility_speed_max', type=float, default=1.5)
    args.add_argument('--mobility_segment_min_s', type=float, default=0.0)
    args.add_argument('--mobility_segment_max_s', type=float, default=5.0)
    args.add_argument('--n_sta_per_ap', type=int, default=4)
    args.add_argument('--room_size', type=float, default=None, help='Default: 10 (residential), 20 (enterprise)')
    args.add_argument('--no_walls', default=False, action='store_true', help='Residential only')
    args.add_argument('--d_ap', type=float, default=75.0, help='Random only')
    args.add_argument('--d_sta', type=float, default=4.0, help='Random only')
    args.add_argument('--n_ap', type=int, default=8, help='Random only')
    args.add_argument('--no_mobility', default=False, action='store_true', help='Enterprise only')
    args.add_argument('--spawn_interval_s', type=float, default=4.0, help='Enterprise only')
    args.add_argument('--spawn_count', type=int, default=10, help='Enterprise only')


def create_env(args):
    if args.sim_dt <= 0.0:
        raise ValueError('--sim_dt must be > 0')
    if not args.no_mobility:
        if args.mobility_speed_min <= 0.0 or args.mobility_speed_max <= 0.0:
            raise ValueError('Mobility speeds must be > 0')
        if args.mobility_speed_min > args.mobility_speed_max:
            raise ValueError('--mobility_speed_min must be <= --mobility_speed_max')
        if args.mobility_segment_min_s < 0.0 or args.mobility_segment_max_s <= 0.0:
            raise ValueError('Mobility segment times must be > 0')
        if args.mobility_segment_min_s > args.mobility_segment_max_s:
            raise ValueError('--mobility_segment_min_s must be <= --mobility_segment_max_s')

    n_steps = args.n_steps if args.n_steps is not None else max(1, int(round(args.sim_time_s / args.sim_dt)))
    mobility = (args.mobility_speed_min, args.mobility_speed_max, args.mobility_segment_min_s, args.mobility_segment_max_s)

    if args.env == 'residential':
        room_size = 10.0 if args.room_size is None else args.room_size
        scenario = residential_scenario(
            seed=args.scenario_seed,
            x_apartments=4,
            y_apartments=4,
            n_sta_per_ap=args.n_sta_per_ap,
            size=room_size,
            n_steps=n_steps,
            channel_width=80,
        )

        if args.no_walls:
            scenario.walls = jnp.zeros_like(scenario.walls)

        env = MobilityEnv(scenario, *room_bounds(4, 4, args.n_sta_per_ap, room_size), args.sim_dt, *mobility)
        metadata = {'no_walls': args.no_walls}
    elif args.env == 'random':
        scenario = random_scenario(
            seed=args.scenario_seed,
            d_ap=args.d_ap,
            n_ap=args.n_ap,
            d_sta=args.d_sta,
            n_sta_per_ap=args.n_sta_per_ap,
            n_steps=n_steps,
            randomize=False,
            channel_width=80,
        )
        env = MobilityEnv(scenario, *area_bounds(scenario, args.d_ap), args.sim_dt, *mobility)
        metadata = {}
    else:
        room_size = 20.0 if args.room_size is None else args.room_size
        scenario = open_enterprise_scenario(
            seed=args.scenario_seed,
            x_apartments=3,
            y_apartments=3,
            n_sta_per_ap=args.n_sta_per_ap,
            size=room_size,
            n_steps=n_steps,
            channel_width=80,
        )
        scenario.walls = jnp.zeros_like(scenario.walls)

        env = EnterpriseEnv(
            scenario,
            ap_indices=[room_idx * (args.n_sta_per_ap + 1) for room_idx in range(3 * 3)],
            bounds_min=np.array([0.0, 0.0], dtype=np.float64),
            bounds_max=np.array([3 * room_size, 3 * room_size], dtype=np.float64),
            sim_dt=args.sim_dt,
            speed_min=args.mobility_speed_min,
            speed_max=args.mobility_speed_max,
            segment_min_s=args.mobility_segment_min_s,
            segment_max_s=args.mobility_segment_max_s,
            spawn_interval_steps=max(1, int(round(args.spawn_interval_s / args.sim_dt))),
            spawn_count=args.spawn_count,
            no_mobility=args.no_mobility,
        )
        metadata = {
            'no_mobility': args.no_mobility,
            'spawn_interval_s': args.spawn_interval_s,
            'spawn_count': args.spawn_count,
        }

    return env, n_steps, metadata
