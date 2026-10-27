from itertools import chain

import pulp as plp
from mapc_optimal import Solver, positions_to_path_loss

from lai4wifi.dataset.utils import TX_POWER_LEVELS


def run_solver(scenario, opt_type, threads=None):
    splits = []

    for static, duration in scenario.split_scenario():
        associations = static.associations
        access_points = list(associations.keys())
        stations = list(chain.from_iterable(associations.values()))
        path_loss = positions_to_path_loss(static.pos, static.walls)

        solver = Solver(
            stations, access_points,
            max_tx_power=TX_POWER_LEVELS[0],
            min_tx_power=TX_POWER_LEVELS[-1],
            opt_type=opt_type,
            solver=plp.CPLEX_CMD(msg=False, threads=threads),
            channel_width=80
        )

        configuration, data_rate = solver(path_loss, associations)
        splits.append({
            'duration': duration,
            'data_rate': data_rate,
            'shares': configuration.get('shares', {}),
            'link_rates': configuration.get('link_rates', {}),
            'tx_power': configuration.get('tx_power', {}),
        })

    return {'splits': splits}
