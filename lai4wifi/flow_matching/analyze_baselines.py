import json
import re
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np


def jains_fairness(rates):
    rates = np.array(rates)
    if len(rates) == 0 or np.sum(rates) == 0:
        return 0.0
    return float((np.sum(rates) ** 2) / (len(rates) * np.sum(rates ** 2)))


def parse_station_id(link_str):
    match = re.search(r"STA_(\d+)", link_str)
    return int(match.group(1)) if match else None


def analyze_solver(agent_data):
    splits_analysis = []

    for split in agent_data['splits']:
        shares = split['shares']
        link_rates = split['link_rates']

        station_rates = defaultdict(float)
        for conf_id, links in link_rates.items():
            share = shares.get(conf_id, 0.0)
            for link_str, rate in links.items():
                sta_id = parse_station_id(link_str)
                if sta_id is not None:
                    station_rates[sta_id] += share * rate

        rates_list = list(station_rates.values())

        splits_analysis.append({
            'data_rate': split['data_rate'],
            'fairness': jains_fairness(rates_list),
            'per_station': dict(sorted(station_rates.items())),
        })

    return {'splits': splits_analysis}


def analyze_h_mab(agent_data):
    runs = agent_data['runs'][0]
    actions = agent_data['actions'][0]
    n_steps = len(runs)

    window_size = min(n_steps, max(200, int(0.2 * n_steps)))
    window_start = n_steps - window_size
    window_runs = runs[window_start:]
    window_actions = actions[window_start:]

    all_stations = set()
    for action in actions:
        for vals in action.values():
            all_stations.add(int(vals[0]))

    station_rate_sums = {sta: 0.0 for sta in all_stations}
    for action in window_actions:
        for vals in action.values():
            sta = int(vals[0])
            rate = vals[1]
            station_rate_sums[sta] += rate

    n_window = len(window_actions)
    per_station = {sta: station_rate_sums[sta] / n_window for sta in sorted(all_stations)}
    rates_list = list(per_station.values())

    return {
        'mean_data_rate': float(np.mean(window_runs)),
        'fairness': jains_fairness(rates_list),
        'per_station': per_station,
        'data_rates_in_time': runs,
    }


def analyze(input_path):
    with open(input_path, 'r') as f:
        all_results = json.load(f)

    analysis = []

    for i, scenario_results in enumerate(all_results):
        scenario_analysis = {}

        for agent_data in scenario_results:
            name = agent_data['agent']

            if name in ('f_optimal', 't_optimal'):
                scenario_analysis[name] = analyze_solver(agent_data)
            elif name == 'h_mab':
                scenario_analysis[name] = analyze_h_mab(agent_data)

        analysis.append(scenario_analysis)

    return analysis


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-i', '--input', type=str, default='baseline_results.json')
    args.add_argument('-o', '--output', type=str, default='baseline_analysis.json')
    args = args.parse_args()

    analysis = analyze(args.input)

    with open(args.output, 'w') as f:
        json.dump(analysis, f, indent=2)

    # Print summary
    for i, scenario in enumerate(analysis):
        print(f'\n=== Scenario {i + 1} ===')
        for agent, data in scenario.items():
            if 'splits' in data:
                for j, split in enumerate(data['splits']):
                    prefix = f'  {agent} split {j + 1}' if len(data['splits']) > 1 else f'  {agent}'
                    print(f'{prefix}: rate={split["data_rate"]:.2f}  fairness={split["fairness"]:.4f}  '
                          f'stations={list(split["per_station"].values())}')
            else:
                print(f'  {agent}: mean_rate={data["mean_data_rate"]:.2f}  fairness={data["fairness"]:.4f}  '
                      f'stations={list(data["per_station"].values())}')
