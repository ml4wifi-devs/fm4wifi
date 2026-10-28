# FM4WiFi: Flow Matching for Multi-AP Coordination in Dense Deployments of Beyond Wi-Fi 8 Networks

FM4WiFi generative ML pipeline for coordinated spatial reuse (Co-SR) in beyond Wi-Fi 8 multi-AP coordination (MAPC). Produces high-quality Co-SR configs (AP/station selection, MCS, tx power) in single inference step, sub-second for 30+ APs. FM4WiFi consists of 3 models:

1. **Autoencoder (`graphs/`)** — GNN learns compact latent embeddings of network state (topology, RSSI, associations) from graph representation.
2. **Flow matching model (`flow_matching/`)** — generative transformer + GNN synthesizes feasible Co-SR configs (AP/station selection, MCS, tx power) conditioned on latent state. Includes rate control, missing from prior work.
3. **Surrogate rate predictor (`surrogate/`)** — GNN + transformer mixture density network scores candidate configs fast, no live system / digital twin needed.

Inference: encode recent network observations → FM model generates candidate configs → surrogate scores them → decode top-k into scheduling decisions.

## Install

Needs Python >=3.12.

```bash
pip install -e .          # CPU
pip install -e ".[gpu]"   # + CUDA 12
```

Key deps: JAX, Jraph (GNN), Flax, Optax, Hydra (configs), Weights & Biases (tracking). Baselines pull `mapc_optimal_research` from ml4wifi-devs.

## Repo layout

```
lai4wifi/
├── dataset/        # training-data generation (random, H-MAB, F-/T-Optimal configurations)
├── graphs/         # GNN autoencoder, graph representation, graph data loading
├── flow_matching/  # FM model, training, training data (data.py), MMD, RL fine-tuning
├── surrogate/      # MDN rate predictor: training, data from FM samples (data.py), evaluation (evaluate.py)
├── pipeline/       # FM4WiFi inference: checkpoint loading, tx <-> graph conversion, candidate generation and scoring
├── envs/           # evaluation scenarios and dynamic environments (mobility, joins, reassociation)
├── experiments/    # FM4WiFi experiments: static, dynamic, inference benchmark
├── baselines/      # F-/T-Optimal solvers, H-MAB (static and dynamic), DCF, result analysis
└── scripts/        # shell scripts running the experiments
```

Each stage depends on the previous checkpoint:

```
graphs/training.py         →  graphs/runs/<name>/          (GNN autoencoder)
flow_matching/data.py      →  flow_matching/datasets/      (encoded configurations with contexts)
flow_matching/training.py  →  flow_matching/runs/<name>/
surrogate/data.py          →  surrogate/datasets/          (FM samples with simulated rates)
surrogate/training.py      →  surrogate/runs/<name>/
```

## Training

Run from module directory, Hydra config overrides on command line:

```bash
cd lai4wifi/graphs && python training.py
cd lai4wifi/flow_matching && python data.py context && python data.py prebatch && python training.py
cd lai4wifi/surrogate && python data.py generate && python training.py

# override config groups (model/optimizer/train each have tiny/small/medium/large + debug)
python training.py model=large optimizer=large train=debug
```

Outputs land in `./runs/<name>/<date>/<time>/`.

RL fine-tuning of the FM model (DiffusionNFT) on a fixed topology, with online surrogate fine-tuning on the simulated rollouts. `n_designated` forces that many random stations (on distinct APs) into every configuration, which keeps top-k selection diverse:

```bash
cd lai4wifi/flow_matching && python rl.py 'scenarios=[residential/4]' train.n_designated=4 train.eval.selection=designated
# evaluate a checkpoint only
python rl.py 'scenarios=[residential/4]' train.n_designated=4 train.eval.selection=designated init_path=runs/<run> train.n_steps=0
```

## Simulation & evaluation

Scenario sets (`scaling`, `residential`, `enterprise`, `random`) are defined in `envs/scenarios.py`.
Experiments load `flow_matching/runs/fm_medium` and `surrogate/runs/sur_medium` by default and can be run from any directory:

```bash
python -m lai4wifi.experiments.static --scenarios residential    # static scenarios
python -m lai4wifi.experiments.dynamic --env residential         # residential with mobility [--no_walls]
python -m lai4wifi.experiments.dynamic --env random              # open space with mobility
python -m lai4wifi.experiments.dynamic --env enterprise          # enterprise with joins and reassociations
python -m lai4wifi.experiments.benchmark                         # inference timing
python -m lai4wifi.baselines.static --scenarios residential      # F-/T-Optimal and H-MAB
python -m lai4wifi.baselines.dynamic --env residential           # H-MAB in dynamic environments
```

Experiment scripts in `lai4wifi/scripts/` write their results to `results/`:

```bash
bash lai4wifi/scripts/generate_datasets.sh       # training datasets (random, H-MAB, F-/T-Optimal)
bash lai4wifi/scripts/run_sim.sh                 # static experiments and inference timing
bash lai4wifi/scripts/run_sim_dynamic.sh         # dynamic residential/random experiments
bash lai4wifi/scripts/run_sim_dynamic_open.sh    # dynamic enterprise experiments
bash lai4wifi/scripts/run_dcf.sh                 # DCF baseline
bash lai4wifi/scripts/run_baselines.sh           # F-/T-Optimal and H-MAB on static scenarios
bash lai4wifi/scripts/prebatch_surrogate.sh      # surrogate datasets
bash lai4wifi/scripts/run_mdn_components.sh      # MDN components ablation
bash lai4wifi/scripts/run_mdn_vs_mse.sh          # MDN vs MSE, AE vs raw ablation
bash lai4wifi/scripts/run_risk_factor.sh         # risk factor ablation
```

## Config system

Hydra, nested YAML per module:

```
configs/
├── {module}.yaml        # top-level (defaults: model, optimizer, train)
├── model/{tiny,small,medium,large}.yaml
├── optimizer/{tiny,small,medium,large}.yaml
└── train/base.yaml
```

## Citation

If you use FM4WiFi in your research, cite:

```bibtex
@article{wojnar2026fm4wifi,
  title   = {FM4WiFi: Flow Matching for Multi-AP Coordination in Dense Deployments of Beyond Wi-Fi 8 Networks},
  author  = {Wojnar, Maksymilian and Rusek, Krzysztof and Kosek-Szott, Katarzyna and Szott, Szymon},
  year    = {2026}
}
```
