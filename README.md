# FM4WiFi: Flow Matching for Multi-AP Coordination in Dense Deployments of Beyond Wi-Fi 8 Networks

FM4WiFi generative ML pipeline for coordinated spatial reuse (Co-SR) in beyond Wi-Fi 8 multi-AP coordination (MAPC). Produces high-quality Co-SR configs (AP/station selection, MCS, tx power) in single inference step, sub-second for 30+ APs. FM4WiFi consists of 3 models:

1. **Autoencoder (`graphs/`)** — GNN learns compact latent embeddings of network state (topology, RSSI, associations) from graph representation.
2. **Flow matching model (`flow_matching/`)** — generative transformer + GNN synthesizes feasible Co-SR configs (AP/station selection, MCS, tx power) conditioned on latent state. Includes rate control, missing from prior work.
3. **Surrogate rate predictor (`surrogate/`)** — lightweight GNN quantile regressor scores candidate configs fast, no live system / digital twin needed.

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
├── dataset/       # scenario/network generation, baseline optimizers (f-optimal, t-optimal, h-MAB, random tx)
├── graphs/        # GNN autoencoder — network state → latent embedding
├── flow_matching/ # FM generative model, sim, MMD eval, RL fine-tuning
├── surrogate/     # GNN quantile-regression throughput predictor
└── baselines/     # baseline runner + analysis (baselines.py, analyze_baselines.py)
```

Each stage depends on the previous checkpoint:

```
graphs/training.py  →  graphs/runs/<name>/         (GNN encoder checkpoint)
flow_matching/training.py  (loads gnn_path)  →  flow_matching/runs/<name>/
surrogate/training.py      (loads gnn_path)  →  surrogate/runs/<name>/
```

## Training

Run from module directory, Hydra config overrides on command line:

```bash
cd lai4wifi/graphs && python training.py
cd lai4wifi/flow_matching && python training.py
cd lai4wifi/surrogate && python training.py

# override config groups (model/optimizer/train each have tiny/small/medium/large + debug)
python training.py model=large optimizer=large train=debug
```

Outputs land in `./runs/<name>/<date>/<time>/`.

## Simulation & evaluation

```bash
cd lai4wifi/flow_matching
python sim.py           # scenario sim
bash run_all_sim.sh     # batch sim runs
```

Baselines: `lai4wifi/baselines/baselines.py` (runner) + `analyze_baselines.py` (analysis); baseline optimizers themselves (`f_optimal.py`, `t_optimal.py`, `h_mab.py`, `random_tx.py`) live in `lai4wifi/dataset/`.

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
