FM4WiFi: GNN autoencoder + flow matching + MDN surrogate for Co-SR configuration generation. Layout, commands and pipeline order are in `README.md`; the paper is in `paper/` (separate git repo, figures in `paper/figures/*.tikz`).

## Reproducibility
- Results depend on `JAX_ENABLE_X64=True` (random node placements and tx sampling change in 32-bit), the seeds, and the JAX version (0.8.0, pinned in `uv.lock`). Every entry point sets `JAX_ENABLE_X64` before importing jax; keep it that way in new scripts.
- GPU runs are not bit-reproducible: graph aggregations use scatter-add, so surrogate scores differ in the last digits between identical runs. Compare results with a float tolerance, not exact equality.
- Use the project venv: `source .venv/bin/activate`.

## Conventions
- Training scripts (`graphs/`, `flow_matching/`, `surrogate/` `training.py` and `data.py`) use Hydra or paths relative to their module directory; run them from there.
- Experiments and baselines (`python -m lai4wifi.experiments.*`, `python -m lai4wifi.baselines.*`) load `flow_matching/runs/fm_medium` and `surrogate/runs/sur_medium` by default and can run from anywhere; `lai4wifi/scripts/*.sh` write to `results/`.
- `graphs/utils.py:make_batch()` pads to power-of-two sizes and adds one padding graph; surrogate scores drop it with `[:-1]`.
- `JaxEnum` and `Connection` in `graphs/attributes.py` are registered pytrees; change them carefully.
- Old results, runs and datasets live in `archive/` (gitignored, layout in `archive/README.md`).
