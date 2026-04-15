# myproject

`myproject` is an Isaac Lab extension/package for the Unitree H1 final project tasks in this repository.
It includes:

- registered Isaac Lab tasks for curriculum, map, baseline, and play variants
- RSL-RL train/play entrypoints
- utility scripts for validating environments with zero/random agents
- terrain-processing scripts used to build curriculum arena assets from source map segments

This README is written for teammates who clone the repo from Git and need a repeatable setup without machine-specific absolute paths.

## Repository Layout

Important paths inside `myproject`:

- `source/myproject/myproject/tasks/`
  Final task registration and environment configs
- `source/myproject/myproject/assets/`
  Project assets used by the tasks
- `source/myproject/myproject/assets/environments/final_map_2.usd`
  Final competition map asset used by the map tasks
- `source/myproject/myproject/assets/environments/curriculum_arenas/`
  Generated stability/crossing arena USDA files and layout JSON
- `scripts/rsl_rl/`
  Train and play scripts for RSL-RL
- `scripts/`
  Helper scripts for listing tasks, testing environments, and building terrain assets

## Prerequisites

You need:

- Isaac Lab installed and working
- an Isaac Sim / Isaac Lab Python environment
- the repository cloned locally

Examples below assume you already activated your Isaac Lab environment, for example:

```bash
conda activate <isaaclab-env>
```

If you use Isaac Lab through `isaaclab.sh -p`, replace `python` with that launcher.

## Installation

From the `myproject` directory:

```bash
python -m pip install -e source/myproject
```

This installs the `myproject` package in editable mode, which is the expected workflow for development and collaboration.

## Verify the Installation

List registered tasks:

```bash
python scripts/list_envs.py
```

Filter by a keyword:

```bash
python scripts/list_envs.py --keyword Stability
```

## Common Commands

Run a quick environment smoke test with zero actions:

```bash
python scripts/zero_agent.py --task Template-Final-Project-Unitree-H1-Stability-Arena-Play-v0 --headless
```

Run with random actions:

```bash
python scripts/random_agent.py --task Template-Final-Project-Unitree-H1-Stability-Arena-Play-v0 --headless
```

Train with RSL-RL:

```bash
python scripts/rsl_rl/train.py --task Template-Final-Project-Unitree-H1-Stability-Arena-v0 --num_envs 1 --max_iterations 100 --headless
```

Play a trained checkpoint:

```bash
python scripts/rsl_rl/play.py --task Template-Final-Project-Unitree-H1-Stability-Arena-Play-v0 --headless
```

Notes:

- `--headless` is recommended for training stability and lower GPU memory usage.
- Non-headless training may require `--num_envs 1` because the viewer increases GPU memory pressure.
- The final-project tasks now resolve their asset paths relative to the package, so teammates do not need to edit hardcoded absolute paths.

## Final Project Tasks

Main task families registered by `myproject`:

- `Template-Final-Project-Unitree-H1-v0`
  Curriculum terrain training
- `Template-Final-Project-Unitree-H1-Map-v0`
  Fine-tuning on the final map
- `Template-Final-Project-Unitree-H1-Baseline-v0`
  Flat baseline on the final map
- `Template-Final-Project-Unitree-H1-Stability-Arena-v0`
  Warm-up on curated easy terrain
- `Template-Final-Project-Unitree-H1-Crossing-Arena-v0`
  Transfer stage on mixed curated terrain

Each family also has a `-Play-v0` variant for evaluation.

## Asset Notes

Project assets are stored inside the package:

- `source/myproject/myproject/assets/environments/final_map_2.usd`
- `source/myproject/myproject/assets/environments/curriculum_arenas/stability_arena.usda`
- `source/myproject/myproject/assets/environments/curriculum_arenas/crossing_arena.usda`

The layout files in `curriculum_arenas/` are metadata describing the tiled arenas and suggested start positions:

- `stability_arena.layout.json`
- `crossing_arena.layout.json`

Generated manifests under `assets/environments/` now use relative metadata so they can be shared across machines.

## Script Catalog

### Environment and RL scripts

`scripts/list_envs.py`

- Prints all registered `Template-...` tasks from this project
- Use it first to confirm task registration

Example:

```bash
python scripts/list_envs.py
```

`scripts/zero_agent.py`

- Runs an environment with zero actions
- Useful for checking spawning, resets, and basic simulation behavior

Example:

```bash
python scripts/zero_agent.py --task Template-Final-Project-Unitree-H1-Stability-Arena-Play-v0 --headless
```

`scripts/random_agent.py`

- Runs an environment with random actions
- Useful for stress-testing contacts and resets

Example:

```bash
python scripts/random_agent.py --task Template-Final-Project-Unitree-H1-Stability-Arena-Play-v0 --headless
```

`scripts/rsl_rl/train.py`

- Main RSL-RL training entrypoint
- Supports task selection, env count, checkpoint loading, video, and headless mode
- Includes extra numeric diagnostics for policy instability

Example:

```bash
python scripts/rsl_rl/train.py --task Template-Final-Project-Unitree-H1-Stability-Arena-v0 --num_envs 1 --max_iterations 100 --headless
```

`scripts/rsl_rl/play.py`

- Loads a trained RSL-RL checkpoint and runs inference
- Can export ONNX/JIT policies and optionally record video

Example:

```bash
python scripts/rsl_rl/play.py --task Template-Final-Project-Unitree-H1-Stability-Arena-Play-v0 --headless
```

### Terrain and arena preparation scripts

These scripts are used to analyze source terrains and build curated curriculum arenas. Most teammates only need them if they are regenerating assets.

`scripts/analyze_test_map_segments.py`

- Read-only analysis of a source terrain USD
- Produces a JSON report of candidate walkable Y-axis segments

Example:

```bash
python scripts/analyze_test_map_segments.py --usd-path test_map.usd
```

`scripts/extract_test_map_segments.py`

- Extracts standalone USDA scenes from selected terrain segments
- Uses the analysis report as input

Example:

```bash
python scripts/extract_test_map_segments.py --usd-path test_map.usd --output-dir source/myproject/myproject/assets/environments/test_map_segments
```

`scripts/recenter_filtered_segments.py`

- Re-centers curated terrain patches and writes a manifest with difficulty levels
- Default output is `source/myproject/myproject/assets/environments/test_segment_filtered_recentered`

Example:

```bash
python scripts/recenter_filtered_segments.py
```

`scripts/compose_filtered_curriculum_arena.py`

- Builds tiled arena layouts from the recentered patch manifest
- Produces composed arena USDA files and `.layout.json` metadata

Example:

```bash
python scripts/compose_filtered_curriculum_arena.py --preset both
```

`scripts/build_filtered_curriculum_mesh_arena.py`

- Builds a merged single-mesh arena USDA from a composed layout
- Useful when you want one merged terrain rather than a tiled stage

Example:

```bash
python scripts/build_filtered_curriculum_mesh_arena.py --layout-path source/myproject/myproject/assets/environments/curriculum_arenas/stability_arena.layout.json --output-path source/myproject/myproject/assets/environments/curriculum_arenas/stability_arena_merged.usda
```

## Recommended Teammate Workflow

1. Clone the repository.
2. Activate the Isaac Lab Python environment.
3. Install `myproject` in editable mode:

```bash
python -m pip install -e source/myproject
```

4. Verify task registration:

```bash
python scripts/list_envs.py
```

5. Run a simple validation:

```bash
python scripts/zero_agent.py --task Template-Final-Project-Unitree-H1-Stability-Arena-Play-v0 --headless
```

6. Start training:

```bash
python scripts/rsl_rl/train.py --task Template-Final-Project-Unitree-H1-Stability-Arena-v0 --num_envs 1 --max_iterations 100 --headless
```

7. Evaluate with play:

```bash
python scripts/rsl_rl/play.py --task Template-Final-Project-Unitree-H1-Stability-Arena-Play-v0 --headless
```

## Troubleshooting

If the task starts but fails in GUI mode with PhysX GPU allocation errors:

- reduce `--num_envs`
- prefer `--headless`
- avoid recording video unless needed

If the robot spawns too low and the feet intersect the ground:

- check the spawn-height tuning in `final_project_env_cfg.py`
- rerun with a single env first to validate spawn behavior

If tasks do not appear in `list_envs.py`:

- confirm editable install succeeded
- confirm you are using the Isaac Lab Python environment
- rerun `python -m pip install -e source/myproject`
