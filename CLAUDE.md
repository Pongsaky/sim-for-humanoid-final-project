# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

`myproject` is an Isaac Lab extension package for Unitree H1 humanoid locomotion RL. The repo registers Gymnasium tasks, ships RSL-RL train/play entry points, and includes asset-preprocessing scripts that build curriculum arenas from source terrain segments. Everything assumes an active Isaac Lab Python environment (e.g. `conda activate <isaaclab-env>`); replace `python` with `isaaclab.sh -p` when using that launcher.

## Common commands

```bash
# Editable install (required after clone, and after modifying setup metadata)
python -m pip install -e source/myproject

# Verify task registration
python scripts/list_envs.py
python scripts/list_envs.py --keyword Stability

# Smoke-test an environment without policy
python scripts/zero_agent.py --task <Task-ID> --headless
python scripts/random_agent.py --task <Task-ID> --headless

# Train (use --headless for stability; non-headless typically needs --num_envs 1)
python scripts/rsl_rl/train.py --task Template-Final-Project-Unitree-H1-Stability-Arena-v0 --num_envs 1 --max_iterations 100 --headless

# Resume / load a prior run
python scripts/rsl_rl/train.py --task <Task-ID> --resume --load_run <run_dir> --checkpoint <ckpt>

# Play a trained checkpoint (Play variants are for evaluation)
python scripts/rsl_rl/play.py --task Template-Final-Project-Unitree-H1-Stability-Arena-Play-v0 --headless

# Lint / format (pre-commit drives ruff + codespell + standard hygiene hooks)
pre-commit run --all-files
ruff check .
ruff format .
```

Train.py adds extras beyond the stock RSL-RL CLI: `--video`, `--video_length`, `--video_interval`, `--shutdown-eval` / `--shutdown-video` (post-shutdown eval w/ video on signal), `--export_io_descriptors`, `--ray-proc-id`, plus distributed flags. Play.py adds `--max_steps`, `--summary-json`, `--real-time`, `--use_pretrained_checkpoint`. Both wrap RSL-RL args from `scripts/rsl_rl/cli_args.py` and Isaac Lab `AppLauncher` args.

## Architecture

### Package entry point

`source/myproject/myproject/__init__.py` triggers gym registration via `from .tasks import *`. Tasks live under `source/myproject/myproject/tasks/manager_based/` and use `isaaclab.envs:ManagerBasedRLEnv` as the entry point. Two families coexist:
- `final_project/` — **active** family, ~14 registered task IDs.
- `myproject/` — legacy single-task template; treat as deprecated unless touching it explicitly.

### Final-project task family

All tasks are registered in `source/myproject/myproject/tasks/manager_based/final_project/__init__.py`. Each gym ID points to a pair of entry points (`env_cfg_entry_point`, `rsl_rl_cfg_entry_point`). Task IDs follow `Template-Final-Project-Unitree-H1-<Variant>[-Play]-v0`.

Variant families and what distinguishes them:
- **`-v0`** (curriculum): rough generated terrain with `FinalProjectUnitreeH1EnvCfg` curriculum.
- **`-Baseline-v0`**: full traversal of `final_map_2.usd` via `H1FlatEnvCfg` with first-passage reward.
- **`-RoughGoal-Baseline-v0`**: `H1RoughEnvCfg` baseline with goal-distance reward shaping.
- **`-SpeedRun-v0`** / **`-FastWalk-v0`**: subclasses of RoughGoalBaseline with elevated linear-velocity ranges.
- **`-Stability-Arena-v0`** / **`-Crossing-Arena-v0`**: shared-arena (single tiled USD) curated curriculum stages.
- **`-Play-v0`** for every family: evaluation/inference variant (typically fewer envs, no domain randomization).

### Single-file env config

`source/myproject/myproject/tasks/manager_based/final_project/final_project_env_cfg.py` (~1700 lines) is the source of truth for **all** env variants and their reward classes. Top of the file defines key constants: `FINAL_MAP_USD_PATH`, `MAP_START_POS`, `GOAL_X` (traversal distance), spawn-z clearance, curriculum-arena USDA paths, and reward weights. When tuning anything env-side, expect to edit this one file.

The main task is **left-to-right traversal**: spawn at the negative-x edge (`MAP_START_POS ≈ (-6.5, 0, ...)`) and reach `GOAL_X ≈ 13.8m` along +x. Reward shaping (`goal_progress`, `goal_reached_bonus`, `time_cost`) and terminations (`goal_reached`, `out_of_bounds`) all assume this +x progression — keep it in mind when reading metrics or modifying spawn/goal logic.

The reward / termination / observation MDP functions used by these configs live in `final_project/mdp/{rewards,terminations,observations,curriculums,events}.py` and are re-exported via `mdp/__init__.py`. Custom rewards like `goal_progress`, `goal_reached_bonus`, and the upright/height-gated wrappers are defined here — base H1 MDP fns are also accessible by import.

Spawn behavior matters: configs use grid spawn, near-start jitter, or shared-map absolute coordinates depending on whether the task uses generated terrain or a single shared USD arena. If the robot spawns with feet through the floor, retune spawn-z in this config and validate with `--num_envs 1` first.

### RSL-RL agent configs

`source/myproject/myproject/tasks/manager_based/final_project/agents/rsl_rl_ppo_cfg.py` defines six `*PPORunnerCfg` classes; each gym ID picks one. There is an **autopilot override system** that reads a JSON file pointed to by `FINAL_PROJECT_AUTOPILOT_PROFILE` to override runner hyperparameters at load time without code edits — useful for sweeps. When debugging unexpected hyperparameters at training start, check whether this env var is set.

### Assets and terrain pipeline

Assets ship inside the package under `source/myproject/myproject/assets/`:
- `environments/final_map_2.usd` — the 13.8m competition arena.
- `environments/curriculum_arenas/{stability,crossing}_arena.{usda,layout.json}` — tiled curated arenas plus metadata describing tile placement and start positions.
- `unitree_humanoid_config.py`, `robot_config.py` — robot articulation cfgs.

The arena `.usda` files are **generated** from a multi-step pipeline driven by `scripts/`:
1. `analyze_test_map_segments.py` → JSON report of walkable Y segments in a source USD.
2. `extract_test_map_segments.py` → standalone USDA scenes per segment.
3. `recenter_filtered_segments.py` → recenter + write difficulty manifest.
4. `compose_filtered_curriculum_arena.py` → tiled arena USDA + `.layout.json`.
5. `build_filtered_curriculum_mesh_arena.py` → optional merged-mesh variant.

Most contributors do not regenerate these; only run the pipeline if you are intentionally rebuilding curriculum assets.

### Path resolution

Final-project configs resolve asset paths relative to the installed package (no hardcoded absolute paths). If you add new USD assets, place them under `source/myproject/myproject/assets/` and resolve via the same package-relative pattern used by the existing constants in `final_project_env_cfg.py`.

## Conventions

- Python 3.11; ruff line-length 120, target py310, with the Isaac-Lab-specific isort sections defined in `pyproject.toml`. The mccabe complexity ceiling is 30.
- Pyright runs in `basic` mode over `source/` and `scripts/`; `reportMissingImports` is silenced because CI cannot install Isaac Sim.
- License headers (BSD-3-Clause) are inserted by pre-commit's `insert-license` hook on `.py` and `.yaml` files.
- `__init__.py` files are exempt from F401 (unused imports).
