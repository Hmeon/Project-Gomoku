# Repository Guidelines

## Project Structure & Module Organization
- `AI-Gomoku-main/gomoku/`: pygame-based Minimax player; launch via `gomoku.py`; board art in `images/`.
- `gomokuAI-py-main/`: menu-driven pygame client; core logic in `source/AI.py`, GUI helpers in `gui/`, assets in `assets/`, entrypoint `play.py`.
- `gomoku_rl-main/`: reinforcement-learning stack; Python package code under `gomoku_rl/`, configs in `cfg/`, training/demo scripts in `scripts/`, docs/assets under `docs/` and `assets/`, C++ Qt GUI sources in `src/`, automated tests in `tests/`.
- Root PDFs capture research notes; keep alongside the repo for reference.

## Build, Test, and Development Commands
- Classic AIs: `python AI-Gomoku-main/gomoku/gomoku.py` and `python gomokuAI-py-main/play.py` (both need `pygame` installed).
- RL environment setup: `cd gomoku_rl-main && pip install -e .[test]` (Python 3.10+; installs torch, torchrl, hydra-core, pytest).
- Train RL agents: `python scripts/train_InRL.py num_env=256 device=cuda` (override Hydra defaults as needed).
- Demo trained agent: `python scripts/demo.py device=cpu checkpoint=pretrained_models/15_15/...`.
- Tests: `cd gomoku_rl-main && pytest`.
- Optional C++ GUI build: `cmake -S src -B build -DCMAKE_PREFIX_PATH=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)") && cmake --build build --config Release`.

## Coding Style & Naming Conventions
- Python: follow PEP 8; snake_case for functions/variables, CapWords for classes; favor type hints and short, focused modules; keep imports explicit.
- Configs: store defaults in `cfg/*.yaml`; avoid hardcoded paths; prefer CLI overrides rather than editing scripts.
- C++ (Qt GUI): keep one class per header, include guards, and descriptive method names consistent with existing `.hpp` files.

## Testing Guidelines
- Pytest suite lives in `gomoku_rl-main/tests/test_*.py`; mirror naming for new cases around collectors, environments, and utilities.
- Add fast unit tests for core logic and describe any manual GUI validation steps (e.g., smoke-run `play.py` or `scripts/demo.py`) in PR notes.
- Before merging, run `cd gomoku_rl-main && pytest`; optionally sanity-check training with a short run (`epochs=1`) when altering training code.

## Commit & Pull Request Guidelines
- Use imperative, scoped subjects (e.g., `gomoku_rl: fix action masking`) under 72 characters; keep diffs focused to the relevant subproject.
- PRs should explain behavior changes, configs used, performance impact, and test evidence; link issues when relevant.
- Attach screenshots or GIFs for GUI changes; call out any new dependencies or config updates in the description.
