# Repository Guidelines

## Project Structure & Module Organization
- Core gameplay/AI in `Battle_Omok_AI/`; entrypoint `main.py`; game loop `Omokgame.py`; rules under `engine/`; search/model code in `ai/`; GUI under `gui/`; shared helpers in `utils/`.
- Reinforcement/training scripts: `selfplay.py`, `train_pv.py`, `auto_train.py`; checkpoints in `Battle_Omok_AI/checkpoints/`; logs in `Battle_Omok_AI/logs/`.
- Tests live in `Battle_Omok_AI/tests/` (files named `test_*.py`); GUI assets in `Battle_Omok_AI/assets/`; `reference*/` folders are historical; leave untouched.

## Build, Test, and Development Commands
- Install deps (Python 3.10+): `pip install -r requirements.txt` (add `torch-directml` for AMD GPU).
- Play locally: `python main.py --mode ai-vs-ai --gui --enable-vcf` or `python main.py --mode human-vs-ai --gui --timeout 5`.
- Generate self-play data: `python selfplay.py --games 10 --board-size 15 --depth 3 --output selfplay_renju.jsonl [--search-backend mcts]`.
- Train PV net: `python train_pv.py --data selfplay_renju.jsonl --epochs 5 --output checkpoints/pv_latest.pt`.
- Full loop: `python auto_train.py --iterations 2 --games 50 --device cpu`.
- Tests: `cd Battle_Omok_AI && pytest` (fast unit suite).

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indent; snake_case for vars/functions, CapWords for classes; explicit imports and type hints preferred.
- Keep functions short; avoid hardcoded paths; use CLI flags and config in `config/*.yaml`.
- Preserve GUI/event loop responsiveness when adding I/O; keep board state and move history consistent.

## Testing Guidelines
- Framework: pytest; files `test_*.py` in `Battle_Omok_AI/tests/`.
- Add targeted, fast tests for rules (Renju fouls), search heuristics, timeout handling, and PV loaders; minimal GUI smoke only.
- Run `cd Battle_Omok_AI && pytest` before pushing.

## Commit & Pull Request Guidelines
- Commits: imperative, focused subjects under 72 chars (e.g., `gomoku: fix renju foul check`); keep diffs scoped to touched subproject.
- PRs: describe behavior change, CLI/config used, performance impact; include test evidence (`pytest` output, self-play sample), link issues; attach screenshots or GIFs for GUI changes; call out new deps or config updates.

## Security & Operational Tips
- Validate checkpoint paths (`checkpoints/`) before training; `policy_value` accepts wrapped or plain `state_dict`.
- Avoid destructive git commands; do not modify `reference*/` archives.
- On Windows terminals, prefer GUI mode to avoid input polling timeouts; respect timeouts and maintain move_count/history consistency.
