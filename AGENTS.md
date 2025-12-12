# Repository Guidelines

## Project Structure & Module Organization
- Core gameplay/AI lives in `Battle_Omok_AI/` (entrypoint `Battle_Omok_AI/main.py`).
- Game loop: `Battle_Omok_AI/Omokgame.py`; rules: `Battle_Omok_AI/engine/`; search/model code: `Battle_Omok_AI/ai/`.
- GUI: `Battle_Omok_AI/gui/`; assets: `Battle_Omok_AI/assets/`; shared helpers: `Battle_Omok_AI/utils/`.
- Reinforcement/training: `Battle_Omok_AI/selfplay.py`, `Battle_Omok_AI/train_pv.py`, `Battle_Omok_AI/auto_train.py`; outputs in `Battle_Omok_AI/checkpoints/` and `Battle_Omok_AI/logs/`.
- Tests: `Battle_Omok_AI/tests/` (`test_*.py`). Folders `reference*/` are historical; leave untouched.

## Build, Test, and Development Commands
- Install (Python 3.10+): `cd Battle_Omok_AI; pip install -r requirements.txt` (optional AMD GPU: `pip install torch-directml`).
- Play (GUI): `cd Battle_Omok_AI; python main.py --mode ai-vs-ai --gui --enable-vcf` or `python main.py --mode human-vs-ai --gui --timeout 5`.
- Self-play data: `cd Battle_Omok_AI; python selfplay.py --games 10 --board-size 15 --depth 3 --output selfplay_renju.jsonl`.
- Train PV net: `cd Battle_Omok_AI; python train_pv.py --data selfplay_renju.jsonl --epochs 5 --output checkpoints/pv_latest.pt`.
- Full loop: `cd Battle_Omok_AI; python auto_train.py --iterations 2 --games 50 --device cpu`.
- Tests (from repo root): `pytest Battle_Omok_AI/tests`.

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indent; `snake_case` vars/functions; `CapWords` classes; explicit imports and type hints preferred.
- Avoid hardcoded paths; prefer CLI flags and YAML config in `Battle_Omok_AI/config/*.yaml`.
- Keep GUI/event loop responsive; keep board state, move history, and timeouts consistent.

## Testing Guidelines
- Framework: pytest; name files `test_*.py` under `Battle_Omok_AI/tests/`.
- Prefer small, fast unit tests (Renju fouls, search heuristics, timeout handling, PV/checkpoint loading); keep GUI tests minimal.

## Commit & Pull Request Guidelines
- Git history contains generic messages; use descriptive, imperative subjects (<= 72 chars), optionally scoped (e.g., `gomoku: fix renju foul check`).
- PRs: describe behavior changes and CLI/config used; include `pytest` output; add screenshots/GIFs for GUI changes; call out new dependencies or config updates.

## Security & Configuration Tips
- Validate checkpoint paths under `Battle_Omok_AI/checkpoints/` and avoid overwriting important runs.
- Be cautious with destructive git commands; keep `reference*/` archives unchanged.
