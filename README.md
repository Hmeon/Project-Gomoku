# Project-Gomoku Workspace

This workspace hosts multiple Gomoku implementations:
- `Battle_Omok_AI/`: Renju rules (black fouls: 3-3, 4-4, overline except when a five is also made), minimax + alpha-beta + TT, lightweight MCTS, Pygame GUI.
- `AI-Gomoku-main/`, `gomokuAI-py-main/`: pygame-based boards/assets and classic minimax reference code.
- `gomoku_rl-main/`: RL/MCTS research stack with optional Qt GUI (separate environment required).
- Root PDFs: research/competition requirement notes.

## Quickstart (Battle_Omok_AI)
```bash
cd Battle_Omok_AI
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python main.py --mode ai-vs-ai --board-size 15 --timeout 5 --gui
```
- Modes: `ai-vs-ai` (default), `human-vs-ai` (human as black), `ai-vs-human` (human as white).
- Board: 15x15 uses `assets/board.jpg`; other sizes draw a grid in code.
- Key configs: `config/settings.yaml`, pattern weights in `config/patterns.yaml`.

## Testing
```bash
cd Battle_Omok_AI
pytest
```
Latest run: 11 tests, all pass (Python 3.13.3).

## Current state / TODO
- Battle_Omok_AI: Renju fouls enforced for black; white wins on 5+. MCTS is lightweight; improve with policy/value guidance if needed. GUI margin corrected for assets; non-15x15 uses drawn grid.
- Future: expand tests (search quality/time-limit), add 19x19 asset or tune `asset_margin`, wire CI to run `pytest`.

## Notes
- RL stack: `cd gomoku_rl-main && pip install -e .[test]` then `pytest` or `python scripts/train_InRL.py ...`.
- Qt GUI build: see `gomoku_rl-main` docs.
