# Battle Gomoku AI (Renju Rules)

Python implementation of a Renju-rule Gomoku bot:
- Asymmetric wins: Black must make exact five (overline is foul unless another direction makes a five); White wins with 5+.
- Forbidden for Black: 3-3, 4-4, overline (unless five is also made); White has no fouls.
- Minimax + alpha-beta + TT, optional MCTS fallback.
- Pygame GUI (mouse input) with board image for 15x15, procedurally drawn grid for other sizes.

## Quickstart
```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
pip install -r requirements.txt

python main.py --mode ai-vs-ai --board-size 15 --timeout 5 --gui
```

Modes: `ai-vs-ai` (default), `human-vs-ai` (human as black), `ai-vs-human` (human as white).  
CLI flags: `--board-size` (15 or 19), `--timeout` seconds per move, `--depth` search depth, `--candidate-limit` move fan-out, `--gui` to enable Pygame.

## Files of Note
- `main.py`: wiring for players, settings, GUI.
- `Omokgame.py`: game loop, timeouts, renderer hook.
- `Board.py`: state + exact-five / 5+ detection helpers.
- `engine/renju_rules.py`: forbidden-move checks (3-3, 4-4, overline) and win logic.
- `ai/search_minimax.py`: alpha-beta with TT and VCF probe.
- `ai/search_mcts.py`: time-bounded MCTS fallback.
- `gui/pygame_view.py`: board render (15x15 asset or drawn grid for 19x19).
- `config/`: default settings and pattern weights.
- `tests/`: pytest coverage for rules, board, timeouts, search.

## Testing
```bash
cd Battle_Omok_AI
pytest
```

## Notes
- For 15x15, the GUI uses `assets/board.jpg`; other sizes use a drawn grid.
- If you add a 19x19 board image, keep the same margin ratio (~23px on 540px) or adjust `asset_margin` accordingly.
- MCTS is lightweight; for stronger play, tune rollout limits or swap in policy/value guidance.
- Opening “Swap” is not implemented; standard Renju fouls apply from move 1.
