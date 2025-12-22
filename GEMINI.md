# Project-Gomoku: Renju Gomoku AI (Battle_Omok_AI)

The primary project in this repository is `Battle_Omok_AI/`.
Historical prototypes live under `reference*/` and are kept unchanged.

## Start Here (Docs)

- `README.md`: overview + quick start
- `Battle_Omok_AI/README.md`: CLI usage and practical option guide
- `Battle_Omok_AI/docs/`: paper-ready docs (architecture/training/logging/rules/plotting)
- `PROJECT_REPORT.md`: technical report (paper-style)
- `PATCH_NOTE_v1.md`: changelog

## Core Features

- **Rules:** Renju (Black forbidden moves: 3-3, 4-4, overline; Black wins only with exact five, White with five or more)
- **Search backends:**
  - Iterative deepening minimax (alpha-beta + TT + optional VCF)
  - PUCT MCTS (soft pi target support)
- **Learning pipeline:** self-play JSONL generation -> PV training -> auto-train loop with evaluation gating
- **Logging:** CSV metrics under `Battle_Omok_AI/logs/` (`selfplay_metrics.csv`, `train_metrics.csv`, `eval_metrics.csv`)

## Key Entrypoints

- `Battle_Omok_AI/main.py`: match runner (CLI + optional GUI)
- `Battle_Omok_AI/selfplay.py`: self-play data (`*.jsonl`) + stats (`*_stats.json`)
- `Battle_Omok_AI/train_pv.py`: PV training + `train_metrics.csv`
- `Battle_Omok_AI/auto_train.py`: full loop + persistent evaluation logging (`eval_metrics.csv`)
