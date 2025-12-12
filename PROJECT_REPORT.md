# Project-Gomoku / Battle_Omok_AI - Technical Report (Renju Gomoku AI)

**Date:** 2025-12-12  
**Scope:** `Battle_Omok_AI/`  
**Keywords:** Renju, Gomoku, Forbidden moves (3-3/4-4/Overline), Minimax, Alpha-Beta, Transposition Table, Zobrist Hashing, PUCT MCTS, Policy-Value Network, Self-play, DirectML

---

## Abstract (EN)

This report describes the design and implementation of a Renju-rule Gomoku AI system that integrates classical search (iterative deepening minimax with alpha-beta pruning and transposition tables) and modern learning components (a ResNet-style policy-value network trained from self-play). The project provides a complete, reproducible pipeline for generating self-play trajectories, training a policy-value model from soft MCTS visit distributions, and iterating improvements via an automated training loop with evaluation gating and persistent metric logging. The rule engine enforces Renju-specific constraints for the first player (Black), including forbidden moves (double-three, double-four, and overline), while maintaining distinct win conditions for Black and White. The system is designed to be robust under time constraints, to avoid board-state corruption during search/self-play by using push/pop simulation primitives, and to support CPU/CUDA/DirectML deployment for training and inference.

---

## 초록 (KO)

본 보고서는 Renju 규칙(흑 금수 포함)을 엄격히 적용하는 오목(Gomoku) AI의 설계 및 구현을 기술한다. 시스템은 고전적 검색 기법(Iterative Deepening Minimax + Alpha-Beta Pruning + Transposition Table)을 기반으로 하되, 연구/학습 목적을 위해 PUCT 기반 MCTS와 Policy-Value(ResNet 스타일) 신경망을 결합한다. Self-play를 통해 데이터(JSONL)를 생성하고, MCTS의 방문 분포(soft pi)를 학습 타깃으로 사용하여 PV 네트워크를 학습하며, 자동 반복 학습(auto-train) 루프에서는 candidate 모델과 incumbent 모델의 대국 평가(score_rate)로 승격 여부를 결정하고 결과를 CSV로 누적 로깅한다. 또한 검색/학습 과정에서 보드 상태가 오염되지 않도록 push/pop 기반 시뮬레이션 API를 제공하고, CPU/CUDA/DirectML 환경을 고려한 실행 옵션을 갖춘다.

---

## 1. Problem Definition & Requirements

### 1.1 Game and Rule Set

- **Board:** `N x N` (default 15; supports other sizes for research)
- **Cell encoding:** `-1` (Black), `0` (Empty), `+1` (White)
- **Renju constraints (Black only):**
  - Forbidden: **double-three (3-3)**, **double-four (4-4)**, **overline (6+)**
  - **Win:** Black wins only with **exactly five in a row**
- **White win:** White wins with **five or more** in a row

Implementation references:
- Rule engine: `Battle_Omok_AI/engine/renju_rules.py`
- Move validation (time/bounds/occupancy/forbidden): `Battle_Omok_AI/engine/referee.py`
- Board line checks: `Battle_Omok_AI/Board.py`

### 1.2 Time Control & Robustness

- Per-move deadline is enforced at the referee level and at the search level.
- The system must handle invalid/late moves without crashing and without corrupting board state.

---

## 2. System Architecture

### 2.1 Module Layout

```text
Battle_Omok_AI/
  main.py                   # Match runner (CLI + optional GUI)
  Omokgame.py               # Game loop
  Board.py                  # Board state + line checks + simulation primitives
  engine/
    renju_rules.py          # Forbidden moves + win conditions
    referee.py              # Move validation (deadline, bounds, occupancy, forbidden)
  ai/
    move_selector.py        # Candidate generation
    heuristic.py            # Pattern-based evaluation + incremental update
    transposition.py        # Zobrist hashing utilities
    search_minimax.py       # Iterative deepening minimax (AB + TT + optional VCF)
    search_mcts.py          # PUCT MCTS (soft pi support, Dirichlet noise)
    pv_model.py             # ResNet-style policy/value net
    policy_value.py         # Checkpoint loader + inference helper
    dataset.py              # JSONL dataset + symmetry augmentation + index caching
  gui/
    pygame_view.py          # Renderer + mouse input
  selfplay.py               # Trajectory generation (JSONL + stats JSON)
  train_pv.py               # PV training with soft pi loss + CSV metrics
  auto_train.py             # Iterative loop + evaluation gate + CSV logs
  tests/                    # Pytest unit suite
```

### 2.2 Board State and Simulation Safety

To support high-performance search and safe rollback, the board provides:
- `Board.place(x, y, color)` for real moves (updates history)
- `Board._push_stone(x, y, color)` / `Board._pop_stone(x, y)` for search/self-play simulation (no history mutation)

This design prevents subtle bugs where search leaves stray stones on the board after timeouts/exceptions.

---

## 3. Rule Engine: Renju Forbidden-Move Detection

### 3.1 Win Conditions

- Black: exact-five only (`has_exact_five`)
- White: five-or-more (`has_five_or_more`)

### 3.2 Forbidden Moves (Black Only)

The rule engine checks, after hypothetically placing a Black stone:
- **Overline:** max contiguous line length > 5
- **Double-three:** creates >=2 distinct open-three patterns (collapsed by direction/run identity)
- **Double-four:** creates >=2 distinct four threats (exact-five in one move)

Key engineering points:
- Use of a simulation context manager to place and remove stones safely.
- Directional line extraction and run-based analysis to reduce false positives.

Implementation reference: `Battle_Omok_AI/engine/renju_rules.py`

---

## 4. Search Algorithms

### 4.1 Candidate Generation

Instead of expanding all empty cells, the AI ranks candidate moves near existing stones:
- Speeds up search significantly on midgame positions.
- Has a full-scan fallback when all local candidates are forbidden for Black.

Implementation reference: `Battle_Omok_AI/ai/move_selector.py`

### 4.2 Iterative Deepening Minimax (Alpha-Beta + TT)

Core features:
- Iterative deepening to return the best move found before deadline.
- Alpha-beta pruning to reduce branching.
- Transposition table keyed by `(zobrist_hash, to_move)` to reuse results.
- Optional VCF probe at root to detect forced wins by continuous-four threats.

Implementation reference: `Battle_Omok_AI/ai/search_minimax.py`

### 4.3 PUCT MCTS (soft pi, Dirichlet noise)

Core features:
- Negamax backup: node values are stored from the node's to-move perspective and sign-flipped when moving up.
- Priors from PV policy head when available; uniform priors otherwise.
- Root exploration noise (Dirichlet) for self-play diversity.
- Optional `return_pi=True` returns the root visit distribution pi (soft labels).

Implementation reference: `Battle_Omok_AI/ai/search_mcts.py`

---

## 5. Policy-Value Network (PV)

### 5.1 Model

- ResNet-style backbone with separate policy and value heads.
- Input planes: black stones, white stones, to-play plane.

Implementation reference: `Battle_Omok_AI/ai/pv_model.py`, `Battle_Omok_AI/ai/dataset.py`

### 5.2 Training Objective

- **Policy loss:** soft pi (from MCTS visits) against `log_softmax(policy_logits)` (KL-style)
- **Value loss:** MSE between predicted value and game outcome value

Implementation reference: `Battle_Omok_AI/train_pv.py`

### 5.3 Practical Compatibility & Safety

- Checkpoints can be either a wrapped dict (`{"model_state": ..., "args": ...}`) or a raw `state_dict`.
- PV loading is rejected if checkpoint `board_size` mismatches the runtime board size to avoid silent shape errors.

Implementation reference: `Battle_Omok_AI/ai/policy_value.py`, `Battle_Omok_AI/main.py`, `Battle_Omok_AI/selfplay.py`

---

## 6. Self-Play Dataset and Augmentation

### 6.1 JSONL Schema

Each line is a training sample:

```json
{
  "board": { "cells": [[0,0,...], ...] },
  "to_play": -1,
  "pi": [0.0, 0.0, ...],
  "value": 1,
  "winner": -1
}
```

Notes:
- `pi` is soft when produced by MCTS; otherwise one-hot for fallback/random moves.
- The dataset module builds an index cache (`*.jsonl.idx`) to allow random access without loading everything into memory.

Implementation reference: `Battle_Omok_AI/selfplay.py`, `Battle_Omok_AI/ai/dataset.py`

### 6.2 Symmetry Augmentation

The training pipeline applies dihedral symmetries (rotations + flips) to both:
- encoded board planes `[C,H,W]`
- policy vector `pi` reshaped to `[H,W]`

Implementation reference: `Battle_Omok_AI/ai/dataset.py`

---

## 7. Automated Training Loop (auto_train)

The loop performs:
1) self-play with current checkpoint (if exists)  
2) train a candidate checkpoint  
3) optional evaluation gate (candidate vs incumbent)  
4) promote or reject candidate, with persistent logging

Implementation reference: `Battle_Omok_AI/auto_train.py`

### 7.1 Persistent Metrics (for paper-grade reporting)

Generated in `Battle_Omok_AI/logs/`:
- `selfplay_metrics.csv`: iteration-wise self-play stats
- `train_metrics.csv`: epoch-wise training losses
- `eval_metrics.csv`: iteration-wise evaluation score_rate and decision
- `eval_iter_<i>_stats.json`: raw evaluation summary

### 7.2 Evaluation Metric

Let `games` be the number of evaluation matches, `wins` and `draws` be results from the candidate's perspective.  

`score_rate = (wins + 0.5 * draws) / games`

If `score_rate >= accept_threshold`, promote candidate.

---

## 8. Reproducibility & Experiment Template

### 8.1 Recommended "Research Run" Command (CPU)

```bash
cd Battle_Omok_AI
python auto_train.py \
  --iterations 10 \
  --games 100 \
  --timeout 5 \
  --candidate-limit 24 \
  --epsilon 0 \
  --epochs 4 \
  --device cpu \
  --search-backend mcts \
  --rollout-limit 256 \
  --explore 1.4 \
  --eval-games 20 \
  --accept-threshold 0.55 \
  --replay-window 5 \
  --seed 42
```

### 8.2 What to Report in the Final Paper

- Environment: OS / Python / torch / device (cpu/cuda/dml)
- Full training command line and seeds
- Curves/tables:
  - `eval_metrics.csv` score_rate by iteration
  - `train_metrics.csv` loss by epoch
  - `selfplay_metrics.csv` win/draw/avg_steps trends

### 8.3 Result Table Template (fill after running)

| Iteration | Eval Games | Wins | Draws | score_rate | Decision |
|---:|---:|---:|---:|---:|---|
| 1 | 20 |  |  |  | promoted/rejected |

---

## 9. Limitations and Future Work

- Candidate generation can miss rare tactical resources if `candidate_limit` is too low; an adaptive candidate policy can improve coverage.
- Deeper VCF integration for MCTS (or neural-guided tactical search) can improve forced-win detection.
- Stronger evaluation: Elo estimation with multiple baselines (random/greedy/minimax variants) and longer match series.
- Model improvements: residual depth, input features (history/ko-like patterns), or rule-aware feature planes.

---

## References

1. D. Silver et al., "Mastering the game of Go without human knowledge," *Nature*, 2017.  
2. D. Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play," *Science*, 2018.  
3. A. L. Zobrist, "A new hashing method with application for game playing," 1970.  
4. L. Kocsis and C. Szepesvari, "Bandit based Monte-Carlo Planning," *ECML*, 2006.
