# :crossed_swords: Battle Gomoku AI (Renju Rules)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![DirectML](https://img.shields.io/badge/DirectML-AMD%20GPU%20Support-red)
![License](https://img.shields.io/badge/License-MIT-green)

A high-performance, **Renju-rule (렌주룰)** compliant Gomoku AI that combines traditional search algorithms with modern Reinforcement Learning.

> **Key Features:** Perfect Renju rule enforcement, Minimax/MCTS hybrid architecture, Self-play Reinforcement Learning pipeline, and AMD GPU acceleration support.

---

## :sparkles: Features

### :balance_scale: Perfect Renju Logic
Unlike standard Gomoku, this AI implements the complex **Renju Rules** used in official competitions:
- **Black (First Player):** Restricted from making 3-3, 4-4, and Overlines (6+). Only an exact 5 wins.
- **White (Second Player):** No restrictions. Wins with 5 or more stones.
- **Failsafe:** The AI understands these constraints perfectly and will never make an illegal move (Foul), forcing the opponent to navigate the board carefully.

### :brain: Hybrid AI Architecture
- **Search Engine:** A highly optimized **Iterative Deepening Minimax** with Alpha-Beta Pruning.
- **Policy-Value Network:** A ResNet-based deep learning model that predicts the best next move (Policy) and the current win probability (Value), guiding the search to prune irrelevant branches.
- **VCF (Victory by Continuous Four):** An optional, specialized search module that detects forced checkmate sequences.
- **MCTS Support:** Includes a PUCT-based Monte Carlo Tree Search implementation for comparative research.

### :rocket: Performance & Optimization
- **Transposition Tables:** Zobrist Hashing caches board states to prevent re-calculating identical positions.
- **DirectML Support:** Supports **AMD Radeon GPUs** via `torch-directml`, enabling fast neural network inference on non-NVIDIA hardware.
- **Robust Training Loop:** An automated self-play pipeline that collects data, trains the model, and updates the AI iteratively.

---

## :computer: Installation

### Prerequisites
- **Python 3.10** is recommended (Required for `torch-directml` AMD GPU support).
- Windows / Linux / macOS

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/your-username/Battle-Gomoku-AI.git
cd Battle_Omok_AI

# 2. Create a virtual environment (Recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# (Optional) For AMD GPU Users:
pip install torch-directml
```

---

## :video_game: How to Play

### 1. Human vs AI (GUI Mode)
Challenge the AI! You play as Black (start first).
```bash
python main.py --mode human-vs-ai --gui --timeout 5 --enable-vcf
```
- **Controls:** Click to place a stone.
- **Visuals:** Red dot indicates the last move. Ghost stones appear on hover.
- **Status Panel:** Displays current turn and game result.

### 2. AI vs AI (Watch Mode)
Watch two AI agents battle it out.
```bash
python main.py --mode ai-vs-ai --gui --board-size 15
```

### 3. Full Strength Mode
Enable VCF (Victory by Continuous Four) for the strongest gameplay.
```bash
python main.py --mode ai-vs-human --gui --enable-vcf --pv-checkpoint checkpoints/pv_latest.pt
```

---

## :chart_with_upwards_trend: Reinforcement Learning (Auto-Train)

Train your own AlphaZero-style model from scratch! The `auto_train.py` script handles the entire loop: **Self-Play $\rightarrow$ Data Generation $\rightarrow$ Training $\rightarrow$ Update Model**.

```bash
python auto_train.py \
    --iterations 10 \
    --games 100 \
    --board-size 15 \
    --depth 3 \
    --device dml  # Use 'dml' for AMD GPU, 'cuda' for NVIDIA, 'cpu' for CPU
```

| Argument | Description |
| :--- | :--- |
| `--iterations` | Number of Training Cycles (Generation + Training) |
| `--games` | Number of self-play games per iteration |
| `--depth` | Search depth for the AI during self-play |
| `--device` | Hardware acceleration (`cpu`, `cuda`, `dml`) |
| `--epsilon` | Random move probability (Exploration) |

---

## :file_folder: Project Structure

```text
Battle_Omok_AI/
├── main.py                 # Entry point for playing games
├── auto_train.py           # Automated RL training loop
├── selfplay.py             # Self-play data generator
├── train_pv.py             # Neural network training script
├── Omokgame.py             # Core game loop & state management
├── Board.py                # Board logic & history
├── Player.py               # Human & AI Agent interfaces
│
├── ai/                     # AI Core
│   ├── search_minimax.py   # Main Search Engine (Minimax)
│   ├── search_mcts.py      # Alternative Search (MCTS)
│   ├── pv_model.py         # PyTorch ResNet Model
│   ├── policy_value.py     # Inference Helper
│   ├── move_selector.py    # Candidate generation logic
│   └── heuristic.py        # Static evaluation board scorer
│
├── engine/                 # Rules Engine
│   ├── renju_rules.py      # 3-3, 4-4, Overline logic
│   └── referee.py          # Move validation
│
├── gui/                    # Visuals
│   └── pygame_view.py      # Pygame renderer
│
└── config/                 # Configuration
    ├── settings.yaml       # Game settings
    └── patterns.yaml       # Heuristic patterns
```

---

## :memo: Technical Details

### Renju Rule Implementation
The project rigorously enforces Renju rules using `engine/renju_rules.py`.
- **Recursion for 3-3/4-4:** It simulates placing stones to check if "open threes" or "four threats" are created in multiple directions simultaneously.
- **Victory Condition:** `Board.py` ensures Black wins only on *exact* 5, while White wins on 5 or more.

### Handling "Timeout" & "Clean Board"
To ensure high-quality training data:
- If the AI times out during self-play, the system performs a **clean rollback** to the pre-move state using snapshots.
- A random fallback move is selected from legal candidates to prevent the game from crashing, ensuring continuous training stability.

---

## :scroll: License
This project is open-sourced under the **MIT License**.
