# :black_medium_square: Project-Gomoku: Advanced Renju AI Workspace
<div align="center">

<!-- PROJECT LOGO OR BANNER (Optional) -->
<!-- <img src="img/banner.png" width="800" alt="Project Gomoku Banner"> -->

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![DirectML](https://img.shields.io/badge/DirectML-AMD%20GPU%20Accel-red?style=for-the-badge)
![Pygame](https://img.shields.io/badge/Pygame-GUI-2ea44f?style=for-the-badge&logo=pygame&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**A State-of-the-Art Renju Gomoku AI combining Minimax Search with Deep Reinforcement Learning.**
<br>
**Minimax 탐색 알고리즘과 심층 강화학습을 결합한 최첨단 렌주 오목 AI 프로젝트입니다.**

[English Guide](#english-guide) • [한국어 가이드](#korean-guide)

</div>

---

<a name="english-guide"></a>
## :earth_americas: English Guide

### :book: Project Overview
**Project-Gomoku** is a high-performance AI workspace specifically designed for **Renju (렌주)** rules. Unlike standard Gomoku, Renju enforces strict restrictions on the first player (Black) to ensure fairness. This project implements a hybrid architecture that integrates **Iterative Deepening Minimax** with a **ResNet-based Policy-Value Network**, accelerated by **AMD DirectML** for efficient training on non-NVIDIA hardware.

<!-- GAMEPLAY GIF PLACEHOLDER -->
<!--
<div align="center">
  <img src="img/gameplay.gif" width="600" alt="Gameplay Demo">
  <p><i>Figure 1: AI vs Human Gameplay Demo</i></p>
</div>
-->

### :balance_scale: Renju Rules Implementation
The AI engine (`engine/renju_rules.py`) rigorously enforces the official Renju restrictions:

| Player | Color | Win Condition | Restrictions (Fouls) |
| :--- | :---: | :--- | :--- |
| **Black** | ⚫ | **Exact 5** stones | **Forbidden:** 3-3, 4-4, Overline (6+) |
| **White** | ⚪ | **5 or more** stones | **None** (White has no restrictions) |

> **Note:** The AI recursively validates moves to prevent fouls as Black, while actively exploiting them as White.

### :brain: AI Architecture
<!-- ARCHITECTURE DIAGRAM PLACEHOLDER -->
<!--
<div align="center">
  <img src="img/architecture.png" width="700" alt="Hybrid AI Architecture">
  <p><i>Figure 2: Hybrid Minimax + ResNet Architecture</i></p>
</div>
-->

1.  **Search Engine:** Minimax with Alpha-Beta Pruning, enhanced by **Transposition Tables (Zobrist Hashing)** for caching and **VCF (Victory by Continuous Four)** for tactical forced wins.
2.  **Neural Network:** A custom ResNet model that predicts move probabilities (Policy) and win rates (Value) to guide the search tree.
3.  **Optimization:** Supports **DirectML** to utilize AMD Radeon GPUs for training and inference.

### :arrows_counterclockwise: RL Pipeline (Auto-Train)
The project features an automated Reinforcement Learning loop (`auto_train.py`):
1.  **Self-Play:** AI plays against itself to generate data.
2.  **Training:** The model learns from the game records.
3.  **Iteration:** The improved model replaces the old one, creating a virtuous cycle of improvement.

---

<a name="korean-guide"></a>
## :kr: 한국어 가이드 (Korean Guide)

### :book: 프로젝트 개요
**Project-Gomoku**는 공정한 오목 대국을 위한 **렌주(Renju) 룰**을 완벽하게 지원하는 고성능 AI 개발 환경입니다. 이 프로젝트는 전통적인 **Minimax 탐색 알고리즘**과 최신 **심층 강화학습(Deep Reinforcement Learning)** 기술을 결합하였으며, **AMD DirectML**을 통해 라데온 GPU에서도 고속 연산이 가능하도록 설계되었습니다.

### :balance_scale: 렌주 룰 (게임 규칙)
AI 엔진(`engine/renju_rules.py`)은 흑번의 유리함을 상쇄하기 위한 렌주 룰의 제약 사항을 정확히 이해하고 있습니다.

*   **흑(Black):** 3-3, 4-4, 육목(6개 이상) 금수(Foul)가 적용되며, 오직 **정확히 5목**을 두어야 승리합니다.
*   **백(White):** 금수가 없으며, 5목 이상(육목 포함)을 두면 승리합니다.

> **특징:** AI는 자신이 흑일 때는 금수 자리를 스스로 피하고, 백일 때는 상대의 금수를 유도하는 전략적인 플레이를 펼칩니다.

### :brain: AI 아키텍처 및 기술적 특징
1.  **고성능 탐색 엔진:** 알파-베타 가지치기(Alpha-Beta Pruning)가 적용된 반복적 심화(Iterative Deepening) Minimax 알고리즘을 사용합니다.
    *   **트랜스포지션 테이블:** 조브리스트 해싱(Zobrist Hashing)을 이용한 트랜스포지션 테이블로 중복된 보드 상태 연산을 방지합니다.
    *   **VCF (강제승 탐색):** 연속된 공격(4연타)을 통한 강제 승리 구간을 빠르게 찾아내는 VCF 모듈을 탑재했습니다.
2.  **정책-가치 네트워크 (Policy-Value Net):** ResNet 기반의 신경망이 다음 수의 확률(Policy)과 현재 승률(Value)을 예측하여 탐색의 효율을 극대화합니다.
3.  **하드웨어 가속:** `torch-directml`을 도입하여 AMD Radeon GPU 환경에서도 PyTorch 학습 및 추론 가속을 지원합니다.

### :arrows_counterclockwise: 강화학습 파이프라인
`auto_train.py`를 통해 알파제로(AlphaZero) 스타일의 자가 학습 루프를 자동으로 수행할 수 있습니다.
1.  **자가 대국 (Self-Play):** AI가 서로 대국하며 데이터를 생성합니다.
2.  **모델 학습 (Training):** 생성된 기보를 바탕으로 정책/가치 네트워크를 학습시킵니다.
3.  **모델 갱신 (Update):** 더 똑똑해진 모델을 다음 자가 대국에 투입하여 실력을 점진적으로 향상시킵니다.

---

## :computer: Installation & Usage (설치 및 실행)

### Prerequisites (필수 요구사항)
*   **Python 3.10** (Recommended for `torch-directml` compatibility / AMD GPU 사용 시 필수)
*   OS: Windows / Linux / macOS

### 1. Setup (설치)
```bash
# Repository Clone
git clone https://github.com/your-repo/Project-Gomoku.git
cd Project-Gomoku/Battle_Omok_AI

# Create Virtual Environment (Recommended)
python -m venv .venv
# Activate: .venv\Scripts\activate (Windows) or source .venv/bin/activate (Linux/Mac)

# Install Dependencies
pip install -r requirements.txt

# (Optional) For AMD GPU Acceleration / AMD GPU 사용자용
pip install torch-directml
```

### 2. Play Game (게임 실행)
**Human vs AI (GUI Mode):**
```bash
python main.py --mode human-vs-ai --gui --timeout 5 --enable-vcf
```
*   **--enable-vcf:** Enables the powerful VCF search module. (강력한 VCF 탐색 모듈을 활성화합니다.)
*   **Visuals:** Wood texture board, hover preview, last move indicator. (나무 질감 보드, 호버 미리보기, 마지막 수 표시 등 향상된 GUI를 제공합니다.)

### 3. Training AI (AI 학습)
Start the automated Reinforcement Learning loop.
<br>자동화된 강화학습 파이프라인을 실행합니다.

```bash
python auto_train.py --iterations 10 --games 100 --depth 3 --device dml
```
*   `--device dml`: Use AMD GPU (DirectML).
*   `--device cuda`: Use NVIDIA GPU.
*   `--device cpu`: Use CPU (Slower).

---

## :file_folder: Project Structure / 폴더 구조

```text
Project-Gomoku/
│
├── README.md                   # Project Documentation (설명서)
│
├── img/                        # [NEW] Images & Assets for Documentation
│   ├── .gitkeep
│   ├── architecture.png        # (Recommended) AI Structure Diagram
│   └── gameplay.gif            # (Recommended) In-game Screenshot/GIF
│
├── Battle_Omok_AI/             # [CORE IMPLEMENTATION]
│   ├── main.py                 # Game Entry Point (게임 실행 파일)
│   ├── auto_train.py           # RL Training Loop (강화학습 루프)
│   ├── selfplay.py             # Data Generator (자가대국 데이터 생성기)
│   ├── train_pv.py             # Model Trainer (모델 학습 스크립트)
│   ├── Omokgame.py             # Game Logic Controller (게임 로직 제어)
│   ├── Board.py                # Board State & History (보드 상태 및 기록)
│   │
│   ├── ai/                     # AI Algorithms
│   │   ├── search_minimax.py   # Minimax Engine (The Brain / 핵심 두뇌)
│   │   ├── search_mcts.py      # MCTS Engine
│   │   ├── pv_model.py         # ResNet Neural Network (신경망 모델)
│   │   ├── policy_value.py     # Inference Helper
│   │   └── heuristic.py        # Static Evaluator (정적 평가 함수)
│   │
│   ├── engine/                 # Rules Engine
│   │   ├── renju_rules.py      # Renju Logic (렌주 룰 구현)
│   │   └── referee.py          # Move Validation (심판/검증)
│   │
│   └── gui/                    # Graphical Interface
│       └── pygame_view.py      # Pygame Renderer (GUI 렌더러)
```

---

<div align="center">
  <sub>Developed with precision and passion for Gomoku AI.</sub>
</div>
