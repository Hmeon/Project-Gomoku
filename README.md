# Project-Gomoku: Renju Gomoku AI (Minimax + PUCT MCTS + Policy-Value Network)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![DirectML](https://img.shields.io/badge/DirectML-optional-8A2BE2?style=for-the-badge)
![Pygame](https://img.shields.io/badge/Pygame-GUI-2ea44f?style=for-the-badge&logo=pygame&logoColor=white)

본 저장소는 **Renju 규칙을 엄격히 적용하는 오목(Gomoku) AI** 프로젝트입니다.  
핵심 구현은 `Battle_Omok_AI/` 아래에 있으며, 플레이(대국)부터 self-play 데이터 생성, PV(Policy-Value) 학습, 자동 반복 학습(auto-train)까지 하나의 파이프라인으로 제공합니다.

주요 구성:
- **Rules**: Renju (흑 금수: 3-3, 4-4, 장목 / 흑은 정확히 5만 승리, 백은 5 이상 승리)
- **Search**: Iterative Deepening Minimax(Alpha-Beta + TT) / PUCT MCTS(soft pi 학습 타깃 지원)
- **Model**: ResNet 기반 Policy-Value 네트워크(PV)
- **Pipeline**: `selfplay.py` -> `train_pv.py` -> `auto_train.py` (평가 게이트 + 지표 CSV 로깅)
- **UI**: Pygame GUI (`--gui`)

---

## 목차

- [빠른 시작](#빠른-시작)
- [문서 안내](#문서-안내)
- [규칙 요약](#규칙-요약)
- [검색/학습 요약](#검색학습-요약)
- [프로젝트 구조](#프로젝트-구조)
- [테스트](#테스트)
- [보안/안전 노트](#보안안전-노트)
- [참고문헌](#참고문헌)

---

## 빠른 시작

### 설치

```bash
cd Battle_Omok_AI
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt

# (선택) AMD GPU
pip install torch-directml
```

### 1분 명령어

| 목적 | 명령 |
| --- | --- |
| GUI 대국 | `cd Battle_Omok_AI; python main.py --mode human-vs-ai --gui --timeout 5` |
| AI vs AI 관전 | `cd Battle_Omok_AI; python main.py --mode ai-vs-ai --gui` |
| self-play 생성(MCTS) | `cd Battle_Omok_AI; python selfplay.py --games 10 --board-size 15 --timeout 1.5 --search-backend mcts --rollout-limit 256 --output logs/selfplay_renju.jsonl` |
| PV 학습 | `cd Battle_Omok_AI; python train_pv.py --data logs/selfplay_renju.jsonl --epochs 4 --device cpu --output checkpoints/pv_latest.pt` |

### auto-train 예시

```bash
cd Battle_Omok_AI
python auto_train.py \
  --iterations 10 \
  --games 100 \
  --timeout 5 \
  --candidate-limit 20 \
  --epsilon 0 \
  --epochs 4 \
  --device cpu \
  --search-backend mcts \
  --rollout-limit 256 \
  --explore 1.4 \
  --eval-games 20 \
  --accept-threshold 0.55
```

기본 설정은 `Battle_Omok_AI/config/settings.yaml`에서 바꿀 수 있습니다.

---

## 문서 안내

- `README.md`: 프로젝트 요약 + 빠른 시작(이 문서)
- `Battle_Omok_AI/README.md`: 실행/옵션 중심 실전 가이드
- `Battle_Omok_AI/docs/`: 논문/보고서용 상세 문서(architecture/training/logging/rules/plotting)
- `PROJECT_REPORT.md`: 기술 보고서(논문 스타일)
- `PATCH_NOTE_v1.md`: 패치 노트/변경 이력

---

## 규칙 요약

- **흑(Black, -1)**: 정확히 5목(exact five)일 때만 승리, 3-3/4-4/장목(6+) 금수 적용
- **백(White, +1)**: 5목 이상이면 승리, 금수 없음

규칙 구현: `Battle_Omok_AI/engine/renju_rules.py`, `Battle_Omok_AI/engine/referee.py`

---

## 검색/학습 요약

- **Minimax**: Iterative Deepening + Alpha-Beta + TT + (옵션) VCF
- **MCTS**: PUCT + soft pi + Dirichlet noise(학습용)
- **PV**: ResNet 기반 Policy-Value 네트워크(옵션)
- **후보 수 생성**: Manhattan/Euclidean 반경 + run-endpoint 보강, 기본 `candidate_limit=20`
- **PV 미사용 시**: MCTS는 heuristic priors와 tanh value 추정으로 탐색을 보강

세부 옵션/동작은 `Battle_Omok_AI/README.md`와 `Battle_Omok_AI/docs/`를 참고하세요.

---

## 프로젝트 구조

```text
Battle_Omok_AI/
  main.py                # 대국 실행 엔트리포인트
  Omokgame.py             # 게임 루프(턴/검증/승패)
  Board.py                # 보드 상태/승리 판정/시뮬레이션 API
  engine/                 # Renju 규칙 + 착수 검증
  ai/                     # minimax / mcts / pv 모델 / dataset / 후보 생성
  gui/                    # pygame 렌더/입력
  config/                 # settings.yaml, patterns.yaml
  selfplay.py             # self-play 데이터 생성
  train_pv.py             # PV 학습
  auto_train.py           # 반복 학습 루프(평가 게이트+CSV 로깅)
  tests/                  # pytest
```

---

## 테스트

```bash
pytest Battle_Omok_AI/tests
```

---

## 보안/안전 노트

- `torch.load`는 신뢰되지 않은 체크포인트를 로드할 경우 위험할 수 있습니다.  
  외부에서 받은 파일은 반드시 출처를 확인하고 샌드박스/격리 환경에서 검증하세요.

---

## 참고문헌

- Silver et al., "Mastering the game of Go without human knowledge" (AlphaGo Zero), Nature (2017)
- Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play" (AlphaZero), Science (2018)
- Zobrist, "A new hashing method with application for game playing" (1970)
- Kocsis & Szepesvari, "Bandit based Monte-Carlo Planning" (UCT), ECML (2006)
