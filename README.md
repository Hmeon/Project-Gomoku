# Project-Gomoku: Renju Gomoku AI (Minimax + PUCT MCTS + Policy-Value Network)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![DirectML](https://img.shields.io/badge/DirectML-optional-8A2BE2?style=for-the-badge)
![Pygame](https://img.shields.io/badge/Pygame-GUI-2ea44f?style=for-the-badge&logo=pygame&logoColor=white)

본 저장소는 **Renju 규칙을 엄격히 적용하는 오목(Gomoku) AI** 프로젝트입니다.  
핵심 구현은 `Battle_Omok_AI/` 아래에 있으며, 다음 구성 요소를 한 프로젝트로 묶어 **플레이(대국) + self-play 데이터 생성 + PV(Policy-Value) 학습 + 자동 반복 학습(auto-train)**까지 재현 가능하게 제공합니다.

- **Rules**: Renju (흑 금수: 3-3, 4-4, 장목 / 흑은 정확히 5만 승리, 백은 5 이상 승리)
- **Search**: Iterative Deepening Minimax(Alpha-Beta + TT) / PUCT MCTS(soft pi 학습 타깃 지원)
- **Model**: ResNet 기반 Policy-Value 네트워크(PV)
- **Pipeline**: `selfplay.py` -> `train_pv.py` -> `auto_train.py` (평가 게이트 + 지표 CSV 로깅)
- **UI**: Pygame GUI (`--gui`)

---

## 목차

- [빠른 시작](#빠른-시작)
- [Renju 규칙 구현](#renju-규칙-구현)
- [AI 구성(검색/모델)](#ai-구성검색모델)
- [학습 파이프라인](#학습-파이프라인)
- [로깅 & 평가 지표](#로깅--평가-지표)
- [데이터 포맷(self-play JSONL)](#데이터-포맷self-play-jsonl)
- [프로젝트 구조](#프로젝트-구조)
- [테스트](#테스트)
- [재현성 체크리스트](#재현성-체크리스트)
- [보안/안전 노트](#보안안전-노트)
- [참고문헌](#참고문헌)

---

## 빠른 시작

### 0) 준비 사항

- Python 3.10+ 권장
- (Windows PowerShell 5.1) 문서가 깨져 보이면: `Get-Content -Encoding UTF8 README.md` (또는 VS Code/GitHub에서 열기)
- (선택) AMD GPU(Windows): `torch-directml`

### 1) 설치

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

### 2) 플레이(대국)

GUI(추천):

```bash
cd Battle_Omok_AI
python main.py --mode human-vs-ai --gui --timeout 5
```

AI vs AI 관전:

```bash
cd Battle_Omok_AI
python main.py --mode ai-vs-ai --gui
```

MCTS 백엔드(soft pi 생성에 유리):

```bash
cd Battle_Omok_AI
python main.py --mode ai-vs-ai --gui --search-backend mcts --rollout-limit 512 --explore 1.4
```

### 3) Self-play 데이터 생성

```bash
cd Battle_Omok_AI
python selfplay.py --games 10 --board-size 15 --timeout 1.5 --search-backend mcts --rollout-limit 256 --output logs/selfplay_renju.jsonl
```

### 4) PV 네트워크 학습

```bash
cd Battle_Omok_AI
python train_pv.py --data logs/selfplay_renju.jsonl --epochs 4 --device cpu --output checkpoints/pv_latest.pt
```

### 5) auto-train (self-play -> 학습 -> 평가 -> 승격)

아래 명령은 **학습이 강화되는지(평가 score_rate)**를 파일로 남깁니다.

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
  --accept-threshold 0.55
```

권장:
- `--search-backend mcts`를 처음부터 사용해도 됩니다(soft pi 학습 지원).
- `--epsilon 0` 권장(MCTS는 Dirichlet/temperature로 탐색을 충분히 유도; epsilon은 one-hot pi를 만들 가능성이 커짐).
- `--enable-vcf`는 **minimax에만 실질적으로 영향**이 있고 MCTS에는 영향이 거의 없습니다.

---

## Renju 규칙 구현

Renju는 첫 플레이어(흑)에게 금수를 적용하여 밸런스를 맞추는 공식 규칙 계열입니다.

- **흑(Black, -1)**
  - 승리: **정확히 5목(exact five)**일 때만 승리
  - 금수: **3-3, 4-4, 장목(overline; 6+)**
- **백(White, +1)**
  - 승리: **5목 이상(5+)**이면 승리
  - 금수 없음

구현 위치:
- 금수/승리 판정: `Battle_Omok_AI/engine/renju_rules.py`
- 착수 검증(시간/범위/중복/금수): `Battle_Omok_AI/engine/referee.py`
- 보드 승리 판정(정확히 5 / 5+): `Battle_Omok_AI/Board.py`

---

## AI 구성(검색/모델)

### Board 표현과 안전한 시뮬레이션

- 보드 셀: `cells[y][x] in {-1, 0, +1}` (흑/빈칸/백)
- 검색 및 self-play에서 "보드 오염"을 막기 위해 빠른 시뮬레이션 API를 사용합니다.
  - `Board._push_stone(x, y, color)` / `Board._pop_stone(x, y)`

### 후보 수 생성(Candidate Generation)

전체 보드를 매번 탐색하지 않고, 기존 돌 주변의 빈 칸을 중심으로 후보를 생성하여 속도를 확보합니다.
- 구현: `Battle_Omok_AI/ai/move_selector.py`
- 주요 파라미터: `candidate_limit`, `radius`

### Minimax (Iterative Deepening + Alpha-Beta + TT)

- 구현: `Battle_Omok_AI/ai/search_minimax.py`
- 핵심 아이디어:
  - Iterative Deepening으로 제한 시간 내에서 깊이를 점진적으로 확장
  - Alpha-Beta pruning으로 가지치기
  - Transposition Table(TT) + Zobrist hashing으로 상태 캐싱
  - 흑 금수 필터링 + "후보가 전부 금수인 희귀 케이스"에서는 full-scan fallback
- VCF(연속 4 위협 기반 강제승 탐색)는 minimax 루트에서 선택적으로 사용됩니다.

### PUCT MCTS (soft pi 지원)

- 구현: `Battle_Omok_AI/ai/search_mcts.py`
- 특징:
  - Negamax 백업(부호 반전) 기반 값 전파
  - Dirichlet noise(루트 priors)로 self-play 탐색성 확보
  - `return_pi=True` 시 루트 방문 분포를 pi로 반환(soft label)
  - 흑 금수 필터링 + 후보가 모두 막히는 경우 full-scan fallback

### Policy-Value 네트워크(PV)

- 모델: `Battle_Omok_AI/ai/pv_model.py`
- 학습: `Battle_Omok_AI/train_pv.py` (soft pi에 대한 KL 형태 loss)
- 추론 헬퍼: `Battle_Omok_AI/ai/policy_value.py`
  - 체크포인트가 다른 보드 크기이면 로드/사용을 거부합니다(크래시 방지).

---

## 학습 파이프라인

### 1) selfplay.py

- 목적: (board, to_play) -> (pi, value) 샘플 생성
- 출력:
  - `*.jsonl` : 한 줄에 한 샘플(JSON)
  - `*_stats.json` : 게임 요약 통계(평균 수, 승/무, timeout/foul/invalid 등)

### 2) train_pv.py

- 목적: self-play JSONL로부터 PV 모델 학습
- 로그:
  - `Battle_Omok_AI/logs/train_metrics.csv` (epoch별 loss 기록)

### 3) auto_train.py

- 목적: self-play -> train -> (optional) eval -> promote 를 반복
- 로그(중요):
  - `Battle_Omok_AI/logs/selfplay_metrics.csv` (iteration별 self-play 요약)
  - `Battle_Omok_AI/logs/train_metrics.csv` (epoch별 학습 loss)
  - `Battle_Omok_AI/logs/eval_metrics.csv` (**iteration별 강화 지표**; score_rate/승격 여부)

---

## 로깅 & 평가 지표

학습 "강화" 여부를 가장 직접적으로 보여주는 것은 **candidate vs incumbent 평가의 score_rate**입니다.

- 평가 방식(요약):
  - iteration마다 새로 학습한 candidate 모델과 기존 incumbent 모델을 `--eval-games` 만큼 대국
  - `score_rate = (wins + 0.5 * draws) / games`
  - `score_rate >= --accept-threshold`이면 candidate를 승격(promote)

저장 위치:
- 평가 원본 통계: `Battle_Omok_AI/logs/eval_iter_<i>_stats.json`
- 누적 CSV(논문/리포트용): `Battle_Omok_AI/logs/eval_metrics.csv`

---

## 데이터 포맷(self-play JSONL)

`selfplay.py`가 생성하는 JSONL의 각 라인은 최소 다음 키를 포함합니다.

```json
{
  "board": { "cells": [[0,0,...], ...] },
  "to_play": -1,
  "pi": [0.0, 0.0, ...],
  "value": 1,
  "winner": -1
}
```

설명:
- `board.cells`: 2D 배열(-1/0/+1)
- `to_play`: 해당 샘플 시점의 다음 수(흑=-1, 백=+1)
- `pi`: 길이 `board_size * board_size`의 정책 분포
  - MCTS에서 `return_pi=True`로 얻는 soft pi가 우선 기록됩니다.
  - fallback/랜덤 등으로 플레이어 pi가 없으면 one-hot pi가 기록됩니다.
- `value`: self-play 종료 후 결과를 현재 플레이어 관점으로 변환한 값(승=+1, 패=-1, 무=0)

참고:
- `Battle_Omok_AI/ai/dataset.py`는 빠른 랜덤 액세스를 위해 `*.jsonl.idx` 인덱스 파일을 자동 생성/캐시합니다.

---

## 프로젝트 구조

핵심 디렉터리:

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

문서:
- `Battle_Omok_AI/README.md`: 실행/옵션 가이드(실전용)
- `Battle_Omok_AI/docs/`: 논문/보고서용 상세 문서(architecture/training/logging/rules)
- `PROJECT_REPORT.md`: 기술 보고서(논문 스타일)
- `PATCH_NOTE_v1.md`: 패치 노트/변경 이력

---

## 테스트

```bash
pytest Battle_Omok_AI/tests
```

---

## 재현성 체크리스트

논문/보고서 작성 시 다음을 함께 기록하는 것을 권장합니다.

- 실행 환경: OS / Python / torch / (cuda/dml 여부)
- 학습 명령: `auto_train.py` 전체 커맨드 라인
- seed: `--seed` 값(가능하면 고정)
- 평가 설정: `--eval-games`, `--accept-threshold`
- 강화 지표: `Battle_Omok_AI/logs/eval_metrics.csv` (iteration별 score_rate)
- self-play 품질: `Battle_Omok_AI/logs/selfplay_metrics.csv`
- 학습 안정성: `Battle_Omok_AI/logs/train_metrics.csv`

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
