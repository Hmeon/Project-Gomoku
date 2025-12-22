# Architecture: Battle_Omok_AI (Renju Gomoku AI)

본 문서는 `Battle_Omok_AI/`의 전체 구조를 "연구/논문" 관점에서 설명합니다.  
코드 레벨의 정확한 구현 위치는 각 섹션의 **Implementation references**를 참고하세요.

---

## 1. High-level overview

시스템은 크게 (1) 규칙 엔진, (2) 검색 AI, (3) 학습(PV), (4) 파이프라인(auto-train), (5) UI로 구성됩니다.

```text
           +-------------------+
           |   main.py         |
           | (CLI + wiring)    |
           +---------+---------+
                     |
                     v
           +-------------------+
           | Omokgame.py       |
           | (game loop)       |
           +---------+---------+
                     |
          move       | validate (deadline, bounds, forbidden)
          request    v
           +-------------------+        +----------------------+
           | Player (Human/AI) | -----> | engine/referee.py    |
           +---------+---------+        +----------+-----------+
                     |                             |
                     v                             v
           +-------------------+        +----------------------+
           | Search (minimax/  | <----> | engine/renju_rules.py|
           | mcts) + heuristic |        +----------------------+
           +---------+---------+
                     |
                     v
           +-------------------+
           | Board.py          |
           | (cells/history)   |
           +-------------------+
```

---

## 2. Core game loop (Omokgame)

`Omokgame.py`는 다음 책임을 가집니다.

- 턴 관리(흑 선공)
- 플레이어로부터 다음 수 요청
- referee를 통한 착수 검증(시간/좌표/중복/금수)
- 보드에 돌을 놓고 승패 판정
- (선택) GUI renderer 호출 및 종료 처리

Implementation references:
- `Omokgame.py`
- `engine/referee.py`
- `engine/renju_rules.py`

---

## 3. Board representation and simulation safety

### 3.1 Representation

- `Board.cells[y][x]`: `-1` black / `0` empty / `+1` white
- `Board.history`: 실제 착수 기록(실제 게임 진행에서만 누적)
- `Board.occupied`: 후보 생성 및 시뮬레이션 효율을 위한 점유 좌표 집합

### 3.2 Simulation API

검색 및 self-play는 많은 가상 착수/되돌리기를 수행하므로, "history 오염"을 막기 위해 빠른 API를 사용합니다.

- `Board._push_stone(x, y, color)` / `Board._pop_stone(x, y)`
  - history를 변경하지 않고 빠르게 돌을 넣고 빼며 `move_count`, `occupied`를 갱신합니다.
  - search/mcts/self-play에서 예외/타임아웃이 발생해도 `finally` 블록에서 안전하게 복구하도록 설계되었습니다.

Implementation references:
- `Board.py`

---

## 4. Renju rule engine

Renju는 "흑"에게만 금수를 적용하는 규칙입니다.

- **승리 조건**
  - 흑: 정확히 5목(exact five)
  - 백: 5목 이상(5+)
- **흑 금수(Forbidden)**
  - double-three (3-3)
  - double-four (4-4)
  - overline (6+)

핵심 설계 포인트:
- 금수 판정은 "돌을 두었을 때"의 보드 상태를 기준으로 해야 하므로 시뮬레이션 컨텍스트(`_simulate`)를 사용합니다.
- open-three/open-four/four 위협은 방향성/연속 run 기반으로 계산하여 중복 카운트를 줄입니다.

Implementation references:
- `engine/renju_rules.py`
- `engine/referee.py`

---

## 5. Search: Minimax

`search_minimax.py`는 대국용 고전 AI 백엔드입니다.

핵심 요소:
- Iterative Deepening(제한 시간 내에서 1..depth까지 확장)
- Alpha-Beta pruning
- Transposition Table(TT) + Zobrist hash로 중복 상태 캐싱
- 후보 수 생성 + 흑 금수 필터링
- (선택) VCF(연속 4 위협 기반 강제승) 루트 탐색

PV 통합(선택):
- 루트에서 PV policy를 이용해 move ordering을 보강할 수 있습니다.
- PV value를 휴리스틱 평가에 소량 혼합할 수 있습니다.

Implementation references:
- `ai/search_minimax.py`
- `ai/transposition.py`
- `ai/move_selector.py`
- `ai/heuristic.py`
- `ai/policy_value.py`

---

## 6. Search: PUCT MCTS

`search_mcts.py`는 학습 및 연구 목적에 유리한 백엔드입니다.

핵심 요소:
- Negamax backup(값 전파 시 부호 반전)
- PUCT 선택 기준: `Q + U` 형태(부모 관점에서 Q 부호 처리)
- PV 정책 priors + PV value(가능하면), 없으면 heuristic priors/value
- 루트 priors에 Dirichlet noise(탐색 다양성)
- `return_pi=True`일 때 루트 방문 분포를 soft pi로 반환(학습 타깃)
- 흑 금수 필터링 + 후보 고갈 시 full-scan fallback

Implementation references:
- `ai/search_mcts.py`
- `ai/move_selector.py`
- `ai/policy_value.py`
- `engine/renju_rules.py`

---

## 7. Policy-Value network

입력:
- 3채널 텐서 `[3, H, W]`
  - 흑 돌 plane
  - 백 돌 plane
  - to-play plane(현재 둘 차례를 상수 plane으로 표시)

출력:
- policy logits: 길이 `H*W`
- value: 스칼라(`[-1, +1]` tanh)

학습:
- policy: soft pi에 대해 `-(pi * log_softmax)` 형태(KL)
- value: MSE

Implementation references:
- `ai/pv_model.py`
- `ai/dataset.py`
- `train_pv.py`

---

## 부록: 실제 동작 요약 (2025-12)

이 섹션은 코드 기준 실제 동작을 요약합니다.

### Minimax (iterative deepening)
- 후보 생성은 Manhattan/Euclidean 반경(기본 2)에서 시작하고, proximity + run-endpoint 보강 + 패턴 델타로 정렬합니다(PV priors는 루트에서 선택적으로 혼합).
- 흑 후보는 금수 필터링을 적용하며, 모두 금수이면 full-scan으로 보드 전체를 확인합니다(`candidate_limit` 적용).
- 루트에서 즉시 승/차단을 전수 스캔하여 전술 누락을 줄입니다.
- VCF는 루트에서만 사용되는 빠른 휴리스틱 탐색입니다(기본 max depth 4).
- PV policy는 루트 move ordering에만 사용되며, PV value는 리프 평가에 고정 스케일로 혼합합니다.

### PUCT MCTS
- 롤아웃은 사용하지 않으며, 리프 값은 PV value가 있으면 사용하고 없으면 heuristic tanh 값을 사용합니다.
- priors는 PV policy가 있으면 사용하고, 없으면 패턴 델타 기반 heuristic priors를 사용합니다.
- 후보 누락을 막기 위해 즉시 승/차단을 full-scan으로 확인합니다.
- 흑 금수 필터링 후 후보가 없으면 합법 수 전체에서 fallback합니다.
- 루트 Dirichlet noise는 `dirichlet_alpha`와 `dirichlet_frac`가 모두 > 0이고 후보가 2개 이상일 때만 적용됩니다.
