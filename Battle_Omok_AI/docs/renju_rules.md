# Renju Rules (Implemented)

본 프로젝트는 "Renju 규칙"을 기준으로 오목을 구현합니다.  
Renju는 흑(선공)에게만 금수를 부여하여 선공 우위를 완화하는 규칙 체계입니다.

---

## 1. 승리 조건

- **흑(Black, -1):** 정확히 5목(exact five)일 때만 승리  
  - 6목 이상(장목, overline)은 승리가 아니라 **금수**입니다.
- **백(White, +1):** 5목 이상이면 승리

Implementation references:
- `Board.has_exact_five`, `Board.has_five_or_more` (`Board.py`)
- `engine/renju_rules.py`

---

## 2. 흑 금수(Forbidden Moves)

### 2.1 장목(Overline)

- 흑이 돌을 두었을 때 어떤 방향이든 연속된 돌의 최대 길이가 6 이상이면 금수.

### 2.2 3-3 (Double-three)

- 한 수로 인해 "open three" 형태가 **둘 이상** 동시에 만들어지면 금수.
- 구현에서는 방향성별 line을 추출하고, "연속 run + 양쪽이 열려 있는지"를 기반으로 open-three를 판정합니다.

### 2.3 4-4 (Double-four)

- 한 수로 인해 "four threat(다음 수로 exact five가 되는 위협)"가 **둘 이상** 동시에 만들어지면 금수.

주의:
- 실제 Renju 규칙의 모든 세부 정의는 매우 복잡할 수 있으며, 본 구현은 테스트 케이스를 기반으로 실용적으로 일치하도록 설계되었습니다.

Implementation reference:
- `engine/renju_rules.py`

---

## 3. 우선순위: 승리 vs 금수

흑은 **정확히 5목을 완성하면 금수보다 우선하여 합법 승리**로 처리됩니다.  
즉, 어떤 패턴이 동시에 생기더라도 해당 수가 exact five라면 금수로 처리하지 않습니다.

Implementation reference:
- `engine/renju_rules.py` 내부의 "five has priority over fouls" 로직

---

## 부록: 구현 메모 (2025-12)

- 금수 판정은 흑에게만 적용되며, 백은 금수의 제약을 받지 않습니다.
- 장목은 연속 길이 > 5로 판정하고, 흑의 exact-five 승리는 금수보다 우선합니다.
- open-three/open-four는 방향성 line 기반 run 분석으로 판정하며, 실전 적합성을 우선한 구현입니다.

