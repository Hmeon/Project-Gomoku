# Requirements (Renju Gomoku AI)

본 문서는 `Battle_Omok_AI/`가 만족해야 하는 기능/환경 요구사항을 정리합니다.  
과거의 Gomoku-Pro 오프닝 제약 등은 본 프로젝트 범위가 아닙니다.

---

## 1. Functional Requirements

### 1.1 Rules

- Renju 규칙을 적용한다.
  - 흑 금수: 3-3, 4-4, 장목(6+)
  - 승리: 흑은 exact five, 백은 5+
- 불법 수(범위 밖/중복/금수/시간 초과)는 패배 처리 가능해야 한다.

### 1.2 Play modes

- AI vs AI
- Human vs AI / AI vs Human
- Human vs Human
- GUI 모드에서 마우스 입력으로 착수 가능해야 한다.

### 1.3 Training pipeline

- Self-play로 (board, to_play, pi, value) 샘플을 JSONL로 생성한다.
- PV 네트워크를 self-play 데이터로 학습한다.
- 반복 학습 루프(auto-train)를 제공한다.
- 평가 게이트를 통해 candidate 승격 여부를 결정할 수 있어야 한다.

### 1.4 Logging (paper-ready)

- iteration별 self-play 요약 지표가 CSV로 남아야 한다.
- epoch별 학습 loss 지표가 CSV로 남아야 한다.
- iteration별 평가 score_rate 및 승격/거부 결정이 CSV로 남아야 한다.

---

## 2. Non-functional Requirements

- **Robustness:** timeouts/예외 시에도 보드 상태가 깨지지 않아야 한다.
- **Reproducibility:** seed 고정 및 로그를 통한 결과 재현이 가능해야 한다.
- **Performance:** 후보 수 생성/TT 등을 통해 현실적인 시간 내 대국 및 self-play가 가능해야 한다.

---

## 3. Environment Requirements

- Python 3.10+
- 주요 라이브러리:
  - PyTorch (`torch`, `torchvision`, `torchaudio`)
  - PyYAML
  - pygame
  - pytest(개발/검증)
- (선택) AMD GPU: `torch-directml` (Windows)
