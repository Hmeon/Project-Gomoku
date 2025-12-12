# Battle_Omok_AI: Renju Gomoku AI (Play + Self-Play + PV Training)

`Battle_Omok_AI/`는 이 저장소의 **핵심 구현 디렉터리**입니다.

- 플레이(대국): `main.py` + `Omokgame.py`
- Renju 규칙(금수/승리 판정): `engine/`
- 검색 AI: `ai/search_minimax.py`, `ai/search_mcts.py`
- Self-play/학습 파이프라인: `selfplay.py`, `train_pv.py`, `auto_train.py`
- GUI: `gui/pygame_view.py`

---

## 1) 설치

```bash
cd Battle_Omok_AI
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# (선택) AMD DirectML (Windows)
pip install torch-directml
```

---

## 2) 플레이(대국 실행)

### GUI (추천)

```bash
cd Battle_Omok_AI
python main.py --mode human-vs-ai --gui --timeout 5
```

### AI vs AI 관전

```bash
cd Battle_Omok_AI
python main.py --mode ai-vs-ai --gui
```

### MCTS 백엔드(연구/학습 타깃용)

```bash
cd Battle_Omok_AI
python main.py --mode ai-vs-ai --gui --search-backend mcts --rollout-limit 512 --explore 1.4
```

### PV 체크포인트 로드(선택)

```bash
cd Battle_Omok_AI
python main.py --mode ai-vs-ai --gui --search-backend mcts --pv-checkpoint checkpoints/pv_latest.pt --pv-device cpu
```

주의:
- PV 체크포인트의 `board_size`가 현재 보드 크기와 다르면 **자동으로 PV를 비활성화**하고 경고만 출력합니다(크래시 방지).

---

## 3) Search Backend 옵션 가이드

이 프로젝트는 self-play 및 플레이에서 `minimax`와 `mcts`를 선택할 수 있습니다.

### Minimax(`--search-backend minimax`)

- 주요 옵션
  - `--depth`: 탐색 깊이 (Iterative Deepening의 최대 깊이)
  - `--candidate-limit`: 후보 수 제한
  - `--enable-vcf`: VCF(연속 4 위협 기반 강제승 탐색) 사용(속도 감소, 전술 증가)
- 부가: `--collect-stats`는 minimax 플레이어의 노드/시간/NPS 요약에 사용됩니다.

### MCTS(`--search-backend mcts`)

- 주요 옵션
  - `--rollout-limit`: 플라이아웃(확장/백업) 횟수
  - `--explore`: PUCT 상수 `c_puct`
  - `--dirichlet-alpha`, `--dirichlet-frac`: 루트 priors에 Dirichlet noise 주입(탐색 증가)
  - `--temperature`: 루트 방문 분포 샘플링 온도(학습용 다양성 증가 / 평가용 결정성 감소)
  - `--candidate-limit`: 후보 수 제한(너무 낮으면 탐색이 막힐 수 있음; 보통 12~24 권장)
- 참고
  - `--enable-vcf`는 MCTS 백엔드에서는 실질 영향이 없습니다.
  - `--epsilon`(self-play에서의 무작위 착수)은 MCTS의 soft pi 기록을 깨뜨릴 수 있어 보통 0을 권장합니다.

---

## 4) Self-play 데이터 생성

```bash
cd Battle_Omok_AI
python selfplay.py \
  --games 50 \
  --board-size 15 \
  --timeout 1.5 \
  --search-backend mcts \
  --rollout-limit 256 \
  --output logs/selfplay_renju.jsonl
```

출력:
- `logs/selfplay_renju.jsonl`
- `logs/selfplay_renju_stats.json` (요약 통계)

---

## 5) PV 학습(train_pv.py)

```bash
cd Battle_Omok_AI
python train_pv.py \
  --data logs/selfplay_renju.jsonl \
  --epochs 4 \
  --batch-size 128 \
  --workers 0 \
  --device cpu \
  --output checkpoints/pv_latest.pt
```

로그:
- `logs/train_metrics.csv` (epoch별 loss 기록)

---

## 6) 자동 반복 학습(auto_train.py)

`auto_train.py`는 **self-play -> PV 학습 -> (선택) 평가 -> 승격**을 반복합니다.

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

필수 로그(논문/리포트용):
- `logs/selfplay_metrics.csv`: iteration별 self-play 요약
- `logs/train_metrics.csv`: epoch별 학습 loss
- `logs/eval_metrics.csv`: iteration별 **강화 지표(score_rate)** + 승격/거부 기록
- `logs/eval_iter_<i>_stats.json`: 평가 원본 요약 통계

---

## 7) 테스트

```bash
pytest tests
```

---

## 8) 트러블슈팅

- (Windows PowerShell 5.1) 문서가 깨져 보이면: `Get-Content -Encoding UTF8 README.md` (또는 VS Code/GitHub에서 열기)
- GUI 에셋이 없거나 보드 크기가 15가 아니면, GUI는 자동으로 단색 보드/그리드 렌더링으로 동작합니다.
- Windows에서 `--gui` 없이 human 입력을 쓰면 입력 폴링 방식이 OS에 따라 다릅니다(`Player.py` 참고).
- DirectML 환경에서 batchnorm이 불안정할 수 있어 학습 스크립트에서 자동 완화 로직이 들어가 있습니다(`train_pv.py`).
