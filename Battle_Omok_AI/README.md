# Battle_Omok_AI: Renju Gomoku AI (Play + Self-Play + PV Training)

`Battle_Omok_AI/`는 이 저장소의 **핵심 구현 디렉터리**입니다.  
기본 설정은 `config/settings.yaml`에서 관리하며, CLI 옵션이 이를 덮어씁니다.

---

## 빠른 실행(요약)

| 목적 | 명령 |
| --- | --- |
| GUI 대국 | `cd Battle_Omok_AI; python main.py --mode human-vs-ai --gui --timeout 5` |
| AI vs AI 관전 | `cd Battle_Omok_AI; python main.py --mode ai-vs-ai --gui` |
| self-play 생성(MCTS) | `cd Battle_Omok_AI; python selfplay.py --games 10 --board-size 15 --timeout 1.5 --search-backend mcts --rollout-limit 256 --output logs/selfplay_renju.jsonl` |
| PV 학습 | `cd Battle_Omok_AI; python train_pv.py --data logs/selfplay_renju.jsonl --epochs 4 --device cpu --output checkpoints/pv_latest.pt` |

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
- PV 체크포인트의 `board_size`가 현재 보드 크기와 다르면 **자동으로 PV를 비활성화**하고 경고만 출력합니다.

---

## 3) Search Backend 옵션 가이드

이 프로젝트는 self-play 및 플레이에서 `minimax`와 `mcts`를 선택할 수 있습니다.

### Minimax(`--search-backend minimax`)

| 옵션 | 설명 | 권장/비고 |
| --- | --- | --- |
| `--depth` | 탐색 깊이(Iterative Deepening의 최대 깊이) | 기본 3 |
| `--candidate-limit` | 후보 수 제한 | 기본 20 |
| `--enable-vcf` | VCF(연속 4 위협 기반 강제승 탐색) | 느려지지만 전술 강화 |

부가: `--collect-stats`는 minimax 플레이어의 노드/시간/NPS 요약에 사용됩니다.

### MCTS(`--search-backend mcts`)

| 옵션 | 설명 | 권장/비고 |
| --- | --- | --- |
| `--rollout-limit` | 확장/백업 횟수 | 128~512 범위 |
| `--explore` | PUCT 상수 `c_puct` | 기본 1.4 |
| `--dirichlet-alpha` | 루트 priors 노이즈 알파 | 학습용(평가 시 0 권장) |
| `--dirichlet-frac` | 루트 priors 노이즈 비율 | 학습용(평가 시 0 권장) |
| `--temperature` | 루트 방문 분포 샘플링 온도 | 학습용 > 0, 평가용 ~0 |
| `--candidate-limit` | 후보 수 제한 | 보통 16~24, 기본 20 |

참고:
- `--enable-vcf`는 MCTS 백엔드에서는 실질 영향이 없습니다.
- `--epsilon`(self-play의 무작위 착수)은 MCTS의 soft pi 기록을 깨뜨릴 수 있어 보통 0을 권장합니다.

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
  --candidate-limit 20 \
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

---

## 구현 메모 (2025-12)

- Minimax는 루트에서 즉시 승/차단을 전수 스캔하여 전술 누락을 줄입니다.
- PV가 없을 때 MCTS는 heuristic priors와 tanh value 추정으로 탐색을 보강합니다.
- 후보 수 생성은 Manhattan/Euclidean 반경 + run-endpoint 보강을 사용합니다.
- PV 체크포인트 `board_size` 불일치 시 PV는 자동 비활성화됩니다.
- self-play는 MCTS가 제공한 `last_pi`가 있을 때만 soft pi를 기록하고, 그 외는 one-hot입니다.
