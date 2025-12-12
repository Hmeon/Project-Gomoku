# Training Guide (Self-play -> PV -> Auto-train)

이 문서는 `Battle_Omok_AI`의 학습 파이프라인을 **재현 가능한 절차**로 정리합니다.  
최종 보고서/논문에서 "학습이 강화되고 있음"을 보여주려면 `logging.md`의 지표/로그 섹션을 함께 사용하세요.

---

## 1. 핵심 개념

- **self-play 데이터**: 상태(board)와 정답(label)을 사람이 주는 대신, AI가 스스로 두며 데이터를 생성합니다.
- **PV(Policy-Value) 네트워크**:
  - Policy: 다음 수 분포(pi)
  - Value: 현재 상태의 승률/기대 결과
- **MCTS 기반 학습**:
  - MCTS 루트 방문 분포를 soft pi로 저장하면 학습 신호가 풍부해집니다.

---

## 2. 권장 학습 흐름

### 2.1 빠른 스모크(파이프라인이 도는지 확인)

```bash
cd Battle_Omok_AI
python auto_train.py \
  --iterations 1 \
  --games 10 \
  --timeout 1.0 \
  --candidate-limit 24 \
  --epsilon 0 \
  --epochs 1 \
  --device cpu \
  --search-backend mcts \
  --rollout-limit 128 \
  --eval-games 0 \
  --seed 0
```

### 2.2 본 학습(CPU 기준 권장)

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

---

## 3. 하이퍼파라미터 해설(중요한 것만)

### 3.1 self-play 품질(탐색)

- `--rollout-limit` (MCTS)
  - 증가: pi 품질 증가, 비용 증가
  - CPU 학습은 128~512 범위에서 타협
- `--candidate-limit`
  - 너무 낮으면 전술적인 수가 누락될 수 있음(특히 중후반)
  - 12~24 권장(초기 24 추천)
- `--dirichlet-alpha`, `--dirichlet-frac`, `--temperature`
  - self-play 다양성 확보용(평가에서는 noise=0, temperature=0(또는 매우 작은 값) 권장)
- `--epsilon`
  - MCTS를 쓰는 경우 보통 0을 권장(soft pi 기록을 유지하기 위함)

### 3.2 학습 안정성

- `--replay-window`
  - 최근 N개 self-play 파일을 섞어서 학습(너무 작으면 편향, 너무 크면 과거 데이터 과다)
  - 3~5 권장
- `--augment/--no-augment`
  - 대칭(회전/반사) 증강을 켜면 표본 효율 증가

---

## 4. 체크포인트 운영

- `--checkpoint checkpoints/pv_latest.pt`:
  - incumbent(현재 모델)
- auto-train은 iteration마다:
  - candidate를 만들고
  - 평가 통과 시 incumbent로 승격하며
  - 기존 incumbent는 `_prev`로 백업합니다.
  - 평가 미통과 시 `_rejected_iter_<i>`로 보관합니다.

주의:
- 체크포인트는 `torch.load`로 읽으므로, 외부 파일을 로드할 때는 출처를 확인하세요.

---

## 5. DirectML(dml) 사용 시 팁(AMD GPU)

- `--device dml`을 사용하면 `torch-directml` 환경에서 학습/추론이 가능합니다.
- 일부 환경에서 BatchNorm이 불안정할 수 있어 학습 스크립트가 완화 옵션을 적용합니다.
- Windows에서는 `--train-workers 0`이 가장 안전합니다.
