# Logging & Evaluation (Paper-ready)

학습이 "강화되고 있다"는 것을 논문/보고서에서 설득력 있게 제시하려면:

1) **평가 지표(score_rate) 커브**  
2) **self-play 품질 지표(승/무/수, foul/timeout/invalid 비율)**  
3) **학습 안정성 지표(loss)**  

를 함께 제시하는 것이 안전합니다.

---

## 1. 로그 파일 위치

기본적으로 다음 디렉터리에 로그가 생성됩니다.

- `Battle_Omok_AI/logs/`

`auto_train.py`는 실행 위치와 무관하게 위 경로에 로그를 남기도록 설계되어 있습니다.

---

## 2. auto_train 로그

### 2.1 Self-play metrics (`selfplay_metrics.csv`)

생성: `auto_train.py`의 각 iteration self-play가 끝날 때 갱신됩니다.

컬럼:
- `timestamp`
- `iteration`
- `games`
- `black_win_rate`
- `white_win_rate`
- `draw_rate`
- `avg_steps`
- `avg_black_fouls`

용도:
- 학습 진행에 따라 승률이 한쪽으로 치우치는지
- 평균 수(게임 길이)가 지나치게 짧아지지 않는지(조기 붕괴 여부)
- 흑 금수 비율이 높아지는지(규칙 위반 경향) 등을 점검

### 2.2 Training metrics (`train_metrics.csv`)

생성: `train_pv.py`의 epoch이 끝날 때 갱신됩니다.

컬럼:
- `timestamp`
- `epoch`
- `total_loss`
- `policy_loss`
- `value_loss`

용도:
- 학습이 발산하지 않는지
- policy/value의 균형이 무너지지 않는지 확인

### 2.3 Evaluation metrics (`eval_metrics.csv`) - 핵심 강화 지표

생성: `auto_train.py`가 iteration마다 승격/거부를 결정할 때 갱신됩니다.

컬럼:
- `timestamp`
- `iteration`
- `games`
- `candidate_wins`
- `candidate_draws`
- `score_rate`
- `threshold`
- `decision` (`promoted` / `rejected` / `bootstrap` / `promoted_no_eval`)
- `eval_stats_file`
- `candidate_ckpt`
- `incumbent_ckpt` (존재하면 `_prev` 경로)

정의:
- `score_rate = (wins + 0.5 * draws) / games`
- `score_rate >= threshold`이면 승격

권장 그래프:
- iteration별 `score_rate` 라인 플롯
- 승격/거부를 색으로 표시

### 2.4 Raw evaluation stats (`eval_iter_<i>_stats.json`)

`selfplay.py --stats-only`의 출력 요약입니다.  
필요하면 여기서 더 자세한 통계를 가져와 논문 표/그림으로 확장할 수 있습니다.

---

## 3. selfplay 단독 실행 로그

`selfplay.py`는 `--output foo.jsonl`로 실행하면 항상 `foo_stats.json`를 함께 생성합니다.

예:
- `logs/selfplay_renju.jsonl`
- `logs/selfplay_renju_stats.json`

요약 키(대표):
- `games`, `black_wins`, `white_wins`, `draws`, `avg_steps`
- `black_timeouts`, `white_timeouts`
- `black_fouls`, `white_fouls`
- `black_invalid_moves`, `white_invalid_moves`
- `agent_wins`, `agent_draws`, `agent_points`

---

## 4. (논문용) 최소 보고 세트

최종 보고서/논문에는 최소 아래 3개를 첨부하는 것을 권장합니다.

1) `logs/eval_metrics.csv`의 iteration별 score_rate 커브  
2) `logs/selfplay_metrics.csv`의 avg_steps 및 foul/timeout 추세  
3) `logs/train_metrics.csv`의 loss 커브  

---

## 5. 분석 예시(Pandas)

```python
import pandas as pd

eval_df = pd.read_csv("Battle_Omok_AI/logs/eval_metrics.csv")
selfplay_df = pd.read_csv("Battle_Omok_AI/logs/selfplay_metrics.csv")
train_df = pd.read_csv("Battle_Omok_AI/logs/train_metrics.csv")

print(eval_df.tail())
```
