# Battle Gomoku AI (Renju Rules)

Python implementation of a Renju-rule Gomoku bot:
- Asymmetric wins: Black must make exact five (overline is foul unless another direction makes a five); White wins with 5+.
- Forbidden for Black: 3-3, 4-4, overline (unless five is also made); White has no fouls.
- Minimax + alpha-beta + TT, optional MCTS fallback.
- Pygame GUI (mouse input) with board image for 15x15, procedurally drawn grid for other sizes.

## Quickstart
```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
pip install -r requirements.txt

python main.py --mode ai-vs-ai --board-size 15 --timeout 5 --gui
# PV 자동 로드: `checkpoints/pv_latest.pt`가 존재하면 CPU로 자동 로드되어 탐색에 적용됩니다.
# 필요시 수동 지정: --pv-checkpoint <path> --pv-device cpu
```

Modes: `ai-vs-ai` (default), `human-vs-ai` (human as black), `ai-vs-human` (human as white).  
CLI flags: `--board-size` (15 or 19), `--timeout` seconds per move, `--depth` search depth, `--candidate-limit` move fan-out, `--gui` to enable Pygame.
PV 옵션: `--pv-checkpoint PATH`로 정책/가치 모델을 자동 로드, `--pv-device cpu|cuda`.

### Self-play 데이터 생성
학습용 정책/가치 라벨을 만들려면:
```bash
python selfplay.py --games 50 --board-size 15 --depth 3 --output selfplay_renju.jsonl
```
출력은 각 착수 상태별 `board`, `to_play`, 선택 수의 원-핫 `pi`, `value`(해당 차례 관점)로 구성된 JSONL입니다.
*참고*: 흑이 금수를 제안할 가능성에 대비해, self-play는 금수 점을 자동 건너뛰는 안전장치를 포함합니다.

### 정책/가치 네트워크 학습
자가대국 JSONL로 오프라인 학습:
```bash
pip install -r requirements.txt  # torch 포함
python train_pv.py --data selfplay_renju.jsonl --epochs 5 --output checkpoints/pv_latest.pt
```
모델은 3채널 인코딩(흑/백/턴) 입력, 작은 ResNet 백본에 정책/가치 헤드로 구성됩니다.

### PV를 MCTS에 사용하기 (옵션)
MCTS에 정책 priors/가치 추정을 쓰려면 실행 전 한 번 설정:
```python
from ai import policy_value, search_mcts
search_mcts.PV_HELPER = policy_value.PolicyValueInfer("checkpoints/pv_latest.pt", device="cpu")
```
설정 후 `choose_move` 호출 시 PV 기반 후보 정렬/롤아웃이 활성화됩니다.

### PV를 Minimax에 사용하기 (옵션)
```python
from ai import policy_value, search_minimax
search_minimax.PV_HELPER = policy_value.PolicyValueInfer("checkpoints/pv_latest.pt", device="cpu")
```
설정하면 리프 평가에 PV value를 혼합하고, 루트 후보 정렬에 정책 priors를 반영합니다(PV_SCALE=5000 기준 가중).

### 자동 루프 (자가대국 → 학습 → 모델 적용)
과적합을 피하려고 기본값을 낮게 설정한 간단한 실행:
```bash
cd Battle_Omok_AI
python auto_train.py --games 200 --epochs 3 --board-size 15 --depth 3 --candidate-limit 20 --timeout 5
```
- 단계: self-play JSONL 생성 → PV 학습 → 새 체크포인트를 MCTS/Minimax에 로드.
- 더 강한 모델을 원하면 `--games`, `--epochs`, `--depth`, `--candidate-limit`을 점진적으로 늘리되, 데이터 다양성(흑/백 스왑, 랜덤 시드 변경)도 함께 고려하세요.

자가대국 다양화 옵션(`selfplay.py`):
- `--swap-colors`: 게임마다 흑/백을 교대하여 한쪽 편향 완화
- `--random-open N`: 초반 N수는 랜덤 합법 수 선택
- `--epsilon P`: 각 수마다 P 확률로 랜덤 합법 수 선택 (epsilon-greedy)
`auto_train.py`에서도 위 옵션을 그대로 전달할 수 있습니다.

## Files of Note
- `main.py`: wiring for players, settings, GUI.
- `Omokgame.py`: game loop, timeouts, renderer hook.
- `Board.py`: state + exact-five / 5+ detection helpers.
- `engine/renju_rules.py`: forbidden-move checks (3-3, 4-4, overline) and win logic.
- `ai/search_minimax.py`: alpha-beta with TT and VCF probe.
- `ai/search_mcts.py`: time-bounded MCTS fallback.
- `gui/pygame_view.py`: board render (15x15 asset or drawn grid for 19x19).
- `config/`: default settings and pattern weights.
- `tests/`: pytest coverage for rules, board, timeouts, search.

## Testing
```bash
cd Battle_Omok_AI
pytest
```

## Notes
- For 15x15, the GUI uses `assets/board.jpg`; other sizes use a drawn grid.
- If you add a 19x19 board image, keep the same margin ratio (~23px on 540px) or adjust `asset_margin` accordingly.
- MCTS is lightweight; for stronger play, tune rollout limits or swap in policy/value guidance.
- Opening “Swap” is not implemented; standard Renju fouls apply from move 1.
