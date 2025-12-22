# Patch Notes / Changelog - Project-Gomoku (Battle_Omok_AI)

이 문서는 `Battle_Omok_AI/` 핵심 구현의 주요 변경 사항을 **재현/검증 가능한 수준**으로 정리합니다.  
실험/논문 작성 시에는 `PROJECT_REPORT.md`와 함께 참고하세요.

---

## [1.1.2] (2025-12-23) - Search quality + docs cleanup

### Added

- MCTS에서 PV가 없을 때 heuristic priors/value로 탐색을 보강하는 fallback 로직
- Minimax 루트에서 즉시 승/차단 전수 스캔 가드레일

### Changed

- 기본 `candidate_limit`을 20으로 조정(settings/self-play/플레이어 기본값)
- 문서 전반 정리 및 기본값/실제 동작 설명 업데이트

### Tests

- Not run (docs + heuristic changes only)

---

## [1.1.1] (2025-12-12) - Research-grade logging & robustness

### Added

- `auto_train.py` 평가 지표 누적 CSV 로깅:
  - `Battle_Omok_AI/logs/eval_metrics.csv`에 iteration별 `score_rate`, `decision(promoted/rejected)`를 기록
  - 평가 원본 요약은 `Battle_Omok_AI/logs/eval_iter_<i>_stats.json`에 저장

- 문서 정리(논문 수준):
  - 루트 `README.md`, `Battle_Omok_AI/README.md`, `PROJECT_REPORT.md` 전면 개편

### Changed

- PV 체크포인트 로딩 안전성 강화:
  - `main.py` / `selfplay.py`에서 PV 로드 실패 또는 `board_size` 불일치 시 크래시 대신 경고 후 PV 비활성화

- DirectML 옵션 판정 정확화:
  - `train_pv.py`에서 "요청 디바이스"가 아닌 "실제 선택된 디바이스" 기준으로 BatchNorm/drop_last 동작을 결정

- GUI 렌더 루프 최적화:
  - `pygame_view.py`에서 입력 대기 루프 시 `display.flip()` 2회 호출을 1회로 감소(기본 API는 backward-compatible)

### Fixed

- `train_pv.py`에서 `drop_last` 사용 시 loss 평균이 `len(ds)` 기준으로 왜곡되던 문제를 실제 처리 샘플 수 기준으로 수정
- `ai/search_mcts.py`에서 노드 확장 시 PV priors 정규화 누락을 보완
- `ai/dataset.py`의 bad JSON 경고 출력 방식을 `print` -> `logging` 기반으로 변경
- `selfplay.py`의 `sys.path` 조작 제거(패키지/스크립트 실행 모두 호환되도록 import fallback으로 정리)

### Tests

- `pytest Battle_Omok_AI/tests` -> **34 passed**

---

## [1.1.0] (2025-12-12) - Renju correctness + search/training stabilization

### Added

- 보드 시뮬레이션 API:
  - `Board._push_stone` / `Board._pop_stone` (검색/self-play 보드 오염 방지)
- self-play 안전장치:
  - timeout/invalid/foul 발생 시 스냅샷 기반 복구 + 합법 수 fallback
  - MCTS에서 얻은 soft pi를 JSONL에 기록(`return_pi=True`)
- PV 모델 학습 파이프라인:
  - `train_pv.py`에서 soft pi 기반 loss(정책 KL 형태)로 학습

### Changed

- Renju 금수 판정(흑 전용) 로직 개선:
  - open-three/open-four/four 카운트의 중복/오탐을 줄이기 위한 run 기반 분석
  - exact-five가 금수보다 우선(흑은 정확히 5로 승리 가능)

- MCTS(PUCT) 안정화:
  - negamax 백업(부호 반전) / PUCT Q 부호 처리 정정
  - 루트 priors 정규화 + Dirichlet noise 옵션 적용
  - full-scan legal fallback(모든 로컬 후보가 금수인 희귀 케이스 방어)

- Minimax 안정화:
  - 흑 금수 필터 + full-scan legal fallback
  - TT/해시 관리 및 보드 undo 일관성 강화

### Tests

- 핵심 규칙/검색/self-play/loader 중심의 pytest 유닛 테스트 추가 및 통과

---

## Notes

- `--enable-vcf`는 minimax에서만 실질 영향이 큽니다(MCTS에는 영향이 거의 없음).
- 학습 강화 지표는 `auto_train.py --eval-games > 0`에서 생성되는 `logs/eval_metrics.csv`의 `score_rate`를 기준으로 추적하는 것을 권장합니다.
