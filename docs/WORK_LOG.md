# Project Work Log

이 문서는 Codex가 완료한 작업을 시간순으로 기록하는 append-only 로그다.
기존 항목을 삭제, 덮어쓰기, 재정렬 또는 수정하지 않고 항상 파일 맨 아래에 새
항목을 추가한다.

## Entry template

### YYYY-MM-DD HH:MM:SS TZ - 작업 제목

- **작업 목적:**
- **변경한 내용:**
- **변경된 주요 파일:**
- **실행한 명령어:**
- **테스트 / 검증 결과:**
- **문제 / 주의사항:**
- **Commit reference:** `SELF (git log -1 -- docs/WORK_LOG.md 로 확인)` 또는 `N/A - 사유`
- **Branch:**
- **Push 결과:**

---

## Entries

### 2026-08-27 13:13:26 KST - Codex 작업 규칙과 작업 로그 초기화

- **작업 목적:** 저장소에서 Codex가 항상 따를 Git·검증·기록 workflow를 정의하고
  append-only 작업 기록을 시작한다.
- **변경한 내용:** 작업 시작 전 상태 확인, 사용자 변경 보호, 요청 범위 제한,
  테스트·빌드 검증, 작업 단위별 단일 commit, Conventional Commit message,
  명시적 stage, 정상 push, 위험한 Git 명령 금지, secret·dataset·artifact 제외,
  종료 보고 규칙을 추가했다. 재사용할 작업 로그 template과 첫 작업 기록을
  생성했다.
- **변경된 주요 파일:** `AGENTS.md`, `docs/WORK_LOG.md`.
- **실행한 명령어:** `git status --short --branch`; `git branch --show-current`;
  `git remote -v`; `find`; `sed`; `.venv/bin/python -m unittest discover -s tests -v`;
  `git diff --check`; `test -f`; `rg`; `date`.
- **테스트 / 검증 결과:** SE(3) geometry·pipeline 단위 테스트 9개가 모두
  통과했다. `git diff --check`가 성공했으며, 두 문서의 존재와 필수 규칙·로그
  필드를 확인했다.
- **문제 / 주의사항:** 작업 시작 시 `main`은 `origin/main`과 동기화된 깨끗한
  상태였다. Commit 자신의 최종 hash와 commit 이후 push 결과는 동일 commit의
  파일 내용에 확정적으로 기록할 수 없으므로 정확한 값은 종료 채팅에서 보고한다.
- **Commit reference:** `SELF (git log -1 -- docs/WORK_LOG.md 로 확인)`.
- **Branch:** `main`.
- **Push 결과:** `origin/main`에 정상 push 예정; 실제 결과는 종료 채팅에서 보고.

### 2026-08-27 16:11:16 KST - Async action-chunk execution protocol 실험

- **작업 목적:** VLA 모델을 추가하지 않고 SE(2) toy execution timeline에서
  observation, inference latency, committed prefix, local transition window 및
  inference-time behavior의 의미를 Exp07/Exp08로 검증한다.
- **변경한 내용:** immutable committed prefix와 observation-relative NEW index
  alignment를 pure function으로 구현했다. SE(2) graph에 기본값 0인 optional
  terminal NEW anchor를 추가했다. Goal-change와 newly-observed-obstacle scenario,
  Exp07의 Continue OLD/Hard switch/Hermite/graph latency sweep, Exp08의
  continue-old/hold-pose safety sweep, CSV metrics와 4-panel figure를 추가했다.
  Execution protocol 단위 테스트 6개와 실제 CSV 관찰을 README에 기록했다.
- **변경된 주요 파일:** `src/action_chunk_graph/execution.py`,
  `src/action_chunk_graph/optimizer.py`, `src/action_chunk_graph/scenarios.py`,
  `scripts/exp07_async_action_update.py`,
  `scripts/exp08_inference_behavior.py`, `tests/test_execution_protocol.py`,
  `README.md`, `docs/WORK_LOG.md`.
- **실행한 명령어:** `git status --short --branch`;
  `git branch --show-current`; `git remote -v`; `find`; `sed`; `tail`; `awk`;
  `.venv/bin/python -m py_compile`; `.venv/bin/python -m unittest discover -s tests -v`;
  `MPLBACKEND=Agg .venv/bin/python scripts/exp01_orientation_wrap.py`부터
  `scripts/exp08_inference_behavior.py`까지 Exp01--Exp08 전체 실행; Pandas 기반
  output/metric invariant 확인; `git diff --check`.
- **테스트 / 검증 결과:** 기존 9개와 신규 6개를 합친 unit test 15개가 모두
  통과했다. Exp01--Exp08 전체 script가 성공했고 Exp07/Exp08의 `metrics.csv`와
  `figure.png`가 생성되었다. Exp07 16개 조합의 committed-prefix position/rotation
  최대 오차는 모두 0이었다. Graph runtime은 8.4--8.7 ms였다. Exp08에서
  continue-old는 latency 0.6 s부터 NEW ready 이전에 충돌했고 hold-pose는 0.8 s까지
  충돌하지 않았다. NEW proposal polyline clearance 0.1755 m는 0.15 m safety
  margin 밖이었다. `py_compile`과 output 존재·크기 검사도 통과했다.
- **문제 / 주의사항:** Exp08의 최초 `max_nfev=200` 설정은 continue-old latency
  0.4 s case에서 evaluation limit에 도달했다. Scenario는 변경하지 않고 segment
  factor용 budget을 600으로 올렸으며 모든 case가 수렴했다. 충돌 prefix에 고정된
  continue-old 0.6 s case runtime은 약 1.1 s였고, optimizer 성공 후에도 이미
  발생한 collision은 남았다. Exp07에서는 Hermite가 graph보다 boundary velocity
  mismatch와 jerk가 낮아 graph의 일관된 smoothness 우위를 주장하지 않는다.
  생성 output은 기존 `.gitignore` 정책에 따라 commit 대상이 아니다.
- **Commit reference:** `SELF (git log -1 -- docs/WORK_LOG.md 로 확인)`.
- **Branch:** `main`.
- **Push 결과:** `origin/main`에 정상 push 예정; 실제 결과는 종료 채팅에서 보고.

### 2026-08-27 17:25:48 KST - Transition window 및 constraint-conditioned reconciliation

- **작업 목적:** H3의 transition reaction-smoothness trade-off와 H4의
  constraint-conditioned graph optimization 필요성을 SE(2) async execution에서
  검증한다.
- **변경한 내용:** Exp09에 `[3, 5, 7, 10, 15]` pose transition-window sweep,
  0.05 m aligned-NEW reaction metric, Hermite/SE(2) graph/Hard-switch reference 및
  CSV·figure를 구현했다. Exp10에 benign regime과 obstacle radius
  `[0.08, 0.14, 0.20, 0.26, 0.32]` m constrained severity suite, Hermite,
  collision-factor 유/무 graph, Hard switch 비교 및 CSV·figure를 구현했다.
  Reusable velocity-boundary/reaction metric과 scenario generator, window/suffix/prefix
  invariant test 3개를 추가하고 README에 실제 결과와 연구 범위를 기록했다.
- **변경된 주요 파일:** `scripts/exp09_transition_window.py`,
  `scripts/exp10_constraint_conditioned_reconciliation.py`,
  `src/action_chunk_graph/metrics.py`, `src/action_chunk_graph/scenarios.py`,
  `tests/test_execution_protocol.py`, `README.md`, `docs/WORK_LOG.md`.
- **실행한 명령어:** `git status --short --branch`;
  `git branch --show-current`; `git remote -v`; `git log --oneline -5`;
  `git fetch origin`; `git rev-parse`; `sed`; `.venv/bin/python -m py_compile`;
  `.venv/bin/python -m unittest discover -s tests -v`;
  `MPLBACKEND=Agg .venv/bin/python scripts/exp07_async_action_update.py`;
  `scripts/exp08_inference_behavior.py`; `scripts/exp09_transition_window.py`;
  `scripts/exp10_constraint_conditioned_reconciliation.py`; Pandas 기반 CSV schema,
  output, committed-prefix invariant 확인; `git diff --check`.
- **테스트 / 검증 결과:** 기존 15개와 신규 3개를 합친 unit test 18개가 모두
  통과했고 Exp07/08 회귀 및 Exp09/10 실행에 성공했다. Exp09 11 rows와 Exp10
  23 rows의 CSV 및 figure가 생성됐고 모든 row의 modification 이전 prefix
  position/rotation error는 0이었다. Exp10의 모든 severity에서 committed prefix와
  NEW proposal은 0.12 m safety margin 밖이었다.
- **정량 결과:** Exp09 Hermite window 3→15에서 reaction 0.2→1.2 s, jerk
  25.13→3.65, mean NEW deviation 0.0065→0.0870 m였다. Graph reaction은
  0.2→1.4 s였지만 jerk는 window 7에서 13.84로 최소 후 15.07로 증가했고 end
  mismatch도 window 5의 0.394에서 window 15의 0.617로 증가했다. Graph runtime은
  약 1.9→90.0 ms였다. Exp10 benign에서 Hermite/graph jerk는 4.24/10.36,
  runtime은 약 0.03/17.76 ms였다. Constrained에서 Hermite는 medium부터,
  collision-factor 없는 graph는 medium-high부터 충돌했으며 segment collision
  graph는 전 severity에서 collision을 피했다.
- **문제 / 주의사항:** H3는 reaction-Hermite smoothness trade-off만 지지하고 긴
  window가 graph의 모든 continuity를 개선한다는 형태는 지지하지 않았다. H4는
  현재 deterministic geometry family에서만 부분 지지된다. Collision graph는 high
  severity에서 0.00075 m safety-margin violation, jerk 47.74와 약 76 ms runtime을
  남겼고 Hard switch도 collision-free지만 0.143 m/0.606 rad pose jump를 만들었다.
  단일 path/obstacle-center family이므로 일반화, 최적 threshold, learned gating,
  실제 VLA/robot 성능은 주장하지 않는다. 생성 output은 `.gitignore`에 따라 commit
  대상이 아니다.
- **Commit reference:** `SELF (git log -1 -- docs/WORK_LOG.md 로 확인)`.
- **Branch:** `main`.
- **Push 결과:** `origin/main`에 정상 push 예정; 실제 결과는 종료 채팅에서 보고.

### 2026-08-29 13:25:16 KST - Context-conditioned multi-update execution decision

- **작업 목적:** H5, 즉 mixed asynchronous regime에서 단일 fixed strategy가 모든
  safety/progress/smoothness/response/compute 축을 지배하지 않으며 OLD-only
  inference decision과 NEW-ready transition cascade를 분리한 context mechanism이
  더 합리적인 trade-off를 만드는지 검증한다.
- **변경한 내용:** NEW나 hidden label을 입력으로 받지 않는 Stage-A
  `continue_old`/`continue_then_hold`/`hold_pose` rule과, NEW-ready 이후 direct,
  Hermite, segment-aware collision graph, validated replan fallback을 고르는 Stage-B
  rule을 추가했다. Seed별 external event schedule은 공유하되 5개 policy가 실제
  executable suffix와 current state를 독립 rollout하는 12 s multi-update episode,
  event/episode metric 집계, summary/decision-count CSV와 6-panel figure를 구현했다.
  Graph success와 실제 polyline collision/margin을 별도로 검증하고 invalid result는
  실행하지 않고 hold한다.
- **정확한 실험 설정:** `dt=0.1`, 120 steps, 31-pose chunk, 30 episodes,
  episode당 4 updates(총 120 external events, policy 평가 row 600개), seed 42--71,
  latency 1--8 steps, commit 1 step, fixed transition window 7 poses, NEW tolerance
  0.05 m이다. Goal delta는 event당 `[-0.85, 0.85]` m, obstacle probability 0.72,
  along-track offset 0.35--1.35 m, lateral offset `[-0.48, 0.48]` m, radius
  0.18--0.34 m, margin 0.12--0.20 m이다. Direct threshold는 0.04 m/0.08 rad,
  graph safety acceptance tolerance는 0.002 m로 사전 고정했다.
- **변경된 주요 파일:** `src/action_chunk_graph/decision.py`,
  `src/action_chunk_graph/multi_update.py`,
  `scripts/exp11_execution_decision.py`, `tests/test_execution_decision.py`,
  `README.md`, `docs/WORK_LOG.md`.
- **실행한 명령어:** `git status --short --branch`; `git branch --show-current`;
  `git remote -v`; `git log --oneline -5`; `git fetch origin`; `sed`; `rg`;
  `.venv/bin/python -m py_compile`; `.venv/bin/python -m unittest discover -s tests -v`;
  `MPLBACKEND=Agg .venv/bin/python scripts/exp07_async_action_update.py`부터
  `scripts/exp10_constraint_conditioned_reconciliation.py`까지 회귀 실행;
  `MPLBACKEND=Agg .venv/bin/python scripts/exp11_execution_decision.py`; Pandas 기반
  CSV schema, causality, prefix, finite metric, collision timing, decision count 확인;
  `file`; image inspection; `git diff --check`.
- **테스트 / 검증 결과:** 기존 18개와 신규 11개를 합친 unit test 29개가 모두
  통과했다. Exp07--Exp10 script 회귀와 Exp11 actual run이 성공했고 figure 및
  event/episode/summary/decision-count CSV가 생성됐다. Exp11의 600 event rows에서
  `new_used_before_new_ready=False`, committed-prefix maximum error 0, 필수 episode
  metric finite invariant를 확인했다. Exp07--Exp10 output도 정상 재생성됐다.
- **정량 결과:** Continue+Hard/Hermite/Graph의 episode collision rate는
  40.0%/56.7%/30.0%, pre-NEW collision event rate는 8.33%/8.33%/5.83%였다.
  Hold+Graph와 Context는 collision 0건이었다. Context Stage A는 continue 64회,
  continue-then-hold 42회, hold 14회였고 Stage B는 Hard 10회, Hermite 71회,
  accepted Graph 20회, replan 19회였다. Context는 fixed Graph의 4.0회 대신 평균
  1.3회 graph를 호출했고 평균 episode compute는 83.6 ms였다(Continue+Graph
  138.0 ms, Hold+Graph 112.1 ms). Hold duration/jerk는 Context 2.29 s/50.66,
  Hold+Graph 3.18 s/94.60이었다.
- **문제 / 제한 / negative result:** Context progress 12.31 m는 Hold+Graph
  12.72 m보다 낮아 H5의 progress 절은 지지되지 않았다. Continue+Hard/Hermite의
  14.08 m progress는 collision과 교환한 결과였다. Context의 19 replan 중 일부는
  observation-relative NEW aligned suffix 자체가 requested margin을 위반한
  transition-infeasible event이고 나머지는 local soft optimizer limitation이므로
  graph failure 하나로 해석하지 않는다. H5는 partially supported이다. 현재
  distribution은 non-overlapping events, known static circle geometry, deterministic
  seeds와 사전 고정 rule을 사용한다. Threshold 최적성, uncertainty robustness,
  dynamic obstacles, real VLA/robot safety 및 SE(3) 일반화는 evidence가 부족하다.
  생성 output은 `.gitignore` 정책에 따라 commit하지 않는다.
- **Commit reference:** `SELF (git log -1 -- docs/WORK_LOG.md 로 확인)`.
- **Branch:** `main`.
- **Push 결과:** `origin/main`에 정상 push 예정; 실제 결과는 종료 채팅에서 보고.
