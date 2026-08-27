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
