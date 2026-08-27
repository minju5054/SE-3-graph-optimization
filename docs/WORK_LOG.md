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
