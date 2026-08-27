# Codex Project Rules

이 파일의 규칙은 사용자가 특정 작업에서 명시적으로 다르게 지시하지 않는 한
이 저장소에서 수행하는 모든 Codex 작업에 적용한다.

## 1. 작업 시작 전 확인

- 모든 작업은 수정 전에 `git status --short --branch`를 실행하는 것으로 시작한다.
- 관련 프로젝트 구조, 현재 branch, remote 설정을 확인하고 작업 범위를 정한다.
- 기존 수정 사항과 untracked 파일은 모두 사용자의 작업으로 간주한다. 요청 범위에
  명백히 포함되지 않는 한 삭제, 덮어쓰기, 되돌리기, reset 또는 stage하지 않는다.
- Git 저장소, 현재 branch 또는 사용할 수 있는 remote가 없다면 임의로 저장소를
  초기화하거나 remote를 추측하지 말고 사용자에게 원인을 보고한다.

## 2. 변경 범위와 안전

- 사용자가 요청한 작업과 직접 관련된 코드, 설정, 문서, 테스트만 수정한다.
- 관련 없는 formatting, refactoring, dependency update, cleanup 또는 생성 파일
  변경을 만들지 않는다.
- 사용자의 기존 설계와 변경을 보존하며, 불가피하게 충돌하면 작업을 중단하고
  사용자에게 확인한다.
- API key, password, token, credential, 개인·비공개 데이터, dataset, build
  artifact, cache, dependency directory 및 대용량 생성 파일을 commit하지 않는다.
- stage 전에 secret과 불필요한 대용량 파일이 포함되지 않았는지 확인한다.

## 3. 수정 후 검증

- 코드 수정 후 가능한 경우 관련 test, build, lint, type-check 또는 실행 검증을
  수행한다.
- 전체 검증이 불가능하면 수행 가능한 가장 강한 부분 검증을 실행하고, 검증하지
  못한 항목과 원인을 명시한다.
- 하나의 논리적 작업 단위가 끝날 때마다 변경한 파일과 결과를 확인한다.

## 4. Append-only 작업 로그

- 하나의 사용자 요청 작업이 끝날 때마다 구현과 검증 결과를
  `docs/WORK_LOG.md`의 맨 아래에 새로운 항목으로 추가한다.
- 기존 로그를 삭제, 덮어쓰기, 재정렬 또는 수정하지 않는다.
- 각 로그에는 최소한 다음 내용을 기록한다.
  - 날짜 및 시간과 timezone
  - 작업 목적
  - 변경한 내용
  - 변경된 주요 파일
  - 실행한 명령어
  - 테스트 또는 검증 결과
  - 발생한 문제, 제한 또는 주의사항
  - commit reference
  - branch 이름
  - push 결과
- 작업 commit은 자신의 최종 hash를 파일 내용으로 포함할 수 없다. 파일 내용이
  바뀌면 hash도 바뀌기 때문이다. 해당 작업과 같은 commit에 포함되는 로그에는
  commit reference로 `SELF (git log -1 -- docs/WORK_LOG.md 로 확인)`를 기록하고,
  정확한 최종 hash는 작업 종료 채팅에서 보고한다.
- 한 번의 작업 commit 안에는 push 이후의 결과를 다시 기록할 수 없으므로 로그에는
  push 대상과 예정 상태를 기록하고, 실제 push 성공 여부는 종료 채팅에서 확정해
  보고한다. 별도의 사용자 요청 없이 로그만 고치기 위한 두 번째 commit은 만들지
  않는다.

## 5. 작업 단위별 commit

- 파일을 수정할 때마다 commit하지 않는다.
- 요청한 하나의 작업 단위를 구현하고 최종 검증과 작업 로그 업데이트까지 마친 후
  한 번의 의미 있는 commit으로 정리한다.
- stage 전에 `git diff`와 `git status --short --branch`를 확인한다.
- 필요한 파일만 명시적인 경로로 stage한다. 관련 없는 변경이 있을 때
  `git add .` 또는 `git add -A` 같은 광범위한 stage 명령을 사용하지 않는다.
- commit 전 `git diff --cached`로 staged diff를 검토한다.
- commit message는 변경 내용에 맞게 다음 Conventional Commit 형식을 우선한다.
  - `feat: ...`
  - `fix: ...`
  - `refactor: ...`
  - `docs: ...`
  - `test: ...`
  - `chore: ...`

## 6. Push와 위험한 Git 명령

- 작업 commit이 성공하면 사용자가 명시적으로 금지하지 않는 한 현재 branch를
  설정된 remote에 일반 push한다.
- `git push --force`, `git push --force-with-lease`, branch 삭제, `git reset`,
  `git rebase`, `git clean` 또는 변경과 history를 버리는 명령은 사용자가 정확한
  작업을 명시적으로 요청하지 않는 한 사용하지 않는다.
- push가 실패하면 remote를 바꾸거나 history를 재작성하는 등 우회하지 말고 즉시
  중단한 뒤 실패 원인과 저장소 상태를 보고한다.

## 7. 기본 workflow

별도의 Git 지시가 없어도 다음 순서를 따른다.

1. 작업 요청 확인
2. 프로젝트 구조와 `git status --short --branch` 확인
3. 요청 범위의 코드 또는 문서 수정
4. 테스트, 빌드 또는 실행 검증
5. `docs/WORK_LOG.md`에 새 로그 append
6. `git diff` 확인
7. `git status --short --branch` 확인
8. 필요한 파일만 명시적으로 stage
9. `git diff --cached` 확인
10. 작업 단위당 한 번 commit
11. 현재 branch를 정상 push
12. 작업 결과 보고

## 8. 작업 종료 보고

작업 종료 시 채팅에서 다음 내용을 간단히 보고한다.

- 무엇을 수정했는지
- 테스트 또는 검증 결과
- 수정한 로그 파일
- commit message
- 정확한 commit hash
- branch 이름
- push 성공 여부
