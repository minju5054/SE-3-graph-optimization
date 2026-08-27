# Action Chunk Graph Optimization — Toy Study

목표: old/new action chunk를 단순 point-wise blending하는 방법과
SE(2)/SE(3) Lie-group residual 기반 graph optimization을 비교합니다.

이 저장소는 첫 proof-of-concept용입니다.
GPU는 필요하지 않으며 SciPy CPU 최적화만 사용합니다.

## Environment

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

## Run

```bash
python scripts/run_toy_experiment.py
```

결과:
- `outputs/toy_comparison.png`
- `outputs/metrics.csv`

현재 실험은 최종 VLA 성능 검증이 아니라,
multi-objective trajectory reconciliation이 의도대로 작동하는지 보는 sanity check입니다.

## Experiment 01 — Orientation Wrap

`179°`에서 `-179°`로 회전할 때 raw Euclidean angle interpolation과
shortest-angle/SE(2) Log interpolation을 비교합니다.

```bash
python scripts/exp01_orientation_wrap.py
```

결과:

- `outputs/exp01_orientation/figure.png`
- `outputs/exp01_orientation/metrics.csv`

이 실험은 raw angle subtraction이 경계에서 `-358°` 회전으로 해석되는 반면,
angle wrapping과 `Log(T_old^-1 T_new)`는 물리적으로 짧은 `+2°` 회전을
선택하는지 검증합니다.

## Experiment 02 — Smoothness Baselines

충돌 요인이 없는 비대칭 old/new trajectory에서 다음 stitching 방법을 비교합니다.

- Linear crossfade
- Cubic Hermite crossfade
- SE(2) graph optimization without collision factors

```bash
python scripts/exp02_smoothness.py
```

결과:

- `outputs/exp02_smoothness/figure.png`
- `outputs/exp02_smoothness/metrics.csv`

Cubic Hermite baseline은 `h(a) = 3a^2 - 2a^3` blend weight를 사용하므로
연속시간에서 시작과 끝의 blend-weight derivative가 0입니다. 실험은 jerk,
rotation increment, body-motion smoothness뿐 아니라 old/new deviation과
양쪽 경계의 velocity mismatch도 함께 기록합니다. 각 방법이 모든 지표에서
우세하다고 가정하지 않고 smoothness 정의 사이의 trade-off를 비교합니다.

## Experiment 03 — Collision Ablation

seed 42로 생성한 12개의 collision stress scenario에서 다음 방법을 비교합니다.

- Linear crossfade
- Cubic Hermite crossfade
- SE(2) graph without collision factors
- SE(2) graph with collision factors

```bash
python scripts/exp03_collision.py
```

결과:

- `outputs/exp03_collision/figure.png`
- `outputs/exp03_collision/metrics.csv`
- `outputs/exp03_collision/per_scenario.csv`

OLD와 NEW proposal은 모든 scenario에서 safety boundary 바깥에 있도록 생성됩니다.
이 suite는 일반 환경의 collision 빈도를 추정하는 데이터가 아니라, 안전한 두
proposal을 섞을 때 충돌이 생기는 경우를 의도적으로 모은 stress test입니다.

Safety 평가는 pose sample뿐 아니라 인접 pose 사이의 line segment까지 검사합니다.
`polyline_collision_rate`와 `polyline_min_clearance`를 주 결과로 사용하며,
sample-only 지표는 discretization 차이를 확인할 수 있도록 함께 저장합니다.

현재 결과에서 collision-aware graph만 모든 실제 장애물 충돌을 피하지만,
soft collision penalty이므로 safety margin을 완전히 만족하지는 않습니다.
또한 collision 회피 과정에서 jerk와 optimization runtime이 증가하는 trade-off가
나타납니다.

## Experiment 04 — Collision-Weight Sweep

Exp03과 동일한 12개 stress scenario에서 collision penalty weight만 변경합니다.

```text
lambda_collision = [0, 10, 30, 100, 300, 1200, 5000, 20000]
```

```bash
python scripts/exp04_collision_weight_sweep.py
```

결과:

- `outputs/exp04_collision_weight/figure.png`
- `outputs/exp04_collision_weight/metrics.csv`
- `outputs/exp04_collision_weight/per_scenario.csv`

현재 suite에서는 `lambda_collision=30`부터 모든 polyline collision이 제거됩니다.
그러나 가중치를 20000까지 높여도 polyline safety margin violation은 완전히
사라지지 않습니다. Pose sample의 violation은 계속 감소하지만 인접 pose 사이의
line segment violation은 일정 수준에서 포화됩니다. 이는 node-only collision
factor의 discretization 한계를 보여줍니다.

가중치가 커질수록 jerk, function evaluation 수, runtime이 증가합니다.
특히 20000에서는 120회 evaluation 제한 내 optimizer success rate가 낮아지므로,
가중치를 계속 키우는 방법보다 segment-aware factor 또는 hard constraint를
검토해야 합니다.

## Experiment 05 — Segment-Aware Collision Factor

기존 node-only collision factor와 각 trajectory segment의 장애물 최근접 거리를
사용하는 segment-aware factor를 비교합니다. 기존 실험 재현성을 위해
`GraphConfig.collision_factor`의 기본값은 `"nodes"`입니다.

```bash
python scripts/exp05_segment_collision_factor.py
```

결과:

- `outputs/exp05_segment_collision/figure.png`
- `outputs/exp05_segment_collision/metrics.csv`
- `outputs/exp05_segment_collision/per_scenario.csv`

동일한 12개 stress scenario와
`lambda_collision = [30, 100, 300, 1200]`을 사용합니다. `lambda_collision=1200`
에서 segment-aware factor는 node-only factor보다 평균 polyline safety-margin
violation과 sample/polyline discretization gap을 크게 줄입니다. Segment 최근접점
계산 때문에 runtime은 증가하며, squared soft penalty를 사용하는 한 작은 margin
violation은 여전히 남습니다.

## Experiment 06 — SE(3) Extension

Pose를 `[x, y, z, rotation-vector(3)]`로 표현하고 다음 기능을 검증합니다.

- SO(3) Exp/Log 기반 SE(3) Exp/Log와 relative residual
- Raw rotation-vector blending과 shortest SO(3) geodesic blending
- SE(3) old/new/smoothness factor
- 구형 장애물에 대한 segment-aware collision factor

```bash
python scripts/exp06_se3_extension.py
```

결과:

- `outputs/exp06_se3_extension/figure.png`
- `outputs/exp06_se3_extension/metrics.csv`
- `outputs/exp06_se3_extension/geometry_validation.csv`

`170°`와 `-170°` rotation vector의 midpoint에서 raw Euclidean blending은 양쪽
orientation으로부터 각각 `170°` 떨어진 자세를 생성하지만, SO(3) geodesic은
각각 `10°` 떨어진 shortest-rotation midpoint를 생성합니다.

3D collision stress case에서는 collision factor가 없는 SE(3) graph가 구형
장애물과 충돌하고, segment-aware factor를 사용한 graph는 실제 polyline 충돌을
피합니다. Squared soft penalty이므로 작은 safety-margin violation은 남습니다.

현재 구현은 9-pose SciPy finite-difference proof-of-concept입니다. Rotation-vector
optimization의 chart boundary, 수치 Jacobian 비용, 실시간 성능은 아직 해결된
것으로 간주하지 않습니다. 실제 VLA 통합 전에 analytic/autodiff Jacobian 또는
검증된 manifold optimization library를 평가해야 합니다.

## Experiment 07 — Async Action-Chunk Update

VLA inference를 직접 실행하는 대신 SE(2) toy chunk로 다음 execution timeline을
명시적으로 검증합니다.

```text
observation -> inference 동안 OLD 계속 실행 -> NEW ready -> committed action
            -> modification point부터 7-pose local transition -> raw NEW suffix
```

```bash
python scripts/exp07_async_action_update.py
```

주요 parameter는 `dt=0.1`, `observation_step=8`,
`latency_steps=[0, 2, 4, 6]`, `commit_horizon_steps=1`,
`optimization_window_poses=7`, `seed=42`입니다. NEW local index는
`global_step - observation_step`으로 명시적으로 정렬합니다. 비교 방법은
Continue OLD, Hard switch, Local cubic Hermite, Local SE(2) graph입니다. Graph의
terminal NEW anchor는 이 실험에서만 활성화되며 기존 실험의 기본값은 0입니다.

결과:

- `outputs/exp07_async_action_update/figure.png`
- `outputs/exp07_async_action_update/metrics.csv`

CSV에는 committed-prefix position/rotation error, transition 양쪽의 jump와 velocity
mismatch, jerk·rotation·body-motion smoothness, aligned NEW deviation, 최종 NEW pose
error, runtime과 optimizer diagnostics를 기록합니다. 현재 결과에서 latency가
0.0 s에서 0.6 s로 증가하면 modification step은 9에서 15로 정확히 이동합니다.
Hard-switch position jump는 0.0044 m에서 0.1862 m로, rotation jump는 0.0652 rad에서
0.3475 rad로 증가했습니다. Hermite와 graph는 start pose jump를 모두 0으로
유지했지만, 이 scenario에서는 Hermite가 graph보다 start/end velocity mismatch와
jerk가 낮았습니다. Graph는 Hermite보다 aligned NEW position deviation이 약간
작았고 runtime은 약 8.4--8.7 ms로 100 ms control period보다 짧았습니다. 모든
method/latency에서 modification 이전 committed-prefix error는 정확히 0이었습니다.
Terminal position anchor만으로 NEW suffix 경계의 velocity continuity가 보장되지는
않으므로 graph가 모든 smoothness metric에서 우세하다고 해석하지 않습니다.

## Experiment 08 — Inference-Time Behavior

관측 시점에 OLD future path 위의 synthetic obstacle을 새로 알게 되었다고 가정하고,
inference 동안 `continue_old`와 `hold_pose`를 비교합니다. Perception, braking
controller 또는 실제 VLA는 구현하지 않습니다. NEW가 ready된 뒤에는 두 policy
모두 동일한 segment-aware Local SE(2) graph transition을 사용합니다.

```bash
python scripts/exp08_inference_behavior.py
```

주요 parameter는 `dt=0.1`, `observation_step=8`,
`latency_steps=[0, 2, 4, 6, 8]`, `commit_horizon_steps=0`,
`optimization_window_poses=7`, `seed=42`입니다.

결과:

- `outputs/exp08_inference_behavior/figure.png`
- `outputs/exp08_inference_behavior/metrics.csv`

`collision_before_new_ready`와 `minimum_clearance_before_new_ready`는 optimizer가
실행되기 전 policy 결과를 분리해 측정합니다. 전체 polyline collision/clearance,
inference 중 이동 거리, smoothness, NEW tracking, optimizer runtime과 diagnostics도
함께 저장합니다. 고정된 현재 scenario에서 `continue_old`는 0.6 s부터 NEW가
ready되기 전에 충돌했고, `hold_pose`는 0.8 s까지 충돌 없이 observation pose의
0.6533 m clearance를 유지했습니다. Hold의 inference 중 이동 거리는 0인 반면
continue-old는 latency 0.2--0.8 s에서 0.267--1.067 m 진행했습니다. Hold의
modification 이후 평균 NEW position deviation은 0.0029 m에서 0.1854 m로
증가했습니다. Continue-old의 충돌 case는 optimizer가 성공을 보고해도 이미
실행된 fixed prefix를 복구할 수 없었으며 full trajectory collision도 남았습니다.
이는 optimizer 결과와 inference-time safety decision을 별도로 해석해야 함을
보여주는 toy counterexample입니다.

두 실험은 실제 VLA 성능이나 최적 gating threshold를 검증하지 않습니다. 현재
결과가 동기를 제공하는 future decision structure는 다음과 같습니다.

```text
OLD safe + OLD/NEW difference small
    -> continue OLD

moderate OLD/NEW disagreement
    -> local reconciliation

OLD unsafe or disagreement too large
    -> do not blindly stitch
       hold / fallback / replan
```

Threshold의 최적성은 현재 결과로 주장하지 않습니다.

## Tests

```bash
python -m unittest discover -s tests -v
```
