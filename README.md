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

## Experiment 09 — Transition Window Ablation

Modification point 이후 OLD→NEW local transition을 몇 pose 동안 적용할지에 따른
reaction-smoothness trade-off를 측정합니다. Exp07의 obstacle-free asynchronous
goal-change scenario와 global/NEW-local index alignment를 그대로 재사용합니다.

```bash
python scripts/exp09_transition_window.py
```

고정 parameter는 `dt=0.1`, `observation_step=8`,
`inference_latency_steps=4`, `commit_horizon_steps=1`, `seed=42`이며,
`new_ready_step=12`, `modification_step=13`입니다. Window는
`[3, 5, 7, 10, 15]` pose, 즉 endpoint 사이 시간으로
`[0.2, 0.4, 0.6, 0.9, 1.4]` s를 사용합니다. OLD/NEW chunk 길이 안에서 15 pose가
유효하므로 clipping하지 않았습니다. Reaction은 modification point 이후 aligned
NEW position error가 처음 0.05 m 이하가 되는 step으로 고정해 측정합니다. Hard
switch는 window가 적용되지 않는 단일 reference row로 기록합니다.

결과:

- `outputs/exp09_transition_window/figure.png`
- `outputs/exp09_transition_window/metrics.csv`

Hermite window가 3에서 15 pose로 증가할 때 reaction time은 0.2 s에서 1.2 s,
mean NEW position deviation은 0.0065 m에서 0.0870 m로 증가한 반면 jerk는
25.13에서 3.65로 단조 감소했습니다. Start velocity mismatch도 0.702에서
0.020으로, end mismatch는 0.654에서 0.152로 감소했습니다. 따라서 Hermite에는
명확한 reaction-smoothness Pareto trade-off가 나타났습니다.

Graph reaction time은 0.2 s에서 1.4 s, mean NEW deviation은 0.0061 m에서
0.0932 m로 증가했습니다. 그러나 jerk는 window 7에서 13.84로 최소가 된 뒤
window 15에서 15.07로 다시 증가했고, end velocity mismatch도 window 5의
0.394에서 window 15의 0.617로 악화됐습니다. Runtime은 약 1.9 ms에서 90.0 ms로
증가했습니다. 따라서 H3는 reaction delay와 Hermite trade-off 측면에서는
지지되지만, 긴 window가 graph의 모든 continuity metric을 개선한다는 형태로는
지지되지 않습니다. 모든 metric에서 우월한 단일 intermediate window도
관찰되지 않았습니다.

## Experiment 10 — Constraint-Conditioned Reconciliation

Graph optimization을 항상 적용해야 하는지, 아니면 active geometric constraint가
있을 때 선택적으로 정당화되는지를 benign/constrained regime으로 나눠 비교합니다.

```bash
python scripts/exp10_constraint_conditioned_reconciliation.py
```

두 regime 모두 `dt=0.1`, `observation_step=8`,
`inference_latency_steps=2`, `commit_horizon_steps=1`,
`modification_step=11`, `transition_window_poses=10`(0.9 s), `seed=42`를
사용합니다. Constrained regime은 obstacle center와 OLD/NEW proposal을 고정하고
radius만 `[0.08, 0.14, 0.20, 0.26, 0.32]` m로 증가시킨 5단계 severity sweep이며
safety margin은 0.12 m입니다. 가장 높은 severity에서도 committed prefix clearance
0.213 m와 NEW proposal clearance 0.135 m로 둘 다 margin 밖이므로 optimizer 실행
이전 collision을 confound로 포함하지 않습니다.

결과:

- `outputs/exp10_constraint_conditioned/figure.png`
- `outputs/exp10_constraint_conditioned/metrics.csv`

Benign regime에서 Hermite는 collision-factor 없는 graph보다 jerk가 낮고
(4.24 vs 10.36), start/end velocity mismatch도 작으며
(0.022/0.123 vs 0.103/0.365), runtime도 짧았습니다(약 0.03 vs 17.76 ms).
Mean NEW position deviation은 0.0226 m와 0.0228 m로 거의 같았습니다. 이 regime은
constraint가 없을 때 graph가 simple transition보다 자동으로 우월하지 않다는
negative result를 재확인합니다.

Constrained regime에서는 Hermite가 medium부터 물리적으로 충돌했고, collision
factor가 없는 graph는 medium-high부터 충돌했습니다. Segment-aware collision
graph는 5단계 모두 물리적 충돌을 피하고 minimum clearance를 약 0.119--0.157 m로
유지했습니다. 다만 squared soft penalty이므로 high severity에서 safety-margin
violation 0.00075 m가 남았고, jerk는 37.30에서 47.74로, runtime은 약
33--95 ms 범위로 증가했습니다. Hard switch도 모든 severity에서 충돌을 피했지만
0.143 m/0.606 rad의 start pose jump와 jerk 40.06을 만들었습니다.

따라서 H4는 이 deterministic scenario family 안에서 부분적으로 지지됩니다.
Constraint-aware graph는 zero start-pose jump를 유지하며 Hermite와 unconstrained
graph가 충돌하는 severity에서도 collision을 회피했지만, hard switch가 유일한
안전 대안은 아니며 graph도 exact safety margin이나 낮은 jerk를 보장하지
않습니다. 현재 결과로 learned gating, 최적 severity threshold, 다양한 geometry에
대한 일반화 또는 실제 robot/VLA 성능은 주장할 수 없습니다.

Exp09/10은 planar robot을 가정하는 연구가 아니라 execution timing, modification
point, transition horizon 및 constraint-conditioned method selection을 isolation한
SE(2) toy study입니다. Mechanism과 boundary continuity가 더 정리된 뒤 별도로
SE(3) validation을 수행해야 합니다.

## Experiment 11 — Context-Conditioned Execution Decision

단일 update가 아니라 12 s episode 안의 반복적인 asynchronous update에서 하나의
fixed policy가 safety, progress, smoothness, response, computation을 동시에 만족하는지,
그리고 관측 가능한 context만 사용하는 2-stage rule이 더 합리적인 trade-off를
만드는지 평가합니다.

```bash
python scripts/exp11_execution_decision.py
```

기본 설정은 `dt=0.1`, 120 control steps, 31-pose local chunk,
`commit_horizon_steps=1`, Exp09의 intermediate point인 7-pose transition window,
30 episodes와 episode당 4 updates(총 120 external events)입니다. Seed는 42--71이고,
latency는 1--8 steps, 첫 observation은 step 12--16, 이후 event 간격은 21--28
steps입니다. Goal-y 변화는 event당 `[-0.85, 0.85]` m에서 누적 후
`[-1.45, 1.45]` m로 제한합니다. Obstacle은 72% 확률로 존재하며 nominal
observation x보다 0.35--1.35 m 앞, lateral offset `[-0.48, 0.48]` m,
radius 0.18--0.34 m, safety margin 0.12--0.20 m의 연속 범위를 사용합니다.
외부 event schedule은 policy들이 공유하지만 robot state와 executable suffix는 각
policy가 독립적으로 rollout합니다.

비교 policy는 Continue+Hard, Continue+Hermite, Continue+collision-aware Graph,
Hold+Graph, Context-conditioned입니다. OLD horizon이 latency와 1-step commit을
덮지 못하면 fixed continue policy도 남은 OLD를 실행한 뒤 hold합니다. Context
policy의 Stage A는 NEW를 입력으로 받지 않고 observation에서 알려진 obstacle과 OLD
future만으로 inference 이전 clearance를 검사해 `continue_old`,
`continue_then_hold`, `hold_pose`를 고릅니다. Stage B는 NEW-ready 이후에만 다음
cascade를 적용합니다.

```text
direct pose disagreement <= 0.04 m and 0.08 rad, and direct candidate safe
    -> Hard switch
otherwise Hermite candidate satisfies the requested obstacle margin
    -> Local cubic Hermite
otherwise
    -> 7-pose segment-aware collision graph
graph failure, collision, or margin failure after independent validation
    -> hold + replan_required
```

Graph acceptance에는 numerical soft-penalty 특성을 고려한 0.002 m tolerance를
명시적으로 사용합니다. 이 값들은 학습하거나 evaluation 결과로 최적화한 threshold가
아니라 사전에 고정한 engineering assumptions입니다. OLD/NEW disagreement는 SE(2)
relative Log의 translation과 `rotation_scale=0.5`인 rotation을 함께 사용한 평균
norm입니다.

결과:

- `outputs/exp11_execution_decision/figure.png`
- `outputs/exp11_execution_decision/event_metrics.csv`
- `outputs/exp11_execution_decision/episode_metrics.csv`
- `outputs/exp11_execution_decision/summary.csv`
- `outputs/exp11_execution_decision/decision_counts.csv`

Event CSV는 context, 선택, pre/post-NEW collision, clearance, progress, boundary
mismatch, local jerk, NEW tracking, optimizer/runtime 및 deadline miss를 기록합니다.
Episode CSV의 task progress는 final-x minus initial-x, hold duration은 hold로 실행된
control interval의 합, collision count는 각 event obstacle의 active interval에서
충돌한 event 수입니다. Response delay는 modification 이후 aligned NEW position
error가 0.05 m 이하가 되는 최초 시간이며, Hard switch가 0 s인 것은 이 metric의
정의상 생기는 expected artifact입니다.

실제 30-episode 결과에서 Continue+Hard, Continue+Hermite, Continue+Graph의
episode collision rate는 각각 40.0%, 56.7%, 30.0%였고, pre-NEW collision event
rate는 8.33%, 8.33%, 5.83%였습니다. Hold+Graph와 Context-conditioned는 모두
actual collision 0건이었습니다. Context policy는 120 events에서 Stage A로
`continue_old` 64회, `continue_then_hold` 42회, `hold_pose` 14회를 선택했고,
Stage B 결과는 Hard 10회, Hermite 71회, accepted Graph 20회,
`replan_required` 19회였습니다. 즉 graph 호출은 평균 1.3회/episode로 fixed Graph의
4.0회보다 적었습니다. 평균 episode computation은 약 83.6 ms로 Continue+Graph
138.0 ms와 Hold+Graph 112.1 ms보다 낮았습니다.

Positive result는 context cascade가 이 distribution에서 avoidable collision을
없애면서 Hold+Graph보다 hold duration(2.29 vs 3.18 s), jerk RMS(50.66 vs 94.60),
graph 호출을 줄였다는 점입니다. 그러나 중요한 negative result로 task progress는
Context 12.31 m가 Hold+Graph 12.72 m보다 낮았고, 19/120 events에서 soft graph를
accept하지 못해 replan/hold fallback이 발생했습니다. 이 중 일부는
observation-relative NEW 자체의 aligned suffix가 requested margin을 만족하지 못한
transition-infeasible event이므로 graph optimizer의 실패만으로 해석하지 않습니다.
Continue+Hard/Hermite는
14.08 m로 더 진행했지만 collision과 교환한 결과입니다. 따라서 H5는
**partially supported**입니다. Fixed policy가 모든 축을 dominate하지 않고 context
selection의 safety/compute 이점은 관찰됐지만, “always hold보다 task progress도
유지한다”는 부분은 지지되지 않았습니다.

현재 generator는 non-overlapping event를 사용하지만 매 event마다 policy의 실제
executable suffix를 OLD로 다시 snapshot하므로 overlapping update를 수용할 구조는
갖습니다. 실제 overlap, sensing/model uncertainty, dynamic obstacle, braking,
threshold sensitivity, 더 넓은 distribution 및 graph fallback planner는 검증하지
않았습니다. Known geometry를 쓰는 SE(2) toy distribution 결과이므로 optimal policy,
robot safety guarantee, 실제 VLA 일반화, 최적 graph-gating threshold는 주장할 수
없습니다. 다음 우선순위는 threshold를 학습하는 것이 아니라, fixed rule을 유지한
채 overlapping updates와 noisy clearance prediction을 독립 변수로 추가하는
robustness experiment입니다.

## Tests

```bash
python -m unittest discover -s tests -v
```
