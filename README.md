# Action Chunk Graph Optimization — Toy Study

목표: old/new action chunk를 단순 point-wise blending하는 방법과
SE(2) Lie-group residual 기반 graph optimization을 비교합니다.

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
