# Robotics

A collection of classic robotics algorithms implemented in pure Python (standard
library only) with a full pytest test suite. Each module is runnable as a demo
(`python src/<module>.py`) and fully unit-tested.

## Algorithms

| Domain | Module | Algorithm |
|---|---|---|
| Kinematics | `src/kinematics/forward_kinematics.py` | Planar serial arm FK via cumulative joint angles |
| Kinematics | `src/kinematics/inverse_kinematics.py` | Closed-form 2-link IK (law of cosines, elbow up/down) |
| Planning | `src/planning/a_star.py` | A* on occupancy grids (4- or 8-connected, octile heuristic, no corner cutting) |
| Planning | `src/planning/rrt.py` | Rapidly-exploring Random Tree in continuous 2D with circular obstacles |
| Control | `src/control/pid_controller.py` | Discrete PID with conditional-integration anti-windup and output saturation |
| Estimation | `src/estimation/kalman_filter.py` | Scalar Kalman filter (predict/correct cycle) for noisy 1D signals |

## Project layout

```
Robotics/
├── src/
│   ├── kinematics/        # forward & inverse kinematics
│   ├── planning/          # A* grid search, RRT sampling-based planning
│   ├── control/           # PID feedback control
│   └── estimation/        # Kalman filtering
├── tests/                 # one pytest suite per module (82 tests)
├── requirements.txt       # dev dependencies (pytest)
└── conftest.py
```

## Quick start

```bash
pip install -r requirements.txt

# run the full test suite
python -m pytest tests -v

# also verify the docstring examples
python -m pytest --doctest-modules src

# run any module as a standalone demo
python src/planning/a_star.py
```

## Usage examples

### Forward / inverse kinematics

```python
from src.kinematics import end_effector_position, inverse_kinematics_2link

pose = end_effector_position([1.0, 1.0], [0.0, 0.0])   # -> (2.0, 0.0)
q1, q2 = inverse_kinematics_2link(1.0, 1.0, 1.0, 1.0)  # elbow-up solution
```

### Grid path planning

```python
from src.planning import a_star

grid = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
path = a_star(grid, start=(0, 0), goal=(2, 3), allow_diagonal=True)
```

### Sampling-based planning

```python
import random
from src.planning import rrt_plan

plan = rrt_plan(
    start=(0.5, 0.5), goal=(9.0, 9.0),
    bounds=(0.0, 0.0, 10.0, 10.0),
    obstacles=[(5.0, 5.0, 1.5)],
    rng=random.Random(42),
)
```

### PID control

```python
from src.control import PIDController

pid = PIDController(kp=8.0, ki=2.0, kd=0.05, dt=0.01, output_limits=(-10, 10))
u = pid.update(setpoint=1.0, measurement=position)
```

### Kalman filtering

```python
from src.estimation import KalmanFilter1D

kf = KalmanFilter1D(process_variance=1e-5, measurement_variance=0.1)
estimate = kf.step(noisy_measurement)
```

## Testing

Every algorithm has a dedicated test suite covering happy paths, edge cases,
error handling, numerical properties (round-trips, optimality, convergence),
and deterministic reproducibility:

```bash
python -m pytest tests -v     # 82 tests
```

## License

See [LICENSE](LICENSE).
