"""Navigation by artificial potential fields (Khatib, 1986).

The goal exerts an attractive quadratic-bowl force and circular obstacles
exert short-range repulsive forces:

    F_att(q) = k_att * (q_goal - q)
    F_rep(q) = k_rep * (1/d - 1/d0) / d^2 * unit(q - o)   for d < d0, else 0

where ``d`` is the distance to the obstacle center and ``d0`` its influence
radius. The planner descends the resulting field with capped proportional
steps. Classic limitation: symmetric head-on configurations can create a
local minimum, in which case the planner reports failure.
"""

import math


def attractive_force(position, goal, k_att):
    """Return the attraction vector pointing from ``position`` to ``goal``."""
    return (
        k_att * (goal[0] - position[0]),
        k_att * (goal[1] - position[1]),
    )


def repulsive_force(position, obstacle, k_rep, influence_radius):
    """Return the repulsion vector exerted by one circular obstacle.

    Zero beyond ``influence_radius``. Raises ``ValueError`` if the position
    coincides exactly with the obstacle center.
    """
    dx = position[0] - obstacle[0]
    dy = position[1] - obstacle[1]
    distance = math.hypot(dx, dy)
    if distance == 0.0:
        raise ValueError("position coincides with the obstacle center")
    if distance >= influence_radius:
        return (0.0, 0.0)
    magnitude = k_rep * (1.0 / distance - 1.0 / influence_radius) / (distance ** 2)
    return (magnitude * dx / distance, magnitude * dy / distance)


def total_force(position, goal, obstacles, k_att, k_rep, influence_radius):
    """Sum attraction over the goal and repulsion over all obstacles."""
    fx, fy = attractive_force(position, goal, k_att)
    for obstacle in obstacles:
        rfx, rfy = repulsive_force(position, obstacle, k_rep, influence_radius)
        fx += rfx
        fy += rfy
    return (fx, fy)


def _validate(start, goal, obstacles, step_size, goal_tolerance, influence_radius):
    for name, point in (("start", start), ("goal", goal)):
        for ox, oy, radius in obstacles:
            if math.hypot(point[0] - ox, point[1] - oy) <= radius:
                raise ValueError(f"{name} lies inside an obstacle")
    if step_size <= 0 or goal_tolerance <= 0:
        raise ValueError("step_size and goal_tolerance must be positive")
    if influence_radius <= 0:
        raise ValueError("influence_radius must be positive")


def potential_field_plan(
    start,
    goal,
    obstacles=(),
    k_att=1.0,
    k_rep=100.0,
    influence_radius=1.5,
    alpha=0.03,
    step_size=0.1,
    goal_tolerance=0.25,
    max_iterations=5000,
    force_epsilon=1e-8,
):
    """Descend the potential field from ``start`` toward ``goal``.

    Args:
        start: ``(x, y)`` start configuration outside every obstacle.
        goal: ``(x, y)`` target configuration outside every obstacle.
        obstacles: Iterable of ``(cx, cy, radius)`` circular obstacles.
        k_att: Attractive gain.
        k_rep: Repulsive gain.
        influence_radius: Range ``d0`` of the repulsive field.
        alpha: Proportional step gain; displacement is ``alpha * F``,
            capped at ``step_size`` per iteration.
        step_size: Maximum displacement per iteration.
        goal_tolerance: Distance at which the goal counts as reached.
        max_iterations: Iteration budget before reporting failure.
        force_epsilon: Net-force magnitude below which the run is declared
            trapped in a local minimum.

    Returns:
        List of ``(x, y)`` waypoints from start to a point within
        ``goal_tolerance`` of the goal, or ``None`` if the descent stalls.

    Raises:
        ValueError: For invalid gains/parameters or configurations inside
            obstacles.
    """
    _validate(start, goal, obstacles, step_size, goal_tolerance, influence_radius)
    current = tuple(start)
    path = [current]
    for _ in range(max_iterations):
        if math.dist(current, goal) <= goal_tolerance:
            return path
        fx, fy = total_force(
            current, goal, obstacles, k_att, k_rep, influence_radius
        )
        norm = math.hypot(fx, fy)
        if norm <= force_epsilon:
            return None
        dx = alpha * fx
        dy = alpha * fy
        displacement = math.hypot(dx, dy)
        if displacement > step_size:
            scale = step_size / displacement
            dx *= scale
            dy *= scale
        current = (current[0] + dx, current[1] + dy)
        path.append(current)
    return None


if __name__ == "__main__":
    plan = potential_field_plan(
        (0.0, 0.0), (5.0, 3.0), obstacles=[(2.5, 0.0, 0.8)]
    )
    print(f"waypoints: {len(plan)}" if plan else "trapped")
