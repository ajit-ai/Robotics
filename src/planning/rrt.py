"""Rapidly-exploring Random Tree (RRT) path planning in continuous 2D.

RRT grows a tree of random samples from the start configuration, steering
each new sample a fixed step size toward randomly drawn points. The plan
succeeds when a leaf comes within ``goal_tolerance`` of the goal.

Obstacles are circles given as ``(center_x, center_y, radius)`` tuples and
the world is an axis-aligned rectangle ``bounds = (x_min, y_min, x_max,
y_max)``.
"""

import math
import random

_EPSILON = 1e-12


def point_in_obstacle(point, obstacle):
    """Return True if ``point`` lies strictly inside ``obstacle``."""
    cx, cy, radius = obstacle
    return (point[0] - cx) ** 2 + (point[1] - cy) ** 2 < radius * radius


def segment_is_collision_free(p, q, obstacles):
    """Check whether the open/closed segment p->q misses every circle."""
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    seg_length_sq = dx * dx + dy * dy
    for obstacle in obstacles:
        cx, cy, radius = obstacle
        t = 0.0
        if seg_length_sq > _EPSILON:
            t = ((cx - p[0]) * dx + (cy - p[1]) * dy) / seg_length_sq
            t = max(0.0, min(1.0, t))
        closest = (p[0] + t * dx, p[1] + t * dy)
        if math.hypot(closest[0] - cx, closest[1] - cy) <= radius:
            return False
    return True


def _validate_inputs(start, goal, bounds, step_size):
    x_min, y_min, x_max, y_max = bounds
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("bounds must define a non-degenerate rectangle")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    for name, point in (("start", start), ("goal", goal)):
        if not (x_min <= point[0] <= x_max and y_min <= point[1] <= y_max):
            raise ValueError(f"{name} {point} lies outside bounds")


def rrt_plan(
    start,
    goal,
    bounds,
    obstacles=(),
    step_size=0.5,
    goal_tolerance=0.25,
    max_iterations=5000,
    rng=None,
):
    """Grow an RRT from ``start`` until the goal region is reached.

    Args:
        start: ``(x, y)`` start configuration.
        goal: ``(x, y)`` goal configuration.
        bounds: ``(x_min, y_min, x_max, y_max)`` sampling rectangle.
        obstacles: Iterable of ``(cx, cy, r)`` circular obstacles.
        step_size: Maximum extension length per iteration.
        goal_tolerance: Distance at which the goal counts as reached.
        max_iterations: Iteration budget before giving up.
        rng: Optional ``random.Random`` instance for reproducibility.

    Returns:
        List of ``(x, y)`` waypoints from start to goal inclusive, or
        ``None`` if no connection was found within ``max_iterations``.

    Raises:
        ValueError: For invalid bounds/step size or endpoints outside bounds.
    """
    _validate_inputs(start, goal, bounds, step_size)
    rng = rng if rng is not None else random.Random()

    if any(point_in_obstacle(start, obstacle) for obstacle in obstacles):
        raise ValueError("start configuration is in collision")
    if any(point_in_obstacle(goal, obstacle) for obstacle in obstacles):
        raise ValueError("goal configuration is in collision")

    parent = {start: None}
    nodes = [start]

    if (
        math.dist(start, goal) <= goal_tolerance
        and segment_is_collision_free(start, goal, obstacles)
    ):
        return [start, goal]

    x_min, y_min, x_max, y_max = bounds
    for _ in range(max_iterations):
        sample = (
            rng.uniform(x_min, x_max),
            rng.uniform(y_min, y_max),
        )
        nearest = min(nodes, key=lambda n: (n[0] - sample[0]) ** 2 + (n[1] - sample[1]) ** 2)
        distance = math.dist(nearest, sample)
        if distance < _EPSILON:
            continue
        scale = min(step_size / distance, 1.0)
        new_node = (
            nearest[0] + (sample[0] - nearest[0]) * scale,
            nearest[1] + (sample[1] - nearest[1]) * scale,
        )
        new_node = (
            max(x_min, min(x_max, new_node[0])),
            max(y_min, min(y_max, new_node[1])),
        )
        if new_node in parent:
            continue
        if not segment_is_collision_free(nearest, new_node, obstacles):
            continue
        parent[new_node] = nearest
        nodes.append(new_node)
        if math.dist(new_node, goal) <= goal_tolerance and segment_is_collision_free(
            new_node, goal, obstacles
        ):
            return _path_to(parent, new_node) + [goal]
    return None


def _path_to(parent, node):
    path = []
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


if __name__ == "__main__":
    plan = rrt_plan(
        start=(0.5, 0.5),
        goal=(9.0, 9.0),
        bounds=(0.0, 0.0, 10.0, 10.0),
        obstacles=[(5.0, 5.0, 1.5)],
        rng=random.Random(42),
    )
    print(f"waypoints: {len(plan)}")
    for point in plan[:5]:
        print(f"  ({point[0]:.3f}, {point[1]:.3f})")
