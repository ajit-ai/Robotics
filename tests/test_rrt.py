import math
import random

import pytest

from src.planning.rrt import point_in_obstacle, rrt_plan, segment_is_collision_free

BOUNDS = (0.0, 0.0, 10.0, 10.0)


def ring_of_circles(center, ring_radius, circle_radius, count=12):
    cx, cy = center
    return [
        (
            cx + ring_radius * math.cos(2.0 * math.pi * i / count),
            cy + ring_radius * math.sin(2.0 * math.pi * i / count),
            circle_radius,
        )
        for i in range(count)
    ]


class TestSegmentCollision:
    def test_point_inside_circle_detected(self):
        assert point_in_obstacle((0.2, 0.0), (0.0, 0.0, 0.5)) is True

    def test_point_outside_circle_not_colliding(self):
        assert point_in_obstacle((2.0, 2.0), (0.0, 0.0, 0.5)) is False

    def test_segment_through_circle_is_blocked(self):
        assert not segment_is_collision_free((-1.0, 0.0), (1.0, 0.0), [(0.0, 0.0, 0.5)])

    def test_segment_missing_circle_is_free(self):
        assert segment_is_collision_free((-1.0, 0.0), (1.0, 0.0), [(0.0, 2.0, 0.5)])

    def test_endpoint_inside_circle_is_blocked(self):
        assert not segment_is_collision_free((0.0, 0.0), (5.0, 0.0), [(1.0, 0.0, 0.25)])

    def test_degenerate_segment_checks_point_only(self):
        free = segment_is_collision_free((3.0, 3.0), (3.0, 3.0), [(0.0, 0.0, 1.0)])
        blocked = segment_is_collision_free((0.0, 0.0), (0.0, 0.0), [(0.0, 0.0, 1.0)])
        assert free is True
        assert blocked is False


class TestRRTPlan:
    def test_finds_path_in_open_space_and_is_deterministic(self):
        first = rrt_plan(
            (0.5, 0.5), (9.0, 9.0), BOUNDS, rng=random.Random(11)
        )
        second = rrt_plan(
            (0.5, 0.5), (9.0, 9.0), BOUNDS, rng=random.Random(11)
        )
        assert first is not None and second is not None
        assert first == second
        assert list(first[0]) == [0.5, 0.5]
        assert tuple(first[-1]) == (9.0, 9.0)

    def test_all_segments_are_step_bounded_and_collision_free(self):
        obstacles = [(5.0, 5.0, 1.5)]
        plan = rrt_plan(
            (0.5, 0.5),
            (9.5, 9.5),
            BOUNDS,
            obstacles=obstacles,
            step_size=0.6,
            goal_tolerance=0.25,
            rng=random.Random(42),
        )
        assert plan is not None
        step_limit = 0.6 + 0.25 + 1e-9
        for p, q in zip(plan, plan[1:]):
            assert math.dist(p, q) <= step_limit
            assert segment_is_collision_free(p, q, obstacles)

    def test_waypoints_avoid_obstacles(self):
        obstacles = [(3.0, 3.0, 1.0), (7.0, 4.0, 1.2)]
        plan = rrt_plan(
            (0.5, 0.5), (9.0, 9.0), BOUNDS, obstacles=obstacles, rng=random.Random(7)
        )
        assert plan is not None
        for x, y in plan[:-1]:
            for ox, oy, radius in obstacles:
                assert (x - ox) ** 2 + (y - oy) ** 2 >= radius**2

    def test_impossible_goal_returns_none(self):
        sealed = ring_of_circles((5.0, 5.0), ring_radius=1.0, circle_radius=0.55)
        plan = rrt_plan(
            (0.5, 0.5),
            (5.0, 5.0),
            BOUNDS,
            obstacles=sealed,
            max_iterations=800,
            rng=random.Random(3),
        )
        assert plan is None

    def test_nearby_start_returns_two_waypoint_plan(self):
        plan = rrt_plan(
            (1.0, 1.0), (1.05, 1.05), BOUNDS, goal_tolerance=0.2, rng=random.Random(5)
        )
        assert plan == [(1.0, 1.0), (1.05, 1.05)]

    def test_start_in_collision_raises(self):
        with pytest.raises(ValueError, match="start configuration"):
            rrt_plan((5.0, 5.0), (9.0, 9.0), BOUNDS, obstacles=[(5.0, 5.0, 1.0)])

    def test_goal_in_collision_raises(self):
        with pytest.raises(ValueError, match="goal configuration"):
            rrt_plan((0.5, 0.5), (5.0, 5.0), BOUNDS, obstacles=[(5.0, 5.0, 1.0)])

    def test_invalid_bounds_raise(self):
        with pytest.raises(ValueError, match="bounds"):
            rrt_plan((0.0, 0.0), (1.0, 1.0), (2.0, 0.0, 0.0, 2.0))

    def test_invalid_step_size_raises(self):
        with pytest.raises(ValueError, match="step_size"):
            rrt_plan((0.0, 0.0), (1.0, 1.0), BOUNDS, step_size=0.0)

    def test_endpoint_outside_bounds_raises(self):
        with pytest.raises(ValueError, match="outside bounds"):
            rrt_plan((-1.0, 0.0), (1.0, 1.0), BOUNDS)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
