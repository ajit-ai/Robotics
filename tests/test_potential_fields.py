import math

import pytest

from src.planning.potential_fields import (
    attractive_force,
    potential_field_plan,
    repulsive_force,
    total_force,
)


class TestForces:
    def test_attraction_points_toward_goal_with_linear_magnitude(self):
        force = attractive_force((0.0, 0.0), (3.0, 4.0), k_att=2.0)
        assert force == pytest.approx((6.0, 8.0))

    def test_attraction_zero_at_goal(self):
        assert attractive_force((2.0, 2.0), (2.0, 2.0), k_att=5.0) == (0.0, 0.0)

    def test_repulsion_zero_beyond_influence_radius(self):
        force = repulsive_force((10.0, 0.0), (0.0, 0.0, 1.0), k_rep=100.0, influence_radius=2.0)
        assert force == (0.0, 0.0)

    def test_repulsion_pushes_directly_away_from_obstacle(self):
        position = (1.5, 0.0)
        force = repulsive_force(position, (0.0, 0.0, 1.0), k_rep=10.0, influence_radius=3.0)
        fx, fy = force
        distance = math.hypot(*position)
        expected = 10.0 * (1.0 / distance - 1.0 / 3.0) / distance**2
        assert fy == pytest.approx(0.0, abs=1e-12)
        assert fx == pytest.approx(expected)

    def test_repulsion_stronger_when_closer(self):
        near = repulsive_force((1.2, 0.0), (0.0, 0.0, 1.0), k_rep=10.0, influence_radius=3.0)
        far = repulsive_force((1.8, 0.0), (0.0, 0.0, 1.0), k_rep=10.0, influence_radius=3.0)
        assert math.hypot(*near) > math.hypot(*far)

    def test_repulsion_at_obstacle_center_raises(self):
        with pytest.raises(ValueError, match="coincides"):
            repulsive_force((0.0, 0.0), (0.0, 0.0, 1.0), k_rep=1.0, influence_radius=2.0)

    def test_total_force_sums_all_contributions(self):
        total = total_force(
            (0.0, 0.0),
            (2.0, 0.0),
            obstacles=[(1.0, 0.5, 0.2)],
            k_att=1.0,
            k_rep=10.0,
            influence_radius=1.0,
        )
        ax, ay = attractive_force((0.0, 0.0), (2.0, 0.0), 1.0)
        rx, ry = repulsive_force((0.0, 0.0), (1.0, 0.5, 0.2), 10.0, 1.0)
        assert total == pytest.approx((ax + rx, ay + ry))


class TestPlanning:
    def test_open_space_reaches_goal(self):
        plan = potential_field_plan((0.0, 0.0), (5.0, 0.0))
        assert plan is not None
        assert math.dist(plan[-1], (5.0, 0.0)) <= 0.25
        assert plan[0] == (0.0, 0.0)

    def test_trajectory_is_contiguous_with_capped_steps(self):
        plan = potential_field_plan((0.0, 0.0), (4.0, 4.0))
        max_step = max(
            math.dist(p, q) for p, q in zip(plan, plan[1:])
        )
        assert max_step <= 0.1 + 1e-9

    def test_slides_around_offset_obstacle(self):
        obstacle = (2.5, 0.0, 0.8)
        plan = potential_field_plan(
            (0.0, 0.0), (5.0, 3.0), obstacles=[obstacle]
        )
        assert plan is not None
        for x, y in plan:
            clearance = math.hypot(x - obstacle[0], y - obstacle[1])
            assert clearance > obstacle[2]

    def test_head_on_obstacle_traps_planner(self):
        plan = potential_field_plan(
            (0.0, 0.0),
            (5.0, 0.0),
            obstacles=[(2.5, 0.0, 0.8)],
        )
        assert plan is None

    def test_start_inside_obstacle_raises(self):
        with pytest.raises(ValueError, match="start lies inside"):
            potential_field_plan((2.5, 0.0), (5.0, 0.0), obstacles=[(2.5, 0.0, 0.8)])

    def test_goal_inside_obstacle_raises(self):
        with pytest.raises(ValueError, match="goal lies inside"):
            potential_field_plan((0.0, 0.0), (5.0, 0.0), obstacles=[(5.0, 0.0, 0.8)])

    def test_start_on_obstacle_boundary_raises(self):
        with pytest.raises(ValueError, match="start lies inside"):
            potential_field_plan((3.3, 0.0), (5.0, 0.0), obstacles=[(2.5, 0.0, 0.8)])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"step_size": 0.0},
            {"goal_tolerance": -1.0},
            {"influence_radius": 0.0},
        ],
    )
    def test_invalid_parameters_raise(self, kwargs):
        with pytest.raises(ValueError):
            potential_field_plan((0.0, 0.0), (1.0, 1.0), **kwargs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
