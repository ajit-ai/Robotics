import pytest

from src.planning.a_star import a_star
from src.planning.d_star_lite import DStarLite

WALL_WORLD = [
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 0],
]
OPEN_5X5 = [[0] * 5 for _ in range(5)]
MAZE_5X5 = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
]
TWO_ROUTES = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
]


def planned(world, start=(0, 0), goal=None):
    planner = DStarLite(world, start=start, goal=goal or (len(world) - 1,) * 2)
    planner.compute_shortest_path()
    return planner


class TestInitialPlanning:
    def test_optimal_path_around_wall(self):
        path = planned(WALL_WORLD).get_path()
        reference = a_star(WALL_WORLD, (0, 0), (2, 2))
        assert path is not None
        assert len(path) == len(reference)

    def test_maze_path_is_valid_and_optimal(self):
        planner = planned(MAZE_5X5)
        path = planner.get_path()
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)
        for row, col in path:
            assert MAZE_5X5[row][col] == 0
        assert len(path) == 9

    def test_start_equals_goal(self):
        planner = DStarLite(OPEN_5X5, (2, 2), (2, 2))
        planner.compute_shortest_path()
        assert planner.get_path() == [(2, 2)]

    def test_sealed_goal_yields_no_path(self):
        sealed = [[0, 1, 0], [1, 1, 0], [0, 0, 0]]
        planner = planned(sealed)
        assert planner.get_path() is None


class TestReplanning:
    def test_repair_when_new_obstacle_blocks_route(self):
        planner = planned(TWO_ROUTES)
        original = planner.get_path()
        for candidate in [(4, 2), (0, 2), (2, 4)]:
            if candidate in original:
                blocked_cell = candidate
                break
        planner.set_blocked(blocked_cell)
        repaired = planner.get_path()
        assert repaired is not None
        assert blocked_cell not in repaired
        assert len(repaired) >= len(original)
        for row, col in repaired:
            assert planner.grid[row][col] == 0

    def test_blocking_cannot_shorten_path(self):
        planner = planned(OPEN_5X5)
        original_len = len(planner.get_path())
        planner.set_blocked((2, 2))
        new_len = len(planner.get_path())
        assert new_len >= original_len

    def test_unblocking_restores_path(self):
        planner = DStarLite(WALL_WORLD, (2, 2), (0, 0))
        planner.compute_shortest_path()
        assert len(planner.get_path()) == 5
        planner.set_blocked((2, 1))
        assert planner.get_path() is None
        planner.set_blocked((2, 1), blocked=False)
        restored = planner.get_path()
        assert restored[-1] == (0, 0)
        assert (2, 1) in restored

    def test_ring_then_release_recovers_path(self):
        planner = planned(TWO_ROUTES)
        planner.set_blocked((4, 2))
        detoured_len = len(planner.get_path())
        assert detoured_len >= 9
        planner.set_blocked((0, 2))
        assert planner.get_path() is None
        planner.set_blocked((0, 2), blocked=False)
        recovered = planner.get_path()
        assert recovered is not None
        assert recovered[-1] == (4, 4)


class TestRobotMovement:
    def test_move_start_keeps_plan_consistent(self):
        planner = planned(MAZE_5X5)
        first_step = planner.get_path()[1]
        planner.move_start(first_step)
        planner.compute_shortest_path()
        path = planner.get_path()
        assert path[0] == first_step
        assert path[-1] == (4, 4)

    def test_multiple_moves_walk_to_goal(self):
        planner = planned(MAZE_5X5)
        steps = 0
        while planner.start != planner.goal and steps < 20:
            planner.move_start(planner.get_path()[1])
            planner.compute_shortest_path()
            steps += 1
        assert planner.start == (4, 4)
        assert steps <= 8

    def test_dynamic_world_during_motion(self):
        planner = planned(TWO_ROUTES)
        steps = 0
        while planner.start != (4, 4) and steps < 30:
            nxt = planner.get_path()[1]
            if nxt == (4, 2):
                planner.set_blocked(nxt)
                nxt = planner.get_path()[1]
            planner.move_start(nxt)
            planner.compute_shortest_path()
            steps += 1
        assert planner.start == (4, 4)


class TestValidation:
    def test_out_of_bounds_endpoint_raises(self):
        with pytest.raises(ValueError, match="outside"):
            DStarLite(OPEN_5X5, (-1, 0), (4, 4))

    def test_obstacle_endpoint_raises(self):
        with pytest.raises(ValueError, match="obstacle"):
            DStarLite(WALL_WORLD, (0, 1), (2, 2))

    def test_ragged_grid_raises(self):
        with pytest.raises(ValueError, match="same length"):
            DStarLite([[0, 0], [0]], (0, 0), (1, 0))

    def test_set_blocked_out_of_bounds_raises(self):
        planner = planned(OPEN_5X5)
        with pytest.raises(ValueError, match="outside"):
            planner.set_blocked((9, 9))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
