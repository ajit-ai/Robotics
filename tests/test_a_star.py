import math

import pytest

from src.planning.a_star import a_star


def path_cost(path, allow_diagonal=False):
    cost = 0.0
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if r1 != r2 and c1 != c2:
            cost += math.sqrt(2.0)
        else:
            cost += 1.0
    return cost


EMPTY_3X3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
WALL_GRID = [
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 0],
]
SEALED_GRID = [
    [0, 1],
    [1, 0],
]


class TestAStar:
    def test_open_grid_path_is_optimal(self):
        path = a_star(EMPTY_3X3, (0, 0), (2, 2))
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        assert len(path) == 5

    def test_start_equals_goal_returns_single_cell(self):
        assert a_star(WALL_GRID, (1, 2), (1, 2)) == [(1, 2)]

    def test_detours_around_vertical_wall(self):
        path = a_star(WALL_GRID, (0, 0), (0, 2))
        assert path is not None
        for row, col in path:
            assert WALL_GRID[row][col] == 0
        assert len(path) == 7

    def test_no_path_when_goal_sealed_returns_none(self):
        assert a_star(SEALED_GRID, (0, 0), (1, 1)) is None

    def test_corner_cutting_is_forbidden_with_diagonals(self):
        assert a_star(SEALED_GRID, (0, 0), (1, 1), allow_diagonal=True) is None

    def test_diagonal_shortens_open_grid_cost(self):
        straight = a_star(EMPTY_3X3, (0, 0), (2, 2))
        diagonal = a_star(EMPTY_3X3, (0, 0), (2, 2), allow_diagonal=True)
        assert path_cost(diagonal, True) == pytest.approx(2 * math.sqrt(2.0))
        assert path_cost(diagonal, True) < path_cost(straight)

    def test_diagonal_path_respects_octile_optimality(self):
        path = a_star(EMPTY_3X3, (0, 0), (2, 2), allow_diagonal=True)
        assert path_cost(path, True) == pytest.approx(octile((0, 0), (2, 2)))

    def test_obstacle_endpoints_raise(self):
        with pytest.raises(ValueError, match="start"):
            a_star(WALL_GRID, (0, 1), (2, 2))
        with pytest.raises(ValueError, match="goal"):
            a_star(WALL_GRID, (0, 0), (0, 1))

    def test_out_of_bounds_endpoints_raise(self):
        with pytest.raises(ValueError, match="outside"):
            a_star(WALL_GRID, (-1, 0), (2, 2))
        with pytest.raises(ValueError, match="outside"):
            a_star(WALL_GRID, (0, 0), (3, 0))

    def test_ragged_grid_raises(self):
        with pytest.raises(ValueError, match="same length"):
            a_star([[0, 0], [0]], (0, 0), (1, 0))

    def test_empty_grid_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            a_star([], (0, 0), (0, 0))


def octile(cell, goal):
    dx = abs(cell[0] - goal[0])
    dy = abs(cell[1] - goal[1])
    return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
