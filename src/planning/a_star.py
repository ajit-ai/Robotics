"""A* shortest-path search on a 2D occupancy grid.

The grid is a list of rows where ``1`` marks an occupied cell and ``0`` a
free cell. Movement uses 4-connectivity by default (cost 1) or optional
8-connectivity with diagonal cost sqrt(2); corner cutting through two
diagonally touching obstacles is forbidden.
"""

import heapq
import math

_MOVES_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_MOVES_8 = _MOVES_4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def manhattan_heuristic(cell, goal):
    """Admissible heuristic for 4-connected grids."""
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def octile_heuristic(cell, goal):
    """Admissible heuristic for 8-connected grids."""
    dx = abs(cell[0] - goal[0])
    dy = abs(cell[1] - goal[1])
    return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)


def _validate(grid, start, goal):
    if not grid or not grid[0]:
        raise ValueError("grid must be a non-empty rectangular matrix")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("grid rows must all have the same length")
    for name, cell in (("start", start), ("goal", goal)):
        row, col = cell
        if not (0 <= row < len(grid) and 0 <= col < width):
            raise ValueError(f"{name} {cell} is outside the grid")
        if grid[row][col] == 1:
            raise ValueError(f"{name} {cell} lies on an obstacle")


def a_star(grid, start, goal, allow_diagonal=False):
    """Find the cheapest path from ``start`` to ``goal``.

    Args:
        grid: 2D list of ints, ``1`` = obstacle, ``0`` = free.
        start: ``(row, col)`` start cell; must be free.
        goal: ``(row, col)`` goal cell; must be free.
        allow_diagonal: Enable 8-connected movement.

    Returns:
        List of ``(row, col)`` cells from start to goal inclusive, or
        ``None`` when no path exists.

    Raises:
        ValueError: For malformed grids or out-of-bounds/obstacle endpoints.
    """
    _validate(grid, start, goal)
    moves = _MOVES_8 if allow_diagonal else _MOVES_4
    heuristic = octile_heuristic if allow_diagonal else manhattan_heuristic

    open_heap = [(heuristic(start, goal), 0.0, start)]
    g_cost = {start: 0.0}
    came_from = {}

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current_g > g_cost.get(current, math.inf):
            continue
        if current == goal:
            return _reconstruct_path(came_from, current)
        for neighbor in _neighbors(current, moves, grid):
            step_cost = (
                math.sqrt(2.0)
                if neighbor[0] != current[0] and neighbor[1] != current[1]
                else 1.0
            )
            tentative_g = current_g + step_cost
            if tentative_g < g_cost.get(neighbor, math.inf):
                g_cost[neighbor] = tentative_g
                came_from[neighbor] = current
                heapq.heappush(
                    open_heap,
                    (tentative_g + heuristic(neighbor, goal), tentative_g, neighbor),
                )
    return None


def _neighbors(cell, moves, grid):
    row, col = cell
    for d_row, d_col in moves:
        n_row, n_col = row + d_row, col + d_col
        if not (0 <= n_row < len(grid) and 0 <= n_col < len(grid[0])):
            continue
        if grid[n_row][n_col] == 1:
            continue
        if d_row != 0 and d_col != 0:
            if grid[row][n_col] == 1 or grid[n_row][col] == 1:
                continue
        yield (n_row, n_col)


def _reconstruct_path(came_from, node):
    path = [node]
    while node in came_from:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path


if __name__ == "__main__":
    world = [
        [0, 0, 0, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    print("path:", a_star(world, (0, 0), (2, 3), allow_diagonal=True))
