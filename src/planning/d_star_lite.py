"""D* Lite incremental path planning on a 2D occupancy grid.

Unlike one-shot A*, D* Lite (Koenig & Likhachev, 2002) repairs an existing
search cheaply when the world changes or when the robot moves forward: only
the vertices whose costs became inconsistent are re-expanded.

The planner keeps for every cell two values:

    g(s)   best known cost-to-goal,
    rhs(s) one-step lookahead of g based on the successors' g values.

A vertex is "locally consistent" when ``g == rhs``. The priority queue holds
inconsistent vertices keyed by ``(min(g, rhs) + h(start, s) + km, min(g,
rhs))``, where ``km`` accumulates heuristic drift as the start moves.
"""

import heapq
import math

_INF = math.inf


class _PriorityQueue:
    """Lazy-deletion binary heap with updatable keys."""

    def __init__(self):
        self._heap = []
        self._keys = {}

    def insert(self, node, key):
        self._keys[node] = key
        heapq.heappush(self._heap, (key, node))

    def remove(self, node):
        self._keys.pop(node, None)

    def contains(self, node):
        return node in self._keys

    def top(self):
        while self._heap:
            key, node = self._heap[0]
            if self._keys.get(node) != key:
                heapq.heappop(self._heap)
                continue
            return node
        return None

    def top_key(self):
        top = self.top()
        return None if top is None else self._keys[top]


class DStarLite:
    """Incremental shortest-path planner with dynamic obstacle updates.

    Args:
        grid: 2D list of ints, ``1`` = obstacle, ``0`` = free. The planner
            works on its own mutable copy; mutate it through
            :meth:`set_blocked`.
        start: ``(row, col)`` start cell.
        goal: ``(row, col)`` goal cell.

    Raises:
        ValueError: For malformed grids or out-of-bounds/obstacle endpoints.
    """

    def __init__(self, grid, start, goal):
        self._validate_grid(grid)
        self._validate_cell(grid, start, "start")
        self._validate_cell(grid, goal, "goal")
        self.grid = [list(row) for row in grid]
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start = tuple(start)
        self.last = tuple(start)
        self.goal = tuple(goal)
        self.km = 0.0
        self._g_values = {}
        self._rhs_values = {self.goal: 0.0}
        self._queue = _PriorityQueue()
        self._queue.insert(self.goal, self._calc_key(self.goal))

    @staticmethod
    def _validate_grid(grid):
        if not grid or not grid[0]:
            raise ValueError("grid must be a non-empty rectangular matrix")
        width = len(grid[0])
        if any(len(row) != width for row in grid):
            raise ValueError("grid rows must all have the same length")

    @staticmethod
    def _validate_cell(grid, cell, name):
        row, col = cell
        if not (0 <= row < len(grid) and 0 <= col < len(grid[0])):
            raise ValueError(f"{name} {tuple(cell)} is outside the grid")
        if grid[row][col] == 1:
            raise ValueError(f"{name} {tuple(cell)} lies on an obstacle")

    def _is_free(self, cell):
        return self.grid[cell[0]][cell[1]] == 0

    def _cost(self, a, b):
        if not (self._is_free(a) and self._is_free(b)):
            return _INF
        return 1.0

    def _neighbors(self, cell):
        row, col = cell
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n_row, n_col = row + d_row, col + d_col
            if 0 <= n_row < self.rows and 0 <= n_col < self.cols:
                yield (n_row, n_col)

    def _h(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _g(self, s):
        return self._g_values.get(s, _INF)

    def _rhs(self, s):
        return self._rhs_values.get(s, _INF)

    def _calc_key(self, s):
        minimum = min(self._g(s), self._rhs(s))
        return (minimum + self._h(self.start, s) + self.km, minimum)

    def _update_vertex(self, u):
        if u != self.goal:
            best = _INF
            for neighbor in self._neighbors(u):
                step = self._cost(u, neighbor)
                if step < best:
                    candidate = step + self._g(neighbor)
                    if candidate < best:
                        best = candidate
            self._rhs_values[u] = best
        if self._queue.contains(u):
            self._queue.remove(u)
        if self._g(u) != self._rhs(u):
            self._queue.insert(u, self._calc_key(u))

    def compute_shortest_path(self):
        """Expand inconsistent vertices until the start is consistent."""
        while True:
            k_old = self._queue.top_key()
            if k_old is None:
                break
            start_inconsistent = self._rhs(self.start) != self._g(self.start)
            if not (
                k_old < self._calc_key(self.start) or start_inconsistent
            ):
                break
            u = self._queue.top()
            k_new = self._calc_key(u)
            if k_old < k_new:
                self._queue.insert(u, k_new)
            elif self._g(u) > self._rhs(u):
                self._g_values[u] = self._rhs(u)
                self._queue.remove(u)
                for predecessor in self._neighbors(u):
                    self._update_vertex(predecessor)
            else:
                g_old = self._g(u)
                self._g_values[u] = _INF
                for node in list(self._neighbors(u)) + [u]:
                    routes_through_u = (
                        node != self.goal
                        and self._rhs(node)
                        == self._cost(node, u) + g_old
                    )
                    if routes_through_u:
                        self._rhs_values[node] = min(
                            self._cost(node, neighbor) + self._g(neighbor)
                            for neighbor in self._neighbors(node)
                        )
                    self._update_vertex(node)

    def get_path(self):
        """Greedy descent from start to goal using current g-values.

        Returns:
            List of cells from start to goal inclusive, or ``None`` when the
            goal is unreachable or no consistent plan exists yet.
        """
        if self._g(self.start) == _INF and self.start != self.goal:
            return None
        path = [self.start]
        visited = {self.start}
        current = self.start
        while current != self.goal:
            best = None
            best_total = _INF
            for neighbor in self._neighbors(current):
                total = self._cost(current, neighbor) + self._g(neighbor)
                if total < best_total:
                    best_total = total
                    best = neighbor
            if best is None or best in visited or best_total == _INF:
                return None
            visited.add(best)
            path.append(best)
            current = best
        return path

    def move_start(self, new_start):
        """Teleport the robot to a free neighbouring cell, updating ``km``."""
        self._validate_cell(self.grid, new_start, "start")
        new_start = tuple(new_start)
        self.km += self._h(self.last, new_start)
        self.last = new_start
        self.start = new_start

    def set_blocked(self, cell, blocked=True):
        """Change one cell's occupancy and incrementally repair the search."""
        row, col = cell
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"cell {tuple(cell)} is outside the grid")
        self.grid[row][col] = 1 if blocked else 0
        changed = [(row, col)] + list(self._neighbors((row, col)))
        for node in changed:
            self._update_vertex(node)
        self.compute_shortest_path()


if __name__ == "__main__":
    world = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    dstar = DStarLite(world, start=(0, 0), goal=(2, 2))
    dstar.compute_shortest_path()
    print("initial path:", dstar.get_path())
    dstar.set_blocked((2, 1))
    print("after blocking (2, 1):", dstar.get_path())
