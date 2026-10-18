"""
maze_engine.py — Pure-Python randomised maze generator and game logic.

Uses Recursive Backtracking (DFS) to carve passages through a grid.
Provides move(), state(), and generate() methods consumed by app.py.
"""

import random
from typing import List, Tuple, Dict, Any


class MazeEngine:
    """
    Randomised maze with a mouse and an exit cell.

    Cell values:
      1 → wall
      0 → open passage
    """

    MOVES = {
        "up":    (-1,  0),
        "down":  ( 1,  0),
        "left":  ( 0, -1),
        "right": ( 0,  1),
    }

    def __init__(self, rows: int = 11, cols: int = 11):
        self.rows = rows
        self.cols = cols
        self.generate(rows, cols)

    # ─────────────────────────────────────────────────────────────────────
    # Generation
    # ─────────────────────────────────────────────────────────────────────

    def generate(self, rows: int = None, cols: int = None):
        """
        Build a new maze using recursive-backtracking DFS.

        Only odd-indexed cells can be carving cells; even indices are walls
        that get removed when the DFS carves a passage between two cells.
        """
        if rows is not None:
            self.rows = rows | 1   # ensure odd
        if cols is not None:
            self.cols = cols | 1

        # Start with all walls
        self.grid: List[List[int]] = [
            [1] * self.cols for _ in range(self.rows)
        ]

        # Carve using DFS from (1, 1)
        self._dfs(1, 1)

        # Start position: top-left open cell
        self.mouse: Tuple[int, int] = (1, 1)
        self.grid[1][1] = 0

        # Exit: bottom-right open cell
        self.exit: Tuple[int, int] = (self.rows - 2, self.cols - 2)
        self.grid[self.rows - 2][self.cols - 2] = 0

        self.won: bool  = False
        self.steps: int = 0

    def _dfs(self, r: int, c: int):
        """Recursive backtracking carver."""
        self.grid[r][c] = 0
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(directions)
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 < nr < self.rows - 1 and 0 < nc < self.cols - 1:
                if self.grid[nr][nc] == 1:
                    # Carve the wall between (r, c) and (nr, nc)
                    self.grid[r + dr // 2][c + dc // 2] = 0
                    self._dfs(nr, nc)

    # ─────────────────────────────────────────────────────────────────────
    # Game logic
    # ─────────────────────────────────────────────────────────────────────

    def move(self, direction: str) -> str:
        """
        Attempt to move the mouse in `direction`.

        Returns a status string: "moved", "blocked", "won", "invalid", "already_won".
        """
        if self.won:
            return "already_won"

        direction = direction.lower().strip()
        if direction not in self.MOVES:
            return "invalid"

        dr, dc = self.MOVES[direction]
        nr, nc = self.mouse[0] + dr, self.mouse[1] + dc

        # Boundary check
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            return "blocked"

        # Wall check
        if self.grid[nr][nc] == 1:
            return "blocked"

        self.mouse = (nr, nc)
        self.steps += 1

        if self.mouse == self.exit:
            self.won = True
            return "won"

        return "moved"

    # ─────────────────────────────────────────────────────────────────────
    # State serialisation
    # ─────────────────────────────────────────────────────────────────────

    def state(self) -> Dict[str, Any]:
        """Return the full maze state as a JSON-serialisable dict."""
        return {
            "grid":  self.grid,
            "rows":  self.rows,
            "cols":  self.cols,
            "mouse": list(self.mouse),
            "exit":  list(self.exit),
            "won":   self.won,
            "steps": self.steps,
        }
