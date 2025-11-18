# dynamic_lidar_map.py

import numpy as np
import math
import os
import matplotlib.pyplot as plt

class DynamicLidarMap:
    """
    Simple global occupancy grid built only from LiDAR.
    -1 = unknown, 0 = free, 1 = occupied
    """

    def __init__(self, width_m=20.0, height_m=20.0, resolution=0.1):
        self.width_m = width_m
        self.height_m = height_m
        self.res = resolution

        self.width_cells = int(width_m / resolution)
        self.height_cells = int(height_m / resolution)

        # World (0,0) is in the middle of the grid
        self.origin_x = -width_m / 2.0
        self.origin_y = -height_m / 2.0

        # Global map: persists across episodes
        self.grid = -1 * np.ones((self.height_cells, self.width_cells),
                                 dtype=np.int8)

        # Per-episode visited flags
        self.visited_episode = np.zeros_like(self.grid, dtype=bool)

    # ---------- basic helpers ----------

    def clear_global_map(self):
        """Call once at the start of training to make the map fully unknown."""
        self.grid[:, :] = -1

    def reset_episode(self):
        """Call at the beginning of each episode."""
        self.visited_episode[:, :] = False

    def world_to_cell(self, x, y):
        """
        Convert world coordinates (meters) to grid indices (row, col).
        Returns None if outside the map.
        """
        col = int((x - self.origin_x) / self.res)
        row = int((y - self.origin_y) / self.res)
        if 0 <= row < self.height_cells and 0 <= col < self.width_cells:
            return row, col
        return None

    # ---------- lidar update ----------

    def update_from_scan(self, robot_x, robot_y, robot_theta,
                         ranges, angle_min, angle_increment, max_range):
        """
        Update occupancy map from a LiDAR scan.
        ranges: list/array of distances, aligned with angles
        """
        robot_cell = self.world_to_cell(robot_x, robot_y)
        if robot_cell is None:
            return

        for i, r in enumerate(ranges):
            if r is None:
                continue
            try:
                r = float(r)
            except (TypeError, ValueError):
                continue
            if math.isinf(r) or math.isnan(r):
                continue

            r = min(r, max_range)

            angle = robot_theta + angle_min + i * angle_increment
            end_x = robot_x + r * math.cos(angle)
            end_y = robot_y + r * math.sin(angle)

            end_cell = self.world_to_cell(end_x, end_y)
            if end_cell is None:
                continue

            self._update_ray(robot_cell, end_cell, r, max_range)

    def _update_ray(self, start_cell, end_cell, r, max_range):
        """
        - All cells along the ray (except the last one) become free (0).
        - If r < max_range, last cell becomes occupied (1).
        """
        x0, y0 = start_cell[1], start_cell[0]
        x1, y1 = end_cell[1], end_cell[0]

        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        x, y = x0, y0
        while True:
            if (x, y) == (x1, y1):
                break

            if 0 <= y < self.height_cells and 0 <= x < self.width_cells:
                if self.grid[y, x] == -1:
                    self.grid[y, x] = 0  # unknown -> free

            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

        # Mark final cell as obstacle (if we hit something)
        if r < max_range:
            if 0 <= y1 < self.height_cells and 0 <= x1 < self.width_cells:
                self.grid[y1, x1] = 1

    # ---------- reward helper ----------

    def reward_for_step(self, robot_x, robot_y,
                        exploration_reward_value=0.5,
                        revisit_penalty_value=-0.1):
        """
        Simple shaping from map:
        +exploration_reward_value when entering a new cell in this episode
        revisit_penalty_value when revisiting the same cell in this episode
        """
        cell = self.world_to_cell(robot_x, robot_y)
        if cell is None:
            return 0.0, 0.0

        r, c = cell
        if not self.visited_episode[r, c]:
            self.visited_episode[r, c] = True
            return exploration_reward_value, 0.0
        else:
            return 0.0, revisit_penalty_value
        
    
    def save_map_png(self, filename, title=None):
        """
        Save the current occupancy grid as a PNG image.

        -1 = unknown (dark)
         0 = free
         1 = occupied (obstacle)
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        plt.figure(figsize=(5, 5))
        plt.imshow(self.grid, origin="lower", vmin=-1, vmax=1)
        plt.colorbar(label="Occupancy (-1 unknown, 0 free, 1 occupied)")
        if title is None:
            title = "Dynamic LiDAR Map"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

