import numpy as np
import math
import random
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle, Arrow

# Epsilon for float comparisons
EPSILON = 1e-6

class Visualizer:
    """
    Handles rendering the simulation state using Matplotlib.
    """
    def __init__(self):
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.first_draw = True

    def draw(self, robot, data, target_circle):
        """
        Draws the current state of the simulation.
        """
        self.ax.clear()

        # --- Draw World Geometry ---
        # Draw Lines
        for line in data.lines:
            self.ax.plot([line.p1[0], line.p2[0]], [line.p1[1], line.p2[1]], 'k-') # 'k-' is black line

        # Draw Circles
        for circle in data.circles:
            patch = MplCircle((circle.center[0], circle.center[1]), circle.radius, color='blue', fill=False, linewidth=2)
            self.ax.add_patch(patch)

        # --- Draw Target ---
        target_patch = MplCircle((target_circle.center[0], target_circle.center[1]), target_circle.radius, color='magenta', fill=True, alpha=0.5)
        self.ax.add_patch(target_patch)

        # --- Draw Robot ---
        robot_patch = MplCircle((robot.x, robot.y), robot.collision_radius, color='red', fill=True, alpha=0.8)
        self.ax.add_patch(robot_patch)
        
        # Draw robot direction arrow
        dx = robot.collision_radius * 1.5 * np.cos(robot.orientation)
        dy = robot.collision_radius * 1.5 * np.sin(robot.orientation)
        arrow = Arrow(robot.x, robot.y, dx, dy, width=0.1, color='red')
        self.ax.add_patch(arrow)

        # --- Draw Lidar Rays ---
        lidar = robot.lidar
        robot_transform = robot.get_pose_transform()
        
        # Get local endpoints
        local_endpoints_x = lidar.laser_vectors[:, 0] * lidar.laser_distances
        local_endpoints_y = lidar.laser_vectors[:, 1] * lidar.laser_distances
        
        # Stack to (N, 3) for homogeneous transformation
        local_points = np.ones((lidar.laser_size, 3), dtype=np.float32)
        local_points[:, 0] = local_endpoints_x
        local_points[:, 1] = local_endpoints_y
        
        # Transform to world space
        world_points = (robot_transform @ local_points.T).T
        
        # Draw lines from robot_pos to each world_point
        robot_pos = np.array([robot.x, robot.y])
        for end_point in world_points:
            self.ax.plot([robot_pos[0], end_point[0]], [robot_pos[1], end_point[1]], 'g-', alpha=0.1) # 'g-' is green line

        # --- Set Plot Limits and Aspect ---
        self.ax.set_aspect('equal')
        if self.first_draw:
            # Set limits based on world boundaries on the first draw
            self.ax.set_xlim(data.area_min_x - 1, data.area_max_x + 1)
            self.ax.set_ylim(data.area_min_y - 1, data.area_max_y + 1)
            self.first_draw = False

        # --- Redraw Canvas ---
        self.fig.canvas.draw()
        plt.pause(0.001) # Important for interactive mode

class Line:
    """
    Represents a 2D line segment, equivalent to the C++ Line class.
    Uses numpy arrays for points.
    """
    def __init__(self, x1, y1, x2, y2):
        self.p1 = np.array([x1, y1], dtype=np.float32)
        self.p2 = np.array([x2, y2], dtype=np.float32)

    def __repr__(self):
        return f"Line({self.p1}, {self.p2})"

class Circle:
    """
    Represents a 2D circle, equivalent to the C++ Circle class.
    Uses a numpy array for the center.
    """
    def __init__(self, x, y, radius):
        self.center = np.array([x, y], dtype=np.float32)
        self.radius = np.float32(radius)

    def __repr__(self):
        return f"Circle(center={self.center}, radius={self.radius})"

class Lidar:
    """
    Python implementation of the C++ Lidar class.
    """
    def __init__(self, hz=40.0, angle_step=0.0043633, angle_min=-2.35619, 
                 angle_max=2.35619, range_min=0.06, range_max=20.0, bias=0.004):
        self.hz = hz
        self.angle_step = angle_step
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.range_min = range_min
        self.range_max = range_max
        self.bias = bias
        
        self.laser_angles = np.arange(angle_min, angle_max + angle_step, angle_step, dtype=np.float32)
        # Pre-calculate unit vectors for each laser beam
        self.laser_vectors = np.array(
            [np.cos(self.laser_angles), np.sin(self.laser_angles)], 
            dtype=np.float32
        ).T
        self.laser_distances = np.full_like(self.laser_angles, self.range_max)
        self.laser_size = len(self.laser_angles)

    def fill_laser_distance_with_range_max(self):
        """Resets all laser distances to the maximum range."""
        self.laser_distances.fill(self.range_max)

    def apply_bias(self):
        """Applies random noise to laser measurements."""
        noise = (np.random.rand(self.laser_size).astype(np.float32) * 2.0 - 1.0) * self.bias
        self.laser_distances += noise

    def calculate_laser_collision_from_line(self, robot_transform, line):
        """
        Calculates lidar collisions with a line segment.
        This ports the logic from the C++ Lidar::calculate_laser_collision_from_line.
        """
        # Transform line to robot's local coordinate frame
        inv_transform = np.linalg.inv(robot_transform)
        p1_local = inv_transform @ np.array([line.p1[0], line.p1[1], 1.0])
        p2_local = inv_transform @ np.array([line.p2[0], line.p2[1], 1.0])
        
        x1, y1 = p1_local[0], p1_local[1]
        x2, y2 = p2_local[0], p2_local[1]

        x3 = self.laser_vectors[:, 0]
        y3 = self.laser_vectors[:, 1]
        
        # Line segment 1 (laser): (x3, y3) to (x4, y4) which is (0, 0)
        # Line segment 2 (wall): (x1, y1) to (x2, y2)
        
        # From C++ code: (x1*y2 - y1*x2)
        x1y2_y1x2 = x1 * y2 - y1 * x2
        # From C++ code: (x1 - x2)
        d_x1x2 = x1 - x2
        # From C++ code: (y1 - y2)
        d_y1y2 = y1 - y2
        
        # From C++ code: (x1 - x2)(y3) - (y1 - y2)(x3)
        # Note: x4, y4 are 0, so (x3-x4) = x3, (y3-y4) = y3
        denominator = d_x1x2 * y3 - d_y1y2 * x3
        
        # Avoid division by zero
        mask_den_zero = np.abs(denominator) < EPSILON
        
        # Calculate intersection point
        # From C++ code: (x1y2_y1x2 * x3) / denominator
        x_intersect = x1y2_y1x2 * x3 / (denominator + EPSILON)
        # From C++ code: (x1y2_y1x2 * y3) / denominator
        y_intersect = x1y2_y1x2 * y3 / (denominator + EPSILON)
        
        # --- Check if intersection is valid ---
        
        # 1. Check if intersection is in the positive direction of the laser
        # (0 <= x_intersect) == (0 <= x3) and (0 <= y_intersect) == (0 <= y3)
        mask_direction = ((x_intersect >= 0) == (x3 >= 0)) & ((y_intersect >= 0) == (y3 >= 0))
        
        # 2. Check if intersection is within the line *segment*
        min_x, max_x = min(x1, x2) - EPSILON, max(x1, x2) + EPSILON
        min_y, max_y = min(y1, y2) - EPSILON, max(y1, y2) + EPSILON
        
        mask_segment = (x_intersect >= min_x) & (x_intersect <= max_x) & \
                       (y_intersect >= min_y) & (y_intersect <= max_y)
        
        # Combine masks
        valid_mask = mask_direction & mask_segment & ~mask_den_zero
        
        if np.any(valid_mask):
            # Calculate distance to valid intersections
            distances = np.sqrt(x_intersect[valid_mask]**2 + y_intersect[valid_mask]**2)
            
            # Update laser distances
            current_distances = self.laser_distances[valid_mask]
            self.laser_distances[valid_mask] = np.minimum(current_distances, distances)

    def calculate_laser_collision_from_circle(self, robot_transform, circle):
        """
        Calculates lidar collisions with a circle.
        Ports logic from C++ Lidar::calculate_laser_collision_from_circle.
        """
        # Transform circle center to robot's local frame
        inv_transform = np.linalg.inv(robot_transform)
        center_local = inv_transform @ np.array([circle.center[0], circle.center[1], 1.0])
        cx, cy = center_local[0], center_local[1]
        r = circle.radius
        r_2 = r * r

        # Vectorized line-circle intersection
        # Laser line: P = t * D, where D = (x3, y3) and t >= 0
        # Circle: (x - cx)^2 + (y - cy)^2 = r^2
        # Substitute line eq into circle eq: (t*x3 - cx)^2 + (t*y3 - cy)^2 = r^2
        # t^2*(x3^2 + y3^2) - 2*t*(cx*x3 + cy*y3) + (cx^2 + cy^2 - r^2) = 0
        # Since D is a unit vector, (x3^2 + y3^2) = 1
        # t^2 - 2*t*(D . C) + (C . C - r^2) = 0
        # Where C = (cx, cy) and D = (x3, y3)
        
        x3 = self.laser_vectors[:, 0]
        y3 = self.laser_vectors[:, 1]
        
        # (D . C)
        dot_dc = cx * x3 + cy * y3
        # (C . C - r^2)
        c_dot_c_minus_r2 = cx*cx + cy*cy - r_2
        
        # Discriminant of quadratic formula: b^2 - 4ac
        # a=1, b=-2*(D.C), c=(C.C - r^2)
        # discriminant = (2*dot_dc)^2 - 4*1*c_dot_c_minus_r2
        # simplified: 4 * (dot_dc^2 - c_dot_c_minus_r2)
        discriminant = dot_dc**2 - c_dot_c_minus_r2
        
        # Only consider rays that can intersect
        mask_intersect = discriminant >= 0
        if not np.any(mask_intersect):
            return

        sqrt_discriminant = np.sqrt(discriminant[mask_intersect])
        
        # Quadratic formula: t = (-b +/- sqrt(discriminant)) / 2a
        # t = (2*dot_dc +/- 2*sqrt_discriminant) / 2
        # t = dot_dc +/- sqrt_discriminant
        t1 = dot_dc[mask_intersect] - sqrt_discriminant
        t2 = dot_dc[mask_intersect] + sqrt_discriminant
        
        # We need the smallest non-negative distance
        # t1 is always <= t2. If t1 is positive, it's the closer intersection.
        # If t1 is negative and t2 positive, robot is inside circle, use t2.
        
        mask_t1_pos = t1 >= 0
        mask_t2_pos = t2 >= 0

        # Valid distances
        valid_t_mask = mask_t1_pos | mask_t2_pos
        if not np.any(valid_t_mask):
            return

        # Get the smallest valid t
        t_values = np.full_like(t1, np.inf)
        t_values[mask_t1_pos] = t1[mask_t1_pos]
        t_values[~mask_t1_pos & mask_t2_pos] = t2[~mask_t1_pos & mask_t2_pos]

        # Final distances to update
        final_distances = t_values[valid_t_mask]
        
        # Update laser distances
        # We need to map 'final_distances' back to the original laser_distances array
        original_indices = np.where(mask_intersect)[0][valid_t_mask]
        
        if original_indices.size > 0:
            current_distances = self.laser_distances[original_indices]
            self.laser_distances[original_indices] = np.minimum(current_distances, final_distances)

class Robot:
    """
    Python implementation of the C++ Robot class.
    """
    def __init__(self, x=0.0, y=0.0, orientation=0.0, collision_radius=0.175, max_velocity=1.0):
        self.x = np.float32(x)
        self.y = np.float32(y)
        self.orientation = np.float32(orientation) # in radians
        self.collision_radius = np.float32(collision_radius)
        self.max_velocity = np.float32(max_velocity)
        self.lidar = Lidar()
    
    def set_pose(self, x, y, orientation):
        self.x = np.float32(x)
        self.y = np.float32(y)
        self.orientation = np.float32(orientation)

    def move(self, linear_velocity, angular_velocity):
        """
        Moves the robot based on velocities.
        Assumes linear_velocity is in the robot's forward direction.
        """
        # Update orientation
        self.orientation += angular_velocity
        # Normalize orientation to [-pi, pi]
        self.orientation = (self.orientation + np.pi) % (2 * np.pi) - np.pi
        
        # Update position
        self.x += linear_velocity * np.cos(self.orientation)
        self.y += linear_velocity * np.sin(self.orientation)

    def get_position(self):
        return np.array([self.x, self.y], dtype=np.float32)

    def get_pose_transform(self):
        """
        Returns a 3x3 2D affine transformation matrix for the robot's pose.
        """
        c = np.cos(self.orientation)
        s = np.sin(self.orientation)
        return np.array([
            [c, -s, self.x],
            [s,  c, self.y],
            [0,  0,  1.0]
        ], dtype=np.float32)

class DataContainer:
    """
    Python implementation of the C++ DataContainer class.
    Holds world geometry (lines and circles).
    """
    def __init__(self):
        self.lines = []
        self.circles = []
        self.area_min_x = np.inf
        self.area_min_y = np.inf
        self.area_max_x = -np.inf
        self.area_max_y = -np.inf

    def set_world(self, lines, circles):
        """Sets the world geometry and calculates boundaries."""
        self.lines = lines
        self.circles = circles
        self.set_area()

    def set_area(self):
        """Calculates the bounding box of the world."""
        if not self.lines and not self.circles:
            return

        for line in self.lines:
            self.area_min_x = min(self.area_min_x, line.p1[0], line.p2[0])
            self.area_min_y = min(self.area_min_y, line.p1[1], line.p2[1])
            self.area_max_x = max(self.area_max_x, line.p1[0], line.p2[0])
            self.area_max_y = max(self.area_max_y, line.p1[1], line.p2[1])

        for circle in self.circles:
            self.area_min_x = min(self.area_min_x, circle.center[0] - circle.radius)
            self.area_min_y = min(self.area_min_y, circle.center[1] - circle.radius)
            self.area_max_x = max(self.area_max_x, circle.center[0] + circle.radius)
            self.area_max_y = max(self.area_max_y, circle.center[1] + circle.radius)

    def calculate_robot_collision(self, robot):
        """
        Checks if the robot is in collision with any world object.
        Ports logic from DataContainer::calculate_robot_collision.
        """
        robot_pos = robot.get_position()
        robot_radius = robot.collision_radius

        # 1. Check circle collisions
        for circle in self.circles:
            dist = np.linalg.norm(robot_pos - circle.center)
            if dist <= (robot_radius + circle.radius):
                return True # Collision

        # 2. Check line collisions
        for line in self.lines:
            # Vector from line.p1 to robot_pos
            p1_to_robot = robot_pos - line.p1
            # Vector for the line segment
            line_vec = line.p2 - line.p1
            
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq < EPSILON:
                # Line is a point
                dist = np.linalg.norm(p1_to_robot)
            else:
                # Project robot_pos onto the line
                # t = dot(p1_to_robot, line_vec) / |line_vec|^2
                t = np.dot(p1_to_robot, line_vec) / line_len_sq
                
                if t < 0.0:
                    # Closest point is p1
                    dist = np.linalg.norm(robot_pos - line.p1)
                elif t > 1.0:
                    # Closest point is p2
                    dist = np.linalg.norm(robot_pos - line.p2)
                else:
                    # Closest point is on the segment
                    projection = line.p1 + t * line_vec
                    dist = np.linalg.norm(robot_pos - projection)
            
            if dist <= robot_radius:
                return True # Collision

        # 3. Check if robot is out of bounds
        if robot.x <= self.area_min_x or robot.x >= self.area_max_x or \
           robot.y <= self.area_min_y or robot.y >= self.area_max_y:
            return True # Collision

        return False # No collision

    def calculate_lidar_collision(self, robot):
        """
        Calculates all lidar collisions for the robot.
        Ports logic from DataContainer::calculate_lidar_collision.
        """
        robot_transform = robot.get_pose_transform()
        robot.lidar.fill_laser_distance_with_range_max()

        # Check collisions with lines
        for line in self.lines:
            robot.lidar.calculate_laser_collision_from_line(robot_transform, line)

        # Check collisions with circles
        for circle in self.circles:
            robot.lidar.calculate_laser_collision_from_circle(robot_transform, circle)
        
        # Apply noise
        robot.lidar.apply_bias()

class Simulation2D:
    """
    Main simulation class, equivalent to the C++ Simulation2D.
    This is the public API for the module.
    """
    def __init__(self):
        self.data = DataContainer()
        self.robot = Robot()
        self.collision = False
        self.time_step_check_collision = 0.0
        self.time_step_check_laser = 0.0
        self.visualizer = None # Initialize visualizer
    
    def _calculate_iteration(self):
        """Ports logic from C++ Simulation2D::calculate_iteration"""
        if self.robot.max_velocity == 0:
             self.robot.max_velocity = 1.0 # Avoid division by zero
        self.time_step_check_collision = self.robot.collision_radius / self.robot.max_velocity
        self.time_step_check_laser = 1.0 / self.robot.lidar.hz

    def _load_world(self, world_path):
        """
        Ports logic from C++ Simulation2D::load_world
        """
        if not os.path.exists(world_path):
            print(f"Error: Could not read file: {world_path}")
            return False
        
        lines = []
        circles = []
        try:
            with open(world_path, 'r') as f:
                for line_str in f:
                    line_str = line_str.strip()
                    if not line_str or line_str.startswith('#'):
                        continue
                    
                    parts = line_str.split()
                    
                    if len(parts) == 3:
                        # Circle: x y r
                        x, y, r = map(float, parts)
                        circles.append(Circle(x, y, r))
                    elif len(parts) == 4:
                        # Line: x1 y1 x2 y2
                        x1, y1, x2, y2 = map(float, parts)
                        lines.append(Line(x1, y1, x2, y2))
                    else:
                        print(f"Warn: unknown line format: {line_str}")
                        
            self.data.set_world(lines, circles)
            return True
        except Exception as e:
            print(f"Error parsing world file {world_path}: {e}")
            return False

    # --- Public API Methods (from pysim2d.cpp) ---

    def init(self, world_path):
        """
        Initializes the simulation with a world file.
        """
        self.collision = False
        self._calculate_iteration()
        if not self._load_world(world_path):
            print(f"Failed to load world: {world_path}")
            return False
        
        # Perform initial lidar scan
        self.data.calculate_lidar_collision(self.robot)
        return True

    def step(self, linear_velocity, angular_velocity, skip_number=1):
        """
        Steps the simulation forward.
        Ports logic from C++ Simulation2D::step
        """
        if skip_number < 1:
            skip_number = 1
        
        # Scale velocities by the time step for collision checking
        # This makes them "velocity per collision check step"
        lv_step = linear_velocity * self.time_step_check_collision
        av_step = angular_velocity * self.time_step_check_collision
        
        time_step_end = self.time_step_check_laser * skip_number
        time_step = 0.0

        while time_step < time_step_end and not self.collision:
            self.robot.move(lv_step, av_step)
            self.collision = self.data.calculate_robot_collision(self.robot)
            time_step += self.time_step_check_collision
        
        if not self.collision:
            # Correct for over-stepping
            time_step_to_laserscan = time_step - time_step_end
            # Move backward by the over-stepped amount
            # Note: C++ code moves forward by this amount, which seems like a bug.
            # Replicating C++ bug:
            lv_over = linear_velocity * time_step_to_laserscan
            av_over = angular_velocity * time_step_to_laserscan
            self.robot.move(lv_over, av_over)
            
        # Update lidar scan
        self.data.calculate_lidar_collision(self.robot)

    def set_robot_pose(self, x, y, orientation):
        """Sets the robot's pose and resets the collision flag."""
        self.robot.set_pose(x, y, orientation)
        self.collision = False
        # Update lidar at new pose
        self.data.calculate_lidar_collision(self.robot)

    def get_robot_pose_x(self):
        """Returns the robot's X coordinate."""
        return self.robot.x

    def get_robot_pose_y(self):
        """Returns the robot's Y coordinate."""
        return self.robot.y

    def get_robot_pose_orientation(self):
        """Returns the robot's orientation in radians."""
        return self.robot.orientation

    def observation_size(self):
        """Returns the number of laser beams in the lidar scan."""
        return self.robot.lidar.laser_size

    def observation_at(self, index):
        """
        Returns the normalized distance for a specific laser beam.
        """
        if 0 <= index < self.robot.lidar.laser_size:
            dist = self.robot.lidar.laser_distances[index]
            return dist / self.robot.lidar.range_max
        return 1.0 # Default value if index is out of bounds

    def observation_min_clustered_at(self, index, cluster_size):
        """
        Returns the minimum normalized distance in a "cluster" of beams.
        """
        value = 1.0
        start_index = index * cluster_size
        end_index = min(start_index + cluster_size, self.robot.lidar.laser_size)
        
        if start_index >= end_index:
            return 1.0
            
        cluster_distances = self.robot.lidar.laser_distances[start_index:end_index]
        if cluster_distances.size > 0:
            value = np.min(cluster_distances) / self.robot.lidar.range_max
        
        return value

    def observation_min_clustered_size(self, cluster_size):
        """Returns the number of clusters."""
        if cluster_size == 0:
            return 0
        return self.robot.lidar.laser_size // cluster_size
        
    def done(self):
        """Returns true if the robot has collided."""
        return self.collision

    def visualize(self, end_x, end_y, end_radius):
        """
        Uses matplotlib to draw the current simulation state.
        """
        if self.visualizer is None:
            # Lazily initialize the visualizer on first call
            self.visualizer = Visualizer()
        
        # Create a temporary target circle object (as C++ version does)
        target_circle = Circle(end_x, end_y, end_radius)
        
        try:
            self.visualizer.draw(self.robot, self.data, target_circle)
        except Exception as e:
            # Handle plot window closed exception
            if 'TclError' in str(e) or 'application has been destroyed' in str(e):
                print("Visualization window closed.")
                # We could set visualizer to None to reopen, or just stop visualizing
                self.visualizer = None 
            else:
                raise e

