import os
import time
from pysim2d import Simulation2D # Import the Simulation2D class from the pysim2d.py file

# --- Create a temporary world file for the example ---
# This is the content from Simulation2d/world/test.world
TEST_WORLD_CONTENT = """
# border lines -> x1 y1 x2 y2
0.0 0.0 12.0 0.0
0.0 0.0 0.0 12.0
12.0 0.0 12.0 12.0
0.0 12.0 12.0 12.0

# rectangle lines -> x1 y1 x2 y2
# circles -> x y r
7.975 4.676 1.270
3.490 3.792 0.637
6.943 9.047 0.620
9.319 7.670 0.543
1.992 10.468 0.275
"""

world_filename = "temp_test_world.world"
with open(world_filename, "w") as f:
    f.write(TEST_WORLD_CONTENT)

# ----------------------------------------------------

# 1. Create an instance of the simulation
sim = Simulation2D()

# 2. Initialize the simulation with the world file
if not sim.init(world_filename):
    print(f"Failed to initialize simulation with {world_filename}")
else:
    print(f"Simulation initialized with {world_filename}")

    # 3. Set the robot's starting pose
    sim.set_robot_pose(1.5, 1.5, 0.0) # x, y, orientation (radians)
    print(f"Robot set to: (x={sim.get_robot_pose_x():.2f}, y={sim.get_robot_pose_y():.2f}, angle={sim.get_robot_pose_orientation():.2f})")

    # 4. Get observation data
    obs_size = sim.observation_size()
    print(f"Lidar observation size: {obs_size}")
    
    # Print the middle laser beam's reading
    middle_beam_index = obs_size // 2
    print(f"Middle beam observation (normalized): {sim.observation_at(middle_beam_index):.3f}")

    # 5. Run the simulation for a few steps
    print("\nRunning simulation...")
    for i in range(10):
        # Move forward (0.5 m/s) and turn left (0.1 rad/s)
        sim.step(0.5, 0.1)
        
        print(f"Step {i+1}: Pos=(x={sim.get_robot_pose_x():.2f}, y={sim.get_robot_pose_y():.2f}), Collided={sim.done()}")
        
        # Stop if collided
        if sim.done():
            print("Collision detected!")
            break
            
        # (Optional) A small delay so you can read the output
        # time.sleep(0.1)

# Clean up the temporary world file
os.remove(world_filename)

