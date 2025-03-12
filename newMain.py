import numpy as np
import random
import matplotlib.pyplot as plt

# Import the sailing environment and MCTS optimizer
# (Assuming the main sailing simulation is imported as needed)
from windAndCurrent import SimplePolarModel, WindGrid, CurrentGrid, SailingEnv
from mcts import optimize_sailing_path


def run_optimization(start_pos, goal_pos, use_currents, base_current_speed, base_current_dir, base_wind_speed, base_wind_dir):
    """Run the sailing path optimization"""
    # Create polar model
    polar_model = SimplePolarModel()

    # Create environment
    grid_size = 40  # Reduced grid size for faster computation
    """
    env = SailingEnv(
        grid_size_x=grid_size,
        grid_size_y=grid_size,
        polar_model=polar_model,
        start_pos=(2.5, 2.5),
        goal_pos=(37.5, 37.5),  # Opposite corner
        use_currents=True,
        base_current_speed=1.0,
        base_current_dir=90,  # Current flowing east
        base_wind_speed=15.0,
        base_wind_dir=120.0  # Wind from the north
    )
    """
    env = SailingEnv(
        grid_size_x=grid_size,
        grid_size_y=grid_size,
        polar_model=polar_model,
        start_pos=start_pos,
        goal_pos=goal_pos,  # Opposite corner
        use_currents=use_currents,
        base_current_speed=base_current_speed,
        base_current_dir=base_current_dir,  # Current flowing east
        base_wind_speed=base_wind_speed,
        base_wind_dir=base_wind_dir  # Wind from the north
    )
    # Manually set a larger goal radius to make it easier to reach
    env.goal_radius = 4.0

    # Reset to initialize
    env.reset()

    # Find optimal path
    path, elapsed_time, time_out = optimize_sailing_path(
        env,
        mcts_iterations=10000  # More iterations for better planning
    )

    return env, path, elapsed_time, time_out

def monte_carlo(iters=1000):
    finish_times = []
    num_timed_out = 0
    for _ in range(iters):
        # Randomize parameters
        goal_x = random.uniform(25, 37.5)
        goal_y = random.uniform(25, 37.5)
        base_current_speed = random.uniform(0.5, 1.5)
        base_current_dir = random.uniform(0, 360)
        base_wind_speed = random.uniform(13, 16)
        base_wind_dir = random.uniform(0, 360)

        env, path, elapsed_time, time_out = run_optimization(
            start_pos=(2.5, 2.5),  # Fixed start position
            goal_pos=(goal_x, goal_y),
            use_currents=True,
            base_current_speed=base_current_speed,
            base_current_dir=base_current_dir,
            base_wind_speed=base_wind_speed,
            base_wind_dir=base_wind_dir
        )

        env.goal_radius = 4.0
        env.reset()

        print(f"Sailing from {env.start_pos} to {env.goal_pos}")

        if time_out:
            num_timed_out += 1
            print("DID NOT REACH GOAL")
        else:
            finish_times.append(elapsed_time)
            print("REACHED GOAL")
    average_finish_time = np.average(finish_times)
    print(f"Average finish time for MCTS: {average_finish_time}")


if __name__ == "__main__":
    """
    env, path, elapsed_time, time_out = run_optimization()

    # Print path statistics
    start_pos = env.start_pos
    goal_pos = env.goal_pos
    path_length = len(path)

    print(f"Sailing from {start_pos} to {goal_pos}")
    print(f"Path found with {path_length} steps")

    # Calculate straight-line distance vs actual path length
    straight_dist = np.sqrt((goal_pos[0] - start_pos[0]) ** 2 + (goal_pos[1] - start_pos[1]) ** 2)
    total_path_dist = 0
    for i in range(1, len(path)):
        total_path_dist += np.sqrt((path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2)

    print(f"Straight-line distance: {straight_dist:.2f}")
    print(f"Actual path distance: {total_path_dist:.2f}")
    print(f"Path efficiency: {straight_dist / total_path_dist:.2%}")
    if time_out:
        print("Did not reach goal")
    else:
        print("Reached goal")
    """
    monte_carlo()