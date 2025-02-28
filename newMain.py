import numpy as np
import matplotlib.pyplot as plt

# Import the sailing environment and MCTS optimizer
# (Assuming the main sailing simulation is imported as needed)
from windAndCurrent import SimplePolarModel, WindGrid, CurrentGrid, SailingEnv
from mcts import optimize_sailing_path

def run_optimization():
    """Run the sailing path optimization"""
    # Create polar model
    polar_model = SimplePolarModel()
    
    # Create environment
    grid_size = 40  # Reduced grid size for faster computation
    env = SailingEnv(
        grid_size_x=grid_size,
        grid_size_y=grid_size,
        polar_model=polar_model,
        start_pos=(5, 5),
        goal_pos=(35, 35),  # Opposite corner
        use_currents=True,
        base_current_speed=1.0,
        base_current_dir=90,  # Current flowing east
        base_wind_speed=15.0,
        base_wind_dir=0.0    # Wind from the north
    )
    
    # Manually set a larger goal radius to make it easier to reach
    env.goal_radius = 3.0
    
    # Reset to initialize
    env.reset()
    
    # Find optimal path
    # Adjusted parameters for better results:
    path = optimize_sailing_path(
        env,
        position_bins=10,     # Higher resolution grid
        heading_bins=16,      # More precise heading control
        mcts_iterations=2000  # More iterations for better planning
    )
    
    return env, path

if __name__ == "__main__":
    env, path = run_optimization()
    
    # Print path statistics
    start_pos = env.start_pos
    goal_pos = env.goal_pos
    path_length = len(path)
    
    print(f"Sailing from {start_pos} to {goal_pos}")
    print(f"Path found with {path_length} steps")
    
    # Calculate straight-line distance vs actual path length
    straight_dist = np.sqrt((goal_pos[0] - start_pos[0])**2 + (goal_pos[1] - start_pos[1])**2)
    total_path_dist = 0
    for i in range(1, len(path)):
        total_path_dist += np.sqrt((path[i][0] - path[i-1][0])**2 + (path[i][1] - path[i-1][1])**2)
    
    print(f"Straight-line distance: {straight_dist:.2f}")
    print(f"Actual path distance: {total_path_dist:.2f}")
    print(f"Path efficiency: {straight_dist/total_path_dist:.2%}")