import numpy as np
import random
import time
from windAndCurrent import SimplePolarModel, WindGrid, CurrentGrid, SailingEnv

def baseline_sailing_path(env, max_steps=100):
    """
    Implements a simple baseline sailing algorithm that follows a basic strategy:
    - Maintain heading toward the goal
    - Adjusts course to maximize VMG (velocity made good) to goal
    - Avoids edges of the grid
    
    Args:
        env: SailingEnv - The sailing environment
        max_steps: int - Maximum number of steps to try
        
    Returns:
        path: list - List of positions
        elapsed_time: float - Time taken to reach goal
        time_out: bool - Whether the algorithm timed out
    """
    # Initialize with current position
    path = [tuple(env.position)]
    
    # Track time out flag
    time_out = True
    
    # Simple waypoints toward goal
    goal_x, goal_y = env.goal_pos
    start_x, start_y = env.start_pos
    
    # Direction to goal
    goal_direction = np.arctan2(goal_x - start_x, goal_y - start_y)
    
    steps_taken = 0
    total_time = 0.0
    
    # Begin sailing
    for step in range(max_steps):
        steps_taken = step
        
        # Current position
        curr_x, curr_y = env.position
        
        # Calculate direction to goal
        dx = goal_x - curr_x
        dy = goal_y - curr_y
        goal_bearing = np.degrees(np.arctan2(dx, dy)) % 360
        
        # Get current wind and current
        wind_speed, wind_dir = env.wind_grid.get_wind_at_position(curr_x, curr_y)
        current_speed, current_dir = env.current_grid.get_current_at_position(curr_x, curr_y)
        
        # Calculate optimal heading based on wind and goal
        # Find action with best VMG toward goal
        best_vmg = -float('inf')
        best_action = 0
        
        for action_idx, heading_delta in enumerate(env.actions):
            # Calculate potential new heading
            potential_heading = (env.heading + heading_delta) % 360
            
            # Calculate VMG to goal
            heading_rad = np.radians(potential_heading)
            goal_rad = np.radians(goal_bearing)
            
            # Calculate true wind angle
            twa = abs((potential_heading - wind_dir) % 360)
            if twa > 180:
                twa = 360 - twa
                
            # Get boat speed from polar model
            boat_speed = env.polar_model.get_boat_speed(twa, wind_speed)
            
            # Account for current
            current_rad = np.radians(current_dir)
            boat_dx = boat_speed * np.sin(heading_rad)
            boat_dy = boat_speed * np.cos(heading_rad)
            current_dx = current_speed * np.sin(current_rad)
            current_dy = current_speed * np.cos(current_rad)
            
            # Combined velocity
            total_dx = boat_dx + current_dx
            total_dy = boat_dy + current_dy
            
            # Calculate VMG
            vmg = total_dx * np.sin(goal_rad) + total_dy * np.cos(goal_rad)
            
            # Penalties for edge proximity
            edge_penalty = 0
            if curr_x < 5 or curr_x > env.grid_size_x - 5 or curr_y < 5 or curr_y > env.grid_size_y - 5:
                edge_penalty = 2.0
                
            # Consider turning penalties for drastic turns
            turn_angle = abs(heading_delta)
            turn_penalty = 0.01 * turn_angle
            
            # Combined score
            adjusted_vmg = vmg - edge_penalty - turn_penalty
            
            if adjusted_vmg > best_vmg:
                best_vmg = adjusted_vmg
                best_action = action_idx
        
        # Take the action with best VMG
        state, reward, done, _ = env.step(best_action)
        path.append(tuple(env.position))
        
        # Check if we've reached the goal
        if done:
            time_out = False
            break
    
    elapsed_time = steps_taken * 1.0  # Assuming each step takes 1 unit of time
    
    return path, elapsed_time, time_out

def run_baseline_optimization(start_pos, goal_pos, use_currents, base_current_speed, base_current_dir, base_wind_speed, base_wind_dir):
    """Setup and run baseline sailing optimization"""
    # Create polar model
    polar_model = SimplePolarModel()
    
    # Create environment
    grid_size = 40
    env = SailingEnv(
        grid_size_x=grid_size,
        grid_size_y=grid_size,
        polar_model=polar_model,
        start_pos=start_pos,
        goal_pos=goal_pos,
        use_currents=use_currents,
        base_current_speed=base_current_speed,
        base_current_dir=base_current_dir,
        base_wind_speed=base_wind_speed,
        base_wind_dir=base_wind_dir
    )
    
    # Set a reasonable goal radius
    env.goal_radius = 4.0
    
    # Reset to initialize
    env.reset()
    
    # Find optimal path
    path, elapsed_time, time_out = baseline_sailing_path(env)
    
    return env, path, elapsed_time, time_out