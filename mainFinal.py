import numpy as np
import random
import matplotlib.pyplot as plt
import time
import os
import multiprocessing as mp
from functools import partial
from matplotlib.patches import Circle

# Import algorithms
from windAndCurrent import SimplePolarModel, WindGrid, CurrentGrid, SailingEnv
from mcts import optimize_sailing_path
from baseline import run_baseline_optimization
from base_q import run_qlearning_optimization

def calculate_travel_time(path, env):
    """
    Calculate accurate travel time using polar model for each path segment
    with specific adjustments for Q-learning paths
    """
    if path is None or len(path) < 2:
        return float('inf')
    
    # Simplify the path if it has too many small steps (common in Q-learning)
    # The threshold for simplification can be adjusted based on your needs
    if len(path) > 20:
        simplified_path = simplify_path(path, tolerance=0.5)
    else:
        simplified_path = path
    
    total_time = 0.0
    
    for i in range(1, len(simplified_path)):
        # Current and previous positions
        curr_x, curr_y = simplified_path[i]
        prev_x, prev_y = simplified_path[i-1]
        
        # Calculate segment distance and heading
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        segment_dist = np.sqrt(dx**2 + dy**2)
        segment_heading = np.degrees(np.arctan2(dx, dy)) % 360
        
        # Get wind at midpoint
        mid_x = (curr_x + prev_x) / 2
        mid_y = (curr_y + prev_y) / 2
        wind_speed, wind_dir = env.wind_grid.get_wind_at_position(mid_x, mid_y)
        
        # Calculate true wind angle
        true_wind_angle = abs((segment_heading - wind_dir) % 360)
        if true_wind_angle > 180:
            true_wind_angle = 360 - true_wind_angle
        
        # Get boat speed from polar model
        boat_speed = env.polar_model.get_boat_speed(true_wind_angle, wind_speed)
        
        # Get current at midpoint
        current_speed, current_dir = env.current_grid.get_current_at_position(mid_x, mid_y)
        
        # Convert to radians
        heading_rad = np.radians(segment_heading)
        current_rad = np.radians(current_dir)
        
        # Calculate boat and current velocity components
        boat_dx = boat_speed * np.sin(heading_rad)
        boat_dy = boat_speed * np.cos(heading_rad)
        current_dx = current_speed * np.sin(current_rad)
        current_dy = current_speed * np.cos(current_rad)
        
        # Calculate effective speed
        effective_dx = boat_dx + current_dx
        effective_dy = boat_dy + current_dy
        
        # Project onto travel direction
        travel_dir_rad = np.arctan2(dx, dy)
        effective_speed = (effective_dx * np.sin(travel_dir_rad) + 
                          effective_dy * np.cos(travel_dir_rad))
        
        # Ensure minimum speed to avoid division by zero
        if effective_speed <= 0.1:
            effective_speed = 0.1
        
        # Calculate time for segment
        segment_time = segment_dist / effective_speed
        total_time += segment_time
    
    return total_time

def simplify_path(path, tolerance=0.5):
    """
    Simplify a path by removing points that don't contribute significantly to the path shape
    Uses the Ramer-Douglas-Peucker algorithm
    
    Args:
        path: List of (x, y) positions
        tolerance: Maximum distance of a point to a line segment to be considered insignificant
        
    Returns:
        Simplified path as a list of (x, y) positions
    """
    if len(path) <= 2:
        return path
    
    # Find the point with the maximum distance from the line segment
    max_distance = 0
    max_index = 0
    
    for i in range(1, len(path) - 1):
        distance = point_line_distance(path[i], path[0], path[-1])
        if distance > max_distance:
            max_distance = distance
            max_index = i
    
    # If max distance is greater than tolerance, recursively simplify
    if max_distance > tolerance:
        # Recursive call
        results1 = simplify_path(path[:max_index + 1], tolerance)
        results2 = simplify_path(path[max_index:], tolerance)
        
        # Merge the results (excluding duplicate)
        return results1[:-1] + results2
    else:
        # All points are within tolerance, simplify to just endpoints
        return [path[0], path[-1]]

def point_line_distance(point, line_start, line_end):
    """Calculate the perpendicular distance from a point to a line segment"""
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Calculate the denominator
    line_length_squared = (x2 - x1)**2 + (y2 - y1)**2
    
    # If the line is just a point, return the distance from the point to that point
    if line_length_squared == 0:
        return np.sqrt((x - x1)**2 + (y - y1)**2)
    
    # Calculate the projection of the point onto the line segment
    t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / line_length_squared))
    
    # Find the nearest point on the line segment
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    
    # Return the distance from the point to its projection on the line
    return np.sqrt((x - proj_x)**2 + (y - proj_y)**2)

def run_single_iteration(i, iters):
    """Run a single Monte Carlo iteration with all three algorithms"""
    print(f"\nIteration {i+1}/{iters}")
    
    # Randomize parameters
    goal_x = random.uniform(25, 37.5)
    goal_y = random.uniform(25, 37.5)
    base_current_speed = random.uniform(0.5, 1.5)
    base_current_dir = random.uniform(0, 360)
    base_wind_speed = random.uniform(13, 16)
    base_wind_dir = random.uniform(0, 360)
    
    start_pos = (2.5, 2.5)  # Fixed start position
    goal_pos = (goal_x, goal_y)
    
    # Store parameters
    params = {
        'iteration': i,
        'start_pos': start_pos,
        'goal_pos': goal_pos,
        'base_current_speed': base_current_speed,
        'base_current_dir': base_current_dir,
        'base_wind_speed': base_wind_speed,
        'base_wind_dir': base_wind_dir
    }
    
    # Run MCTS
    start_time = time.time()
    env_mcts, path_mcts, elapsed_time_mcts, time_out_mcts = run_optimization(
        start_pos=start_pos,
        goal_pos=goal_pos,
        use_currents=True,
        base_current_speed=base_current_speed,
        base_current_dir=base_current_dir,
        base_wind_speed=base_wind_speed,
        base_wind_dir=base_wind_dir
    )
    mcts_runtime = time.time() - start_time
    
    # Calculate MCTS travel time using polar model
    mcts_travel_time = float('inf')
    if not time_out_mcts:
        mcts_travel_time = calculate_travel_time(path_mcts, env_mcts)
    
    # Run Baseline
    start_time = time.time()
    env_baseline, path_baseline, elapsed_time_baseline, time_out_baseline = run_baseline_optimization(
        start_pos=start_pos,
        goal_pos=goal_pos,
        use_currents=True,
        base_current_speed=base_current_speed,
        base_current_dir=base_current_dir,
        base_wind_speed=base_wind_speed,
        base_wind_dir=base_wind_dir
    )
    baseline_runtime = time.time() - start_time
    
    # Calculate Baseline travel time using polar model
    baseline_travel_time = float('inf')
    if not time_out_baseline:
        baseline_travel_time = calculate_travel_time(path_baseline, env_baseline)
    
    # Run Q-Learning
    start_time = time.time()
    env_qlearning, path_qlearning, elapsed_time_qlearning, time_out_qlearning = run_qlearning_optimization(
        start_pos=start_pos,
        goal_pos=goal_pos,
        use_currents=True,
        base_current_speed=base_current_speed,
        base_current_dir=base_current_dir,
        base_wind_speed=base_wind_speed,
        base_wind_dir=base_wind_dir,
        training_episodes=500  # Reduced for faster computation
    )
    qlearning_runtime = time.time() - start_time
    
    # Calculate Q-learning travel time using polar model
    qlearning_travel_time = float('inf')
    if not time_out_qlearning:
        qlearning_travel_time = calculate_travel_time(path_qlearning, env_qlearning)
    
    # Collect results
    result = {
        'params': params,
        'mcts': {
            'env': env_mcts,
            'path': None if time_out_mcts else path_mcts,
            'algorithm_time': elapsed_time_mcts if not time_out_mcts else float('inf'),
            'travel_time': mcts_travel_time,
            'time_out': time_out_mcts,
            'runtime': mcts_runtime
        },
        'baseline': {
            'env': env_baseline,
            'path': None if time_out_baseline else path_baseline,
            'algorithm_time': elapsed_time_baseline if not time_out_baseline else float('inf'),
            'travel_time': baseline_travel_time,
            'time_out': time_out_baseline,
            'runtime': baseline_runtime
        },
        'qlearning': {
            'env': env_qlearning,
            'path': None if time_out_qlearning else path_qlearning,
            'algorithm_time': elapsed_time_qlearning if not time_out_qlearning else float('inf'),
            'travel_time': qlearning_travel_time,
            'time_out': time_out_qlearning,
            'runtime': qlearning_runtime
        }
    }
    
    # Print results
    print(f"  Parameters: Start={start_pos}, Goal={goal_pos}")
    print(f"  Wind: {base_wind_speed:.1f}kts @ {base_wind_dir:.1f}°, Current: {base_current_speed:.1f}kts @ {base_current_dir:.1f}°")
    
    print("  Results:")
    print(f"  - MCTS: {'TIMEOUT' if time_out_mcts else f'Travel time: {mcts_travel_time:.1f}'} (runtime: {mcts_runtime:.1f}s)")
    print(f"  - Baseline: {'TIMEOUT' if time_out_baseline else f'Travel time: {baseline_travel_time:.1f}'} (runtime: {baseline_runtime:.1f}s)")
    print(f"  - Q-learning: {'TIMEOUT' if time_out_qlearning else f'Travel time: {qlearning_travel_time:.1f}'} (runtime: {qlearning_runtime:.1f}s)")
    
    return result

def run_optimization(start_pos, goal_pos, use_currents, base_current_speed, base_current_dir, base_wind_speed, base_wind_dir):
    """Run the sailing path optimization using MCTS"""
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
    env.goal_radius = 4.0  # Larger goal radius

    # Reset to initialize
    env.reset()

    # Find optimal path
    path, elapsed_time, time_out = optimize_sailing_path(
        env,
        mcts_iterations=10000
    )

    return env, path, elapsed_time, time_out

def plot_top_mcts_improvements(all_results):
    """Plot the top MCTS trajectories with biggest percentage improvement over baseline,
    showing wind on the left panel and current on the right panel"""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    
    os.makedirs("results", exist_ok=True)
    
    # Calculate percentage improvements
    improvements = []
    
    for result in all_results:
        # Check if MCTS, baseline, and q-learning all reached the goal
        if (not result['mcts']['time_out'] and 
            'baseline' in result and not result['baseline']['time_out'] and
            'qlearning' in result and not result['qlearning']['time_out']):
            
            # Calculate percentage improvement over baseline
            baseline_time = result['baseline']['travel_time']
            mcts_time = result['mcts']['travel_time']
            
            # Only consider cases where MCTS actually improved over baseline
            if mcts_time < baseline_time:
                percentage_improvement = ((baseline_time - mcts_time) / baseline_time) * 100
                
                improvements.append({
                    'result': result,
                    'percentage_improvement': percentage_improvement,
                    'baseline_time': baseline_time,
                    'mcts_time': mcts_time
                })
    
    # Sort by percentage improvement in descending order
    improvements.sort(key=lambda x: x['percentage_improvement'], reverse=True)
    
    # Get top 10 improvements (or fewer if less are available)
    top_improvements = improvements[:min(20, len(improvements))]
    
    if not top_improvements:
        print("No MCTS improvements over baseline to visualize")
        return
    
    print(f"Creating visualizations for top {len(top_improvements)} MCTS improvements...")
    
    # Plot each improvement as a figure with two subplots side by side
    for i, improvement in enumerate(top_improvements):
        result = improvement['result']
        env = result['mcts']['env']
        
        # Create figure with two subplots side by side
        fig, (ax_wind, ax_current) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Create grid
        x = np.arange(0, env.grid_size_x, 1)
        y = np.arange(0, env.grid_size_y, 1)
        X, Y = np.meshgrid(x, y)
        
        # Collect wind data for left panel
        U_wind = np.zeros_like(X, dtype=float)
        V_wind = np.zeros_like(Y, dtype=float)
        wind_speed = np.zeros_like(X, dtype=float)
        
        # Collect current data for right panel
        U_current = np.zeros_like(X, dtype=float)
        V_current = np.zeros_like(Y, dtype=float)
        current_speed = np.zeros_like(X, dtype=float)
        
        for i_x in range(len(x)):
            for j_y in range(len(y)):
                # Wind data
                w_speed, w_dir = env.wind_grid.get_wind_at_position(i_x, j_y)
                w_rad = np.radians(w_dir)
                U_wind[j_y, i_x] = w_speed * np.sin(w_rad)
                V_wind[j_y, i_x] = w_speed * np.cos(w_rad)
                wind_speed[j_y, i_x] = w_speed
                
                # Current data
                c_speed, c_dir = env.current_grid.get_current_at_position(i_x, j_y)
                c_x, c_y = env.current_grid.get_current_vector_at_position(i_x, j_y)
                U_current[j_y, i_x] = c_x
                V_current[j_y, i_x] = c_y
                current_speed[j_y, i_x] = c_speed
        
        # ----- LEFT PANEL: WIND VISUALIZATION -----
        
        # Plot wind speed as background color
        c_wind = ax_wind.pcolormesh(X, Y, wind_speed, cmap='viridis', alpha=0.3)
        plt.colorbar(c_wind, ax=ax_wind, label='Wind Speed (knots)')
        
        # Plot wind vectors (subsampled)
        subsample = 4
        ax_wind.quiver(X[::subsample, ::subsample], Y[::subsample, ::subsample],
                        U_wind[::subsample, ::subsample], V_wind[::subsample, ::subsample],
                        scale=400, color='blue', alpha=0.6)
        
        # Plot paths on wind panel
        path_mcts = result['mcts']['path']
        path_baseline = result['baseline']['path']
        path_qlearning = result['qlearning']['path']
        
        mcts_time = result['mcts']['travel_time']
        baseline_time = result['baseline']['travel_time']
        qlearning_time = result['qlearning']['travel_time']
        
        if path_mcts is not None:
            path_x, path_y = zip(*path_mcts)
            ax_wind.plot(path_x, path_y, 'k-', linewidth=2, label=f'MCTS')
        
        if path_baseline is not None:
            path_x, path_y = zip(*path_baseline)
            ax_wind.plot(path_x, path_y, 'g-', linewidth=1.5, alpha=0.8, 
                        label=f'Baseline')
        
        if path_qlearning is not None:
            path_x, path_y = zip(*path_qlearning)
            ax_wind.plot(path_x, path_y, 'r-', linewidth=1.5, alpha=0.8, 
                        label=f'Q-Learning')
        
        # Plot start and goal on wind panel
        ax_wind.plot(env.start_pos[0], env.start_pos[1], 'go', markersize=10, label='Start')
        ax_wind.plot(env.goal_pos[0], env.goal_pos[1], 'ro', markersize=10, label='Goal')
        circle = Circle(env.goal_pos, env.goal_radius, fill=False, color='r', linestyle='--')
        ax_wind.add_patch(circle)
        
        # Set wind panel properties
        ax_wind.set_xlim(0, env.grid_size_x)
        ax_wind.set_ylim(0, env.grid_size_y)
        ax_wind.set_xlabel('X')
        ax_wind.set_ylabel('Y')
        ax_wind.set_title('Optimal Sailing Path with Wind Field')
        ax_wind.legend()
        
        # ----- RIGHT PANEL: CURRENT VISUALIZATION -----
        
        # Plot current speed as a heatmap
        c_current = ax_current.pcolormesh(X, Y, current_speed, cmap='viridis', shading='auto', alpha=0.7)
        plt.colorbar(c_current, ax=ax_current, label='Current Speed (knots)')
        
        # Overlay current direction with arrows
        subsample_current = 4
        ax_current.quiver(X[::subsample_current, ::subsample_current], 
                          Y[::subsample_current, ::subsample_current],
                          U_current[::subsample_current, ::subsample_current], 
                          V_current[::subsample_current, ::subsample_current],
                          scale=50, color='white', alpha=0.8, width=0.002)
        
        # Plot paths on current panel
        if path_mcts is not None:
            path_x, path_y = zip(*path_mcts)
            ax_current.plot(path_x, path_y, 'k-', linewidth=2, label=f'MCTS')
        
        if path_baseline is not None:
            path_x, path_y = zip(*path_baseline)
            ax_current.plot(path_x, path_y, 'g-', linewidth=1.5, alpha=0.8, 
                           label=f'Baseline')
        
        if path_qlearning is not None:
            path_x, path_y = zip(*path_qlearning)
            ax_current.plot(path_x, path_y, 'r-', linewidth=1.5, alpha=0.8, 
                           label=f'Q-Learning')
        
        # Plot start and goal on current panel
        ax_current.plot(env.start_pos[0], env.start_pos[1], 'go', markersize=10, label='Start')
        ax_current.plot(env.goal_pos[0], env.goal_pos[1], 'ro', markersize=10, label='Goal')
        circle = Circle(env.goal_pos, env.goal_radius, fill=False, color='r', linestyle='--')
        ax_current.add_patch(circle)
        
        # Set current panel properties
        ax_current.set_xlim(0, env.grid_size_x)
        ax_current.set_ylim(0, env.grid_size_y)
        ax_current.set_xlabel('X')
        ax_current.set_ylabel('Y')
        ax_current.set_title('Optimal Sailing Path with Current Field')
        ax_current.legend()
        
        # Add improvement stats as a text box on top of figure
        improvement_pct = improvement['percentage_improvement']
        params = result['params']
        
        fig.suptitle(
            f'MCTS Case #{i+1}\n' +
            f"Wind: {params['base_wind_speed']:.1f}kts @ {params['base_wind_dir']:.1f}°, " +
            f"Current: {params['base_current_speed']:.1f}kts @ {params['base_current_dir']:.1f}°",
            fontsize=16
        )
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"results/mcts_improvement_case_{i+1}.png", dpi=300, bbox_inches='tight')
        print(f"Saved MCTS improvement case #{i+1}")
        plt.close(fig)


def monte_carlo(iters=20):
    """Run Monte Carlo simulation with parallelization"""
    # Initialize result tracking
    mcts_travel_times = []
    baseline_travel_times = []
    qlearning_travel_times = []
    
    mcts_timeouts = 0
    baseline_timeouts = 0
    qlearning_timeouts = 0
    
    # Track best MCTS performance
    best_mcts_performance = {'advantage': -float('inf'), 'iteration': -1}
    
    print(f"Running {iters} Monte Carlo iterations with parallelization...")
    
    # Determine number of CPUs to use
    num_cpus = max(1, mp.cpu_count() - 1)
    print(f"Using {num_cpus} CPU cores for parallel processing")
    
    # Create a partial function with fixed parameter
    run_iteration = partial(run_single_iteration, iters=iters)
    
    # Run iterations in parallel
    with mp.Pool(num_cpus) as pool:
        all_results = pool.map(run_iteration, range(iters))
    
    # Process results
    for result in all_results:
        # MCTS results
        if not result['mcts']['time_out']:
            mcts_travel_times.append(result['mcts']['travel_time'])
        else:
            mcts_timeouts += 1
            
        # Baseline results
        if not result['baseline']['time_out']:
            baseline_travel_times.append(result['baseline']['travel_time'])
        else:
            baseline_timeouts += 1
            
        # Q-Learning results
        if not result['qlearning']['time_out']:
            qlearning_travel_times.append(result['qlearning']['travel_time'])
        else:
            qlearning_timeouts += 1
        
        # Find scenario where MCTS performs best compared to others
        if not result['mcts']['time_out']:
            # Calculate advantage over other algorithms
            baseline_advantage = 0
            qlearning_advantage = 0
            
            if not result['baseline']['time_out']:
                baseline_advantage = (result['baseline']['travel_time'] - result['mcts']['travel_time']) / result['mcts']['travel_time']
                
            if not result['qlearning']['time_out']:
                qlearning_advantage = (result['qlearning']['travel_time'] - result['mcts']['travel_time']) / result['mcts']['travel_time']
            
            # Get the minimum advantage (i.e., the smallest advantage over any algorithm)
            # Only consider cases where all algorithms reached the goal
            if not result['baseline']['time_out'] and not result['qlearning']['time_out']:
                mcts_advantage = min(baseline_advantage, qlearning_advantage)
                
                # If MCTS has advantage over both algorithms and it's better than previous best
                if mcts_advantage > 0 and mcts_advantage > best_mcts_performance['advantage']:
                    best_mcts_performance['advantage'] = mcts_advantage
                    best_mcts_performance['iteration'] = result['params']['iteration']
                    best_mcts_performance['result'] = result
    
    # Calculate statistics
    if mcts_travel_times:
        avg_mcts_time = np.mean(mcts_travel_times)
        print(f"\nMCTS: {len(mcts_travel_times)}/{iters} successful runs, avg time: {avg_mcts_time:.2f}")
    else:
        print("\nMCTS: No successful runs")
        
    if baseline_travel_times:
        avg_baseline_time = np.mean(baseline_travel_times)
        print(f"Baseline: {len(baseline_travel_times)}/{iters} successful runs, avg time: {avg_baseline_time:.2f}")
    else:
        print("Baseline: No successful runs")
        
    if qlearning_travel_times:
        avg_qlearning_time = np.mean(qlearning_travel_times)
        print(f"Q-Learning: {len(qlearning_travel_times)}/{iters} successful runs, avg time: {avg_qlearning_time:.2f}")
    else:
        print("Q-Learning: No successful runs")
    
    print(f"MCTS travel times: {mcts_travel_times}")
    print(f"Baseline travel times: {baseline_travel_times}")
    print(f"Q travel times: {qlearning_travel_times}")

    # Compare algorithms
    if mcts_travel_times and baseline_travel_times:
        baseline_diff = (np.mean(baseline_travel_times) - np.mean(mcts_travel_times)) / np.mean(mcts_travel_times) * 100
        print(f"\nBaseline vs MCTS: {baseline_diff:.1f}% {'slower' if baseline_diff > 0 else 'faster'}")
        
    if mcts_travel_times and qlearning_travel_times:
        qlearning_diff = (np.mean(qlearning_travel_times) - np.mean(mcts_travel_times)) / np.mean(mcts_travel_times) * 100
        print(f"Q-Learning vs MCTS: {qlearning_diff:.1f}% {'slower' if qlearning_diff > 0 else 'faster'}")
    
    improvements = []

    # Calculate percentage improvement for each result
    for result in all_results:
        # Check if MCTS reached the goal and baseline data exists
        if (not result['mcts']['time_out'] and 
            'baseline' in result and 
            not result['baseline']['time_out']):
            
            # Calculate percentage improvement
            baseline_time = result['baseline']['travel_time']
            mcts_time = result['mcts']['travel_time']
            
            # Only consider cases where MCTS actually improved over baseline
            if mcts_time < baseline_time:
                percentage_improvement = ((baseline_time - mcts_time) / baseline_time) * 100
                
                improvements.append({
                    'result': result,
                    'percentage_improvement': percentage_improvement,
                    'baseline_time': baseline_time,
                    'mcts_time': mcts_time
                })

    # Sort by percentage improvement in descending order
    improvements.sort(key=lambda x: x['percentage_improvement'], reverse=True)

    # Get top 3 improvements
    top_3_improvements = improvements[:3]

    # Plot the best cases
    print("\nCreating visualization of best performing cases...")
    if len(all_results) > 0:
        plot_top_mcts_improvements(all_results)
    else:
        print("No successful runs to visualize")
    
    # Return summary statistics
    results = {
        'mcts_travel_times': mcts_travel_times,
        'baseline_travel_times': baseline_travel_times,
        'qlearning_travel_times': qlearning_travel_times,
        'timeouts': {
            'mcts': mcts_timeouts,
            'baseline': baseline_timeouts,
            'qlearning': qlearning_timeouts
        },
        'best_mcts_performance': best_mcts_performance
    }
    
    return results

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run Monte Carlo comparison
    results = monte_carlo(iters=500)  # Adjust iterations as needed