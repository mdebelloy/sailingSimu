import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List
from boat import Position, BoatState, Boat
from static_wind import StaticWindField, WindState
from course import Course
import matplotlib.pyplot as plt
from numba import jit
import time
import math 
from boat import get_tacking_penalty

@dataclass
class GridDimensions:
    x_cells: int
    y_cells: int
    grid_size: float
    width: float
    length: float

@jit(nopython=True)
def calculate_rewards(x_positions, y_positions, next_x, next_y, 
                     is_upwind, course_width, course_length, grid_size):
    """Vectorized reward calculation"""
    progress = np.where(is_upwind, 
                       next_y - y_positions,
                       y_positions - next_y)
    
    # Boundary penalties
    out_of_bounds = ((next_x < 0) | (next_x > course_width) |
                     (next_y < 0) | (next_y > course_length))
    
    rewards = np.where(out_of_bounds, -1000, progress)
    
    # Goal rewards
    at_top = (next_y >= course_length - grid_size)
    at_bottom = (next_y <= grid_size)
    rewards = np.where(is_upwind & at_top, rewards + 1000, rewards)
    rewards = np.where(~is_upwind & at_bottom, rewards + 1000, rewards)
    
    return rewards

class VectorizedSailingMDP:
    def __init__(self, 
            course: Course,
            boat: Boat,
            wind_field: StaticWindField,
            grid_size: float = 100.0,
            n_angles: int = 36):

        self.course = course
        self.boat = boat
        self.wind_field = wind_field
        self.grid_size = grid_size
        self.discount = 0.95
        
        # Create grid
        self.dims = GridDimensions(
            x_cells=int(course.width / grid_size) + 1,
            y_cells=int(course.length / grid_size) + 1,
            grid_size=grid_size,
            width=course.width,
            length=course.length
        )
        
        # Create state space arrays
        self.x_positions = np.linspace(0, course.width, self.dims.x_cells)
        self.y_positions = np.linspace(0, course.length, self.dims.y_cells)
        self.X, self.Y = np.meshgrid(self.x_positions, self.y_positions)
        
        # Generate smoothed wind variations
        print("Computing wind field with smoothed variations...")
        self.wind_speeds = np.zeros_like(self.X)
        self.wind_directions = np.zeros_like(self.X)
        
        # Create Perlin-like noise for smooth transitions
        n_clusters = 8  # Increased for more variation points
        cluster_centers = []
        for _ in range(n_clusters):
            x = np.random.uniform(0, course.width)
            y = np.random.uniform(0, course.length)
            # Randomly assign either positive or negative variation
            variation = np.random.choice([0, 0])
            cluster_centers.append((x, y, variation))
        
        # Apply smoothed variations using inverse distance weighting
        for i in range(self.dims.x_cells):
            for j in range(self.dims.y_cells):
                pos = Position(self.x_positions[i], self.y_positions[j])
                wind = wind_field.get_wind_state(pos)
                
                # Calculate weighted variation based on distance to all clusters
                total_weight = 0
                weighted_variation = 0
                
                for cx, cy, var in cluster_centers:
                    dist = ((pos.x - cx)**2 + (pos.y - cy)**2)**0.5
                    # Use inverse square distance for smoother transitions
                    weight = 1 / (1 + (dist/500)**2)  # Increased distance scaling for smoother transitions
                    weighted_variation += var * weight
                    total_weight += weight
                
                if total_weight > 0:
                    variation = weighted_variation / total_weight
                else:
                    variation = 0
                    
                # Apply smoothed variation to base wind direction
                self.wind_speeds[j,i] = wind.speed
                self.wind_directions[j,i] = wind.direction + variation
        
        print(f"Wind direction range: {np.min(self.wind_directions):.1f}° to {np.max(self.wind_directions):.1f}°")
        print(f"Wind speed range: {np.min(self.wind_speeds):.1f} to {np.max(self.wind_speeds):.1f} knots")
        
        # Initialize value functions and policies
        self.values_upwind = np.zeros((self.dims.y_cells, self.dims.x_cells))
        self.values_downwind = np.zeros((self.dims.y_cells, self.dims.x_cells))
        self.policy_upwind = np.zeros((self.dims.y_cells, self.dims.x_cells))
        self.policy_downwind = np.zeros((self.dims.y_cells, self.dims.x_cells))
        
        # Define action space
        # Split angles into upwind and downwind segments
        upwind_angles = np.linspace(-160, -30, n_angles//2)  # No angles between -30 and 30
        downwind_angles = np.linspace(30, 160, n_angles//2)
        self.actions = np.concatenate([upwind_angles, downwind_angles])
        
        # Store last action for tacking/gybing detection
        self.last_action_upwind = np.zeros((self.dims.y_cells, self.dims.x_cells))
        self.last_action_downwind = np.zeros((self.dims.y_cells, self.dims.x_cells))
        
        # Initialize penalty factors for rewards
        self.boundary_penalty = -20000  # Main out-of-bounds penalty
        self.outer_buffer_penalty = -3000  # Penalty for outer buffer zone
        self.inner_buffer_penalty = -5000  # Penalty for inner buffer zone
        self.no_go_penalty = -2000  # Penalty for no-go zones
        self.speed_reward_factor = 2.0  # Multiplier for speed rewards
        self.progress_reward_factor = 15.0  # Multiplier for progress rewards
        self.gate_reward = 2000  # Reward for passing through gates
        self.tacking_penalty_factor = 15.0  # Multiplier for tacking/gybing penalties
        
        # Buffer zone size (in grid cells)
        self.buffer_size = 3

    def get_upwind_reward(self, current_y: np.ndarray, next_y: np.ndarray, 
                        next_x: np.ndarray, speed: np.ndarray, action: float) -> np.ndarray:
        """Calculate rewards for upwind leg with enhanced tacking penalties"""
        # Progress reward (going up)
        progress_reward = (next_y - current_y) * self.progress_reward_factor
        
        # Speed bonus
        speed_reward = speed * self.speed_reward_factor
        
        # No-go zone penalty (sailing too close to wind)
        no_go_penalty = np.zeros_like(speed)
        min_twa = 30
        for i in range(self.wind_speeds.shape[0]):
            for j in range(self.wind_speeds.shape[1]):
                if abs(action) < min_twa:
                    no_go_penalty[i,j] = self.no_go_penalty
        
        # Calculate tacking penalties with increased severity
        tacking_penalty = np.zeros_like(speed)
        for i in range(self.wind_speeds.shape[0]):
            for j in range(self.wind_speeds.shape[1]):
                if self.check_for_tack(self.last_action_upwind[i,j], action, True):
                    base_penalty = get_tacking_penalty(self.wind_speeds[i,j])
                    # Increase penalty significance
                    tacking_penalty[i,j] = -base_penalty * self.tacking_penalty_factor * 10  # Increased multiplier
        
        # Gate reward
        gate_y = self.course.top_marks[0].y
        gate_x_min = self.course.top_marks[0].x
        gate_x_max = self.course.top_marks[1].x
        
        at_gate = ((next_y >= gate_y - self.grid_size) & 
                (next_y <= gate_y + self.grid_size) & 
                (next_x >= gate_x_min) & 
                (next_x <= gate_x_max))
        
        gate_reward = np.where(at_gate, self.gate_reward, 0)
        
        # Boundary penalties
        buffer = self.grid_size * self.buffer_size
        out_of_bounds = ((next_x < 0) | (next_x > self.course.width) |
                        (next_y < 0) | (next_y > self.course.length))
        
        outer_buffer = ((next_x < buffer) | (next_x > self.course.width - buffer))
        inner_buffer = ((next_x < buffer/2) | (next_x > self.course.width - buffer/2))
        
        boundary_penalty = np.zeros_like(speed)
        boundary_penalty[outer_buffer] = self.outer_buffer_penalty
        boundary_penalty[inner_buffer] = self.inner_buffer_penalty
        
        total_reward = (progress_reward + 
                    speed_reward + 
                    gate_reward + 
                    tacking_penalty +  # Now more significant
                    no_go_penalty + 
                    boundary_penalty)
        
        return np.where(out_of_bounds, self.boundary_penalty, total_reward)

    def get_downwind_reward(self, current_y: np.ndarray, next_y: np.ndarray, 
                        next_x: np.ndarray, speed: np.ndarray, action: float) -> np.ndarray:
        """Calculate rewards for downwind leg with enhanced gybing penalties"""
        # Progress reward (going down)
        progress_reward = (current_y - next_y) * self.progress_reward_factor
        
        # Speed bonus
        speed_reward = speed * self.speed_reward_factor
        
        # No-go zone penalty (sailing too close to direct downwind)
        no_go_penalty = np.zeros_like(speed)
        min_angle_from_downwind = 15
        for i in range(self.wind_speeds.shape[0]):
            for j in range(self.wind_speeds.shape[1]):
                downwind_angle = 180
                if abs(abs(action) - downwind_angle) < min_angle_from_downwind:
                    no_go_penalty[i,j] = self.no_go_penalty
        
        # Calculate gybing penalties with increased severity
        gybing_penalty = np.zeros_like(speed)
        for i in range(self.wind_speeds.shape[0]):
            for j in range(self.wind_speeds.shape[1]):
                if self.check_for_tack(self.last_action_downwind[i,j], action, False):
                    base_penalty = get_tacking_penalty(self.wind_speeds[i,j])
                    # Increase penalty significance
                    gybing_penalty[i,j] = -base_penalty * self.tacking_penalty_factor * 10  # Increased multiplier
        
        # Finish line reward
        at_finish = ((next_y <= self.grid_size) & 
                    (next_x >= self.course.start_line[0].x) & 
                    (next_x <= self.course.start_line[1].x))
        
        finish_reward = np.where(at_finish, self.gate_reward, 0)
        
        # Boundary penalties
        buffer = self.grid_size * self.buffer_size
        out_of_bounds = ((next_x < 0) | (next_x > self.course.width) |
                        (next_y < 0) | (next_y > self.course.length))
        
        outer_buffer = ((next_x < buffer) | (next_x > self.course.width - buffer))
        inner_buffer = ((next_x < buffer/2) | (next_x > self.course.width - buffer/2))
        
        boundary_penalty = np.zeros_like(speed)
        boundary_penalty[outer_buffer] = self.outer_buffer_penalty
        boundary_penalty[inner_buffer] = self.inner_buffer_penalty
        
        total_reward = (progress_reward + 
                    speed_reward + 
                    finish_reward + 
                    gybing_penalty +  # Now more significant
                    no_go_penalty + 
                    boundary_penalty)
        
        return np.where(out_of_bounds, self.boundary_penalty, total_reward)

    def check_for_tack(self, old_twa: float, new_twa: float, is_upwind: bool) -> bool:
        """Enhanced check for tacking or gybing with clearer conditions"""
        if is_upwind:
            # Tacking (crossing through 0 degrees)
            # Consider it a tack if we cross between positive and negative angles
            return (old_twa * new_twa < 0 and abs(old_twa) < 90 and abs(new_twa) < 90)
        else:
            # Gybing (crossing through 180 degrees)
            return (abs(old_twa) > 90 and abs(new_twa) > 90 and 
                ((old_twa > 0 and new_twa < 0) or (old_twa < 0 and new_twa > 0)))


    def value_iteration(self, max_iterations: int = 100, tolerance: float = 0.1, dt: float = 10.0):
        """Vectorized value iteration"""
        t0 = time.time()
        
        for iteration in range(max_iterations):
            delta = 0
            
            # Update upwind values
            new_values_upwind = np.copy(self.values_upwind)
            new_policy_upwind = np.copy(self.policy_upwind)
            new_last_action_upwind = np.copy(self.last_action_upwind)
            
            # Update downwind values
            new_values_downwind = np.copy(self.values_downwind)
            new_policy_downwind = np.copy(self.policy_downwind)
            new_last_action_downwind = np.copy(self.last_action_downwind)
            
            # Try each action
            for action in self.actions:
                # Calculate velocities for this action
                vx, vy = self.get_velocity_field(action)
                speed = np.sqrt(vx**2 + vy**2)
                
                # Calculate next positions
                next_x = self.X + vx * dt
                next_y = self.Y + vy * dt
                
                # Calculate rewards including tacking/gybing penalties
                rewards_up = self.get_upwind_reward(self.Y, next_y, next_x, speed, action)
                rewards_down = self.get_downwind_reward(self.Y, next_y, next_x, speed, action)
                
                # Convert next positions to indices
                next_x_idx = np.clip((next_x / self.grid_size).astype(int), 0, self.dims.x_cells-1)
                next_y_idx = np.clip((next_y / self.grid_size).astype(int), 0, self.dims.y_cells-1)
                
                # Calculate values
                values_up = rewards_up + self.discount * self.values_upwind[next_y_idx, next_x_idx]
                values_down = rewards_down + self.discount * self.values_downwind[next_y_idx, next_x_idx]
                
                # Update if better
                better_up = values_up > new_values_upwind
                better_down = values_down > new_values_downwind
                
                new_values_upwind[better_up] = values_up[better_up]
                new_policy_upwind[better_up] = action
                new_last_action_upwind[better_up] = action
                
                new_values_downwind[better_down] = values_down[better_down]
                new_policy_downwind[better_down] = action
                new_last_action_downwind[better_down] = action
            
            # Calculate maximum change
            delta = max(
                np.max(np.abs(new_values_upwind - self.values_upwind)),
                np.max(np.abs(new_values_downwind - self.values_downwind))
            )
            
            # Update values and policies
            self.values_upwind = new_values_upwind
            self.values_downwind = new_values_downwind
            self.policy_upwind = new_policy_upwind
            self.policy_downwind = new_policy_downwind
            self.last_action_upwind = new_last_action_upwind
            self.last_action_downwind = new_last_action_downwind
            
            print(f"Iteration {iteration}, delta: {delta:.3f}, time: {time.time()-t0:.1f}s")
            
            if delta < tolerance:
                print(f"Converged after {iteration+1} iterations")
                break

   
    def get_velocity_field(self, twa):
        """Vectorized velocity calculation for all grid points with corrected heading calculation"""
        speeds = self.wind_speeds
        directions = self.wind_directions
        
        # Calculate boat speed for all points
        boat_speeds = np.zeros_like(speeds)
        for i in range(speeds.shape[0]):
            for j in range(speeds.shape[1]):
                boat_speeds[i,j] = self.boat.polars.get_boat_speed(abs(twa), speeds[i,j])
        
        # Convert to m/s
        boat_speeds_ms = boat_speeds * 0.51444
        
        # For upwind TWAs (between -90 and 90), heading should be:
        # wind_direction + TWA 
        # For downwind TWAs, heading should be:
        # wind_direction + TWA
        # We don't need different cases because it's the same formula
        heading_rad = np.radians((directions + twa))
        
        
        # In nautical/compass coordinates:
        # 0° is North (positive y)
        # 90° is East (positive x)
        # So, vx = sin(heading), vy = cos(heading)
        vx = boat_speeds_ms * np.sin(heading_rad)
        vy = boat_speeds_ms * np.cos(heading_rad)
        
        return vx, vy


    def get_optimal_action(self, position: Position, leg: str) -> float:
        """Get optimal action for given position with bounds checking"""
        x_idx = int(position.x / self.grid_size)
        y_idx = int(position.y / self.grid_size)
        x_idx = np.clip(x_idx, 0, self.dims.x_cells-1)
        y_idx = np.clip(y_idx, 0, self.dims.y_cells-1)
        
        try:
            if leg == "upwind":
                action = self.policy_upwind[y_idx, x_idx]
                if action == 0:  # No valid policy found
                    # Default to reasonable upwind angle
                    return -45 if position.x < self.course.width/2 else 45
                return action
            else:
                action = self.policy_downwind[y_idx, x_idx]
                if action == 0:  # No valid policy found
                    # Default to reasonable downwind angle
                    return 135 if position.x < self.course.width/2 else -135
                return action
        except IndexError:
            # If somehow we get an index error, return safe default angles
            return -45 if leg == "upwind" else 135
    
    def visualize_policy(self):
        """Visualize the learned policy with debug vectors"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot wind field
        im1 = ax1.pcolormesh(self.X, self.Y, self.wind_speeds, cmap='YlOrRd', alpha=0.5)
        im2 = ax2.pcolormesh(self.X, self.Y, self.wind_speeds, cmap='YlOrRd', alpha=0.5)
        plt.colorbar(im1, ax=ax1, label='Wind Speed (knots)')
        plt.colorbar(im2, ax=ax2, label='Wind Speed (knots)')
        
        # Plot wind directions with longer, more visible arrows
        skip = 2
        wind_scale = 100  # Increased from 50
        wind_alpha = 0.5  # Increased from 0.3
        for i in range(0, self.dims.y_cells, skip):
            for j in range(0, self.dims.x_cells, skip):
                wind_dir = np.radians(self.wind_directions[i,j])
                wx = -np.sin(wind_dir) * wind_scale
                wy = -np.cos(wind_dir) * wind_scale
                # Draw wind arrows in black for better visibility
                ax1.arrow(self.X[i,j], self.Y[i,j], wx, wy, color='black', alpha=wind_alpha, width=2)
                ax2.arrow(self.X[i,j], self.Y[i,j], wx, wy, color='black', alpha=wind_alpha, width=2)
        
        # Plot policies with debug info
        skip = 2
        boat_scale = 150  # Increased from 100
        for i in range(0, self.dims.y_cells, skip):
            for j in range(0, self.dims.x_cells, skip):
                # Upwind policy
                twa = self.policy_upwind[i,j]
                wind_dir = self.wind_directions[i,j]
                heading = (wind_dir + twa) % 360
                
                # Print debug info for some points
                if i % 10 == 0 and j % 10 == 0:
                    print(f"\nUpwind Point ({i},{j}):")
                    print(f"Wind direction: {wind_dir:.1f}°")
                    print(f"TWA from policy: {twa:.1f}°")
                    print(f"Resulting heading: {heading:.1f}°")
                
                dx = np.sin(np.radians(heading)) * boat_scale
                dy = np.cos(np.radians(heading)) * boat_scale
                ax1.arrow(self.X[i,j], self.Y[i,j], dx, dy, 
                        color='blue', alpha=0.7, head_width=20)
                
                # Downwind policy
                twa = self.policy_downwind[i,j]
                heading = (wind_dir + twa) % 360
                dx = np.sin(np.radians(heading)) * boat_scale
                dy = np.cos(np.radians(heading)) * boat_scale
                ax2.arrow(self.X[i,j], self.Y[i,j], dx, dy,
                        color='blue', alpha=0.7, head_width=20)
        
        ax1.set_title("Upwind Policy (black=wind, blue=boat)")
        ax2.set_title("Downwind Policy (black=wind, blue=boat)")
        
        for ax in [ax1, ax2]:
            ax.set_xlim(0, self.course.width)
            ax.set_ylim(0, self.course.length)
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()


def test_vectorized_mdp():
    """Test the vectorized MDP solver"""
    course = Course(
        start_line=(Position(500, 0), Position(1500, 0)),
        top_marks=(Position(900, 3000), Position(1100, 3000)),
        length=3000,
        width=2000,
        extension=500
    )
    
    wind_field = StaticWindField(
        course_width=course.width,
        course_length=course.length,
        base_direction=0,
        base_speed=15,
        n_patterns=3
    )
    
    from polars import PolarData
    polars = PolarData()
    boat = Boat(Position(1000, 0), 0, "port", polars)
    
    solver = VectorizedSailingMDP(course, boat, wind_field)
    solver.value_iteration()
    solver.visualize_policy()
    
    return solver, wind_field

def simulate_mdp_policy(solver: VectorizedSailingMDP, 
                       time_step: float = 1.0,
                       max_time: float = 3600) -> Tuple[List[BoatState], List[WindState], List[float], List[float]]:
    """Simulate race using MDP policy"""
    # Initialize lists to store history
    boat_states = []
    wind_states = []
    boat_speeds = []
    wind_angles = []
    
    # Start position and initial states
    start_pos = Position((solver.course.start_line[0].x + solver.course.start_line[1].x)/2, 0)
    boat = Boat(start_pos, 0, 'port', solver.boat.polars)  # Create fresh boat
    
    # Initialize with proper upwind heading
    initial_wind = solver.wind_field.get_wind_state(start_pos)
    initial_twa = solver.get_optimal_action(start_pos, "upwind")
    initial_heading = (initial_wind.direction + initial_twa) % 360
    
    print(f"Initial wind: {initial_wind.direction:.1f}° @ {initial_wind.speed:.1f}kts")
    print(f"Initial TWA: {initial_twa:.1f}°")
    print(f"Initial heading: {initial_heading:.1f}°")
    
    boat.state = BoatState(
        position=start_pos,
        heading=initial_heading,
        tack='port' if initial_twa > 0 else 'starboard',
        leg="upwind"
    )
    
    # Racing states
    current_leg = "upwind"
    gate_passed = False
    
    # Store initial state
    boat_states.append(boat.state.copy())
    wind_states.append(initial_wind)
    boat_speeds.append(0.0)
    wind_angles.append(initial_twa)
    
    gate_y = solver.course.top_marks[0].y
    gate_x_min = solver.course.top_marks[0].x
    gate_x_max = solver.course.top_marks[1].x
    
    print("Starting simulation...")
    
    time = 0
    last_pos = start_pos
    stuck_count = 0
    
    while time < max_time:
        current_pos = boat.state.position
        
        # Check if boat is stuck
        if abs(current_pos.x - last_pos.x) < 0.01 and abs(current_pos.y - last_pos.y) < 0.01:
            stuck_count += 1
            if stuck_count > 10:
                print(f"Boat appears stuck at ({current_pos.x:.1f}, {current_pos.y:.1f})")
                break
        else:
            stuck_count = 0
        
        last_pos = Position(current_pos.x, current_pos.y)
        
        # Get current wind
        wind = solver.wind_field.get_wind_state(current_pos)
        
        # Get optimal TWA from policy
        optimal_twa = solver.get_optimal_action(current_pos, current_leg)
        
        # Update boat
        new_state, penalty = boat.step(time_step, wind, optimal_twa)
        boat.state = new_state  # Important: update the boat's state
        
        # Calculate boat speed
        vx, vy = boat.get_velocity(wind, optimal_twa)
        boat_speed = math.sqrt(vx**2 + vy**2) / 0.51444  # Convert m/s to knots
        
        # Store state
        boat_states.append(new_state)
        wind_states.append(wind)
        boat_speeds.append(boat_speed)
        wind_angles.append(optimal_twa)
        
        # Check for gate passage (upwind to downwind transition)
        if not gate_passed and current_leg == "upwind":
            if (current_pos.y >= gate_y - solver.grid_size and 
                gate_x_min <= current_pos.x <= gate_x_max):
                print(f"Passed through gate at time {time:.1f}")
                current_leg = "downwind"
                gate_passed = True
                boat.state.leg = "downwind"
        
        # Check finish
        if gate_passed and current_leg == "downwind" and current_pos.y <= solver.grid_size:
            if (solver.course.start_line[0].x <= current_pos.x <= 
                solver.course.start_line[1].x):
                print(f"Finished race at time {time:.1f}")
                break
        
        time += time_step
        
        # Debug print every 100 steps
        if len(boat_states) % 100 == 0:
            print(f"Time: {time:.1f}")
            print(f"Position: ({current_pos.x:.1f}, {current_pos.y:.1f})")
            print(f"Leg: {current_leg}")
            print(f"TWA: {optimal_twa:.1f}°")
            print(f"Boat speed: {boat_speed:.1f}kts")
            print(f"Wind: {wind.direction:.1f}° @ {wind.speed:.1f}kts")
            print("---")
    
    print(f"\nRace summary:")
    print(f"Total time: {time:.1f}s")
    print(f"Final position: ({current_pos.x:.1f}, {current_pos.y:.1f})")
    print(f"Average speed: {np.mean(boat_speeds):.1f}kts")
    
    return boat_states, wind_states, boat_speeds, wind_angles

def visualize_full_solution(solver: VectorizedSailingMDP):
    """Visualize wind field, policy, and simulated path with detailed data points"""
    # Run simulation
    boat_states, wind_states, boat_speeds, wind_angles = simulate_mdp_policy(solver, time_step=1.0)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot wind field
    im1 = ax1.pcolormesh(solver.X, solver.Y, solver.wind_speeds, cmap='YlOrRd', alpha=0.5)
    im2 = ax2.pcolormesh(solver.X, solver.Y, solver.wind_speeds, cmap='YlOrRd', alpha=0.5)
    plt.colorbar(im1, ax=ax1, label='Wind Speed (knots)')
    plt.colorbar(im2, ax=ax2, label='Wind Speed (knots)')
    
    # Plot wind directions
    skip = 3
    for i in range(0, solver.dims.y_cells, skip):
        for j in range(0, solver.dims.x_cells, skip):
            wind_dir = np.radians(solver.wind_directions[i,j])
            wx = -np.sin(wind_dir) * 80
            wy = -np.cos(wind_dir) * 80
            ax1.arrow(solver.X[i,j], solver.Y[i,j], wx, wy, 
                     color='gray', alpha=0.3, width=5, head_width=20)
            ax2.arrow(solver.X[i,j], solver.Y[i,j], wx, wy, 
                     color='gray', alpha=0.3, width=5, head_width=20)
    
    # Plot policies only on first plot
    for i in range(0, solver.dims.y_cells, skip):
        for j in range(0, solver.dims.x_cells, skip):
            twa = solver.policy_upwind[i,j]
            heading = (solver.wind_directions[i,j] + twa) % 360
            dx = np.sin(np.radians(heading)) * 100
            dy = np.cos(np.radians(heading)) * 100
            ax1.arrow(solver.X[i,j], solver.Y[i,j], dx, dy, 
                     color='blue', alpha=0.3, head_width=20)
    
    # Plot boat path
    xs = [state.position.x for state in boat_states]
    ys = [state.position.y for state in boat_states]
    
    # Split path into upwind and downwind legs
    max_y_idx = np.argmax([state.position.y for state in boat_states])
    
    path_color = 'red'
    path_width = 3
    
    # Plot paths
    ax1.plot(xs[:max_y_idx+1], ys[:max_y_idx+1], '-', 
             color=path_color, linewidth=path_width)
    ax2.plot(xs[max_y_idx:], ys[max_y_idx:], '-', 
             color=path_color, linewidth=path_width)
    
    # Plot marked points with data
    n_markers = 10
    marker_size = 10
    
    def add_data_point(ax, idx, alpha):
        """Helper function to add a data point with text annotation"""
        state = boat_states[idx]
        wind = wind_states[idx]
        boat_speed = boat_speeds[idx]
        wind_angle = wind_angles[idx]
        time = idx  # Since time_step is 1.0
        
        # Plot marker
        ax.plot(state.position.x, state.position.y, 'o', 
                color=path_color, alpha=alpha, markersize=marker_size)
        
        # Add text annotation
        text = f"Time: {time}s\nTWA: {wind_angle:.0f}°\nWind: {wind.speed:.1f}kts\nBoat: {boat_speed:.1f}kts"
        
        # Alternate text position above/below point to avoid overlap
        if idx % 2 == 0:
            y_offset = 100
        else:
            y_offset = -200
            
        ax.annotate(text, 
                   (state.position.x, state.position.y),
                   xytext=(0, y_offset),
                   textcoords='offset points',
                   ha='center',
                   va='center',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                   alpha=alpha)
    
    # Add data points for upwind leg
    indices = np.linspace(0, max_y_idx, n_markers, dtype=int)
    alphas = np.linspace(1, 0.2, n_markers)
    for idx, alpha in zip(indices, alphas):
        add_data_point(ax1, idx, alpha)
    
    # Add data points for downwind leg
    indices = np.linspace(max_y_idx, len(xs)-1, n_markers, dtype=int)
    for idx, alpha in zip(indices, alphas):
        add_data_point(ax2, idx, alpha)
    
    # Plot course boundaries
    for ax in [ax1, ax2]:
        # Start/finish line
        ax.plot([solver.course.start_line[0].x, solver.course.start_line[1].x],
                [0, 0], 'g-', linewidth=2)
        
        # Gate
        ax.plot([solver.course.top_marks[0].x, solver.course.top_marks[1].x],
                [solver.course.top_marks[0].y, solver.course.top_marks[1].y],
                'r--', linewidth=2)
        
        # Course boundaries
        ax.plot([0, 0], [0, solver.course.length], 'k--', alpha=0.5)
        ax.plot([solver.course.width, solver.course.width], 
                [0, solver.course.length], 'k--', alpha=0.5)
    
    ax1.set_title("Policy and Wind Field")
    ax2.set_title("Wind Field and Boat Path")
    
    for ax in [ax1, ax2]:
        ax.set_xlim(0, solver.course.width)
        ax.set_ylim(0, solver.course.length)
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return boat_states, wind_states, boat_speeds, wind_angles

def distance_to_point(pos1: Position, pos2: Position) -> float:
    """Calculate distance between two positions"""
    return math.sqrt((pos2.x - pos1.x)**2 + (pos2.y - pos1.y)**2)


if __name__ == "__main__":
    solver, wind_field = test_vectorized_mdp()
    boat_states, wind_states, boat_speeds, wind_angles = visualize_full_solution(solver)