import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
from boat import Boat, Position, BoatState, get_tacking_penalty
from course import Course, create_standard_course
from polars import PolarData
from vis2 import plot_course
from copy import deepcopy
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import random
import time
import datetime
import os

# Discretization parameters
N_X_BINS = 20  # Width divisions - increased for better granularity
N_Y_BINS = 40  # Length divisions - increased for better resolution upwind
N_ACTIONS = 2  # Actions: 0=keep current tack, 1=tack

@dataclass
class WindField:
    """Represents the spatial wind field across the course"""
    speeds: np.ndarray  # 2D array of wind speeds
    x_coords: np.ndarray  # X coordinates for the grid
    y_coords: np.ndarray  # Y coordinates for the grid
    base_speed: float  # Base wind speed
    
    def get_wind_speed(self, x: float, y: float) -> float:
        """Get interpolated wind speed at any point"""
        x_idx = np.abs(self.x_coords - x).argmin()
        y_idx = np.abs(self.y_coords - y).argmin()
        return self.speeds[y_idx, x_idx]

@dataclass
class MarkovWindState:
    """State of wind in Markov model"""
    direction: float  # Current wind direction
    mean_direction: float  # Mean direction state
    speed: float  # Wind speed at current position
    base_speed: float  # Base wind speed
    transition_matrix: np.ndarray  # Matrix of direction state transitions

def create_wind_field(course: Course, 
                     base_speed: float = 20.0,
                     n_gusts: int = 7,
                     resolution: int = 50) -> WindField:
    
    """Create a wind field with spatial variations"""
    x_coords = np.linspace(0, course.width, resolution)
    y_coords = np.linspace(0, course.length + course.extension, resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    wind_speeds = np.full_like(X, base_speed)
    
    for _ in range(n_gusts):
        center_x = np.random.uniform(0, course.width)
        center_y = np.random.uniform(0, course.length + course.extension)
        intensity = np.random.uniform(-0.6, 0.6) * base_speed
        x_distance = (X - center_x)**2 / (400**2)
        y_distance = (Y - center_y)**2 / (600**2)
        gust = intensity * np.exp(-(x_distance + y_distance))
        wind_speeds += gust
    
    wind_speeds = gaussian_filter(wind_speeds, sigma=2)
    wind_speeds = np.maximum(wind_speeds, base_speed * 0.5)
    
    return WindField(wind_speeds, x_coords, y_coords, base_speed)

class MarkovWindModel:
    """Implementation of wind model"""
    def __init__(self, 
                 course: Course,
                 base_speed: float = 16.0,
                 n_direction_states: int = 19,
                 direction_range: Tuple[float, float] = (-45, 45),
                 seed: int = None):
        
        if seed is not None:
            np.random.seed(seed)
            
        self.base_speed = base_speed
        self.wind_field = create_wind_field(course, base_speed)
        
        self.direction_states = np.linspace(
            direction_range[0], 
            direction_range[1], 
            n_direction_states
        )
        
        n = len(self.direction_states)
        P = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                diff = abs(i - j)
                if diff == 0:
                    P[i,j] = 0.5
                elif diff == 1:
                    P[i,j] = 0.2
                elif diff == 2:
                    P[i,j] = 0.05
        
        self.P = P / P.sum(axis=1)[:,None]
        
        center_idx = n // 2
        self.current_state = MarkovWindState(
            direction=self.direction_states[center_idx],
            mean_direction=self.direction_states[center_idx],
            speed=base_speed,
            base_speed=base_speed,
            transition_matrix=self.P
        )
    
    def step(self, dt: float, position: Position) -> MarkovWindState:
        """Update wind state"""
        current_idx = np.abs(
            self.direction_states - self.current_state.mean_direction
        ).argmin()
        
        next_idx = np.random.choice(
            len(self.direction_states), 
            p=self.P[current_idx]
        )
        
        new_mean = self.direction_states[next_idx]
        new_direction = new_mean + np.random.uniform(-5, 5)
        local_speed = self.wind_field.get_wind_speed(position.x, position.y)
        
        self.current_state = MarkovWindState(
            direction=new_direction,
            mean_direction=new_mean,
            speed=local_speed,
            base_speed=self.base_speed,
            transition_matrix=self.P
        )
        
        return self.current_state

# Simplified discretization
N_X_BINS = 20
N_Y_BINS = 40
N_ACTIONS = 2  # 0=keep current tack, 1=tack

@dataclass
class QState:
    """Enhanced state representation including wind information"""
    x_bin: int
    y_bin: int
    tack: str
    near_mark: bool
    wind_strength_bin: int  # Discretized local wind strength
    wind_gradient_bin: int  # Discretized wind gradient direction

def discretize_state(pos: Position, course: Course, tack: str, 
                    local_wind: float, base_wind: float,
                    n_wind_bins: int = 5) -> QState:
    """Convert continuous state to discrete with wind information"""
    x_bin = min(N_X_BINS - 1, max(0, int(pos.x / (course.width/N_X_BINS))))
    y_bin = min(N_Y_BINS - 1, max(0, int(pos.y / (course.length/N_Y_BINS))))
    near_mark = pos.y > course.length * 0.8
    
    # Discretize wind strength relative to base wind
    wind_ratio = local_wind / base_wind
    wind_strength_bin = min(n_wind_bins - 1, 
                          max(0, int(wind_ratio * (n_wind_bins/2))))
    
    # Simple wind gradient measurement (could be enhanced with surrounding cells)
    wind_gradient_bin = 1 if wind_ratio > 1.05 else (0 if wind_ratio < 0.95 else 2)
    
    return QState(x_bin, y_bin, tack, near_mark, 
                 wind_strength_bin, wind_gradient_bin)

def get_state_index(state: QState) -> int:
    """Convert enhanced QState to integer index"""
    n_wind_bins = 5
    n_gradient_bins = 3
    
    tack_idx = 0 if state.tack == 'starboard' else 1
    near_mark_idx = 1 if state.near_mark else 0
    
    return (state.x_bin * N_Y_BINS * 2 * 2 * n_wind_bins * n_gradient_bins + 
            state.y_bin * 2 * 2 * n_wind_bins * n_gradient_bins + 
            tack_idx * 2 * n_wind_bins * n_gradient_bins +
            near_mark_idx * n_wind_bins * n_gradient_bins +
            state.wind_strength_bin * n_gradient_bins +
            state.wind_gradient_bin)

def calculate_reward(prev_pos: Position, new_pos: Position, 
                    wind_speed: float, local_wind_speed: float,
                    tack_penalty: float, course: Course,
                    time_elapsed: float) -> float:
    """Reward function with enhanced wind-seeking behavior"""
    reward = 0.0
    
    # Calculate progress components
    gate_center_x = (course.top_marks[0].x + course.top_marks[1].x) / 2
    gate_center_y = course.top_marks[0].y
    
    to_gate_x = gate_center_x - prev_pos.x
    to_gate_y = gate_center_y - prev_pos.y
    gate_distance = np.sqrt(to_gate_x**2 + to_gate_y**2)
    
    movement_x = new_pos.x - prev_pos.x
    movement_y = new_pos.y - prev_pos.y
    movement_distance = np.sqrt(movement_x**2 + movement_y**2)
    
    # --- Wind-seeking rewards (enhanced) ---
    
    # Exponential reward for finding better wind
    wind_ratio = local_wind_speed / wind_speed
    if wind_ratio > 1:
        # Strong reward for being in better wind
        wind_bonus = (wind_ratio - 1) * 200
        # Additional multiplier for significant improvements
        if wind_ratio > 1.1:  # More than 10% better wind
            wind_bonus *= 2
    else:
        # Linear penalty for worse wind
        wind_bonus = (wind_ratio - 1) * 100
    
    reward += wind_bonus
    
    # Reward for maintaining position in good wind
    if local_wind_speed > wind_speed * 1.05:
        reward += 50  # Bonus for staying in good wind
    
    # --- Progress rewards (maintained but adjusted) ---
    if movement_distance > 0 and gate_distance > 0:
        vmg = (movement_x * to_gate_x + movement_y * to_gate_y) / gate_distance
        
        # Scale VMG reward with wind strength
        vmg_reward = vmg * 15.0 * wind_ratio  # Better wind = more effective progress
        reward += vmg_reward
        
        # Distance improvement reward
        prev_dist = np.sqrt((gate_center_x - prev_pos.x)**2 + (gate_center_y - prev_pos.y)**2)
        new_dist = np.sqrt((gate_center_x - new_pos.x)**2 + (gate_center_y - new_pos.y)**2)
        distance_improvement = prev_dist - new_dist
        reward += distance_improvement * 8.0 * wind_ratio  # Also scaled with wind
    
    # --- Penalties ---
    
    # Dynamic boundary penalty based on wind conditions
    edge_distance = min(new_pos.x, course.width - new_pos.x)
    edge_margin = course.width * 0.15
    if edge_distance < edge_margin:
        penalty_factor = ((edge_margin - edge_distance) / edge_margin) ** 2
        # Reduced penalty if there's significantly better wind near the edge
        if wind_ratio > 1.1:
            penalty_factor *= 0.5
        reward -= penalty_factor * 80
    
    # Severe penalty for going backwards
    if new_pos.y < prev_pos.y:
        # Reduced penalty if moving backwards into much better wind
        backwards_penalty = 40 if wind_ratio <= 1.1 else 20
        reward -= backwards_penalty
    
    # Time and tacking penalties
    reward -= 0.5  # Reduced time penalty
    if tack_penalty > 0:
        # Scale tacking penalty with wind - more costly in light wind
        scaled_penalty = tack_penalty * (1.5 - min(wind_ratio, 1))
        reward -= scaled_penalty * 3
    
    # --- Goal reward ---
    if course.has_passed_gate(prev_pos, new_pos) and new_pos.y > prev_pos.y:
        # Scale completion reward with time and wind efficiency
        time_bonus = max(0, (1200 - time_elapsed) * 2)
        wind_efficiency = sum(wind_ratio > 1.05 for _ in range(int(time_elapsed/5))) / (time_elapsed/5)
        reward += 3000 + time_bonus * (1 + wind_efficiency)
    
    return reward

def train_q_learning(n_episodes: int = 500,
                    learning_rate: float = 0.1,
                    discount_factor: float = 0.99,
                    epsilon_start: float = 1.0,
                    epsilon_end: float = 0.1,
                    epsilon_decay: float = 0.99,
                    seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Q-learning with convergence metrics tracking and visualization"""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    course = create_standard_course()
    polars = PolarData()
    
    # Initialize Q-tables
    n_states = N_X_BINS * N_Y_BINS * 2 * 2 * 5 * 3
    Q = np.ones((n_states, N_ACTIONS)) * 100.0
    best_Q = np.copy(Q)
    best_episode_Q = None  # Store the entire Q-table for the best episode
    
    # Metrics tracking
    q_values = []  # Track Q-values over time
    rewards = []   # Track rewards over time
    
    # Track performance metrics
    best_completion_time = float('inf')
    best_episode = -1
    running_success_rate = 0.0
    success_decay = 0.95
    
    # Enhanced exploration tracking
    wind_visited = np.zeros((N_X_BINS, N_Y_BINS))
    exploration_bonus = 100.0
    
    # Experience replay
    buffer_size = 20000
    replay_buffer = []
    min_replay_size = 1000
    batch_size = 128
    
    epsilon = epsilon_start
    
    for episode in range(n_episodes):
        episode_start = time.time()
        
        # Reset Q to best episode's Q-table periodically to prevent degradation
        if best_episode_Q is not None and episode % 10 == 0:
            Q = np.copy(best_episode_Q)
        
        force_explore = episode % 10 == 0
        if force_explore:
            current_epsilon = 0.8
        else:
            current_epsilon = epsilon
            
        start_pos = Position(course.width/2, 0)
        boat = Boat(start_pos, 45, 'starboard', polars)
        wind_model = MarkovWindModel(course, seed=seed+episode if seed else None)
        
        time_elapsed = 0
        last_tack_time = -float('inf')
        total_reward = 0
        n_tacks = 0
        episode_buffer = []
        completed = False
        
        episode_start_Q = np.copy(Q)
        
        while time_elapsed < 1200.0:
            wind_state = wind_model.step(5.0, boat.state.position)
            prev_pos = deepcopy(boat.state.position)
            
            current_state = discretize_state(
                pos=boat.state.position,
                course=course,
                tack=boat.state.tack,
                local_wind=wind_state.speed,
                base_wind=wind_state.base_speed
            )
            state_idx = get_state_index(current_state)
            
            x_bin = min(N_X_BINS - 1, max(0, int(boat.state.position.x / (course.width/N_X_BINS))))
            y_bin = min(N_Y_BINS - 1, max(0, int(boat.state.position.y / (course.length/N_Y_BINS))))
            
            if np.random.random() < current_epsilon:
                if force_explore:
                    if wind_visited[x_bin, y_bin] < 5:
                        action = np.random.randint(N_ACTIONS)
                    else:
                        wind_ratio = wind_state.speed / wind_state.base_speed
                        if wind_ratio > 1.1:
                            action = 0
                        else:
                            action = np.random.randint(N_ACTIONS)
                else:
                    if np.random.random() < 0.3:
                        action = np.argmax(best_Q[state_idx])
                    else:
                        action = np.random.randint(N_ACTIONS)
            else:
                action = np.argmax(Q[state_idx])
            
            can_tack = (time_elapsed - last_tack_time) >= 30
            tack_penalty = 0
            
            if action == 1 and can_tack:
                boat.state.tack = 'port' if boat.state.tack == 'starboard' else 'starboard'
                last_tack_time = time_elapsed
                tack_penalty = get_tacking_penalty(wind_state.speed)
                n_tacks += 1
            
            optimal_upwind, _ = boat.get_optimal_angles(wind_state.speed)
            current_twa = optimal_upwind if boat.state.tack == 'starboard' else -optimal_upwind
            new_state, step_penalty = boat.step(5.0, wind_state, current_twa)
            
            exploration_reward = 0
            if wind_visited[x_bin, y_bin] < 5:
                exploration_reward = exploration_bonus / (wind_visited[x_bin, y_bin] + 1)
                wind_visited[x_bin, y_bin] += 1
            
            reward = calculate_reward(
                prev_pos, new_state.position,
                wind_state.base_speed, wind_state.speed,
                tack_penalty, course, time_elapsed
            ) + exploration_reward
            
            total_reward += reward
            
            next_state = discretize_state(
                pos=new_state.position,
                course=course,
                tack=new_state.tack,
                local_wind=wind_state.speed,
                base_wind=wind_state.base_speed
            )
            next_state_idx = get_state_index(next_state)
            
            episode_buffer.append((state_idx, action, reward, next_state_idx))
            
            wind_ratio = wind_state.speed / wind_state.base_speed
            local_lr = learning_rate * (1 + max(0, wind_ratio - 1))
            
            Q[state_idx, action] += local_lr * (
                reward + discount_factor * np.max(Q[next_state_idx]) - 
                Q[state_idx, action]
            )
            
            boat.state = new_state
            time_elapsed += 5.0
            
            if course.has_passed_gate(prev_pos, new_state.position) and new_state.position.y > prev_pos.y:
                completed = True
                total_time = time_elapsed + n_tacks * 10
                
                if total_time < best_completion_time:
                    best_completion_time = total_time
                    best_episode = episode
                    best_episode_Q = np.copy(episode_start_Q)
                    best_Q = np.copy(Q)
                    print(f"New best policy! Episode {episode}: "
                          f"time={time_elapsed:.1f}s, penalties={n_tacks*10}s, "
                          f"total={total_time:.1f}s, wind={wind_state.speed:.1f}kts")
                break
        
        # Store metrics for plotting
        q_values.append(np.mean(Q))
        rewards.append(total_reward)
        
        success = 1.0 if completed else 0.0
        running_success_rate = running_success_rate * success_decay + success * (1 - success_decay)
        
        replay_buffer.extend(episode_buffer)
        if len(replay_buffer) > buffer_size:
            replay_buffer = replay_buffer[-buffer_size:]
        
        if len(replay_buffer) >= min_replay_size:
            batch = random.sample(replay_buffer, batch_size)
            for s, a, r, next_s in batch:
                Q[s, a] += learning_rate * (
                    r + discount_factor * np.max(Q[next_s]) - Q[s, a]
                )
        
        if not force_explore:
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
    
    print(f"\nBest policy found in episode {best_episode} with completion time {best_completion_time:.1f}s")

    # Create convergence plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot average Q-values
    ax1.plot(q_values, label='Average Q-value')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Q-value')
    ax1.set_title('Q-value Convergence')
    ax1.grid(True)
    ax1.legend()
    
    # Plot episode rewards
    ax2.plot(rewards, label='Episode Reward', color='green')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.set_title('Episode Rewards')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plots_dir = "sailing_plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{plots_dir}/q_convergence_{timestamp}_seed{seed}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return Q, best_episode_Q if best_episode_Q is not None else best_Q

def simulate_q_learning(Q: np.ndarray, seed: int = None) -> Tuple[List[BoatState], List[MarkovWindState]]:
    """Run simulation using trained Q-table with enhanced state representation"""
    if seed is not None:
        np.random.seed(seed)
        
    course = create_standard_course()
    polars = PolarData()
    wind_model = MarkovWindModel(course, seed=seed)
    start_pos = Position(course.width/2, 0)
    boat = Boat(start_pos, 45, 'starboard', polars)
    
    boat_states = [boat.state.copy()]
    wind_states = [wind_model.current_state]
    
    time = 0
    last_tack_time = -float('inf')
    total_penalty = 0
    n_tacks = 0
    
    while time < 1200.0:  # 20 minute limit
        wind_state = wind_model.step(5.0, boat.state.position)
        optimal_upwind, _ = boat.get_optimal_angles(wind_state.speed)
        prev_pos = deepcopy(boat.state.position)
        
        # Get current state and action using enhanced state representation
        current_state = discretize_state(
            pos=boat.state.position,
            course=course,
            tack=boat.state.tack,
            local_wind=wind_state.speed,
            base_wind=wind_state.base_speed
        )
        state_idx = get_state_index(current_state)
        action = np.argmax(Q[state_idx])
        
        # Execute action
        can_tack = (time - last_tack_time) >= 30
        
        if action == 1 and can_tack:
            boat.state.tack = 'port' if boat.state.tack == 'starboard' else 'starboard'
            last_tack_time = time
            penalty = get_tacking_penalty(wind_state.speed)
            total_penalty += penalty
            n_tacks += 1
            #print(f"Tacking at time {time:.1f}, penalty: {penalty:.1f}s, "f"wind: {wind_state.speed:.1f}kts")
        
        # Use optimal angle
        current_twa = optimal_upwind if boat.state.tack == 'starboard' else -optimal_upwind
        new_state, step_penalty = boat.step(5.0, wind_state, current_twa)
        
        boat_states.append(new_state)
        wind_states.append(wind_state)
        
        if (course.has_passed_gate(prev_pos, new_state.position) and 
            new_state.position.y > prev_pos.y):
            print(f"Successfully finished in {time:.1f}s")
            print(f"Number of tacks: {n_tacks}")
            print(f"Tacking penalties: {total_penalty:.1f}s")
            print(f"Total time: {time + total_penalty:.1f}s")
            #print(f"Final wind speed: {wind_state.speed:.1f}kts")
            break
        
        boat.state = new_state
        time += 5.0
        
        # Print progress every 60 seconds
        #if len(boat_states) % 12 == 0:  # Every minute
        #    print(f"Time: {time:.1f}s, Position: ({new_state.position.x:.1f}, {new_state.position.y:.1f}), "
        #          f"Wind: {wind_state.speed:.1f}kts")
    
    return boat_states, wind_states


def simulate_upwind_leg(time_step: float = 5.0, 
                       max_time: float = 1200.0,
                       seed: int = None) -> Tuple[List[BoatState], List[MarkovWindState], WindField]:
    """Simulate upwind leg using simple tacking strategy"""
    if seed is not None:
        np.random.seed(seed)
        
    course = create_standard_course()
    polars = PolarData()
    wind_model = MarkovWindModel(course, seed=seed)
    
    start_pos = Position(course.width/2, 0)
    boat = Boat(start_pos, 45, 'starboard', polars)
    
    boat_states = [boat.state.copy()]
    wind_states = [deepcopy(wind_model.current_state)]
    
    time = 0
    last_tack_time = -float('inf')
    total_tacking_penalty = 0
    n_tacks = 0
    
    while time < max_time:
        current_pos = boat.state.position
        
        # Check if we've passed through the gate
        if course.has_passed_gate(
            previous_pos=boat_states[-2].position if len(boat_states) > 1 else start_pos,
            current_pos=current_pos
        ):
            print(f"Passed through gate at time {time:.1f}")
            print(f"Number of tacks: {n_tacks}")
            print(f"Total tacking penalty: {total_tacking_penalty:.1f} seconds")
            print(f"Total time including penalties: {(time + total_tacking_penalty):.1f} seconds")
            break
        
        wind_state = wind_model.step(time_step, current_pos)
        optimal_upwind, _ = boat.get_optimal_angles(wind_state.speed)
        
        # Simple tacking logic: tack when we get near the course boundaries
        min_tack_interval = 30  # Minimum time between tacks
        can_tack = (time - last_tack_time) >= min_tack_interval
        
        if can_tack:
            # Tack when we get close to course boundaries
            near_boundary = (current_pos.x < course.width * 0.15 or 
                           current_pos.x > course.width * 0.85)
            
            if near_boundary:
                boat.state.tack = 'port' if boat.state.tack == 'starboard' else 'starboard'
                last_tack_time = time
                n_tacks += 1
                penalty = get_tacking_penalty(wind_state.speed)

                total_tacking_penalty += penalty
                #print(f"Tacking at time {time:.1f}, penalty: {penalty:.1f} seconds")
        
        # Set the true wind angle based on tack
        current_twa = optimal_upwind if boat.state.tack == 'starboard' else -optimal_upwind
        new_state, penalty = boat.step(time_step, wind_state, current_twa)
        
        boat_states.append(new_state)
        wind_states.append(deepcopy(wind_state))
        
        time += time_step
        
        #if len(boat_states) % 20 == 0:
        #    print(f"Time: {time:.1f}, Position: ({current_pos.x:.1f}, {current_pos.y:.1f})")
        #    print(f"Local wind: {wind_state.speed:.1f}kts @ {wind_state.direction:.1f}°")
    
    return boat_states, wind_states, wind_model.wind_field


def create_race_visualization(course, wind_field, baseline_states, baseline_winds, 
                            q_states, q_winds, q_learning_speeds, seed):
    """Create improved race visualization with better scaling and wind representation"""
    
    # Create figure with adjusted size
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Plot wind speed as colored background
    mesh = ax.pcolormesh(wind_field.x_coords, wind_field.y_coords, wind_field.speeds, 
                        shading='auto', 
                        cmap='YlOrRd', 
                        alpha=0.5)
    
    # Create wind vector grid with varying directions
    skip = 6  # Increased skip for less crowded arrows
    X, Y = np.meshgrid(wind_field.x_coords[::skip], wind_field.y_coords[::skip])
    
    # Calculate wind directions at each point
    # Interpolate wind directions from the wind states
    y_positions = np.array([w.direction for w in baseline_winds])
    y_times = np.linspace(0, course.length + course.extension, len(baseline_winds))
    
    # Create interpolated wind direction field
    wind_directions = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Find closest time index for this y position
            idx = np.abs(y_times - Y[i,j]).argmin()
            wind_directions[i,j] = np.radians(baseline_winds[idx].direction)
    
    # Create wind direction field (wind arrows point FROM the wind direction)
    U = -np.sin(wind_directions)
    V = -np.cos(wind_directions)
    
    # Scale arrows by local wind speed
    speeds = wind_field.speeds[::skip, ::skip]
    scale_factor = speeds / wind_field.base_speed
    U = U * scale_factor
    V = V * scale_factor
    
    # Plot wind vectors with adjusted size
    Q = ax.quiver(X, Y, U, V,
                 scale=20,  # Adjusted scale
                 width=0.004,  # Thicker arrows
                 headwidth=4,
                 headlength=5,
                 headaxislength=4.5,
                 alpha=0.6,
                 color='gray')
    
    # Add wind vector reference with larger font
    ref_speed = wind_field.base_speed
    ax.quiverkey(Q, 0.95, 0.95, 1.0, 
                 f'{ref_speed:.0f} knots',
                 labelpos='E',
                 coordinates='axes',
                 fontproperties={'size': 12})
    
    # Add colorbar with larger font
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.ax.tick_params(labelsize=12)  # Correct way to set tick label size
    cbar.set_label('Wind Speed (knots)', fontsize=12)
    
    # Plot course elements with thicker lines
    ax.plot([0, 0], [0, course.length + course.extension], 'k--', alpha=0.5, linewidth=2)
    ax.plot([course.width, course.width], 
            [0, course.length + course.extension], 'k--', alpha=0.5, linewidth=2)
    
    # Plot start/finish line
    start_x = [course.start_line[0].x, course.start_line[1].x]
    start_y = [course.start_line[0].y, course.start_line[1].y]
    ax.plot(start_x, start_y, 'g-', linewidth=3, label='Start/Finish Line')
    
    # Plot top marks with larger size
    ax.plot(course.top_marks[0].x, course.top_marks[0].y, 'ro', 
            markersize=12, label='Left Mark')
    ax.plot(course.top_marks[1].x, course.top_marks[1].y, 'ro', 
            markersize=12, label='Right Mark')
    ax.plot([course.top_marks[0].x, course.top_marks[1].x],
            [course.top_marks[0].y, course.top_marks[1].y],
            'r--', alpha=0.5, linewidth=2, label='Gate')
    
    # Plot boat tracks with thicker lines
    xs_baseline = [state.position.x for state in baseline_states]
    ys_baseline = [state.position.y for state in baseline_states]
    ax.plot(xs_baseline, ys_baseline, 'b-', linewidth=2.5, label='Baseline Track')
    
    xs_q = [state.position.x for state in q_states]
    ys_q = [state.position.y for state in q_states]
    ax.plot(xs_q, ys_q, 'r-', linewidth=2.5, label='Q-Learning Track')
    
    # Add Q-learning markers with larger fonts
    viz_points = 6
    time_indices = np.linspace(0, len(q_states)-1, viz_points, dtype=int)
    for i in time_indices[1:]:
        ax.plot(xs_q[i], ys_q[i], 'r.', markersize=12)
        label_text = (f'Q-Learning\nBoat: {q_learning_speeds[i]:.1f}kts\n'
                     f'TWA: {q_states[i].last_twa:.0f}°\n'
                     f'Wind: {q_winds[i].speed:.1f}kts @ {q_winds[i].direction:.0f}°')
        ax.annotate(label_text, 
                   (xs_q[i], ys_q[i]),
                   xytext=(10, -40), 
                   textcoords='offset points',
                   fontsize=10,
                   bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))
    
    # Set title and labels with larger fonts
    ax.set_title(f"Baseline vs Q-Learning Strategy Comparison (Seed: {seed})", 
                fontsize=14, pad=20)
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Distance (m)', fontsize=12)
    
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Adjust legend
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set axis limits with some padding
    ax.set_xlim(-50, course.width + 50)
    ax.set_ylim(-50, course.length + course.extension + 50)
    
    ax.set_aspect('equal')
    plt.tight_layout()
    
    return fig, ax

def create_baseline_visualization(course, wind_field, baseline_states, baseline_winds, seed):
    """Create race visualization showing only the baseline simulation"""
    
    # Create figure with adjusted size
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Plot wind speed as colored background
    mesh = ax.pcolormesh(wind_field.x_coords, wind_field.y_coords, wind_field.speeds, 
                        shading='auto', 
                        cmap='YlOrRd', 
                        alpha=0.5)
    
    # Create wind vector grid with varying directions
    skip = 6  # Increased skip for less crowded arrows
    X, Y = np.meshgrid(wind_field.x_coords[::skip], wind_field.y_coords[::skip])
    
    # Calculate wind directions at each point
    y_positions = np.array([w.direction for w in baseline_winds])
    y_times = np.linspace(0, course.length + course.extension, len(baseline_winds))
    
    # Create interpolated wind direction field
    wind_directions = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            idx = np.abs(y_times - Y[i,j]).argmin()
            wind_directions[i,j] = np.radians(baseline_winds[idx].direction)
    
    # Create wind direction field
    U = -np.sin(wind_directions)
    V = -np.cos(wind_directions)
    
    # Scale arrows by local wind speed
    speeds = wind_field.speeds[::skip, ::skip]
    scale_factor = speeds / wind_field.base_speed
    U = U * scale_factor
    V = V * scale_factor
    
    # Plot wind vectors
    Q = ax.quiver(X, Y, U, V,
                 scale=20,
                 width=0.004,
                 headwidth=4,
                 headlength=5,
                 headaxislength=4.5,
                 alpha=0.6,
                 color='gray')
    
    # Add wind vector reference
    ref_speed = wind_field.base_speed
    ax.quiverkey(Q, 0.95, 0.95, 1.0, 
                 f'{ref_speed:.0f} knots',
                 labelpos='E',
                 coordinates='axes',
                 fontproperties={'size': 12})
    
    # Add colorbar
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Wind Speed (knots)', fontsize=12)
    
    # Plot course elements
    ax.plot([0, 0], [0, course.length + course.extension], 'k--', alpha=0.5, linewidth=2)
    ax.plot([course.width, course.width], 
            [0, course.length + course.extension], 'k--', alpha=0.5, linewidth=2)
    
    # Plot start/finish line
    start_x = [course.start_line[0].x, course.start_line[1].x]
    start_y = [course.start_line[0].y, course.start_line[1].y]
    ax.plot(start_x, start_y, 'g-', linewidth=3, label='Start/Finish Line')
    
    # Plot top marks
    ax.plot(course.top_marks[0].x, course.top_marks[0].y, 'ro', 
            markersize=12, label='Left Mark')
    ax.plot(course.top_marks[1].x, course.top_marks[1].y, 'ro', 
            markersize=12, label='Right Mark')
    ax.plot([course.top_marks[0].x, course.top_marks[1].x],
            [course.top_marks[0].y, course.top_marks[1].y],
            'r--', alpha=0.5, linewidth=2, label='Gate')
    
    # Plot baseline boat track
    xs_baseline = [state.position.x for state in baseline_states]
    ys_baseline = [state.position.y for state in baseline_states]
    ax.plot(xs_baseline, ys_baseline, 'b-', linewidth=2.5, label='Baseline Track')
    
    # Add baseline boat markers and annotations
    polars = PolarData()
    viz_points = 6
    time_indices = np.linspace(0, len(baseline_states)-1, viz_points, dtype=int)
    for i in time_indices[1:]:
        boat_speed = polars.get_boat_speed(abs(baseline_states[i].last_twa), 
                                         baseline_winds[i].speed)
        ax.plot(xs_baseline[i], ys_baseline[i], 'b.', markersize=12)
        label_text = (f'Baseline\nBoat: {boat_speed:.1f}kts\n'
                     f'TWA: {baseline_states[i].last_twa:.0f}°\n'
                     f'Wind: {baseline_winds[i].speed:.1f}kts @ {baseline_winds[i].direction:.0f}°')
        ax.annotate(label_text, 
                   (xs_baseline[i], ys_baseline[i]),
                   xytext=(10, -40), 
                   textcoords='offset points',
                   fontsize=10,
                   bbox=dict(facecolor='white', edgecolor='blue', alpha=0.7))
    
    # Set title and labels
    ax.set_title(f"Baseline Strategy Visualization (Seed: {seed})", 
                fontsize=14, pad=20)
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Distance (m)', fontsize=12)
    
    # Adjust tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set axis limits with padding
    ax.set_xlim(-50, course.width + 50)
    ax.set_ylim(-50, course.length + course.extension + 50)
    
    ax.set_aspect('equal')
    plt.tight_layout()
    
    return fig, ax



if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    plots_dir = "sailing_plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        

    for i in range(10):
        print("ITERATION number " + str(i))

        # Set main random seed for reproducibility
        MAIN_SEED = random.randint(0, 2**32 - 1)

        print("SEED this run: " + str(MAIN_SEED))
        np.random.seed(MAIN_SEED)
        
        # Run baseline simulation
        print("\nRunning baseline simulation...")
        course = create_standard_course()
        baseline_states, baseline_winds, wind_field = simulate_upwind_leg(seed=MAIN_SEED)
        
        # Train Q-learning agent
        print("\nTraining Q-learning agent...")
        final_Q, best_Q = train_q_learning(seed=MAIN_SEED)

        # Run simulation with the best Q-table instead of final
        print("\nRunning Q-learning simulation with best policy...")
        q_states, q_winds = simulate_q_learning(best_Q, seed=MAIN_SEED)
        
        # Calculate boat speeds using polars
        polars = PolarData()
        baseline_speeds = []
        q_learning_speeds = []
        
        # Calculate speeds for baseline
        for state, wind in zip(baseline_states, baseline_winds):
            speed = polars.get_boat_speed(abs(state.last_twa), wind.speed)
            baseline_speeds.append(speed)
        
        # Calculate speeds for Q-learning
        for state, wind in zip(q_states, q_winds):
            speed = polars.get_boat_speed(abs(state.last_twa), wind.speed)
            q_learning_speeds.append(speed)
        
        # Generate filename with timestamp and seed
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{plots_dir}/sailing_comparison_{timestamp}_seed{MAIN_SEED}.png"
        
        # Create visualization
        fig, ax = create_race_visualization(course, wind_field, baseline_states, baseline_winds,
                                        q_states, q_winds, q_learning_speeds, MAIN_SEED)
        
        # Save the figure
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as: {filename}")
        
        
        # Show the plot
        #plt.show()
        
        # Clean up
        #plt.close(fig)

        # Create baseline visualization
        fig, ax = create_baseline_visualization(course, wind_field, baseline_states, baseline_winds, MAIN_SEED)

        # Save the figure
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{plots_dir}/sailing_baseline_{timestamp}_seed{MAIN_SEED}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')

        # Optional: display the plot
        #plt.show()

        # Clean up
        #plt.close(fig)