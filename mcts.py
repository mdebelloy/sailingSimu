import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from collections import defaultdict
import heapq
from tqdm import tqdm

#########################
# Discretized State Space
#########################

class DiscreteStateConverter:
    """Converts continuous sailing states to discrete states for efficient MCTS"""
    def __init__(self, grid_size_x, grid_size_y, position_bins=10, heading_bins=8):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.position_bins = position_bins  # Number of bins for x and y position
        self.heading_bins = heading_bins    # Number of bins for heading (e.g., N, NE, E, SE, S, SW, W, NW)
        
        # Create position bin edges
        self.x_bin_edges = np.linspace(0, grid_size_x, position_bins + 1)
        self.y_bin_edges = np.linspace(0, grid_size_y, position_bins + 1)
        
        # Create heading bin edges (in degrees)
        self.heading_bin_edges = np.linspace(0, 360, heading_bins + 1)
        
        # Cache for discretized wind and current data
        self.wind_discretized = None
        self.current_discretized = None
        
    def discretize_position(self, x, y):
        """Convert continuous position to discrete bin indices"""
        x_bin = max(0, min(self.position_bins - 1, np.digitize(x, self.x_bin_edges) - 1))
        y_bin = max(0, min(self.position_bins - 1, np.digitize(y, self.y_bin_edges) - 1))
        return (x_bin, y_bin)
    
    def discretize_heading(self, heading):
        """Convert continuous heading to discrete bin index"""
        heading_normalized = heading % 360
        heading_bin = max(0, min(self.heading_bins - 1, 
                                np.digitize(heading_normalized, self.heading_bin_edges) - 1))
        return heading_bin
    
    def precompute_environment_features(self, wind_grid, current_grid):
        """Precompute discretized wind and current data for faster lookups"""
        # Create discretized matrices for wind and current
        wind_speed = np.zeros((self.position_bins, self.position_bins))
        wind_dir = np.zeros((self.position_bins, self.position_bins))
        
        current_speed = np.zeros((self.position_bins, self.position_bins))
        current_dir = np.zeros((self.position_bins, self.position_bins))
        
        # Calculate average values for each discrete cell
        for x_bin in range(self.position_bins):
            for y_bin in range(self.position_bins):
                # Get continuous region boundaries
                x_min = self.x_bin_edges[x_bin]
                x_max = self.x_bin_edges[x_bin + 1]
                y_min = self.y_bin_edges[y_bin]
                y_max = self.y_bin_edges[y_bin + 1]
                
                # Sample points within the region
                x_samples = np.linspace(x_min, x_max, 3)
                y_samples = np.linspace(y_min, y_max, 3)
                
                # Collect wind and current samples
                wind_speed_samples = []
                wind_dir_samples = []
                current_speed_samples = []
                current_dir_samples = []
                
                for x in x_samples:
                    for y in y_samples:
                        # Get wind data
                        ws, wd = wind_grid.get_wind_at_position(x, y)
                        wind_speed_samples.append(ws)
                        wind_dir_samples.append(wd)
                        
                        # Get current data if available
                        if current_grid:
                            cs, cd = current_grid.get_current_at_position(x, y)
                            current_speed_samples.append(cs)
                            current_dir_samples.append(cd)
                
                # Compute average values
                wind_speed[y_bin, x_bin] = np.mean(wind_speed_samples)
                
                # Average angles properly (circular mean)
                wind_dir_rad = np.radians(wind_dir_samples)
                mean_sin = np.mean(np.sin(wind_dir_rad))
                mean_cos = np.mean(np.cos(wind_dir_rad))
                wind_dir[y_bin, x_bin] = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360
                
                if current_grid:
                    current_speed[y_bin, x_bin] = np.mean(current_speed_samples)
                    
                    # Average angles properly
                    current_dir_rad = np.radians(current_dir_samples)
                    mean_sin = np.mean(np.sin(current_dir_rad))
                    mean_cos = np.mean(np.cos(current_dir_rad))
                    current_dir[y_bin, x_bin] = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360
        
        # Store discretized environment data
        self.wind_discretized = (wind_speed, wind_dir)
        self.current_discretized = (current_speed, current_dir)
    
    def get_discrete_state(self, position, heading):
        """Convert full continuous state to discrete state tuple"""
        x, y = position
        discrete_pos = self.discretize_position(x, y)
        discrete_heading = self.discretize_heading(heading)
        
        return (discrete_pos[0], discrete_pos[1], discrete_heading)
    
    def get_environment_at_position(self, discrete_pos):
        """Get discretized wind and current data for a position"""
        x_bin, y_bin = discrete_pos
        
        wind_speed = self.wind_discretized[0][y_bin, x_bin]
        wind_dir = self.wind_discretized[1][y_bin, x_bin]
        
        if self.current_discretized:
            current_speed = self.current_discretized[0][y_bin, x_bin]
            current_dir = self.current_discretized[1][y_bin, x_bin]
        else:
            current_speed = 0.0
            current_dir = 0.0
        
        return wind_speed, wind_dir, current_speed, current_dir
    
    def reconstruct_continuous_position(self, discrete_pos):
        """Get continuous position from discrete position"""
        x_bin, y_bin = discrete_pos
        
        # Get bin centers
        x_center = (self.x_bin_edges[x_bin] + self.x_bin_edges[x_bin + 1]) / 2
        y_center = (self.y_bin_edges[y_bin] + self.y_bin_edges[y_bin + 1]) / 2
        
        return (x_center, y_center)
    
    def reconstruct_continuous_heading(self, discrete_heading):
        """Get continuous heading from discrete heading"""
        heading_min = self.heading_bin_edges[discrete_heading]
        heading_max = self.heading_bin_edges[discrete_heading + 1]
        
        return (heading_min + heading_max) / 2

#########################
# MCTS Implementation
#########################

class MCTSNode:
    """Node for Monte Carlo Tree Search"""
    def __init__(self, state, parent=None, action=None):
        self.state = state  # Discrete state tuple
        self.parent = parent
        self.action = action  # Action that led to this state
        
        self.children = {}  # Map from action to child node
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None  # Will be initialized on first visit
    
    def is_fully_expanded(self):
        """Check if all possible actions have been tried"""
        return self.untried_actions is not None and len(self.untried_actions) == 0
    
    def select_child(self, exploration_weight=1.0):
        """Select child with highest UCB score"""
        # UCB1 formula: value/visits + exploration_weight * sqrt(log(parent visits) / visits)
        log_visits = np.log(self.visits)
        
        def ucb_score(child):
            # Avoid division by zero by ensuring visits > 0
            if child.visits == 0:
                return float('inf')  # Unvisited nodes have infinite potential
            exploitation = child.value / child.visits
            exploration = exploration_weight * np.sqrt(log_visits / child.visits)
            return exploitation + exploration
        
        return max(self.children.values(), key=ucb_score)
    
    def expand(self, action, next_state):
        """Add a new child node for the given action and state"""
        child = MCTSNode(next_state, parent=self, action=action)
        self.children[action] = child
        
        # Remove action from untried_actions
        if self.untried_actions is not None:
            self.untried_actions.remove(action)
            
        return child
    
    def update(self, reward):
        """Update node statistics with reward from simulation"""
        self.visits += 1
        self.value += reward

class SailingMCTS:
    """MCTS algorithm adapted for sailing path optimization"""
    def __init__(self, sailing_env, state_converter, iterations=1000, exploration_weight=1.0):
        self.sailing_env = sailing_env
        self.state_converter = state_converter
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        
        # Initialize discretized environment data
        self.state_converter.precompute_environment_features(
            sailing_env.wind_grid, 
            sailing_env.current_grid if sailing_env.use_currents else None
        )
        
        # Cache for dynamics simulation
        self.dynamics_cache = {}
        
        # Cache for goal checking
        self.goal_pos_discrete = self.state_converter.discretize_position(
            sailing_env.goal_pos[0], sailing_env.goal_pos[1]
        )
    
    def get_action(self, env):
        """Run MCTS and return the best action for the current state"""
        discrete_state = self.state_converter.get_discrete_state(
            env.position, env.heading
        )
        
        # Create root node
        root = MCTSNode(discrete_state)
        # Initialize untried actions for root
        root.untried_actions = set(range(len(self.sailing_env.actions)))
        
        # Run MCTS iterations
        for _ in range(self.iterations):
            # Selection phase
            node = self._select(root)
            
            # Expansion and simulation
            if not self._is_terminal(node.state):
                # If node is not fully expanded, try an untried action
                if node.untried_actions is None:
                    # Get valid actions for this state
                    node.untried_actions = self._get_valid_actions(node.state)
                
                if not node.is_fully_expanded() and node.untried_actions:
                    # Choose a random untried action
                    action = random.choice(list(node.untried_actions))
                    
                    # Get next state using simplified dynamics
                    next_state = self._get_next_state(node.state, action)
                    
                    # Expand tree with new node
                    child = node.expand(action, next_state)
                    
                    # Simulate from new state
                    reward = self._simulate(next_state)
                else:
                    # If fully expanded, select best child and simulate
                    child = node.select_child(self.exploration_weight)
                    reward = self._simulate(child.state)
                    node = child
            else:
                # Terminal node, use its value
                reward = self._get_terminal_reward(node.state)
            
            # Backpropagation
            self._backpropagate(node, reward)
        
        # Return best action based on most visits
        if not root.children:
            # If no children, return a random action
            return random.randint(0, len(self.sailing_env.actions) - 1)
        
        # Validate that action is within valid range before returning
        best_action = max(root.children.items(), key=lambda x: x[1].visits)[0]
        if best_action < 0 or best_action >= len(self.sailing_env.actions):
            # If invalid, return middle action (0 change in heading)
            return len(self.sailing_env.actions) // 2
            
        return best_action
    
    def _select(self, node):
        """Select a node to expand using UCB"""
        while not self._is_terminal(node.state) and node.children:
            if not node.is_fully_expanded():
                # If not fully expanded, this is the node to expand
                return node
            
            # Select the child with highest UCB score
            node = node.select_child(self.exploration_weight)
        
        return node
    
    def _get_valid_actions(self, state):
        """Get valid actions for the given state"""
        # All heading changes are valid
        # We could restrict based on state if needed
        return set(range(len(self.sailing_env.actions)))
    
    def _get_next_state(self, state, action):
        """Get next state using simplified dynamics model"""
        # Check cache first
        cache_key = (state, action)
        if cache_key in self.dynamics_cache:
            return self.dynamics_cache[cache_key]
        
        # Unpack state
        x_bin, y_bin, heading_bin = state
        
        # Get continuous values
        x, y = self.state_converter.reconstruct_continuous_position((x_bin, y_bin))
        heading = self.state_converter.reconstruct_continuous_heading(heading_bin)
        
        # Get environment data
        wind_speed, wind_dir, current_speed, current_dir = self.state_converter.get_environment_at_position(
            (x_bin, y_bin)
        )
        
        # Apply heading change
        new_heading = (heading + self.sailing_env.actions[action]) % 360
        
        # Calculate true wind angle
        twa = (new_heading - wind_dir) % 360
        if twa > 180:
            twa = 360 - twa
        
        # Get boat speed
        boat_speed = self.sailing_env.polar_model.get_boat_speed(twa, wind_speed)
        
        # Calculate movement
        heading_rad = np.radians(new_heading)
        dx = boat_speed * np.sin(heading_rad) * self.sailing_env.time_step
        dy = boat_speed * np.cos(heading_rad) * self.sailing_env.time_step
        
        # Add current effect
        if current_speed > 0:
            current_rad = np.radians(current_dir)
            dx += current_speed * np.sin(current_rad) * self.sailing_env.time_step
            dy += current_speed * np.cos(current_rad) * self.sailing_env.time_step
        
        # Calculate new position
        new_x = x + dx
        new_y = y + dy
        
        # Bound to grid
        new_x = max(0, min(new_x, self.sailing_env.grid_size_x - 1))
        new_y = max(0, min(new_y, self.sailing_env.grid_size_y - 1))
        
        # Check if too close to goal
        goal_x, goal_y = self.sailing_env.goal_pos
        dist_to_goal = np.sqrt((new_x - goal_x)**2 + (new_y - goal_y)**2)
        
        # If within goal radius, adjust position to be exactly at goal
        if dist_to_goal <= self.sailing_env.goal_radius:
            new_x, new_y = goal_x, goal_y
        
        # Convert to discrete state
        new_discrete_pos = self.state_converter.discretize_position(new_x, new_y)
        new_discrete_heading = self.state_converter.discretize_heading(new_heading)
        next_state = (new_discrete_pos[0], new_discrete_pos[1], new_discrete_heading)
        
        # Cache result
        self.dynamics_cache[cache_key] = next_state
        
        return next_state
    
    def _is_terminal(self, state):
        """Check if state is terminal (reached goal or out of bounds)"""
        x_bin, y_bin, _ = state
        
        # Check if reached goal
        if (x_bin, y_bin) == self.goal_pos_discrete:
            return True
        
        # Check if out of bounds
        if (x_bin < 0 or x_bin >= self.state_converter.position_bins or
            y_bin < 0 or y_bin >= self.state_converter.position_bins):
            return True
        
        return False
    
    def _get_terminal_reward(self, state):
        """Get reward for terminal state"""
        x_bin, y_bin, _ = state
        
        # Check if reached goal
        if (x_bin, y_bin) == self.goal_pos_discrete:
            return 20000.0  # Very high reward for reaching goal
        
        # Out of bounds penalty
        return -50.0
    
    def _simulate(self, state, max_steps=30):
        """Run a simulation from state using random policy"""
        current_state = state
        total_reward = 0.0
        discount = 1.0
        step = 0
        
        # Get initial position and goal
        x_bin, y_bin, _ = current_state
        x, y = self.state_converter.reconstruct_continuous_position((x_bin, y_bin))
        goal_x, goal_y = self.sailing_env.goal_pos
        
        # Initial distance to goal
        initial_dist = np.sqrt((goal_x - x)**2 + (goal_y - y)**2)
        best_dist = initial_dist  # Keep track of best distance achieved
        
        # Store initial state to check for loops
        visited_states = set([current_state])
        
        # Track if we've reached the goal during simulation
        reached_goal = False
        
        while not self._is_terminal(current_state) and step < max_steps:
            # Choose action (bias toward goal)
            if random.random() < 0.7:  # 70% of the time, use a smarter policy
                # Get bearing to goal
                x_bin, y_bin, heading_bin = current_state
                x, y = self.state_converter.reconstruct_continuous_position((x_bin, y_bin))
                heading = self.state_converter.reconstruct_continuous_heading(heading_bin)
                
                # Calculate desired heading toward goal
                goal_x, goal_y = self.sailing_env.goal_pos
                goal_angle = np.degrees(np.arctan2(goal_y - y, goal_x - x)) % 360
                
                # Get wind at position
                wind_speed, wind_dir, _, _ = self.state_converter.get_environment_at_position((x_bin, y_bin))
                
                # Account for no-go zone - can't sail directly upwind
                twa_to_goal = abs((goal_angle - wind_dir) % 360)
                if twa_to_goal > 180:
                    twa_to_goal = 360 - twa_to_goal
                
                min_angle = self.sailing_env.polar_model.min_upwind_angle
                
                # If trying to go upwind in no-go zone, adjust heading
                if twa_to_goal < min_angle:
                    # Choose to go on port or starboard tack
                    if random.random() < 0.5:
                        goal_angle = (wind_dir + min_angle) % 360
                    else:
                        goal_angle = (wind_dir - min_angle) % 360
                
                # Calculate heading difference
                heading_diff = (goal_angle - heading + 180) % 360 - 180
                
                # Choose action that gets closest to desired heading
                best_action = None
                best_score = float('-inf')
                
                valid_actions = self._get_valid_actions(current_state)
                for action in valid_actions:
                    new_heading = (heading + self.sailing_env.actions[action]) % 360
                    
                    # Calculate heading difference score
                    diff = abs((new_heading - goal_angle + 180) % 360 - 180)
                    heading_score = (180 - diff) / 180.0
                    
                    # Calculate boat speed with this heading
                    twa = abs((new_heading - wind_dir) % 360)
                    if twa > 180:
                        twa = 360 - twa
                    boat_speed = self.sailing_env.polar_model.get_boat_speed(twa, wind_speed)
                    speed_score = boat_speed / self.sailing_env.polar_model.max_speed
                    
                    # Combined score (prioritize speed slightly more)
                    score = 0.4 * heading_score + 0.6 * speed_score
                    
                    if score > best_score:
                        best_score = score
                        best_action = action
                
                if best_action is not None:
                    action = best_action
                else:
                    # Fallback to random action
                    action = random.choice(list(valid_actions))
            else:
                # Random action
                valid_actions = self._get_valid_actions(current_state)
                if not valid_actions:
                    break
                action = random.choice(list(valid_actions))
            
            # Get next state
            next_state = self._get_next_state(current_state, action)
            
            # Check for loops or revisits
            if next_state in visited_states:
                # Penalty for revisiting states
                reward = -5.0
                total_reward += discount * reward
                break
            
            visited_states.add(next_state)
            
            # Calculate reward
            reward = self._calculate_reward(current_state, next_state)
            
            # Update closest distance to goal
            x_bin, y_bin, _ = next_state
            x, y = self.state_converter.reconstruct_continuous_position((x_bin, y_bin))
            dist_to_goal = np.sqrt((goal_x - x)**2 + (goal_y - y)**2)
            best_dist = min(best_dist, dist_to_goal)
            
            # Check if we've reached the goal
            if (x_bin, y_bin) == self.goal_pos_discrete:
                reached_goal = True
            
            # Update
            total_reward += discount * reward
            discount *= 0.95  # Discount factor
            current_state = next_state
            step += 1
            
            # Early termination if goal reached
            if reached_goal:
                break
        
        # Add terminal reward if simulation ended in terminal state
        if self._is_terminal(current_state):
            terminal_reward = self._get_terminal_reward(current_state)
            total_reward += discount * terminal_reward
            
            # Extra reward for reaching goal in fewer steps
            if reached_goal:
                total_reward += discount * (200.0 * (max_steps - step) / max_steps)
        else:
            # If didn't reach goal, add reward based on how close we got
            improvement = initial_dist - best_dist
            total_reward += discount * (10 * improvement)
            
            # Extra penalty for not reaching goal
            if best_dist > self.sailing_env.goal_radius:
                total_reward -= discount * 50.0
        
        return total_reward
    
    def _calculate_reward(self, state, next_state):
        """Calculate reward for transition from state to next_state"""
        x_bin1, y_bin1, heading_bin1 = state
        x_bin2, y_bin2, heading_bin2 = next_state
        
        # Get continuous positions
        x1, y1 = self.state_converter.reconstruct_continuous_position((x_bin1, y_bin1))
        x2, y2 = self.state_converter.reconstruct_continuous_position((x_bin2, y_bin2))
        
        # Goal position
        goal_x, goal_y = self.sailing_env.goal_pos
        
        # Calculate distances to goal
        dist1 = np.sqrt((goal_x - x1)**2 + (goal_y - y1)**2)
        dist2 = np.sqrt((goal_x - x2)**2 + (goal_y - y2)**2)
        
        # Movement distance
        movement_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Get environment conditions
        wind_speed1, wind_dir1, current_speed1, current_dir1 = self.state_converter.get_environment_at_position((x_bin1, y_bin1))
        wind_speed2, wind_dir2, current_speed2, current_dir2 = self.state_converter.get_environment_at_position((x_bin2, y_bin2))
        
        # Reward boat speed - strongly prefer higher speeds
        heading2 = self.state_converter.reconstruct_continuous_heading(heading_bin2)
        twa = abs((heading2 - wind_dir2) % 360)
        if twa > 180:
            twa = 360 - twa
        boat_speed = self.sailing_env.polar_model.get_boat_speed(twa, wind_speed2)
        speed_reward = 5.0 * boat_speed / self.sailing_env.polar_model.max_speed
        
        # Strong reward for making progress toward goal
        progress_reward = 150.0 * (dist1 - dist2)
        
        # Penalty for distance from goal (to encourage direct paths)
        goal_distance_penalty = -0.1 * dist2
        
        # Penalty for zigzagging (heading changes)
        heading1 = self.state_converter.reconstruct_continuous_heading(heading_bin1)
        heading_change = abs((heading2 - heading1 + 180) % 360 - 180)
        zigzag_penalty = -0.1 * heading_change / 45.0
        
        # Small penalty for each step to encourage efficiency
        step_penalty = -0.2
        
        # Check if hit boundary
        if (x_bin2 == 0 or x_bin2 == self.state_converter.position_bins - 1 or
            y_bin2 == 0 or y_bin2 == self.state_converter.position_bins - 1):
            boundary_penalty = -20.0
        else:
            boundary_penalty = 0.0
            
        # Combine rewards
        total_reward = (
            progress_reward + 
            speed_reward +
            goal_distance_penalty + 
            zigzag_penalty + 
            step_penalty + 
            boundary_penalty
        )
        
        return total_reward
    
    def _backpropagate(self, node, reward):
        """Backpropagate reward up the tree"""
        while node is not None:
            node.update(reward)
            node = node.parent
    
    def find_optimal_path(self, max_steps=100):
        """Find optimal path from start to goal"""
        # Reset environment
        state = self.sailing_env.reset()
        
        # Initialize path with start position
        path = [tuple(self.sailing_env.position)]
        
        for _ in range(max_steps):
            # Get best action
            action_idx = self.get_action(self.sailing_env)
            
            # Ensure action index is valid
            if action_idx < 0 or action_idx >= len(self.sailing_env.actions):
                action_idx = len(self.sailing_env.actions) // 2  # Default to no heading change
            
            # Take step
            state, reward, done, _ = self.sailing_env.step(action_idx)
            
            # Record position
            path.append(tuple(self.sailing_env.position))
            
            if done:
                break
        
        return path

    def visualize_optimal_path(self, path):
        """Visualize optimal path on the environment"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot wind field as background
        x = np.arange(0, self.sailing_env.grid_size_x, 1)
        y = np.arange(0, self.sailing_env.grid_size_y, 1)
        X, Y = np.meshgrid(x, y)
        
        # Collect wind data
        U = np.zeros_like(X, dtype=float)
        V = np.zeros_like(Y, dtype=float)
        speed = np.zeros_like(X, dtype=float)
        
        for i in range(len(y)):
            for j in range(len(x)):
                wind_speed, wind_dir = self.sailing_env.wind_grid.get_wind_at_position(x[j], y[i])
                wind_rad = np.radians(wind_dir)
                U[i, j] = wind_speed * np.sin(wind_rad)
                V[i, j] = wind_speed * np.cos(wind_rad)
                speed[i, j] = wind_speed
        
        # Plot wind as background color
        c = ax.pcolormesh(X, Y, speed, cmap='viridis', alpha=0.3)
        plt.colorbar(c, ax=ax, label='Wind Speed (knots)')
        
        # Plot wind vectors (subsampled)
        subsample = 4
        ax.quiver(X[::subsample, ::subsample], Y[::subsample, ::subsample], 
                 U[::subsample, ::subsample], V[::subsample, ::subsample],
                 scale=400, color='blue', alpha=0.6)
        
        # Plot start and goal
        ax.plot(self.sailing_env.start_pos[0], self.sailing_env.start_pos[1], 'go', markersize=10, label='Start')
        ax.plot(self.sailing_env.goal_pos[0], self.sailing_env.goal_pos[1], 'ro', markersize=10, label='Goal')
        circle = Circle(self.sailing_env.goal_pos, self.sailing_env.goal_radius, fill=False, color='r', linestyle='--')
        ax.add_patch(circle)
        
        # Plot path
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'k-', linewidth=2, label='Optimal Path')
        
        # Plot grid lines for discretization
        for i in range(self.state_converter.position_bins + 1):
            ax.axhline(y=self.state_converter.y_bin_edges[i], color='gray', linestyle='--', alpha=0.3)
            ax.axvline(x=self.state_converter.x_bin_edges[i], color='gray', linestyle='--', alpha=0.3)
        
        ax.set_xlim(0, self.sailing_env.grid_size_x)
        ax.set_ylim(0, self.sailing_env.grid_size_y)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Optimal Sailing Path with Wind Field')
        ax.legend()
        
        plt.savefig('optimal_sailing_path.png', dpi=300, bbox_inches='tight')
        plt.show()

#########################
# Main Optimization Function
#########################

def optimize_sailing_path(sailing_env, position_bins=10, heading_bins=8, mcts_iterations=1000):
    """Find optimal sailing path using MCTS"""
    print("Initializing discretized state converter...")
    state_converter = DiscreteStateConverter(
        sailing_env.grid_size_x,
        sailing_env.grid_size_y,
        position_bins=position_bins,
        heading_bins=heading_bins
    )
    
    print(f"Running MCTS with {mcts_iterations} iterations per step...")
    mcts = SailingMCTS(
        sailing_env,
        state_converter,
        iterations=mcts_iterations,
        exploration_weight=1.0
    )
    
    print("Finding optimal path...")
    start_time = time.time()
    path = mcts.find_optimal_path()
    elapsed_time = time.time() - start_time
    
    print(f"Path found in {elapsed_time:.2f} seconds")
    print(f"Path length: {len(path)} steps")
    
    # Visualize path
    print("Visualizing optimal path...")
    mcts.visualize_optimal_path(path)
    
    return path