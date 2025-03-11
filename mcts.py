import numpy as np
import random
import time
import scipy
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from collections import defaultdict
import heapq
from tqdm import tqdm

#########################
# Discretized State Space
#########################

class DiscreteStateConverter:
    """Converts continuous sailing states to a reduced state space focusing on gusts/lulls."""

    def __init__(self, grid_size_x, grid_size_y, gust_threshold=16.0, lull_threshold=13.0, grid_resolution=10):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.gust_threshold = gust_threshold  # High wind speed threshold
        self.lull_threshold = lull_threshold  # Low wind speed threshold
        self.grid_resolution= grid_resolution

        # Cache gust and lull locations
        self.gust_positions = []  # List of (x, y) locations with strong gusts
        self.lull_positions = []  # List of (x, y) locations with weak wind

    def precompute_gusts_and_lulls(self, wind_grid):
        """Identify gusts and lulls at exact points where wind speed is above or below the threshold."""
        self.gust_positions = []
        self.lull_positions = []

        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                wind_speed, _ = wind_grid.get_wind_at_position(x, y)
                if x == 24 and y == 0:
                    print(f"WIND SPEED WHEN PRECOMPUTING GUSTS AND LULLS: {wind_speed}")
                if wind_speed > self.gust_threshold:  # Strictly above the threshold
                    self.gust_positions.append((x, y))
                elif wind_speed < self.lull_threshold:  # Strictly below the threshold
                    self.lull_positions.append((x, y))

        # Ensure gust and lull lists are not empty
        if not self.gust_positions:
            self.gust_positions.append((0, 0))  # Default gust position to avoid errors
        if not self.lull_positions:
            self.lull_positions.append((self.grid_size_x - 1, self.grid_size_y - 1))  # Default lull position

    def get_nearest_gust_lull(self, x, y):
        """Find the nearest gust and lull to the given position."""
        nearest_gust = min(self.gust_positions, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2, default=None)
        nearest_lull = min(self.lull_positions, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2, default=None)

        gust_distance = ((nearest_gust[0] - x) ** 2 + (nearest_gust[1] - y) ** 2) ** 0.5 if nearest_gust else float(
            'inf')
        lull_distance = ((nearest_lull[0] - x) ** 2 + (nearest_lull[1] - y) ** 2) ** 0.5 if nearest_lull else float(
            'inf')

        return gust_distance, lull_distance

    def get_reduced_state(self, position, heading):
        """Convert continuous state into a reduced representation."""
        x, y = position
        gust_dist, lull_dist = self.get_nearest_gust_lull(x, y)
        return (round(x), round(y), round(heading % 360, -1), round(gust_dist, 1), round(lull_dist, 1))


#########################
# MCTS Implementation
#########################

class MCTSNode:
    """Node for Monte Carlo Tree Search using reduced state space."""

    def __init__(self, state, parent=None, action=None):
        self.state = state  # Reduced state representation
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
        log_visits = np.log(self.visits)

        def ucb_score(child):
            if child.visits == 0:
                return float('inf')
            exploitation = child.value / child.visits
            exploration = exploration_weight * np.sqrt(log_visits / child.visits)
            return exploitation + exploration

        return max(self.children.values(), key=ucb_score)

    def expand(self, action, next_state):
        """Bias expansion toward better sailing conditions."""
        child = MCTSNode(next_state, parent=self, action=action)
        self.children[action] = child

        if self.untried_actions is not None:
            # Prioritize actions that move toward gusts and away from lulls
            gust_dist, lull_dist = next_state[3], next_state[4]
            if gust_dist < lull_dist:  # Prefer gusts
                self.untried_actions = sorted(self.untried_actions, key=lambda a: random.random() - 0.3)
            else:  # Avoid lulls
                self.untried_actions = sorted(self.untried_actions, key=lambda a: random.random() + 0.3)
            self.untried_actions.remove(action)

        return child

    def update(self, reward):
        """Update node statistics with reward from simulation"""
        self.visits += 1
        self.value += reward


class SailingMCTS:
    """MCTS algorithm adapted for sailing path optimization using gusts/lulls."""

    def __init__(self, sailing_env, state_converter, iterations=1000, exploration_weight=1.0):
        self.sailing_env = sailing_env
        self.state_converter = state_converter
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.state_converter.precompute_gusts_and_lulls(self.sailing_env.wind_grid)

    def calculate_reward(self, state, next_state):
        """Calculate reward based on distance to gusts/lulls and progress to goal."""
        x1, y1, heading1, gust_dist1, lull_dist1 = state
        x2, y2, heading2, gust_dist2, lull_dist2 = next_state

        goal_x, goal_y = self.sailing_env.goal_pos
        dist1 = ((goal_x - x1) ** 2 + (goal_y - y1) ** 2) ** 0.5
        dist2 = ((goal_x - x2) ** 2 + (goal_y - y2) ** 2) ** 0.5

        progress_reward = 200.0 * (dist1 - dist2)  # Increased weight for moving forward
        gust_reward = 100.0 / (gust_dist2 + 1e-3)  # Larger reward when closer to gust
        lull_penalty = -100.0 / (lull_dist2 + 1e-3)  # Larger penalty when closer to lull
        heading_penalty = -0.5 * abs(heading2 - heading1)  # Reduce unnecessary heading changes

        # Encourage sailing at optimal wind angles
        wind_speed, wind_dir = self.sailing_env.wind_grid.get_wind_at_position(x2, y2)
        twa = abs((heading2 - wind_dir) % 360)
        if twa > 180:
            twa = 360 - twa
        wind_angle_penalty = -20.0 if twa < self.sailing_env.polar_model.min_upwind_angle else 0  # Discourage sailing into no-go zone

        return progress_reward + gust_reward + lull_penalty + heading_penalty + wind_angle_penalty

    def find_optimal_path(self, max_steps=100):
        """Find optimal path from start to goal."""
        # state = self.sailing_env.reset()
        path = [tuple(self.sailing_env.position)]

        for _ in range(max_steps):
            action_idx = self.get_action(self.sailing_env)
            state, reward, done, _ = self.sailing_env.step(action_idx)
            path.append(tuple(self.sailing_env.position))
            if done:
                break

        return path

    def visualize_optimal_path_wind(self, path):
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

        for i in range(len(x)):
            for j in range(len(y)):
                wind_speed, wind_dir = self.sailing_env.wind_grid.get_wind_at_position(i, j)
                if i == 24 and j == 0:
                    print(f"WIND SPEED WHEN VISUALIZING PATH: {wind_speed}")
                wind_rad = np.radians(wind_dir)
                U[i, j] = wind_speed * np.sin(wind_rad)
                V[i, j] = wind_speed * np.cos(wind_rad)
                speed[j, i] = wind_speed

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
        x_ticks = np.linspace(0, self.state_converter.grid_size_x, self.state_converter.grid_resolution + 1)
        y_ticks = np.linspace(0, self.state_converter.grid_size_y, self.state_converter.grid_resolution + 1)

        for x in x_ticks:
            ax.axvline(x=x, color='gray', linestyle='--', alpha=0.3)
        for y in y_ticks:
            ax.axhline(y=y, color='gray', linestyle='--', alpha=0.3)

        # Plot gusts and lulls
        if self.state_converter.gust_positions:
            gust_x, gust_y = zip(*self.state_converter.gust_positions)
            ax.scatter(gust_x, gust_y, color='blue', label='Gusts')

        if self.state_converter.lull_positions:
            lull_x, lull_y = zip(*self.state_converter.lull_positions)
            ax.scatter(lull_x, lull_y, color='red', label='Lulls')

        ax.set_xlim(0, self.sailing_env.grid_size_x)
        ax.set_ylim(0, self.sailing_env.grid_size_y)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Optimal Sailing Path with Wind Field')
        ax.legend()

        plt.savefig('optimal_sailing_path.png', dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_optimal_path_current(self, path):
        """Visualize optimal sailing path with a current field heatmap as background."""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Get grid size
        x = np.arange(0, self.sailing_env.grid_size_x, 1)
        y = np.arange(0, self.sailing_env.grid_size_y, 1)
        X, Y = np.meshgrid(x, y)

        # Collect current data
        U = np.zeros_like(X, dtype=float)  # X-component of current
        V = np.zeros_like(Y, dtype=float)  # Y-component of current
        speed = np.zeros_like(X, dtype=float)  # Current speed magnitude

        for i in range(len(x)):
            for j in range(len(y)):
                current_speed, current_dir = self.sailing_env.current_grid.get_current_at_position(i, j)
                current_x, current_y = self.sailing_env.current_grid.get_current_vector_at_position(i, j)
                U[j, i] = current_x  # X-component
                V[j, i] = current_y  # Y-component
                speed[j, i] = current_speed  # Current magnitude

        # Plot current speed as a heatmap
        c = ax.pcolormesh(X, Y, speed, cmap='viridis', shading='auto', alpha=0.7)
        plt.colorbar(c, ax=ax, label='Current Speed (knots)')

        # Overlay current direction with arrows
        subsample = 4  # Reduce arrow density
        ax.quiver(X[::subsample, ::subsample], Y[::subsample, ::subsample],
                  U[::subsample, ::subsample], V[::subsample, ::subsample],
                  scale=50, color='white', alpha=0.8, width=0.002)

        # Plot start and goal
        ax.plot(self.sailing_env.start_pos[0], self.sailing_env.start_pos[1], 'go', markersize=10, label='Start')
        ax.plot(self.sailing_env.goal_pos[0], self.sailing_env.goal_pos[1], 'ro', markersize=10, label='Goal')
        circle = Circle(self.sailing_env.goal_pos, self.sailing_env.goal_radius, fill=False, color='r', linestyle='--')
        ax.add_patch(circle)

        # Plot optimal path
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'k-', linewidth=2, label='Optimal Path')


        # Set axis limits and labels
        ax.set_xlim(0, self.sailing_env.grid_size_x)
        ax.set_ylim(0, self.sailing_env.grid_size_y)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Optimal Sailing Path with Current Field Heatmap')
        ax.legend()

        # Save and show plot
        plt.savefig('optimal_sailing_path_currents.png', dpi=300, bbox_inches='tight')
        plt.show()

    def get_action(self, env):
        """Run MCTS and return the best action for the current state."""
        discrete_state = self.state_converter.get_reduced_state(env.position, env.heading)
        root = MCTSNode(discrete_state)
        root.untried_actions = set(range(len(self.sailing_env.actions)))

        for _ in range(self.iterations):
            node = root
            while node.is_fully_expanded() and node.children:
                node = node.select_child(self.exploration_weight)

            if node.untried_actions:
                action = random.choice(list(node.untried_actions))
                next_state = self.simulate_action(node.state, action)
                node = node.expand(action, next_state)

            reward = self.calculate_reward(node.parent.state if node.parent else node.state, node.state)
            while node:
                node.update(reward)
                node = node.parent

        return max(root.children.items(), key=lambda x: x[1].visits)[0] if root.children else random.choice(
            list(root.untried_actions))


    def simulate_action(self, state, action):
        """Simulate taking an action from the current state using wind, current, and polar model."""
        x, y, heading, _, _ = state
        heading = (heading + self.sailing_env.actions[action]) % 360

        # Get wind data
        wind_speed, wind_dir = self.sailing_env.wind_grid.get_wind_at_position(x, y)
        true_wind_angle = abs((heading - wind_dir) % 360)
        if true_wind_angle > 180:
            true_wind_angle = 360 - true_wind_angle

        # Get boat speed from polar model
        boat_speed = self.sailing_env.polar_model.get_boat_speed(true_wind_angle, wind_speed)

        # Get current data
        if self.sailing_env.use_currents:
            current_speed, current_dir = self.sailing_env.current_grid.get_current_at_position(x, y)
            current_x, current_y = self.sailing_env.current_grid.get_current_vector_at_position(x, y)
        else:
            current_x, current_y = 0.0, 0.0

        # Calculate movement
        heading_rad = np.radians(heading)
        boat_dx = boat_speed * np.sin(heading_rad)
        boat_dy = boat_speed * np.cos(heading_rad)

        # Combine boat movement with current
        new_x = max(0, min(self.sailing_env.grid_size_x - 1, x + round(boat_dx + current_x)))
        new_y = max(0, min(self.sailing_env.grid_size_y - 1, y + round(boat_dy + current_y)))

        return self.state_converter.get_reduced_state((new_x, new_y), heading)


#########################
# Main Optimization Function
#########################

def optimize_sailing_path(sailing_env, mcts_iterations=10000):
    """Find optimal sailing path using MCTS"""
    print("Initializing discretized state converter...")
    state_converter = DiscreteStateConverter(
        sailing_env.grid_size_x,
        sailing_env.grid_size_y
    )
    
    print(f"Running MCTS with {mcts_iterations} iterations per step...")
    mcts = SailingMCTS(
        sailing_env,
        state_converter,
        iterations=mcts_iterations,
        exploration_weight=0.7
    )
    
    print("Finding optimal path...")
    start_time = time.time()
    path = mcts.find_optimal_path()
    elapsed_time = time.time() - start_time
    
    print(f"Path found in {elapsed_time:.2f} seconds")
    print(f"Path length: {len(path)} steps")
    
    # Visualize path
    print("Visualizing optimal path...")
    mcts.visualize_optimal_path_current(path)
    
    return path