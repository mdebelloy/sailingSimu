#avoidance.py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
from boat import Boat, Position, BoatState
from polars import PolarData
import math
import random
from scipy.stats import chi2
from typing import Tuple, Optional

@dataclass
class EncounterState:
    """State representation for collision encounters"""
    relative_distance: float  # meters
    relative_bearing: float  # degrees
    own_tack: str
    other_tack: str
    time_to_collision: float
    closing_speed: float

class VesselKalmanFilter:
    """Kalman filter for vessel tracking with uncertainty"""
    def __init__(self, dt: float, process_noise: float = 0.1):
        # State: [x, y, vx, vy]
        self.dt = dt
        self.n_states = 4
        
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only observe position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise
        q = process_noise
        self.Q = q * np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ])
        
        # Measurement noise
        self.R = np.array([
            [10.0, 0],    # Position uncertainty (meters)
            [0, 10.0]
        ])
        
        # Initial state uncertainty
        self.P = np.eye(4) * 100
        
        self.state = None
        self.initialized = False
    
    def initialize(self, pos_x: float, pos_y: float, vel_x: float, vel_y: float):
        """Initialize state"""
        self.state = np.array([pos_x, pos_y, vel_x, vel_y])
        self.initialized = True
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state and covariance"""
        if not self.initialized:
            raise ValueError("Filter not initialized")
            
        # Predict state
        self.state = self.F @ self.state
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state, self.P
    
    def update(self, measurement: np.ndarray):
        """Update state with measurement"""
        if not self.initialized:
            raise ValueError("Filter not initialized")
            
        # Calculate Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        y = measurement - self.H @ self.state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(self.n_states)
        self.P = (I - K @ self.H) @ self.P
        
        return self.state, self.P
    
    def get_uncertainty_ellipse(self, confidence: float = 0.95) -> Tuple[float, float, float]:
        """Get uncertainty ellipse parameters for plotting"""
        if not self.initialized:
            raise ValueError("Filter not initialized")
            
        # Get position covariance
        pos_cov = self.P[:2, :2]
        
        # Calculate eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eig(pos_cov)
        
        # Sort by eigenvalue
        idx = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Calculate ellipse parameters
        chi2_val = chi2.ppf(confidence, df=2)
        a = np.sqrt(chi2_val * eigenvals[0])
        b = np.sqrt(chi2_val * eigenvals[1])
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        
        return a, b, angle

class CollisionTrainer:
    def __init__(self, polars: PolarData):
        self.polars = polars
        
        # Training parameters
        self.n_distance_bins = 20  # Distance discretization
        self.n_bearing_bins = 16   # Bearing discretization
        self.n_actions = 3         # maintain, tack, bear_away
        
        # State space definition
        self.max_distance = 500.0  # meters
        self.time_horizon = 300.0  # seconds
        
        # Initialize Q-table
        # States: distance bins × bearing bins × own_tack × other_tack
        self.Q = np.zeros((
            self.n_distance_bins,
            self.n_bearing_bins,
            2,  # port/starboard
            2,  # port/starboard
            self.n_actions
        ))
        
        # Rewards
        self.collision_penalty = -1000    # Reduced to prevent overshadowing other rewards
        self.tacking_penalty = -20        # Reduced to encourage more tacking when needed
        self.bearing_away_penalty = -10   # Reduced to encourage course changes
        self.maintain_reward = 2          # Increased to reward maintaining course when safe
        self.safe_distance_reward = 5     # New reward for maintaining safe distance
        
        # Other vessel behavior
        self.other_vessel_action_prob = 0.4  # Increased probability of other vessel action
        
        # Training parameters
        self.min_safe_distance = 100  # meters
        self.preferred_distance = 200  # meters

    def calculate_reward(self, state: EncounterState, action: int, next_state: EncounterState) -> float:
        """Enhanced reward calculation"""
        reward = 0
        
        # Distance-based rewards
        if next_state.relative_distance < 20:  # Collision
            reward += self.collision_penalty
        elif next_state.relative_distance > self.min_safe_distance:
            # Reward for maintaining safe distance
            reward += self.safe_distance_reward * min(1.0, 
                next_state.relative_distance / self.preferred_distance)
        
        # Action penalties
        if action == 1:  # tack
            # Scale tacking penalty based on distance - cheaper to tack early
            distance_factor = max(0.2, min(1.0, 
                next_state.relative_distance / self.preferred_distance))
            reward += self.tacking_penalty * distance_factor
        elif action == 2:  # bear away
            reward += self.bearing_away_penalty
        else:  # maintain course
            reward += self.maintain_reward
        
        # Priority vessel rewards/penalties
        if state.own_tack == 'starboard' and action > 0:
            # Extra penalty for priority vessel changing course unnecessarily
            reward -= 10
        elif state.own_tack == 'port' and state.other_tack == 'starboard':
            # Reward for giving way when required
            if action > 0:
                reward += 15
            else:
                reward -= 5
                
        return reward
        
    def discretize_state(self, state: EncounterState) -> Tuple[int, int, int, int]:
        """Convert continuous state to discrete indices"""
        dist_idx = min(self.n_distance_bins - 1, 
                      int(state.relative_distance / self.max_distance * self.n_distance_bins))
        bearing_idx = min(self.n_bearing_bins - 1, 
                         int((state.relative_bearing + 180) / 360 * self.n_bearing_bins))
        own_tack_idx = 0 if state.own_tack == 'starboard' else 1
        other_tack_idx = 0 if state.other_tack == 'starboard' else 1
        
        return dist_idx, bearing_idx, own_tack_idx, other_tack_idx

    def simulate_step(self, state: EncounterState, own_action: int, 
                     dt: float = 1.0) -> Tuple[EncounterState, float, bool]:
        """Simulate one timestep of the encounter"""
        # Unpack state
        distance = state.relative_distance
        bearing = state.relative_bearing
        closing_speed = state.closing_speed
        
        # Initialize reward
        reward = 0
        
        # Apply own action
        if own_action == 1:  # tack
            reward += self.tacking_penalty
            state.own_tack = 'port' if state.own_tack == 'starboard' else 'starboard'
            closing_speed *= 0.5  # Reduced closing speed after tack
            
        elif own_action == 2:  # bear away
            reward += self.bearing_away_penalty
            closing_speed *= 0.7  # Reduced closing speed after bearing away
            
        else:  # maintain course
            reward += self.maintain_reward
            
        # Other vessel's stochastic behavior
        if random.random() < self.other_vessel_action_prob:
            if state.other_tack == 'port':  # Non-priority vessel more likely to take action
                state.other_tack = 'starboard'
                closing_speed *= 0.5
                
        # Update distance and time
        new_distance = distance - closing_speed * dt
        new_time = state.time_to_collision - dt
        
        # Check for collision
        collision = new_distance < 20  # 20 meters collision threshold
        if collision:
            reward += self.collision_penalty
            
        # Update state
        new_state = EncounterState(
            relative_distance=new_distance,
            relative_bearing=bearing,  # Simplified - bearing changes omitted
            own_tack=state.own_tack,
            other_tack=state.other_tack,
            time_to_collision=new_time,
            closing_speed=closing_speed
        )
        
        return new_state, reward, collision

    def train_policy(self, n_episodes: int = 10000, 
                    learning_rate: float = 0.1,
                    discount_factor: float = 0.95,
                    epsilon_start: float = 1.0,
                    epsilon_end: float = 0.01,
                    epsilon_decay: float = 0.995) -> List[float]:
        """Train the collision avoidance policy"""


        learning_rate = 0.1
        discount_factor = 0.99  # Increased to value future rewards more
        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.997  # Slower decay

        rewards_history = []
        epsilon = epsilon_start
        
        for episode in range(n_episodes):
            # Initialize random encounter
            initial_state = EncounterState(
                relative_distance=random.uniform(100, self.max_distance),
                relative_bearing=random.uniform(-180, 180),
                own_tack=random.choice(['port', 'starboard']),
                other_tack=random.choice(['port', 'starboard']),
                time_to_collision=random.uniform(60, self.time_horizon),
                closing_speed=random.uniform(5, 15)
            )
            
            total_reward = 0
            state = initial_state
            done = False
            
            while not done:
                # Discretize state
                state_idx = self.discretize_state(state)
                
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = random.randint(0, self.n_actions - 1)
                else:
                    action = np.argmax(self.Q[state_idx])
                
                # Take action
                next_state, reward, collision = self.simulate_step(state, action)
                next_state_idx = self.discretize_state(next_state)
                
                # Q-learning update
                best_next_action = np.argmax(self.Q[next_state_idx])
                td_target = reward + discount_factor * self.Q[next_state_idx][best_next_action]
                td_error = td_target - self.Q[state_idx][action]
                self.Q[state_idx][action] += learning_rate * td_error
                
                total_reward += reward
                state = next_state
                
                # Check termination
                done = collision or state.time_to_collision <= 0 or state.relative_distance > self.max_distance
                
            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            rewards_history.append(total_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.1f}, Epsilon: {epsilon:.3f}")
                
        return rewards_history

    def simulate_encounter(self, initial_state: EncounterState) -> List[Tuple[EncounterState, int]]:
        """Simulate an encounter using the learned policy"""
        trajectory = []
        state = initial_state
        done = False
        
        while not done:
            state_idx = self.discretize_state(state)
            action = np.argmax(self.Q[state_idx])
            
            trajectory.append((state, action))
            next_state, _, collision = self.simulate_step(state, action)
            state = next_state
            
            done = collision or state.time_to_collision <= 0 or state.relative_distance > self.max_distance
            
        return trajectory

    def plot_encounter(self, trajectory: List[Tuple[EncounterState, int]], title: str, wind_direction: float = 0):
        """
        Visualize an encounter simulation with wind direction, collision path, and uncertainty
        Args:
            trajectory: List of (state, action) tuples
            title: Plot title
            wind_direction: degrees, 0 = wind from north, clockwise positive
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Extract trajectory data
        distances = [state.relative_distance for state, _ in trajectory]
        times = [state.time_to_collision for state, _ in trajectory]
        actions = [action for _, action in trajectory]
        
        # Plot distance over time (top plot)
        ax1.plot(times, distances, 'b-', label='Relative Distance', linewidth=2)
        ax1.set_xlabel('Time to Potential Collision (s)')
        ax1.set_ylabel('Distance (m)')
        ax1.grid(True, alpha=0.3)
        
        # Mark actions on distance plot
        action_names = ['maintain', 'tack', 'bear_away']
        colors = ['gray', 'red', 'blue']
        for i, (action, time) in enumerate(zip(actions, times)):
            if action > 0:  # Skip maintain course
                ax1.axvline(x=time, color=colors[action], linestyle='--', alpha=0.5,
                        label=f'{action_names[action]}' if i == 0 else "")
        
        ax1.legend()
        
        # Calculate vessel tracks
        initial_state = trajectory[0][0]
        
        # Own vessel track
        own_x = [0]
        own_y = [0]
        own_heading = 0  # Start pointing north
        
        # Other vessel track
        other_x = [initial_state.relative_distance * math.sin(math.radians(initial_state.relative_bearing))]
        other_y = [initial_state.relative_distance * math.cos(math.radians(initial_state.relative_bearing))]
        other_heading = initial_state.relative_bearing + 180
        
        # Initialize Kalman filter for other vessel
        dt = trajectory[0][0].time_to_collision / len(trajectory)
        kf = VesselKalmanFilter(dt=dt)
        kf.initialize(other_x[0], other_y[0], 
                    initial_state.closing_speed * math.sin(math.radians(other_heading)),
                    initial_state.closing_speed * math.cos(math.radians(other_heading)))
        
        # Store uncertainty ellipses
        uncertainty_ellipses = []
        
        # Generate tracks
        for i, ((state, action), next_state_action) in enumerate(zip(trajectory[:-1], trajectory[1:])):
            next_state = next_state_action[0]
            
            # Update own vessel position based on action
            if action == 1:  # tack
                own_heading = own_heading + (90 if state.own_tack == 'starboard' else -90)
            elif action == 2:  # bear away
                own_heading = own_heading + (15 if state.own_tack == 'starboard' else -15)
                
            own_speed = 6.0  # Example fixed speed
            own_dx = own_speed * math.sin(math.radians(own_heading)) * dt
            own_dy = own_speed * math.cos(math.radians(own_heading)) * dt
            
            own_x.append(own_x[-1] + own_dx)
            own_y.append(own_y[-1] + own_dy)
            
            # Update other vessel
            other_speed = state.closing_speed
            if next_state.other_tack != state.other_tack:
                other_heading = other_heading + (90 if state.other_tack == 'starboard' else -90)
                
            other_dx = other_speed * math.sin(math.radians(other_heading)) * dt
            other_dy = other_speed * math.cos(math.radians(other_heading)) * dt
            
            other_x.append(other_x[-1] + other_dx)
            other_y.append(other_y[-1] + other_dy)
            
            # Predict next state with Kalman filter
            kf.predict()
            
            # Create noisy measurement
            noise_x = np.random.normal(0, 5)  # 5m std dev
            noise_y = np.random.normal(0, 5)
            measurement = np.array([other_x[-1] + noise_x, other_y[-1] + noise_y])
            
            # Update filter
            kf.update(measurement)
            
            # Get uncertainty ellipse
            a, b, angle = kf.get_uncertainty_ellipse()
            uncertainty_ellipses.append((other_x[-1], other_y[-1], a, b, angle))
        
        # Plot potential collision path
        collision_x = np.linspace(own_x[0], other_x[0], 100)
        collision_y = np.linspace(own_y[0], other_y[0], 100)
        ax2.plot(collision_x, collision_y, 'r:', alpha=0.5, label='Potential Collision Path')
        
        # Plot vessel tracks
        ax2.plot(own_x, own_y, 'b-', label='Own Vessel Track', linewidth=2)
        ax2.plot(other_x, other_y, 'r-', label='Other Vessel Track', linewidth=2)
        
        # Plot uncertainty ellipses at regular intervals
        n_ellipses = 10  # Number of ellipses to show
        step = len(uncertainty_ellipses) // n_ellipses
        for i in range(0, len(uncertainty_ellipses), step):
            x, y, a, b, angle = uncertainty_ellipses[i]
            
            ellipse = plt.matplotlib.patches.Ellipse(
                (x, y), 2*a, 2*b, angle=angle,
                fill=False, color='red', linestyle='--', alpha=0.3
            )
            ax2.add_patch(ellipse)
            
            time = trajectory[i][0].time_to_collision
            ax2.text(x, y+b, f't-{time:.0f}s', 
                    horizontalalignment='center', 
                    verticalalignment='bottom',
                    alpha=0.5)
        
        # Plot vessels
        vessel_length = 20  # meters
        def plot_vessel(x, y, heading, color, alpha=1.0):
            """Plot boat as triangle"""
            boat_length = vessel_length
            boat_width = vessel_length/2
            
            # Calculate triangle points
            dx = boat_length * math.sin(math.radians(heading))
            dy = boat_length * math.cos(math.radians(heading))
            wx = boat_width/2 * math.sin(math.radians(heading + 90))
            wy = boat_width/2 * math.cos(math.radians(heading + 90))
            
            triangle_x = [x - wx, x + dx, x + wx]
            triangle_y = [y - wy, y + dy, y + wy]
            
            ax2.fill(triangle_x, triangle_y, color=color, alpha=alpha)
        
        # Plot initial and final positions
        plot_vessel(own_x[0], own_y[0], own_heading, 'blue')
        plot_vessel(other_x[0], other_y[0], other_heading, 'red')
        plot_vessel(own_x[-1], own_y[-1], own_heading, 'blue', alpha=0.3)
        plot_vessel(other_x[-1], other_y[-1], other_heading, 'red', alpha=0.3)
        
        # Plot wind direction
        wind_arrow_length = vessel_length * 2
        wind_base_x = max(max(own_x), max(other_x)) * 0.8
        wind_base_y = max(max(own_y), max(other_y)) * 0.8
        
        wind_dx = -wind_arrow_length * math.sin(math.radians(wind_direction))
        wind_dy = -wind_arrow_length * math.cos(math.radians(wind_direction))
        
        ax2.arrow(wind_base_x, wind_base_y, 
                wind_dx, wind_dy,
                head_width=vessel_length/2, 
                head_length=vessel_length,
                fc='gray', ec='gray',
                alpha=0.7,
                label='Wind Direction')
        
        # Mark tacking points
        for i, (x, y, action) in enumerate(zip(own_x, own_y, actions)):
            if action == 1:  # tack
                ax2.plot(x, y, 'r*', markersize=10, label='Tack Point' if i == 0 else "")
        
        # Final plot settings
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Distance (m)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_aspect('equal')
        
        # Set reasonable axis limits
        max_range = max(
            max(abs(min(own_x)), abs(max(own_x)), abs(min(own_y)), abs(max(own_y))),
            max(abs(min(other_x)), abs(max(other_x)), abs(min(other_y)), abs(max(other_y)))
        )
        margin = max_range * 0.2
        ax2.set_xlim(-max_range-margin, max_range+margin)
        ax2.set_ylim(-max_range-margin, max_range+margin)
        
        # Add uncertainty to legend
        ax2.plot([], [], 'r--', alpha=0.3, label='Position Uncertainty (95%)')
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig, (ax1, ax2)

    def test_scenarios():
        """Run and visualize test scenarios with wind"""
        polars = PolarData()
        trainer = CollisionTrainer(polars)
        
        # Train the policy
        print("Training policy...")
        rewards = trainer.train_policy()
        
        # Test scenarios with different wind directions
        scenarios = [
            ("Starboard vs Port (Wind North)", EncounterState(
                relative_distance=400,
                relative_bearing=45,
                own_tack='starboard',
                other_tack='port',
                time_to_collision=200,
                closing_speed=10
            ), 0),  # Wind from north
            
            ("Port vs Starboard (Wind NE)", EncounterState(
                relative_distance=400,
                relative_bearing=-45,
                own_tack='port',
                other_tack='starboard',
                time_to_collision=200,
                closing_speed=10
            ), 45),  # Wind from NE
        ]
        
        for title, initial_state, wind_direction in scenarios:
            trajectory = trainer.simulate_encounter(initial_state)
            fig, _ = trainer.plot_encounter(trajectory, title, wind_direction)
            fig.savefig(f'{title.lower().replace(" ", "_")}.png')
            plt.close(fig)

def test_scenarios():
    """Run and visualize test scenarios"""
    polars = PolarData()
    trainer = CollisionTrainer(polars)
    
    # Train the policy
    print("Training policy...")
    rewards = trainer.train_policy()
    
    # Plot learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve')
    plt.savefig('learning_curve.png')
    plt.close()
    
    # Test scenarios
    scenarios = [
        ("Starboard vs Port", EncounterState(
            relative_distance=400,
            relative_bearing=45,
            own_tack='starboard',
            other_tack='port',
            time_to_collision=200,
            closing_speed=10
        )),
        ("Port vs Starboard", EncounterState(
            relative_distance=400,
            relative_bearing=-45,
            own_tack='port',
            other_tack='starboard',
            time_to_collision=200,
            closing_speed=10
        )),
    ]
    
    for title, initial_state in scenarios:
        trajectory = trainer.simulate_encounter(initial_state)
        fig, _ = trainer.plot_encounter(trajectory, title)
        fig.savefig(f'{title.lower().replace(" ", "_")}.png')
        plt.close(fig)

if __name__ == "__main__":
    test_scenarios()