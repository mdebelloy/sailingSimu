import numpy as np
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Tuple
import random

class Action(Enum):
    CONTINUE = 0
    TACK = 1

@dataclass
class BoatState:
    x: float  # meters
    y: float  # meters
    heading: float  # radians
    speed: float  # meters/second

@dataclass
class State:
    time_to_collision: float
    collision_prob: float
    has_priority: bool

class ParticleFilter:
    def __init__(self, num_particles: int = 100):
        self.num_particles = num_particles
        self.particles: List[Tuple[BoatState, float]] = []  # (state, weight) pairs
        
    def initialize(self, initial_state: BoatState, position_std: float = 10.0, heading_std: float = 0.1):
        """Initialize particles around an initial state estimate"""
        self.particles = []
        for _ in range(self.num_particles):
            state = BoatState(
                x=initial_state.x + np.random.normal(0, position_std),
                y=initial_state.y + np.random.normal(0, position_std),
                heading=initial_state.heading + np.random.normal(0, heading_std),
                speed=initial_state.speed * (1 + np.random.normal(0, 0.1))
            )
            self.particles.append((state, 1.0 / self.num_particles))
            
    def predict(self, dt: float, turn_rate_std: float = 0.1, speed_var_std: float = 0.2):
        """Move particles forward in time"""
        new_particles = []
        for state, weight in self.particles:
            # Add noise to heading and speed
            heading = state.heading + np.random.normal(0, turn_rate_std)
            speed = state.speed * (1 + np.random.normal(0, speed_var_std))
            
            # Move boat forward
            x = state.x + speed * dt * np.cos(heading)
            y = state.y + speed * dt * np.sin(heading)
            
            new_state = BoatState(x=x, y=y, heading=heading, speed=speed)
            new_particles.append((new_state, weight))
            
        self.particles = new_particles
        
    def update(self, measurement: Tuple[float, float], measurement_std: float):
        """Update weights based on measurement"""
        measured_x, measured_y = measurement
        weights_sum = 0
        
        # Update weights based on measurement likelihood
        new_particles = []
        for state, weight in self.particles:
            # Compute likelihood of measurement given particle state
            dist = np.sqrt((state.x - measured_x)**2 + (state.y - measured_y)**2)
            likelihood = np.exp(-dist**2 / (2 * measurement_std**2))
            new_weight = weight * likelihood
            weights_sum += new_weight
            new_particles.append((state, new_weight))
            
        # Normalize weights
        if weights_sum > 0:
            self.particles = [(state, w/weights_sum) for state, w in new_particles]
        
        # Resample if effective particle count is too low
        self._resample_if_needed()
        
    def _resample_if_needed(self, threshold: float = 0.5):
        """Resample particles if effective sample size is too low"""
        weights = [w for _, w in self.particles]
        eff_particles = 1.0 / sum(w**2 for w in weights)
        
        if eff_particles < threshold * self.num_particles:
            cumsum = np.cumsum(weights)
            new_particles = []
            
            for _ in range(self.num_particles):
                u = random.random()
                idx = np.searchsorted(cumsum, u)
                state = BoatState(
                    x=self.particles[idx][0].x,
                    y=self.particles[idx][0].y,
                    heading=self.particles[idx][0].heading,
                    speed=self.particles[idx][0].speed
                )
                new_particles.append((state, 1.0/self.num_particles))
                
            self.particles = new_particles
            
    def estimate_state(self) -> BoatState:
        """Get weighted average state estimate"""
        total_weight = sum(w for _, w in self.particles)
        if total_weight == 0:
            return self.particles[0][0]  # Return any particle if weights are zero
            
        x = sum(s.x * w for s, w in self.particles) / total_weight
        y = sum(s.y * w for s, w in self.particles) / total_weight
        
        # For heading, we need to handle circular average correctly
        sin_sum = sum(np.sin(s.heading) * w for s, w in self.particles)
        cos_sum = sum(np.cos(s.heading) * w for s, w in self.particles)
        heading = np.arctan2(sin_sum, cos_sum)
        
        speed = sum(s.speed * w for s, w in self.particles) / total_weight
        
        return BoatState(x=x, y=y, heading=heading, speed=speed)




    
class SailingCollisionAvoidance:
    def __init__(self):
        # Physical parameters
        self.boat_speed = 5.0  # m/s
        self.tack_time = 3.0   # seconds
        self.safety_distance = 10.0  # meters - distance for collision detection
        
        # State estimation
        self.particle_filter = ParticleFilter(num_particles=200)
        self.measurement_std = 5.0  # meters
        
        # State space discretization
        self.ttc_points = np.linspace(0, 60, 40)
        self.prob_points = np.linspace(0, 1, 30)
        self.priorities = [True, False]
        
        # Initialize value and policy arrays
        shape = (len(self.ttc_points), len(self.prob_points), len(self.priorities))
        self.values = np.zeros(shape)
        self.policy = np.zeros(shape, dtype=int)
        
        # Reward structure
        self.collision_penalty = -100.0
        self.tack_cost = -1.0
        self.successful_passage_reward = 5.0
        
    def get_reward(self, state: State, action: Action, next_state: State) -> float:
        """Calculate reward for a state transition with balanced risk aversion"""
        reward = 0.0
        
        # Immediate collision state
        if state.time_to_collision <= 0:
            return self.collision_penalty * state.collision_prob
            
        # Only care about probabilities above a threshold
        if state.collision_prob > 0.2:  # Ignore very low probabilities
            # Exponentially increasing penalty as collision gets closer
            time_urgency = np.exp(-max(0, state.time_to_collision - 5) / 5.0)  # Peaks in last 5 seconds
            collision_risk = (state.collision_prob - 0.2) * time_urgency  # Shift probability impact
            reward += self.collision_penalty * collision_risk
        
        # Basic cost for tacking
        if action == Action.TACK:
            reward += self.tack_cost
        
        # Reward for reducing collision probability (only for significant probabilities)
        if next_state.collision_prob < state.collision_prob and state.collision_prob > 0.2:
            reward += self.successful_passage_reward * (state.collision_prob - next_state.collision_prob)
                
        return reward

    def _get_transitions(self, state: State, action: Action) -> List[Tuple[float, State, float]]:
        """Get possible transitions with improved priority-dependent behavior"""
        transitions = []
        
        if state.time_to_collision <= 0:
            return [(1.0, state, self.get_reward(state, action, state))]
            
        if action == Action.CONTINUE:
            next_ttc = max(0, state.time_to_collision - 1)
            base_prob = state.collision_prob
            
            if state.has_priority:
                # We have priority - other boat less likely to move
                transitions.extend([
                    (0.2, State(next_ttc, base_prob * 0.7, state.has_priority), 0),  # They give way somewhat
                    (0.3, State(next_ttc, min(1, base_prob * 1.2), state.has_priority), 0),  # Situation worsens
                    (0.5, State(next_ttc, base_prob, state.has_priority), 0)  # No change
                ])
            else:
                # We don't have priority - other boat very likely to take action
                transitions.extend([
                    (0.8, State(next_ttc, base_prob * 0.3, state.has_priority), 0),  # They take strong avoiding action
                    (0.1, State(next_ttc, min(1, base_prob * 1.1), state.has_priority), 0),  # Slight worsening
                    (0.1, State(next_ttc, base_prob, state.has_priority), 0)  # No change
                ])
        else:  # TACK
            # More conservative probability reduction from tacking
            next_ttc = state.time_to_collision + self.tack_time
            transitions.extend([
                (0.8, State(next_ttc, max(0, state.collision_prob * 0.4), state.has_priority), 0),  # Successful tack
                (0.2, State(next_ttc, max(0, state.collision_prob * 0.7), state.has_priority), 0)  # Less effective tack
            ])
            
        return [(p, s, self.get_reward(state, action, s)) for p, s, _ in transitions]

    def optimize_policy(self, num_iterations: int = 200):
        """Value iteration to find optimal policy"""
        gamma = 0.95
        
        for iteration in tqdm(range(num_iterations)):
            delta = 0
            new_values = np.zeros_like(self.values)
            
            for i, ttc in enumerate(self.ttc_points):
                for j, prob in enumerate(self.prob_points):
                    for k, priority in enumerate(self.priorities):
                        state = State(ttc, prob, priority)
                        values = []
                        
                        for action in Action:
                            value = self._compute_action_value(state, action, gamma)
                            values.append(value)
                            
                        best_value = max(values)
                        best_action = values.index(best_value)
                        
                        new_values[i, j, k] = best_value
                        self.policy[i, j, k] = best_action
                        delta = max(delta, abs(best_value - self.values[i, j, k]))
            
            self.values = new_values
            
            if delta < 0.01:
                print(f"Converged after {iteration} iterations")
                break

    def estimate_collision_probability(self, own_state: BoatState) -> Tuple[float, float]:
        """Estimate collision probability and time using particle distribution"""
        other_state = self.particle_filter.estimate_state()
        
        # Compute relative velocity components
        rel_vx = own_state.speed * np.cos(own_state.heading) - other_state.speed * np.cos(other_state.heading)
        rel_vy = own_state.speed * np.sin(own_state.heading) - other_state.speed * np.sin(other_state.heading)
        rel_v = np.sqrt(rel_vx**2 + rel_vy**2)
        
        if rel_v < 0.1:  # Practically zero relative velocity
            return 0.0, float('inf')
            
        # Compute closest point of approach
        dx = other_state.x - own_state.x
        dy = other_state.y - own_state.y
        
        # Time to closest point of approach
        t_cpa = -(dx*rel_vx + dy*rel_vy) / (rel_v**2)
        
        if t_cpa < 0:  # Boats are moving apart
            return 0.0, float('inf')
            
        # Distance at closest point of approach
        x_cpa = dx + rel_vx * t_cpa
        y_cpa = dy + rel_vy * t_cpa
        d_cpa = np.sqrt(x_cpa**2 + y_cpa**2)
        
        # Estimate collision probability based on particle distribution
        collision_count = 0
        for particle, weight in self.particle_filter.particles:
            dx = particle.x - own_state.x
            dy = particle.y - own_state.y
            rel_vx = own_state.speed * np.cos(own_state.heading) - particle.speed * np.cos(particle.heading)
            rel_vy = own_state.speed * np.sin(own_state.heading) - particle.speed * np.sin(particle.heading)
            
            if np.sqrt(rel_vx**2 + rel_vy**2) > 0.1:
                t = -(dx*rel_vx + dy*rel_vy) / (rel_vx**2 + rel_vy**2)
                if t > 0:
                    x = dx + rel_vx * t
                    y = dy + rel_vy * t
                    if np.sqrt(x**2 + y**2) < self.safety_distance:
                        collision_count += weight
                        
        collision_prob = collision_count
        
        return collision_prob, t_cpa
        
    def update_state_estimate(self, measurement: Tuple[float, float], dt: float):
        """Update state estimate with new measurement"""
        self.particle_filter.predict(dt)
        self.particle_filter.update(measurement, self.measurement_std)
        

    
                
    def _compute_action_value(self, state: State, action: Action, gamma: float) -> float:
        """Compute value for a state-action pair"""
        transitions = self._get_transitions(state, action)
        value = 0.0
        
        for prob, next_state, reward in transitions:
            next_i = np.searchsorted(self.ttc_points, next_state.time_to_collision)
            next_j = np.searchsorted(self.prob_points, next_state.collision_prob)
            next_k = int(next_state.has_priority)
            
            next_i = min(next_i, len(self.ttc_points)-1)
            next_j = min(next_j, len(self.prob_points)-1)
            
            value += prob * (reward + gamma * self.values[next_i, next_j, next_k])
            
        return value
        

    def get_action(self, state: State) -> Action:
        """Get optimal action for given state"""
        i = np.searchsorted(self.ttc_points, state.time_to_collision)
        j = np.searchsorted(self.prob_points, state.collision_prob)
        k = int(state.has_priority)
        
        i = min(i, len(self.ttc_points)-1)
        j = min(j, len(self.prob_points)-1)
        
        return Action(self.policy[i, j, k])
    
    def plot_policy(self, priority: bool = True):
        """Plot the policy with smoothed decision boundary"""
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter

        # Create meshgrid for smooth visualization
        ttc_grid, prob_grid = np.meshgrid(self.ttc_points, self.prob_points)
        
        # Get policy for the specified priority
        k = int(priority)
        policy_grid = self.policy[:, :, k].T
        
        # Apply Gaussian smoothing to the policy
        smoothed_policy = gaussian_filter(policy_grid.astype(float), sigma=1.0)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot smoothed contour
        contour = plt.contourf(ttc_grid, prob_grid, smoothed_policy, 
                            levels=np.linspace(-0.5, 1.5, 20),
                            cmap='RdYlBu')
        
        # Add contour lines for decision boundary
        decision_boundary = plt.contour(ttc_grid, prob_grid, smoothed_policy,
                                    levels=[0.5],
                                    colors='black',
                                    linewidths=2,
                                    linestyles='dashed')
        
        # Customize plot
        plt.xlabel('Time to Collision (seconds)')
        plt.ylabel('Collision Probability')
        plt.title(f'Collision Avoidance Policy ({"With" if priority else "Without"} Priority)')
        
        # Add colorbar with custom labels
        cbar = plt.colorbar(contour)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Continue', 'Tack'])
        
        # Add annotations
        plt.text(0.02, 0.98, 
                'Decision Boundary',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        return plt

    def plot_policy_comparison(self):
        """Plot decision boundaries with shaded regions for both priority cases"""
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.interpolate import UnivariateSpline

        def smooth_boundary(ttc, prob):
            """Helper function to smooth boundary points with appropriate constraints"""
            # Add constraint points
            ttc_ext = list(ttc)
            prob_ext = list(prob)
            
            # If needed, add point at t=0 with appropriate probability
            if min(ttc) > 0:
                ttc_ext = [0] + ttc_ext
                # Use linear extrapolation for the p value at t=0
                if len(ttc) >= 2:
                    p0 = prob[0] + (prob[0] - prob[1]) * (ttc[0]) / (ttc[1] - ttc[0])
                    p0 = min(max(p0, 0), 1)  # Clip to [0,1]
                    prob_ext = [p0] + prob_ext
                else:
                    prob_ext = [prob[0]] + prob_ext
            
            # Create smooth spline with appropriate smoothing
            spl = UnivariateSpline(ttc_ext, prob_ext, k=3, s=0.05)
            return spl

        plt.figure(figsize=(12, 8))
        
        # Store curves for later shading
        curves = []
        
        for priority, (color, label) in enumerate([('blue', 'Without Priority'), ('red', 'With Priority')]):
            # Get policy
            policy_grid = self.policy[:, :, priority].T
            
            # Find boundary points
            boundary_points = []
            for i, ttc in enumerate(self.ttc_points):
                prob_indices = np.where(np.diff(policy_grid[:, i]))[0]
                if len(prob_indices) > 0:
                    boundary_points.append((ttc, self.prob_points[prob_indices[0]]))
            
            if boundary_points:
                # Sort points by ttc
                boundary_points.sort(key=lambda x: x[0])
                ttc_boundary, prob_boundary = zip(*boundary_points)
                
                # Plot raw points
                plt.plot(ttc_boundary, prob_boundary, 'o', color=color, alpha=0.2, markersize=4)
                
                if len(ttc_boundary) > 3:
                    # Create smooth curve
                    spl = smooth_boundary(ttc_boundary, prob_boundary)
                    ttc_smooth = np.linspace(0, 45, 200)
                    prob_smooth = np.clip(spl(ttc_smooth), 0, 1)
                    curves.append((ttc_smooth, prob_smooth, color))
                    
                    # Plot smooth boundary
                    plt.plot(ttc_smooth, prob_smooth, color=color, linewidth=2, label=label)

        # Add shading
        if len(curves) == 2:
            ttc_smooth = curves[0][0]  # Same for both curves
            
            # Fill above curves
            plt.fill_between(ttc_smooth, curves[0][1], np.ones_like(ttc_smooth), 
                            color='blue', alpha=0.1)
            plt.fill_between(ttc_smooth, curves[1][1], np.ones_like(ttc_smooth), 
                            color='red', alpha=0.1)
            
            # Find and fill overlap region
            plt.fill_between(ttc_smooth, 
                            np.maximum(curves[0][1], curves[1][1]), 
                            np.ones_like(ttc_smooth),
                            color='purple', alpha=0.1)
        
        plt.xlabel('Time to Collision (seconds)')
        plt.ylabel('Collision Probability')
        plt.title('Collision Avoidance Policy Decision Boundaries')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.xlim(0, 35)
        plt.ylim(0, 1)
        
        plt.text(0.02, 0.98, 
                'Tack (above)\nContinue (below)',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                verticalalignment='top')
        
        return plt


    def plot_state_estimation(self, own_state: BoatState):
        """Visualize current state estimation with improved uncertainty representation"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Ellipse
        import numpy as np
        from scipy.stats import gaussian_kde
        
        plt.figure(figsize=(8, 8))
        
        # Plot particles with density estimation
        particle_xs = np.array([p[0].x for p in self.particle_filter.particles])
        particle_ys = np.array([p[0].y for p in self.particle_filter.particles])
        weights = np.array([p[1] for p in self.particle_filter.particles])
        
        # Calculate weighted covariance for uncertainty ellipse
        mean_x = np.average(particle_xs, weights=weights)
        mean_y = np.average(particle_ys, weights=weights)
        cov_xx = np.average((particle_xs - mean_x)**2, weights=weights)
        cov_yy = np.average((particle_ys - mean_y)**2, weights=weights)
        cov_xy = np.average((particle_xs - mean_x)*(particle_ys - mean_y), weights=weights)
        
        # Calculate ellipse parameters
        covariance = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
        eigenvals, eigenvecs = np.linalg.eig(covariance)
        angle = np.degrees(np.arctan2(eigenvecs[1,0], eigenvecs[0,0]))
        
        # Plot 95% confidence ellipse
        uncertainty_ellipse = Ellipse(xy=(mean_x, mean_y),
                                    width=2*np.sqrt(5.991*eigenvals[0]),  # 95% confidence
                                    height=2*np.sqrt(5.991*eigenvals[1]),
                                    angle=angle,
                                    fill=True,
                                    color='red',
                                    alpha=0.1,
                                    label='95% Confidence Region')
        plt.gca().add_patch(uncertainty_ellipse)
        
        # Plot particles with improved visibility
        plt.scatter(particle_xs, particle_ys, 
                c='gray', s=30*weights/max(weights), 
                alpha=0.4, label='Particles')
        
        # Estimated other boat (red)
        estimated_state = self.particle_filter.estimate_state()
        
        # Draw dotted line between boats FIRST (so it appears behind other elements)
        plt.plot([own_state.x, estimated_state.x], 
                [own_state.y, estimated_state.y], 
                'k:', linewidth=1.5, alpha=0.6, 
                zorder=1,  # Ensure it's drawn behind boats
                label='Current Separation')
        
        # Plot boats as smaller circles with direction indicators
        boat_radius = 1.5
        arrow_length = 7
        
        # Own boat (blue)
        own_circle = Circle((own_state.x, own_state.y), boat_radius, 
                        color='blue', alpha=0.7, label='Own Boat')
        plt.gca().add_patch(own_circle)
        
        # Own boat direction arrow
        plt.arrow(own_state.x, own_state.y,
                arrow_length * np.cos(own_state.heading),
                arrow_length * np.sin(own_state.heading),
                head_width=1.5, head_length=1.5, fc='blue', ec='blue')
        
        # Estimated other boat (red)
        other_circle = Circle((estimated_state.x, estimated_state.y), boat_radius, 
                            color='red', alpha=0.7, label='Estimated Other Boat')
        plt.gca().add_patch(other_circle)
        
        # Estimated boat direction arrow
        plt.arrow(estimated_state.x, estimated_state.y,
                arrow_length * np.cos(estimated_state.heading),
                arrow_length * np.sin(estimated_state.heading),
                head_width=1.5, head_length=1.5, fc='red', ec='red')
        
        # Calculate and plot potential collision point
        rel_vx = own_state.speed * np.cos(own_state.heading) - estimated_state.speed * np.cos(estimated_state.heading)
        rel_vy = own_state.speed * np.sin(own_state.heading) - estimated_state.speed * np.sin(estimated_state.heading)
        rel_v = np.sqrt(rel_vx**2 + rel_vy**2)
        
        if rel_v > 0.1:  # Only if there's relative motion
            dx = estimated_state.x - own_state.x
            dy = estimated_state.y - own_state.y
            t_cpa = -(dx*rel_vx + dy*rel_vy) / (rel_v**2)
            
            if t_cpa > 0:  # Only if collision is in the future
                # Calculate collision point
                cpa_x = own_state.x + own_state.speed * np.cos(own_state.heading) * t_cpa
                cpa_y = own_state.y + own_state.speed * np.sin(own_state.heading) * t_cpa
                
                # Plot paths to collision point
                plt.plot([own_state.x, cpa_x], [own_state.y, cpa_y], 
                        'b--', alpha=0.5, label='Own Path')
                plt.plot([estimated_state.x, cpa_x], [estimated_state.y, cpa_y], 
                        'r--', alpha=0.5, label='Estimated Path')
                
                # Mark collision point
                plt.plot(cpa_x, cpa_y, 'kx', markersize=8, label='Potential Collision')
        
        # Customize plot
        plt.axis('equal')
        plt.grid(True, linestyle=':')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('State Estimation with Uncertainty')
        
        # Add time to collision annotation if applicable
        if rel_v > 0.1 and t_cpa > 0:
            plt.text(0.02, 0.98, 
                    f'Time to CPA: {t_cpa:.1f}s\nCPA Distance: {np.sqrt((cpa_x-estimated_state.x)**2 + (cpa_y-estimated_state.y)**2):.1f}m',
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                    verticalalignment='top')
        
        # Adjust layout to minimize white space
        plt.tight_layout()
        
        # Adjust margins to reduce white space
        plt.subplots_adjust(right=0.85)
        
        return plt



def test_collision_avoidance():
    """Test the collision avoidance system with state estimation"""
    # Initialize system
    system = SailingCollisionAvoidance()
    
    # Initialize own state
    own_state = BoatState(x=0, y=0, heading=np.pi/4, speed=5.0)
    
    # Initialize other boat state for particle filter
    other_state = BoatState(x=50, y=50, heading=-3*np.pi/4, speed=5.0)
    system.particle_filter.initialize(other_state)
    
    # Optimize policy
    system.optimize_policy(num_iterations=200)
    
    # Plot initial policy
    print("Initial learned policy:")
    system.plot_policy_comparison()
    plt.show()
    
    # Simulate a few steps
    print("\nSimulating collision scenario...")
    for t in range(5):
        # Update state estimate with noisy measurement
        true_x = other_state.x + other_state.speed * np.cos(other_state.heading) * t
        true_y = other_state.y + other_state.speed * np.sin(other_state.heading) * t
        measured_x = true_x + np.random.normal(0, 5.0)
        measured_y = true_y + np.random.normal(0, 5.0)
        
        system.update_state_estimate((measured_x, measured_y), dt=1.0)
        
        # Estimate collision probability and time
        collision_prob, ttc = system.estimate_collision_probability(own_state)
        
        # Get recommended action
        state = State(ttc, collision_prob, has_priority=True)
        action = system.get_action(state)
        
        print(f"\nTime step {t}:")
        print(f"Estimated collision probability: {collision_prob:.3f}")
        print(f"Time to collision: {ttc:.1f}s")
        print(f"Recommended action: {action}")
        
        # Visualize current state
        system.plot_state_estimation(own_state)
        plt.show()
        
        
        # Update own state based on action
        if action == Action.TACK:
            own_state.heading += np.pi/2  # 90-degree tack
            
        # Move own boat forward
        own_state.x += own_state.speed * np.cos(own_state.heading)
        own_state.y += own_state.speed * np.sin(own_state.heading)


def monte_carlo_simulation(num_trials=10000):
    """Run Monte Carlo simulation to evaluate collision avoidance policy"""
    print(f"\nRunning {num_trials} Monte Carlo trials...")
    
    # Initialize system
    system = SailingCollisionAvoidance()
    
    # Optimize policy first
    system.optimize_policy(num_iterations=200)
    
    # Statistics tracking
    collisions = 0
    close_calls = 0  # Track near misses (within 5m)
    tacks_made = 0
    simulation_times = []
    
    # Run trials with progress bar
    for trial in tqdm(range(num_trials)):
        # Randomize initial conditions for each trial
        own_state = BoatState(
            x=0, 
            y=0,
            heading=np.random.uniform(0, 2*np.pi),
            speed=5.0
        )
        
        # Initialize other boat with random approach angle
        approach_angle = np.random.uniform(0, 2*np.pi)
        initial_distance = 100  # meters
        other_state = BoatState(
            x=initial_distance * np.cos(approach_angle),
            y=initial_distance * np.sin(approach_angle),
            heading=approach_angle + np.pi + np.random.normal(0, 0.2),  # Roughly heading towards own boat
            speed=5.0
        )
        
        system.particle_filter.initialize(other_state)
        
        # Simulation parameters
        max_time = 60  # seconds
        dt = 1.0
        collision_occurred = False
        min_distance = float('inf')
        trial_tacks = 0
        
        # Run single trial
        for t in range(int(max_time/dt)):
            # Update state estimates
            true_x = other_state.x + other_state.speed * np.cos(other_state.heading) * dt
            true_y = other_state.y + other_state.speed * np.sin(other_state.heading) * dt
            measured_x = true_x + np.random.normal(0, 2.0)
            measured_y = true_y + np.random.normal(0, 2.0)
            
            system.update_state_estimate((measured_x, measured_y), dt)
            
            # Get current situation assessment
            collision_prob, ttc = system.estimate_collision_probability(own_state)
            
            # Check current distance
            dx = other_state.x - own_state.x
            dy = other_state.y - own_state.y
            current_distance = np.sqrt(dx**2 + dy**2)
            min_distance = min(min_distance, current_distance)
            
            # Check for collision
            if current_distance < 3.0:  # 3m collision threshold
                collision_occurred = True
                break
                
            # Get and apply action
            state = State(ttc, collision_prob, has_priority=True)
            action = system.get_action(state)
            
            if action == Action.TACK:
                own_state.heading += np.pi/2
                trial_tacks += 1
            
            # Move boats
            own_state.x += own_state.speed * np.cos(own_state.heading) * dt
            own_state.y += own_state.speed * np.sin(own_state.heading) * dt
            
            other_state.x = true_x
            other_state.y = true_y
            
            # Check if boats are far apart (early termination)
            if current_distance > 200:  # meters
                break
        
        # Update statistics
        if collision_occurred:
            collisions += 1
        if min_distance < 5.0:  # Close call threshold
            close_calls += 1
        tacks_made += trial_tacks
        simulation_times.append(t * dt)
    
    # Compute statistics
    collision_rate = (collisions / num_trials) * 100
    close_call_rate = (close_calls / num_trials) * 100
    avg_tacks = tacks_made / num_trials
    avg_time = np.mean(simulation_times)
    
    print("\nSimulation Results:")
    print(f"Collision Rate: {collision_rate:.2f}%")
    print(f"Close Call Rate: {close_call_rate:.2f}%")
    print(f"Average Tacks per Trial: {avg_tacks:.2f}")
    print(f"Average Scenario Duration: {avg_time:.1f} seconds")
    
    return collision_rate, close_call_rate, avg_tacks, avg_time


if __name__ == '__main__':
    test_collision_avoidance()
    monte_carlo_simulation(1000)