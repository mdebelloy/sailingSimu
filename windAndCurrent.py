import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

#########################
# Simplified Polar Model
#########################

class SimplePolarModel:
    """A simplified boat performance model"""
    def __init__(self):
        # Create a simplified performance model using sin function
        # Max speed at 90° to the wind, min speed at 0° and 180°
        self.max_speed = 10.0  # Maximum boat speed in knots
        self.min_upwind_angle = 30  # Minimum angle to the wind
    
    def get_boat_speed(self, twa, wind_speed):
        """Get boat speed based on true wind angle and speed
        
        Args:
            twa: True wind angle in degrees (0-180)
            wind_speed: Wind speed in knots
        
        Returns:
            Boat speed in knots
        """
        # Normalize angle to 0-180 range
        twa = abs(twa) % 360
        if twa > 180:
            twa = 360 - twa
        
        # Can't sail too close to the wind
        if twa < self.min_upwind_angle:
            return 0.0
        
        # Simple sinusoidal model with max at 90-110°
        # Boat speed is proportional to wind speed with a cap
        normalized_angle = np.radians(twa)
        speed_factor = 0.6 + 0.4 * np.sin(normalized_angle * 0.9)
        
        # Apply wind speed factor (capped at certain value)
        wind_factor = min(wind_speed / 15.0, 1.5)
        
        return self.max_speed * speed_factor * wind_factor

#########################
# Fixed Wind Model
#########################

class WindGrid:
    """A grid-based representation of wind conditions with spatial variation but fixed in time"""
    def __init__(self, grid_size_x, grid_size_y, base_wind_speed=15.0, base_wind_dir=0.0):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.base_wind_speed = base_wind_speed
        self.base_wind_dir = base_wind_dir
        
        # Create grids for wind speed and direction
        self.wind_speed = np.full((grid_size_y, grid_size_x), base_wind_speed)
        self.wind_dir = np.full((grid_size_y, grid_size_x), base_wind_dir)
        
        # Add variation to both speed and direction
        self._add_wind_variation()
    
    def _add_wind_variation(self, num_patches=5, max_speed_var=0.3, max_dir_var=30):
        """Add random variations to wind field"""
        # Create speed variations
        for _ in range(num_patches):
            # Random center point
            center_x = random.randint(0, self.grid_size_x - 1)
            center_y = random.randint(0, self.grid_size_y - 1)
            
            # Random radius and intensity
            radius = random.uniform(self.grid_size_x / 8, self.grid_size_x / 3)
            intensity = random.uniform(-max_speed_var, max_speed_var) * self.base_wind_speed
            
            # Apply gaussian-like patch
            for y in range(self.grid_size_y):
                for x in range(self.grid_size_x):
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if distance < 2 * radius:
                        factor = np.exp(-(distance/radius)**2)
                        self.wind_speed[y, x] += intensity * factor
        
        # Create direction variations (similar patches)
        for _ in range(num_patches // 2):
            center_x = random.randint(0, self.grid_size_x - 1)
            center_y = random.randint(0, self.grid_size_y - 1)
            
            radius = random.uniform(self.grid_size_x / 6, self.grid_size_x / 2)
            angle_shift = random.uniform(-max_dir_var, max_dir_var)
            
            for y in range(self.grid_size_y):
                for x in range(self.grid_size_x):
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if distance < 2 * radius:
                        factor = np.exp(-(distance/radius)**2)
                        self.wind_dir[y, x] += angle_shift * factor
        
        # Ensure minimum wind speed
        self.wind_speed = np.maximum(self.wind_speed, self.base_wind_speed * 0.5)
        # Normalize wind directions to 0-360
        self.wind_dir = self.wind_dir % 360
    
    def get_wind_at_position(self, x, y):
        """Get wind speed and direction at a position
        
        Args:
            x: X position in grid coordinates (0 to grid_size_x-1)
            y: Y position in grid coordinates (0 to grid_size_y-1)
            
        Returns:
            Tuple of (wind_speed, wind_direction)
        """
        # Convert to grid indices with bounds checking
        grid_x = max(0, min(int(x), self.grid_size_x - 1))
        grid_y = max(0, min(int(y), self.grid_size_y - 1))
        
        return self.wind_speed[grid_y, grid_x], self.wind_dir[grid_y, grid_x]

#########################
# Current Model
#########################

class CurrentGrid:
    """An Eulerian Flow Model for underwater currents"""
    def __init__(self, grid_size_x, grid_size_y, base_current_speed=1.0, base_current_dir=45.0):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.base_current_speed = base_current_speed  # in knots
        self.base_current_dir = base_current_dir      # in degrees (0-360)
        
        # Create grids for current speed and direction
        self.current_speed = np.full((grid_size_y, grid_size_x), base_current_speed)
        self.current_dir = np.full((grid_size_y, grid_size_x), base_current_dir)
        
        # Add spatial variation to the current field
        self._generate_current_field()
    
    def _generate_current_field(self, num_features=3, max_speed_var=0.5, max_dir_var=30):
        """Generate a spatially varying but temporally fixed current field using flow features"""
        # Create primary circulation features (eddies, streams, etc.)
        for _ in range(num_features):
            # Random center point for the feature
            center_x = random.randint(0, self.grid_size_x - 1)
            center_y = random.randint(0, self.grid_size_y - 1)
            
            # Feature parameters
            feature_type = random.choice(['eddy', 'stream', 'gradient'])
            feature_radius = random.uniform(self.grid_size_x / 10, self.grid_size_x / 3)
            
            # Apply the feature to the current field
            for y in range(self.grid_size_y):
                for x in range(self.grid_size_x):
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    
                    if feature_type == 'eddy':
                        # Circular flow feature (eddy)
                        if distance < 2 * feature_radius:
                            # Direction follows circular pattern
                            angle_to_center = np.degrees(np.arctan2(y - center_y, x - center_x))
                            eddy_direction = (angle_to_center + 90) % 360  # Perpendicular to radial
                            
                            # Speed peaks at certain radius
                            speed_factor = (distance / feature_radius) * np.exp(1 - distance / feature_radius)
                            speed_change = speed_factor * max_speed_var * self.base_current_speed
                            
                            # Blend with existing field
                            blend_factor = np.exp(-0.5 * (distance / feature_radius)**2)
                            self.current_dir[y, x] = (
                                (1 - blend_factor) * self.current_dir[y, x] + 
                                blend_factor * eddy_direction
                            )
                            self.current_speed[y, x] += speed_change * blend_factor
                    
                    elif feature_type == 'stream':
                        # Linear flow feature (stream/jet)
                        stream_dir = random.uniform(0, 360)
                        stream_width = feature_radius / 2
                        
                        # Calculate perpendicular distance to stream centerline
                        angle_rad = np.radians(stream_dir)
                        stream_vec = np.array([np.cos(angle_rad), np.sin(angle_rad)])
                        point_vec = np.array([x - center_x, y - center_y])
                        
                        # Project point onto stream direction to get perpendicular distance
                        proj = np.dot(point_vec, stream_vec)
                        closest_point = center_x + proj * stream_vec[0], center_y + proj * stream_vec[1]
                        perp_dist = np.sqrt(
                            (x - closest_point[0])**2 + (y - closest_point[1])**2
                        )
                        
                        # Apply stream effect if within width
                        if perp_dist < stream_width and 0 <= proj <= feature_radius * 2:
                            # Gaussian profile across stream width
                            intensity = np.exp(-0.5 * (perp_dist / (stream_width/2))**2)
                            speed_change = max_speed_var * self.base_current_speed * intensity
                            
                            # Blend direction with stream direction
                            self.current_dir[y, x] = (
                                (1 - intensity) * self.current_dir[y, x] + 
                                intensity * stream_dir
                            )
                            self.current_speed[y, x] += speed_change
                    
                    elif feature_type == 'gradient':
                        # Gradual change in current across region
                        if distance < 2 * feature_radius:
                            # Linear gradient from center
                            gradient_factor = (distance / (2 * feature_radius))
                            
                            # Apply to speed
                            speed_change = ((2 * random.random() - 1) * 
                                           max_speed_var * self.base_current_speed * 
                                           (1 - gradient_factor))
                            
                            # Apply to direction
                            dir_change = ((2 * random.random() - 1) * 
                                         max_dir_var * (1 - gradient_factor))
                            
                            self.current_speed[y, x] += speed_change
                            self.current_dir[y, x] = (self.current_dir[y, x] + dir_change) % 360
        
        # Ensure minimum current speed and normalize directions
        self.current_speed = np.maximum(self.current_speed, 0.1)
        self.current_dir = self.current_dir % 360
    
    def get_current_at_position(self, x, y):
        """Get current speed and direction at a position
        
        Args:
            x: X position in grid coordinates (0 to grid_size_x-1)
            y: Y position in grid coordinates (0 to grid_size_y-1)
            
        Returns:
            Tuple of (current_speed, current_direction)
        """
        # Convert to grid indices with bounds checking
        grid_x = max(0, min(int(x), self.grid_size_x - 1))
        grid_y = max(0, min(int(y), self.grid_size_y - 1))
        
        return self.current_speed[grid_y, grid_x], self.current_dir[grid_y, grid_x]
    
    def get_current_vector_at_position(self, x, y):
        """Get current as a vector (x, y components) at a position
        
        Args:
            x: X position in grid coordinates (0 to grid_size_x-1)
            y: Y position in grid coordinates (0 to grid_size_y-1)
            
        Returns:
            Tuple of (current_x, current_y) vector components
        """
        current_speed, current_dir = self.get_current_at_position(x, y)
        
        # Convert direction to radians (0 degrees is North/+y, 90 is East/+x)
        current_rad = np.radians(current_dir)
        
        # Calculate vector components
        current_x = current_speed * np.sin(current_rad)
        current_y = current_speed * np.cos(current_rad)
        
        return current_x, current_y

#########################
# Sailing Environment
#########################

class SailingEnv:
    """A simplified grid-based sailing environment with wind and current effects"""
    def __init__(self, grid_size_x, grid_size_y, polar_model, 
                 start_pos=(0, 0), goal_pos=None, 
                 use_currents=True, base_current_speed=1.0, base_current_dir=45.0,
                 base_wind_speed=15.0, base_wind_dir=0.0):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.polar_model = polar_model
        
        # Set default goal at opposite corner if not specified
        self.start_pos = start_pos
        self.goal_pos = goal_pos if goal_pos else (grid_size_x - 1, grid_size_y - 1)
        self.goal_radius = max(grid_size_x, grid_size_y) / 20
        
        # Wind settings
        self.base_wind_speed = base_wind_speed
        self.base_wind_dir = base_wind_dir
        
        # Current settings
        self.use_currents = use_currents
        self.base_current_speed = base_current_speed
        self.base_current_dir = base_current_dir
        
        # Current state
        self.position = None
        self.heading = None
        self.wind_grid = None
        self.current_grid = None
        self.steps = 0
        self.max_steps = grid_size_x * grid_size_y * 2  # Reasonable upper bound
        
        # Action space: Change in heading in degrees
        # -45, -30, -15, 0, 15, 30, 45
        self.actions = np.array([-45, -30, -15, 0, 15, 30, 45])
        self.n_actions = len(self.actions)
        
        # Parameters
        self.time_step = 1.0  # seconds
        self.prev_heading = None  # Used to detect tacking/gybing
    
    def reset(self):
        """Reset environment to initial state"""
        # Reset position and heading
        self.position = np.array(self.start_pos, dtype=float)
        self.heading = 0  # Start facing "north" (towards increasing y)
        self.prev_heading = self.heading
        
        # Create fixed wind field
        self.wind_grid = WindGrid(
            self.grid_size_x, 
            self.grid_size_y,
            base_wind_speed=self.base_wind_speed,
            base_wind_dir=self.base_wind_dir
        )
        
        # Create current field (if enabled)
        if self.use_currents:
            self.current_grid = CurrentGrid(
                self.grid_size_x, 
                self.grid_size_y,
                base_current_speed=self.base_current_speed,
                base_current_dir=self.base_current_dir
            )
        else:
            self.current_grid = None
        
        self.steps = 0
        
        return self._get_state()
    
    def _get_state(self):
        """Create state representation"""
        # Get current wind
        wind_speed, wind_dir = self.wind_grid.get_wind_at_position(
            self.position[0], self.position[1])
        
        # Get current (if enabled)
        if self.use_currents and self.current_grid:
            current_speed, current_dir = self.current_grid.get_current_at_position(
                self.position[0], self.position[1])
        else:
            current_speed, current_dir = 0.0, 0.0
        
        # Calculate relative position to goal
        dx = self.goal_pos[0] - self.position[0]
        dy = self.goal_pos[1] - self.position[1]
        
        # Distance and bearing to goal
        distance = np.sqrt(dx**2 + dy**2)
        bearing_to_goal = np.degrees(np.arctan2(dx, dy)) % 360
        
        # Calculate true wind angle
        twa = (self.heading - wind_dir) % 360
        if twa > 180:
            twa = 360 - twa
        
        # Calculate normalized values
        norm_x = self.position[0] / (self.grid_size_x - 1)
        norm_y = self.position[1] / (self.grid_size_y - 1)
        norm_distance = distance / np.sqrt(self.grid_size_x**2 + self.grid_size_y**2)
        
        # Create state vector (extended to include current information)
        state = np.array([
            norm_x,                            # X position (0-1)
            norm_y,                            # Y position (0-1)
            np.sin(np.radians(self.heading)),  # Sin of current heading
            np.cos(np.radians(self.heading)),  # Cos of current heading
            np.sin(np.radians(bearing_to_goal)), # Sin of bearing to goal
            np.cos(np.radians(bearing_to_goal)), # Cos of bearing to goal
            norm_distance,                     # Normalized distance to goal
            wind_speed / 30.0,                 # Normalized wind speed
            np.sin(np.radians(wind_dir)),      # Sin of wind direction
            np.cos(np.radians(wind_dir)),      # Cos of wind direction
            twa / 180.0,                       # Normalized true wind angle
            current_speed / 5.0,               # Normalized current speed
            np.sin(np.radians(current_dir)),   # Sin of current direction
            np.cos(np.radians(current_dir))    # Cos of current direction
        ], dtype=np.float32)
        
        return state
    
    def step(self, action_idx):
        """Take a step with the given action"""
        self.steps += 1
        
        # Get change in heading
        heading_change = self.actions[action_idx]
        
        # Save previous heading to detect tacks/gybes
        self.prev_heading = self.heading
        
        # Apply heading change
        self.heading = (self.heading + heading_change) % 360
        
        # Get wind at current position
        wind_speed, wind_dir = self.wind_grid.get_wind_at_position(
            self.position[0], self.position[1])
        
        # Calculate true wind angle
        twa = (self.heading - wind_dir) % 360
        if twa > 180:
            twa = 360 - twa
        
        # Get boat speed from polar model
        boat_speed = self.polar_model.get_boat_speed(twa, wind_speed)
        
        # Calculate movement from boat velocity
        heading_rad = np.radians(self.heading)
        boat_dx = boat_speed * np.sin(heading_rad) * self.time_step
        boat_dy = boat_speed * np.cos(heading_rad) * self.time_step
        
        # Get current vector (if enabled)
        if self.use_currents and self.current_grid:
            current_x, current_y = self.current_grid.get_current_vector_at_position(
                self.position[0], self.position[1])
            
            # Apply current effect
            current_dx = current_x * self.time_step
            current_dy = current_y * self.time_step
        else:
            current_dx = 0.0
            current_dy = 0.0
        
        # Combined movement (boat + current)
        dx = boat_dx + current_dx
        dy = boat_dy + current_dy
        
        # Update position
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy
        
        # Check if out of bounds and constrain
        if new_x < 0 or new_x >= self.grid_size_x or new_y < 0 or new_y >= self.grid_size_y:
            # Clip to boundaries
            new_x = max(0, min(new_x, self.grid_size_x - 1))
            new_y = max(0, min(new_y, self.grid_size_y - 1))
            
            # Penalty for hitting boundary
            boundary_penalty = -20
        else:
            boundary_penalty = 0
        
        self.position = np.array([new_x, new_y])
        
        # Check if reached goal
        dx = self.goal_pos[0] - self.position[0]
        dy = self.goal_pos[1] - self.position[1]
        distance_to_goal = np.sqrt(dx**2 + dy**2)
        
        if distance_to_goal <= self.goal_radius:
            # Reward for reaching goal
            reward = 100
            done = True
            return self._get_state(), reward, done, {"reason": "goal_reached"}
        
        # Check for tack or gybe
        old_twa = (self.prev_heading - wind_dir) % 360
        if old_twa > 180:
            old_twa = 360 - old_twa
        
        tack_penalty = 0
        if (old_twa < 90 and twa < 90 and 
            ((self.prev_heading < wind_dir and self.heading > wind_dir) or 
             (self.prev_heading > wind_dir and self.heading < wind_dir))):
            # Tacking penalty
            tack_penalty = -5
        elif (old_twa > 90 and twa > 90 and 
              ((self.prev_heading < wind_dir and self.heading > wind_dir) or 
               (self.prev_heading > wind_dir and self.heading < wind_dir))):
            # Gybing penalty
            tack_penalty = -3
        
        # Calculate reward based on progress toward goal
        # We want to reward movement in the right direction
        prev_distance = np.sqrt(
            (self.goal_pos[0] - (self.position[0] - dx))**2 + 
            (self.goal_pos[1] - (self.position[1] - dy))**2
        )
        progress = prev_distance - distance_to_goal
        
        # Basic reward is proportional to progress
        reward = progress * 10
        
        # Add penalties
        reward += boundary_penalty + tack_penalty
        
        # Small penalty for each step to encourage efficiency
        reward -= 0.1
        
        # Check if out of steps
        if self.steps >= self.max_steps:
            done = True
            reward -= 10  # Additional penalty for timeout
            return self._get_state(), reward, done, {"reason": "timeout"}
        
        done = False
        return self._get_state(), reward, done, {"current_info": {
            "speed": current_dx**2 + current_dy**2,
            "direction": np.degrees(np.arctan2(current_dx, current_dy)) if current_dx != 0 or current_dy != 0 else 0
        }}

#########################
# Visualization Functions
#########################

def plot_vector_field(grid, title, ax, is_wind=True, subsample=2):
    """Plot a vector field for wind or current"""
    # Create coordinate matrices
    y_coords, x_coords = np.mgrid[0:grid.grid_size_y:subsample, 0:grid.grid_size_x:subsample]
    
    # Extract subsampled data for plotting
    if is_wind:
        speeds = grid.wind_speed[::subsample, ::subsample]
        directions = grid.wind_dir[::subsample, ::subsample]
    else:
        speeds = grid.current_speed[::subsample, ::subsample]
        directions = grid.current_dir[::subsample, ::subsample]
    
    # Convert directions to vector components
    u = speeds * np.sin(np.radians(directions))
    v = speeds * np.cos(np.radians(directions))
    
    # Normalize for plotting
    magnitude = np.sqrt(u**2 + v**2)
    norm = Normalize()
    norm.autoscale(magnitude)
    
    # Plot vector field
    cmap = cm.viridis
    scale_value = 400 if is_wind else 50
    ax.quiver(x_coords, y_coords, u, v, magnitude, 
              cmap=cmap, pivot='mid', scale=scale_value)
    
    # Plot speed as background color - using imshow instead of pcolormesh to avoid dimension issues
    full_speeds = grid.wind_speed if is_wind else grid.current_speed
    speed_grid = ax.imshow(full_speeds, origin='lower', extent=[0, grid.grid_size_x, 0, grid.grid_size_y],
                          alpha=0.3, cmap=cmap, aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(speed_grid, ax=ax)
    cbar.set_label("Speed (knots)")
    
    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.set_aspect('equal')
    
    return ax

def visualize_environment(env):
    """Create a visualization of the sailing environment with wind and current"""
    # Create figure with two subplots side by side
    fig = plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    # Plot wind field
    ax_wind = plt.subplot(gs[0])
    plot_vector_field(env.wind_grid, "Wind Field (Spatially Varied, Fixed in Time)", ax_wind, is_wind=True)
    
    # Plot current field
    ax_current = plt.subplot(gs[1])
    plot_vector_field(env.current_grid, "Current Field (Eulerian Flow Model)", ax_current, is_wind=False)
    
    # Mark start and goal positions on both plots
    for ax in [ax_wind, ax_current]:
        # Start position
        ax.plot(env.start_pos[0], env.start_pos[1], 'go', markersize=10, label='Start')
        
        # Goal position with radius
        ax.plot(env.goal_pos[0], env.goal_pos[1], 'ro', markersize=10, label='Goal')
        circle = Circle((env.goal_pos[0], env.goal_pos[1]), env.goal_radius, 
                        fill=False, color='r', linestyle='--')
        ax.add_patch(circle)
        
        # Add legend
        ax.legend()
    
    plt.tight_layout()
    return fig

#########################
# Main Simulation Example
#########################

def run_simulation_example():
    # Create polar model
    polar_model = SimplePolarModel()
    
    # Create environment
    grid_size = 50
    env = SailingEnv(
        grid_size_x=grid_size,
        grid_size_y=grid_size,
        polar_model=polar_model,
        start_pos=(10, 10),
        goal_pos=(40, 40),
        use_currents=True,
        base_current_speed=1.5,
        base_current_dir=120,
        base_wind_speed=15.0,
        base_wind_dir=45.0
    )
    
    # Reset to initialize
    env.reset()
    
    # Visualize the environment
    fig = visualize_environment(env)
    plt.savefig('wind_current_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Wind direction varies spatially but is fixed in time.")
    print("Current field follows Eulerian Flow Model with spatial variation.")
    
    return env

# Example usage
if __name__ == "__main__":
    env = run_simulation_example()
    print("Simulation environment created and visualized successfully.")