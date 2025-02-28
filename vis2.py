# visualization.py
import matplotlib.pyplot as plt
import numpy as np
from boat import Position, BoatState
from course import Course
from wind import WindState
from typing import List
import matplotlib.colors as mcolors
from matplotlib.quiver import QuiverKey
from dataclasses import dataclass


@dataclass
class WindField:
    """Represents the spatial wind field across the course"""
    speeds: np.ndarray  # 2D array of wind speeds
    x_coords: np.ndarray  # X coordinates for the grid
    y_coords: np.ndarray  # Y coordinates for the grid
    base_speed: float  # Base wind speed
    
    def get_wind_speed(self, x: float, y: float) -> float:
        """Get interpolated wind speed at any point"""
        # Find nearest grid points
        x_idx = np.abs(self.x_coords - x).argmin()
        y_idx = np.abs(self.y_coords - y).argmin()
        return self.speeds[y_idx, x_idx]

def create_wind_grid(course: Course, resolution: int = 40):
    """Create grid points for wind visualization"""
    x = np.linspace(0, course.width, resolution)
    y = np.linspace(0, course.length, resolution)
    return np.meshgrid(x, y)

def interpolate_wind_field(wind_states: List[WindState], 
                         times: List[float],
                         query_points: np.ndarray,
                         query_time: float) -> tuple:
    """Interpolate wind at specific points and time"""
    # Find closest time points
    time_idx = min(range(len(times)), key=lambda i: abs(times[i] - query_time))
    wind = wind_states[time_idx]
    
    # Create uniform wind field
    wind_dir_rad = np.radians(wind.direction)
    U = np.sin(wind_dir_rad) * np.ones_like(query_points[0])
    V = np.cos(wind_dir_rad) * np.ones_like(query_points[1])
    
    return U, V, np.full_like(U, wind.speed)

def create_wind_grid(course: Course, resolution: int = 40):
    """Create grid points for wind visualization"""
    x = np.linspace(0, course.width, resolution)
    y = np.linspace(0, course.length, resolution)
    return np.meshgrid(x, y)

def interpolate_wind_to_background(X: np.ndarray, Y: np.ndarray, 
                                 boat_states: List[BoatState],
                                 wind_states: List[WindState]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create wind speed and direction field showing wind variation over the course"""
    speeds = np.zeros_like(X)
    u_components = np.zeros_like(X)
    v_components = np.zeros_like(X)
    weights = np.zeros_like(X)
    
    # For each boat position, create a wind influence region
    for boat_state, wind_state in zip(boat_states, wind_states):
        # Create a distance-based weight matrix centered on boat position
        dx = X - boat_state.position.x
        dy = Y - boat_state.position.y
        distance = np.sqrt(dx**2 + dy**2)
        weight = np.exp(-distance / 500)  # 500m decay distance
        
        # Add weighted wind components
        wind_dir_rad = np.radians(wind_state.direction)
        u = -np.sin(wind_dir_rad)  # FROM direction
        v = -np.cos(wind_dir_rad)  # FROM direction
        
        speeds += wind_state.speed * weight
        u_components += u * weight
        v_components += v * weight
        weights += weight
    
    # Normalize by weights
    valid_mask = weights > 0
    speeds[valid_mask] /= weights[valid_mask]
    u_components[valid_mask] /= weights[valid_mask]
    v_components[valid_mask] /= weights[valid_mask]
    
    # Fill in areas with no data
    speeds[~valid_mask] = np.mean([w.speed for w in wind_states])
    mean_dir_rad = np.radians(np.mean([w.direction for w in wind_states]))
    u_components[~valid_mask] = -np.sin(mean_dir_rad)
    v_components[~valid_mask] = -np.cos(mean_dir_rad)
    
    return speeds, u_components, v_components

def plot_course(course: Course, 
                boat_states: List[BoatState],
                wind_states: List[WindState],
                boat_speeds: List[float],
                wind_angles: List[float],
                wind_field: WindField,
                title: str = "Race Course"):
    
    fig, ax = plt.subplots(figsize=(12, 18))
    
    # Plot wind speed as colored background using the pre-computed wind field
    mesh = ax.pcolormesh(wind_field.x_coords, wind_field.y_coords, wind_field.speeds, 
                        shading='auto', 
                        cmap='YlOrRd', 
                        alpha=0.5)
    
    # Create wind vector grid
    skip = 4  # Adjust for desired arrow density
    X, Y = np.meshgrid(wind_field.x_coords[::skip], wind_field.y_coords[::skip])
    
    # Get mean wind direction from states
    mean_direction = np.mean([w.direction for w in wind_states])
    wind_dir_rad = np.radians(mean_direction)
    
    # Create uniform direction field (since direction variations are small)
    U = -np.sin(wind_dir_rad) * np.ones_like(X)
    V = -np.cos(wind_dir_rad) * np.ones_like(X)
    
    # Scale arrows by local wind speed
    speeds = wind_field.speeds[::skip, ::skip]
    U = U * speeds / wind_field.base_speed
    V = V * speeds / wind_field.base_speed
    
    # Plot wind vectors
    Q = ax.quiver(X, Y, U, V,
                 scale=30,
                 width=0.002,
                 headwidth=3,
                 headlength=4,
                 headaxislength=3.5,
                 alpha=0.6,
                 color='gray')
    
    # Add colorbar
    cbar = fig.colorbar(mesh, ax=ax, label='Wind Speed (knots)')
    
    # Plot course elements
    ax.plot([0, 0], [0, course.length + course.extension], 'k--', alpha=0.5)
    ax.plot([course.width, course.width], 
            [0, course.length + course.extension], 'k--', alpha=0.5)
    
    # Plot start/finish line
    start_x = [course.start_line[0].x, course.start_line[1].x]
    start_y = [course.start_line[0].y, course.start_line[1].y]
    ax.plot(start_x, start_y, 'g-', linewidth=2, label='Start/Finish Line')
    
    # Plot top marks
    ax.plot(course.top_marks[0].x, course.top_marks[0].y, 'ro', 
            markersize=10, label='Left Mark')
    ax.plot(course.top_marks[1].x, course.top_marks[1].y, 'ro', 
            markersize=10, label='Right Mark')
    ax.plot([course.top_marks[0].x, course.top_marks[1].x],
            [course.top_marks[0].y, course.top_marks[1].y],
            'r--', alpha=0.5, label='Gate')
    
    # Plot boat track
    xs = [state.position.x for state in boat_states]
    ys = [state.position.y for state in boat_states]
    ax.plot(xs, ys, 'b-', linewidth=2, label='Boat Track')
    
    # Add boat position markers and labels
    viz_points = 6
    time_indices = np.linspace(0, len(boat_states)-1, viz_points, dtype=int)
    
    for i in time_indices:
        ax.plot(xs[i], ys[i], 'b.', markersize=10)
        label_text = (f'Boat: {boat_speeds[i]:.1f}kts\n'
                     f'TWA: {wind_angles[i]:.0f}°\n'
                     f'Wind: {wind_states[i].speed:.1f}kts @ {wind_states[i].direction:.0f}°')
        ax.annotate(label_text, 
                   (xs[i], ys[i]),
                   xytext=(10, 10), 
                   textcoords='offset points',
                   fontsize=8,
                   bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    # Add boat direction arrows
    arrow_indices = np.linspace(0, len(boat_states)-1, 20, dtype=int)
    for i in arrow_indices:
        if i+1 < len(boat_states):
            dx = xs[i+1] - xs[i]
            dy = ys[i+1] - ys[i]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx = dx/length * 30
                dy = dy/length * 30
                ax.arrow(xs[i], ys[i], dx, dy,
                        head_width=15, head_length=20,
                        fc='blue', ec='blue', alpha=0.5)
    
    ax.set_title(title)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Distance (m)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits with some padding
    ax.set_xlim(-100, course.width + 100)
    ax.set_ylim(-100, course.length + course.extension + 100)
    
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()