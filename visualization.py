# visualization.py
import matplotlib.pyplot as plt
import numpy as np
from boat import Position, BoatState
from course import Course
from wind import WindState
from typing import List
import matplotlib.colors as mcolors
from matplotlib.quiver import QuiverKey

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
                title: str = "Race Course"):
    
    fig, ax = plt.subplots(figsize=(12, 18))
    
    # Create wind visualization grid
    X, Y = create_wind_grid(course)
    
    # Create background wind speed and direction plot
    wind_speeds, U, V = interpolate_wind_to_background(X, Y, boat_states, wind_states)
    min_speed = min(w.speed for w in wind_states)
    max_speed = max(w.speed for w in wind_states)
    norm = mcolors.Normalize(vmin=min_speed, vmax=max_speed)
    
    # Plot wind speed as colored background
    mesh = ax.pcolormesh(X, Y, wind_speeds, shading='auto', 
                        norm=norm, cmap='YlOrRd', alpha=0.5)
    
    # Plot wind vectors evenly across the course
    skip = 2  # Adjust this value to control arrow density
    Q = ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                 U[::skip, ::skip], V[::skip, ::skip],
                 scale=30,
                 width=0.002,
                 headwidth=3,
                 headlength=4,
                 headaxislength=3.5,
                 alpha=0.6,
                 color='gray')
    
    # Add colorbar
    cbar = fig.colorbar(mesh, ax=ax, label='Wind Speed (knots)')
    cbar.set_ticks(np.linspace(min_speed, max_speed, 10))
    
    # Plot extended course boundaries
    ax.plot([0, 0], [0, course.length + course.extension], 'k--', alpha=0.5)
    ax.plot([course.width, course.width], 
            [0, course.length + course.extension], 'k--', alpha=0.5)
    
    # Plot start/finish line
    start_x = [course.start_line[0].x, course.start_line[1].x]
    start_y = [course.start_line[0].y, course.start_line[1].y]
    ax.plot(start_x, start_y, 'g-', linewidth=2, label='Start/Finish Line')
    
    # Plot top marks and gate
    ax.plot(course.top_marks[0].x, course.top_marks[0].y, 'ro', 
            markersize=10, label='Left Mark')
    ax.plot(course.top_marks[1].x, course.top_marks[1].y, 'ro', 
            markersize=10, label='Right Mark')
    ax.plot([course.top_marks[0].x, course.top_marks[1].x],
            [course.top_marks[0].y, course.top_marks[1].y],
            'r--', alpha=0.5, label='Gate')
    
    # Extract boat track coordinates
    xs = [state.position.x for state in boat_states]
    ys = [state.position.y for state in boat_states]
    
    # Plot boat track
    ax.plot(xs, ys, 'b-', linewidth=2, label='Boat Track')
    
    # Select visualization points
    viz_points = 6  # Increased number of points
    time_indices = np.linspace(0, len(boat_states)-1, viz_points, dtype=int)
    
    # Add boat position markers and labels
    for i in time_indices:
        ax.plot(xs[i], ys[i], 'b.', markersize=10)
        
        # Add speed, TWA, and wind info label
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
    
    # Set axis limits
    ax.set_xlim(-100, course.width + 100)
    ax.set_ylim(-100, course.length + course.extension + 100)
    
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()