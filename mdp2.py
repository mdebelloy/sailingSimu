import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from boat import Position, BoatState, Boat
from static_wind import StaticWindField, WindState
from polars import PolarData
from course import Course, create_standard_course
import time

@dataclass
class GridCell:
    x: float
    y: float
    wind_speed: float
    wind_dir: float

class Grid:
    def __init__(self, 
                 course: Course,
                 x_cells: int = 40,
                 y_cells: int = 60):
        """Initialize computational grid"""
        self.x_cells = x_cells
        self.y_cells = y_cells
        
        # Create grid points
        x = np.linspace(0, course.width, x_cells)
        y = np.linspace(0, course.length, y_cells)
        self.x, self.y = np.meshgrid(x, y)
        
        # Initialize wind arrays
        self.wind_speed = np.zeros((y_cells, x_cells))
        self.wind_dir = np.zeros((y_cells, x_cells))
        
    def update_wind(self, wind_field: StaticWindField):
        """Update wind values at all grid points"""
        for i in range(self.y_cells):
            for j in range(self.x_cells):
                pos = Position(self.x[i,j], self.y[i,j])
                wind = wind_field.get_wind_state(pos)
                self.wind_speed[i,j] = wind.speed
                self.wind_dir[i,j] = wind.direction

class SailingMDP:
    def __init__(self,
                 course: Course,
                 polars: PolarData,
                 wind_field: StaticWindField,
                 dt: float = 1.0):
        """Initialize sailing MDP"""
        self.course = course
        self.polars = polars
        self.wind_field = wind_field
        self.dt = dt
        
        # Setup computational grid
        self.grid = Grid(course)
        self.grid.update_wind(wind_field)
        
        # Initialize policy arrays (y_cells, x_cells, 2 tacks)
        self.policy_upwind = np.zeros((self.grid.y_cells, self.grid.x_cells, 2))
        self.policy_downwind = np.zeros((self.grid.y_cells, self.grid.x_cells, 2))
        
        # Value functions
        self.value_upwind = np.full((self.grid.y_cells, self.grid.x_cells), np.inf)
        self.value_downwind = np.full((self.grid.y_cells, self.grid.x_cells), np.inf)
        
        # Initialize terminal states
        self._init_terminal_states()
    
    def _init_terminal_states(self):
        """Initialize terminal state values"""
        # Gate line is terminal for upwind
        gate_y_idx = np.searchsorted(self.grid.y[0,:], self.course.top_marks[0].y)
        gate_x_range = (self.course.top_marks[0].x, self.course.top_marks[1].x)
        gate_x_idxs = np.where((self.grid.x[0,:] >= gate_x_range[0]) & 
                              (self.grid.x[0,:] <= gate_x_range[1]))[0]
        self.value_upwind[gate_y_idx, gate_x_idxs] = 0
        
        # Start line is terminal for downwind
        start_x_range = (self.course.start_line[0].x, self.course.start_line[1].x)
        start_x_idxs = np.where((self.grid.x[0,:] >= start_x_range[0]) & 
                               (self.grid.x[0,:] <= start_x_range[1]))[0]
        self.value_downwind[0, start_x_idxs] = 0
    
    def _get_next_state(self, 
                        x: float, 
                        y: float, 
                        wind_dir: float,
                        wind_speed: float,
                        twa: float) -> Tuple[float, float]:
        """Get next state given current state and action"""
        # Get boat speed
        bsp = self.polars.get_boat_speed(abs(twa), wind_speed)
        bsp_ms = bsp * 0.51444  # Convert to m/s
        
        # Calculate heading and velocity components
        heading = (wind_dir + twa) % 360
        heading_rad = np.radians(heading)
        vx = bsp_ms * np.sin(heading_rad)
        vy = bsp_ms * np.cos(heading_rad)
        
        # Update position
        new_x = x + vx * self.dt
        new_y = y + vy * self.dt
        
        return new_x, new_y
    
    def solve_upwind(self, max_iter: int = 1000, tol: float = 0.1):
        """Solve upwind leg using vectorized value iteration"""
        # Possible TWA values for port and starboard tacks
        twa_range = np.arange(30, 60, 2)  # Typical upwind angles
        
        # Pre-compute grid of positions
        x_grid = self.grid.x
        y_grid = self.grid.y
        
        # Create meshgrid of next states for all TWAs
        twa_mesh = twa_range[:, np.newaxis, np.newaxis]  # Shape: (n_angles, 1, 1)
        
        # Expand wind information
        wind_dir_expanded = self.grid.wind_dir[np.newaxis, :, :]  # Shape: (1, y_cells, x_cells)
        wind_speed_expanded = self.grid.wind_speed[np.newaxis, :, :]
        
        # Pre-compute boat speeds for all TWAs and wind speeds
        bsp_matrix = np.array([[self.polars.get_boat_speed(abs(twa), ws) 
                            for ws in np.unique(self.grid.wind_speed)] 
                            for twa in twa_range])
        wind_speed_indices = np.searchsorted(np.unique(self.grid.wind_speed), 
                                        self.grid.wind_speed.flatten()).reshape(self.grid.wind_speed.shape)
        
        for iter in range(max_iter):
            value_old = self.value_upwind.copy()
            
            # Vectorized calculation of next states for all angles
            # Convert boat speeds to m/s
            bsp_expanded = bsp_matrix[..., wind_speed_indices] * 0.51444  # Shape: (n_angles, y_cells, x_cells)
            
            # Calculate headings for both tacks
            headings_port = (wind_dir_expanded + twa_mesh) % 360
            headings_starboard = (wind_dir_expanded - twa_mesh) % 360
            
            # Convert to radians
            headings_port_rad = np.radians(headings_port)
            headings_starboard_rad = np.radians(headings_starboard)
            
            # Calculate velocity components
            vx_port = bsp_expanded * np.sin(headings_port_rad)
            vy_port = bsp_expanded * np.cos(headings_port_rad)
            vx_starboard = bsp_expanded * np.sin(headings_starboard_rad)
            vy_starboard = bsp_expanded * np.cos(headings_starboard_rad)
            
            # Calculate next positions
            next_x_port = x_grid + vx_port * self.dt
            next_y_port = y_grid + vy_port * self.dt
            next_x_starboard = x_grid + vx_starboard * self.dt
            next_y_starboard = y_grid + vy_starboard * self.dt
            
            # Mask for valid positions
            valid_port = ((next_x_port >= 0) & (next_x_port <= self.course.width) &
                        (next_y_port >= 0) & (next_y_port <= self.course.length))
            valid_starboard = ((next_x_starboard >= 0) & (next_x_starboard <= self.course.width) &
                            (next_y_starboard >= 0) & (next_y_starboard <= self.course.length))
            
            # Initialize arrays for storing interpolated values
            values_port = np.full_like(next_x_port, np.inf)
            values_starboard = np.full_like(next_x_starboard, np.inf)
            
            # Vectorized interpolation for valid positions
            x_indices_port = np.clip(np.searchsorted(self.grid.x[0,:], next_x_port) - 1, 0, self.grid.x_cells-2)
            y_indices_port = np.clip(np.searchsorted(self.grid.y[:,0], next_y_port) - 1, 0, self.grid.y_cells-2)
            x_indices_starboard = np.clip(np.searchsorted(self.grid.x[0,:], next_x_starboard) - 1, 0, self.grid.x_cells-2)
            y_indices_starboard = np.clip(np.searchsorted(self.grid.y[:,0], next_y_starboard) - 1, 0, self.grid.y_cells-2)
            
            # Calculate interpolation weights
            wx_port = (next_x_port - self.grid.x[0,x_indices_port]) / (self.grid.x[0,x_indices_port+1] - self.grid.x[0,x_indices_port])
            wy_port = (next_y_port - self.grid.y[y_indices_port,0]) / (self.grid.y[y_indices_port+1,0] - self.grid.y[y_indices_port,0])
            wx_starboard = (next_x_starboard - self.grid.x[0,x_indices_starboard]) / (self.grid.x[0,x_indices_starboard+1] - self.grid.x[0,x_indices_starboard])
            wy_starboard = (next_y_starboard - self.grid.y[y_indices_starboard,0]) / (self.grid.y[y_indices_starboard+1,0] - self.grid.y[y_indices_starboard,0])
            
            # Get corner values and interpolate
            for i in range(len(twa_range)):
                v00_port = self.value_upwind[y_indices_port[i], x_indices_port[i]]
                v01_port = self.value_upwind[y_indices_port[i], x_indices_port[i]+1]
                v10_port = self.value_upwind[y_indices_port[i]+1, x_indices_port[i]]
                v11_port = self.value_upwind[y_indices_port[i]+1, x_indices_port[i]+1]
                
                values_port[i] = ((1-wx_port[i])*(1-wy_port[i])*v00_port + 
                                wx_port[i]*(1-wy_port[i])*v01_port + 
                                (1-wx_port[i])*wy_port[i]*v10_port + 
                                wx_port[i]*wy_port[i]*v11_port)
                
                v00_starboard = self.value_upwind[y_indices_starboard[i], x_indices_starboard[i]]
                v01_starboard = self.value_upwind[y_indices_starboard[i], x_indices_starboard[i]+1]
                v10_starboard = self.value_upwind[y_indices_starboard[i]+1, x_indices_starboard[i]]
                v11_starboard = self.value_upwind[y_indices_starboard[i]+1, x_indices_starboard[i]+1]
                
                values_starboard[i] = ((1-wx_starboard[i])*(1-wy_starboard[i])*v00_starboard + 
                                    wx_starboard[i]*(1-wy_starboard[i])*v01_starboard + 
                                    (1-wx_starboard[i])*wy_starboard[i]*v10_starboard + 
                                    wx_starboard[i]*wy_starboard[i]*v11_starboard)
            
            # Apply validity masks
            values_port[~valid_port] = np.inf
            values_starboard[~valid_starboard] = np.inf
            
            # Add time step to values
            values_port += self.dt
            values_starboard += self.dt
            
            # Find best actions and update value function
            min_values_port = np.min(values_port, axis=0)
            min_values_starboard = np.min(values_starboard, axis=0)
            best_port_indices = np.argmin(values_port, axis=0)
            best_starboard_indices = np.argmin(values_starboard, axis=0)
            
            # Update policy and value function
            self.value_upwind = np.minimum(min_values_port, min_values_starboard)
            
            mask_port = min_values_port <= min_values_starboard
            self.policy_upwind[:,:,0] = np.where(mask_port, twa_range[best_port_indices], 0)
            self.policy_upwind[:,:,1] = np.where(~mask_port, -twa_range[best_starboard_indices], 0)
            
            # Check convergence
            delta = np.max(np.abs(self.value_upwind - value_old))
            if delta < tol:
                break
    
    def solve_downwind(self, max_iter: int = 1000, tol: float = 0.1):
        """Solve downwind leg using vectorized value iteration"""
        # Possible TWA values for port and starboard gybes
        twa_range = np.arange(120, 180, 2)  # Typical downwind angles
        
        # Pre-compute grid of positions
        x_grid = self.grid.x
        y_grid = self.grid.y
        
        # Create meshgrid of next states for all TWAs
        twa_mesh = twa_range[:, np.newaxis, np.newaxis]  # Shape: (n_angles, 1, 1)
        
        # Expand wind information
        wind_dir_expanded = self.grid.wind_dir[np.newaxis, :, :]  # Shape: (1, y_cells, x_cells)
        wind_speed_expanded = self.grid.wind_speed[np.newaxis, :, :]
        
        # Pre-compute boat speeds for all TWAs and wind speeds
        bsp_matrix = np.array([[self.polars.get_boat_speed(abs(twa), ws) 
                            for ws in np.unique(self.grid.wind_speed)] 
                            for twa in twa_range])
        wind_speed_indices = np.searchsorted(np.unique(self.grid.wind_speed), 
                                        self.grid.wind_speed.flatten()).reshape(self.grid.wind_speed.shape)
        
        for iter in range(max_iter):
            value_old = self.value_downwind.copy()
            
            # Vectorized calculation of next states for all angles
            # Convert boat speeds to m/s
            bsp_expanded = bsp_matrix[..., wind_speed_indices] * 0.51444  # Shape: (n_angles, y_cells, x_cells)
            
            # Calculate headings for both gybes
            headings_port = (wind_dir_expanded + twa_mesh) % 360
            headings_starboard = (wind_dir_expanded - twa_mesh) % 360
            
            # Convert to radians
            headings_port_rad = np.radians(headings_port)
            headings_starboard_rad = np.radians(headings_starboard)
            
            # Calculate velocity components
            vx_port = bsp_expanded * np.sin(headings_port_rad)
            vy_port = bsp_expanded * np.cos(headings_port_rad)
            vx_starboard = bsp_expanded * np.sin(headings_starboard_rad)
            vy_starboard = bsp_expanded * np.cos(headings_starboard_rad)
            
            # Calculate next positions
            next_x_port = x_grid + vx_port * self.dt
            next_y_port = y_grid + vy_port * self.dt
            next_x_starboard = x_grid + vx_starboard * self.dt
            next_y_starboard = y_grid + vy_starboard * self.dt
            
            # Mask for valid positions
            valid_port = ((next_x_port >= 0) & (next_x_port <= self.course.width) &
                        (next_y_port >= 0) & (next_y_port <= self.course.length))
            valid_starboard = ((next_x_starboard >= 0) & (next_x_starboard <= self.course.width) &
                            (next_y_starboard >= 0) & (next_y_starboard <= self.course.length))
            
            # Initialize arrays for storing interpolated values
            values_port = np.full_like(next_x_port, np.inf)
            values_starboard = np.full_like(next_x_starboard, np.inf)
            
            # Vectorized interpolation for valid positions
            x_indices_port = np.clip(np.searchsorted(self.grid.x[0,:], next_x_port) - 1, 0, self.grid.x_cells-2)
            y_indices_port = np.clip(np.searchsorted(self.grid.y[:,0], next_y_port) - 1, 0, self.grid.y_cells-2)
            x_indices_starboard = np.clip(np.searchsorted(self.grid.x[0,:], next_x_starboard) - 1, 0, self.grid.x_cells-2)
            y_indices_starboard = np.clip(np.searchsorted(self.grid.y[:,0], next_y_starboard) - 1, 0, self.grid.y_cells-2)
            
            # Calculate interpolation weights
            wx_port = (next_x_port - self.grid.x[0,x_indices_port]) / (self.grid.x[0,x_indices_port+1] - self.grid.x[0,x_indices_port])
            wy_port = (next_y_port - self.grid.y[y_indices_port,0]) / (self.grid.y[y_indices_port+1,0] - self.grid.y[y_indices_port,0])
            wx_starboard = (next_x_starboard - self.grid.x[0,x_indices_starboard]) / (self.grid.x[0,x_indices_starboard+1] - self.grid.x[0,x_indices_starboard])
            wy_starboard = (next_y_starboard - self.grid.y[y_indices_starboard,0]) / (self.grid.y[y_indices_starboard+1,0] - self.grid.y[y_indices_starboard,0])
            
            # Get corner values and interpolate
            for i in range(len(twa_range)):
                v00_port = self.value_downwind[y_indices_port[i], x_indices_port[i]]
                v01_port = self.value_downwind[y_indices_port[i], x_indices_port[i]+1]
                v10_port = self.value_downwind[y_indices_port[i]+1, x_indices_port[i]]
                v11_port = self.value_downwind[y_indices_port[i]+1, x_indices_port[i]+1]
                
                values_port[i] = ((1-wx_port[i])*(1-wy_port[i])*v00_port + 
                                wx_port[i]*(1-wy_port[i])*v01_port + 
                                (1-wx_port[i])*wy_port[i]*v10_port + 
                                wx_port[i]*wy_port[i]*v11_port)
                
                v00_starboard = self.value_downwind[y_indices_starboard[i], x_indices_starboard[i]]
                v01_starboard = self.value_downwind[y_indices_starboard[i], x_indices_starboard[i]+1]
                v10_starboard = self.value_downwind[y_indices_starboard[i]+1, x_indices_starboard[i]]
                v11_starboard = self.value_downwind[y_indices_starboard[i]+1, x_indices_starboard[i]+1]
                
                values_starboard[i] = ((1-wx_starboard[i])*(1-wy_starboard[i])*v00_starboard + 
                                    wx_starboard[i]*(1-wy_starboard[i])*v01_starboard + 
                                    (1-wx_starboard[i])*wy_starboard[i]*v10_starboard + 
                                    wx_starboard[i]*wy_starboard[i]*v11_starboard)
            
            # Apply validity masks
            values_port[~valid_port] = np.inf
            values_starboard[~valid_starboard] = np.inf
            
            # Add time step to values
            values_port += self.dt
            values_starboard += self.dt
            
            # Find best actions and update value function
            min_values_port = np.min(values_port, axis=0)
            min_values_starboard = np.min(values_starboard, axis=0)
            best_port_indices = np.argmin(values_port, axis=0)
            best_starboard_indices = np.argmin(values_starboard, axis=0)
            
            # Update policy and value function
            self.value_downwind = np.minimum(min_values_port, min_values_starboard)
            
            mask_port = min_values_port <= min_values_starboard
            self.policy_downwind[:,:,0] = np.where(mask_port, twa_range[best_port_indices], 0)
            self.policy_downwind[:,:,1] = np.where(~mask_port, -twa_range[best_starboard_indices], 0)
            
            # Check convergence
            delta = np.max(np.abs(self.value_downwind - value_old))
            if delta < tol:
                print(f"Converged after {iter+1} iterations")
                break
    
    def _interpolate_value(self, x: float, y: float, value_grid: np.ndarray) -> float:
        """Bilinearly interpolate value at arbitrary position"""
        # Find grid cell indices
        x_idx = np.searchsorted(self.grid.x[0,:], x) - 1
        y_idx = np.searchsorted(self.grid.y[:,0], y) - 1
        
        # Handle boundary cases
        if x_idx < 0 or x_idx >= self.grid.x_cells - 1 or \
           y_idx < 0 or y_idx >= self.grid.y_cells - 1:
            return np.inf
        
        # Get cell corners
        x0, x1 = self.grid.x[0,x_idx], self.grid.x[0,x_idx+1]
        y0, y1 = self.grid.y[y_idx,0], self.grid.y[y_idx+1,0]
        
        # Interpolation weights
        wx = (x - x0) / (x1 - x0)
        wy = (y - y0) / (y1 - y0)
        
        # Get corner values
        v00 = value_grid[y_idx, x_idx]
        v01 = value_grid[y_idx, x_idx+1]
        v10 = value_grid[y_idx+1, x_idx]
        v11 = value_grid[y_idx+1, x_idx+1]
        
        # Bilinear interpolation
        return (1-wx)*(1-wy)*v00 + wx*(1-wy)*v01 + (1-wx)*wy*v10 + wx*wy*v11
    
    def plot_policy(self):
        """Plot policy with wind gradient"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        for ax in [ax1, ax2]:
            im = ax.pcolormesh(self.grid.x, self.grid.y, self.grid.wind_speed, 
                             cmap='YlOrRd', alpha=0.5)
            plt.colorbar(im, ax=ax, label='Wind Speed (knots)')
            
            # Wind arrows
            skip = 4
            wind_scale = 50
            for i in range(0, self.grid.y_cells, skip):
                for j in range(0, self.grid.x_cells, skip):
                    wind_dir = np.radians(self.grid.wind_dir[i,j])
                    wx = -np.sin(wind_dir) * wind_scale
                    wy = -np.cos(wind_dir) * wind_scale
                    ax.arrow(self.grid.x[i,j], self.grid.y[i,j], wx, wy,
                            color='gray', alpha=0.3, width=2)
        
        # Plot policies
        skip = 2
        boat_scale = 80
        
        # Upwind policies
        for i in range(0, self.grid.y_cells, skip):
            for j in range(0, self.grid.x_cells, skip):
                x, y = self.grid.x[i,j], self.grid.y[i,j]
                
                for tack_idx, color in enumerate(['blue', 'red']):
                    twa = self.policy_upwind[i,j,tack_idx]
                    if abs(twa) > 0:
                        heading = (self.grid.wind_dir[i,j] + twa) % 360
                        dx = np.sin(np.radians(heading)) * boat_scale
                        dy = np.cos(np.radians(heading)) * boat_scale
                        ax1.arrow(x, y, dx, dy, color=color, alpha=0.5,
                                head_width=20, head_length=20)
        
        # Downwind policies
        for i in range(0, self.grid.y_cells, skip):
            for j in range(0, self.grid.x_cells, skip):
                x, y = self.grid.x[i,j], self.grid.y[i,j]
                
                for tack_idx, color in enumerate(['blue', 'red']):
                    twa = self.policy_downwind[i,j,tack_idx]
                    if abs(twa) > 0:
                        heading = (self.grid.wind_dir[i,j] + twa) % 360
                        dx = np.sin(np.radians(heading)) * boat_scale
                        dy = np.cos(np.radians(heading)) * boat_scale
                        ax2.arrow(x, y, dx, dy, color=color, alpha=0.5,
                                head_width=20, head_length=20)
        
        # Course boundaries and marks
        for ax in [ax1, ax2]:
            ax.plot([0, 0], [0, self.course.length], 'k--', alpha=0.5)
            ax.plot([self.course.width, self.course.width], 
                   [0, self.course.length], 'k--', alpha=0.5)
            ax.plot([self.course.top_marks[0].x, self.course.top_marks[1].x],
                   [self.course.top_marks[0].y, self.course.top_marks[1].y],
                   'r-', linewidth=2)
            ax.plot([self.course.start_line[0].x, self.course.start_line[1].x],
                   [0, 0], 'g-', linewidth=2)
            ax.set_xlim(0, self.course.width)
            ax.set_ylim(0, self.course.length)
            ax.grid(True, alpha=0.3)
        
        ax1.set_title("Upwind Policy (blue=port, red=starboard)")
        ax2.set_title("Downwind Policy (blue=port, red=starboard)")
        plt.tight_layout()
        plt.show()

def main():
    # Create course and polar data
    course = create_standard_course()
    polars = PolarData()
    
    # Create wind field
    wind_field = StaticWindField(
        course_width=course.width,
        course_length=course.length,
        base_direction=0,
        base_speed=15,
        n_patterns=3
    )
    
    # Create and solve MDP
    mdp = SailingMDP(course, polars, wind_field)
    
    print("Solving upwind leg...")
    start_time = time.time()
    mdp.solve_upwind()
    print(f"Upwind solved in {time.time() - start_time:.1f} seconds")
    
    print("Solving downwind leg...")
    start_time = time.time()
    mdp.solve_downwind()
    print(f"Downwind solved in {time.time() - start_time:.1f} seconds")
    
    # Plot results
    mdp.plot_policy()

if __name__ == "__main__":
    main()