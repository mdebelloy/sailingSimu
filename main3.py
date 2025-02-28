import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from boat import Boat, Position, BoatState, get_tacking_penalty
from course import Course, create_standard_course
from polars import PolarData
from vis2 import plot_course
from copy import deepcopy
from scipy.ndimage import gaussian_filter

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
                     n_gusts: int = 4,
                     resolution: int = 50) -> WindField:
    """
    Create a wind field with spatial variations (gusts)
    """
    # Create coordinate grids
    x_coords = np.linspace(0, course.width, resolution)
    y_coords = np.linspace(0, course.length + course.extension, resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Start with base wind speed
    wind_speeds = np.full_like(X, base_speed)
    
    # Add random gusts
    for _ in range(n_gusts):
        # Random center point for gust
        center_x = np.random.uniform(0, course.width)
        center_y = np.random.uniform(0, course.length + course.extension)
        
        # Random gust intensity (±30% of base speed)
        intensity = np.random.uniform(-0.6, 0.6) * base_speed
        
        # Create gust pattern
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        gust = intensity * np.exp(-distance**2 / (600**2))  # 200m characteristic size
        wind_speeds += gust
    
    # Smooth the field
    wind_speeds = gaussian_filter(wind_speeds, sigma=2)
    
    # Ensure minimum wind speed is positive
    wind_speeds = np.maximum(wind_speeds, base_speed * 0.5)
    
    return WindField(wind_speeds, x_coords, y_coords, base_speed)

class MarkovWindModel:
    """
    Implementation of wind model with:
    - Mean-reverting wind direction following hidden Markov model
    - Spatial wind speed variations (gusts)
    - Discrete states for wind direction
    """
    def __init__(self, 
                 course: Course,
                 base_speed: float = 16.0,
                 n_direction_states: int = 19,
                 direction_range: Tuple[float, float] = (-45, 45)):
        """Initialize wind model"""
        self.base_speed = base_speed
        self.wind_field = create_wind_field(course, base_speed)
        
        # Create direction states
        self.direction_states = np.linspace(
            direction_range[0], 
            direction_range[1], 
            n_direction_states
        )
        
        # Create transition matrix favoring small changes
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
                
        # Normalize rows
        self.P = P / P.sum(axis=1)[:,None]
        
        # Initialize state
        center_idx = n // 2
        self.current_state = MarkovWindState(
            direction=self.direction_states[center_idx],
            mean_direction=self.direction_states[center_idx],
            speed=base_speed,
            base_speed=base_speed,
            transition_matrix=self.P
        )
    
    def step(self, dt: float, position: Position) -> MarkovWindState:
        """Update wind state according to Markov model"""
        # Find current state index
        current_idx = np.abs(
            self.direction_states - self.current_state.mean_direction
        ).argmin()
        
        # Sample next state index based on transition probabilities
        next_idx = np.random.choice(
            len(self.direction_states), 
            p=self.P[current_idx]
        )
        
        # Update mean direction
        new_mean = self.direction_states[next_idx]
        
        # Add random fluctuation around mean
        new_direction = new_mean + np.random.uniform(-5, 5)
        
        # Get wind speed at current position
        local_speed = self.wind_field.get_wind_speed(position.x, position.y)
        
        self.current_state = MarkovWindState(
            direction=new_direction,
            mean_direction=new_mean,
            speed=local_speed,
            base_speed=self.base_speed,
            transition_matrix=self.P
        )
        
        return self.current_state

def simulate_upwind_leg(time_step: float = 5.0, 
                       max_time: float = 1200.0) -> Tuple[List[BoatState], List[MarkovWindState], WindField]:
    """Simulate upwind leg and return states and wind field"""
    course = create_standard_course()
    polars = PolarData()
    wind_model = MarkovWindModel(course)
    
    start_pos = Position(course.width/2, 0)
    boat = Boat(start_pos, 45, 'starboard', polars)
    
    boat_states = [boat.state.copy()]
    wind_states = [deepcopy(wind_model.current_state)]
    
    time = 0
    last_tack_time = -float('inf')
    
    while time < max_time:
        current_pos = boat.state.position
        
        if course.has_passed_gate(
            previous_pos=boat_states[-2].position if len(boat_states) > 1 else start_pos,
            current_pos=current_pos
        ):
            print(f"Passed through gate at time {time:.1f}")
            break
        
        wind_state = wind_model.step(time_step, current_pos)
        optimal_upwind, _ = boat.get_optimal_angles(wind_state.speed)
        
        min_tack_interval = 30
        can_tack = (time - last_tack_time) >= min_tack_interval
        
        if can_tack:
            current_twa = optimal_upwind if boat.state.tack == 'starboard' else -optimal_upwind
            opposite_twa = -current_twa
            
            vx_current, vy_current = boat.get_velocity(wind_state, current_twa)
            vx_opposite, vy_opposite = boat.get_velocity(wind_state, opposite_twa)
            
            target = Position(
                (course.top_marks[0].x + course.top_marks[1].x)/2,
                course.top_marks[0].y
            )
            
            dx = target.x - current_pos.x
            dy = target.y - current_pos.y
            dist = np.sqrt(dx*dx + dy*dy)
            
            vmg_current = (dx*vx_current + dy*vy_current) / dist
            vmg_opposite = (dx*vx_opposite + dy*vy_opposite) / dist
            
            near_boundary = (current_pos.x < course.width * 0.1 or 
                           current_pos.x > course.width * 0.9)
            
            if vmg_opposite > vmg_current * 1.1 or near_boundary:
                boat.state.tack = 'port' if boat.state.tack == 'starboard' else 'starboard'
                current_twa = -current_twa
                last_tack_time = time
                print(f"Tacking at time {time:.1f}")
        
        current_twa = optimal_upwind if boat.state.tack == 'starboard' else -optimal_upwind
        new_state, penalty = boat.step(time_step, wind_state, current_twa)
        
        boat_states.append(new_state)
        wind_states.append(deepcopy(wind_state))
        
        time += time_step
        
        if len(boat_states) % 20 == 0:
            print(f"Time: {time:.1f}, Position: ({current_pos.x:.1f}, {current_pos.y:.1f})")
            print(f"Local wind: {wind_state.speed:.1f}kts @ {wind_state.direction:.1f}°")
    
    return boat_states, wind_states, wind_model.wind_field

if __name__ == "__main__":
    # Simulate upwind leg
    boat_states, wind_states, wind_field = simulate_upwind_leg()
    
    # Initialize polar data for boat speed calculation
    polars = PolarData()
    boat_speeds = []
    wind_angles = []
    
    for i, state in enumerate(boat_states):
        # Get absolute true wind angle from state
        # last_twa is already the true wind angle relative to wind direction
        wind_angle = abs(state.last_twa)  # Use absolute value as polar diagram is symmetric
        wind_angles.append(state.last_twa)  # Keep signed angle for display
        
        # Get local wind speed
        wind_speed = wind_states[i].speed
        
        # Calculate boat speed using the same method as in boat.get_velocity()
        boat_speed = polars.get_boat_speed(wind_angle, wind_speed)
        boat_speeds.append(boat_speed)
        
        if i % 20 == 0:  # Log some speeds for verification
            print(f"Time step {i}: TWA={wind_angle:.1f}°, "
                  f"Wind={wind_speed:.1f}kts, Boat={boat_speed:.1f}kts")
    
    # Visualize results
    course = create_standard_course()
    plot_course(course, boat_states, wind_states, boat_speeds, wind_angles, wind_field)