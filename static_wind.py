# static_wind.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from boat import Position

@dataclass
class WindState:
    direction: float  # degrees, 0 = from north
    speed: float     # knots

class StaticWindField:
    def __init__(self, 
                 course_width: float,
                 course_length: float,
                 base_direction: float = 0,
                 base_speed: float = 15,
                 n_patterns: int = 3):
        """
        Create a wind field with static patterns of varying wind speed and direction
        
        Args:
            course_width: Width of course in meters
            course_length: Length of course in meters
            base_direction: Base wind direction in degrees
            base_speed: Base wind speed in knots
            n_patterns: Number of wind patterns to generate
        """
        self.course_width = course_width
        self.course_length = course_length
        self.base_direction = base_direction
        self.base_speed = base_speed
        
        # Create wind patterns using Gaussian functions
        self.patterns = []
        for _ in range(n_patterns):
            # Random center point
            center_x = np.random.uniform(0, course_width)
            center_y = np.random.uniform(0, course_length)
            
            # Random size of pattern
            sigma = np.random.uniform(300, 1000)  # Size in meters
            
            # Random wind modifications
            speed_change = np.random.uniform(-3, 3)  # Speed change in knots
            dir_change = np.random.uniform(-20, 20)   # Direction change in degrees
            
            self.patterns.append({
                'center_x': center_x,
                'center_y': center_y,
                'sigma': sigma,
                'speed_change': speed_change,
                'dir_change': dir_change
            })
    
    def get_wind_state(self, position: Position) -> WindState:
        """Get wind state at given position"""
        # Start with base wind
        total_direction = self.base_direction
        total_speed = self.base_speed
        total_weight = 1.0
        
        # Add effect of each pattern
        for pattern in self.patterns:
            # Calculate distance to pattern center
            dx = position.x - pattern['center_x']
            dy = position.y - pattern['center_y']
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Calculate influence using Gaussian function
            influence = np.exp(-(distance**2) / (2 * pattern['sigma']**2))
            
            # Add weighted contributions
            total_direction += pattern['dir_change'] * influence
            total_speed += pattern['speed_change'] * influence
            total_weight += influence
        
        # Normalize
        avg_direction = (total_direction / total_weight) % 360
        avg_speed = max(5, min(25, total_speed / total_weight))  # Clip to reasonable range
        
        return WindState(direction=avg_direction, speed=avg_speed)
    
    def visualize_wind_field(self, resolution: int = 40) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate visualization data for wind field
        Returns (X, Y, U, V) for quiver plots and wind speeds
        """
        x = np.linspace(0, self.course_width, resolution)
        y = np.linspace(0, self.course_length, resolution)
        X, Y = np.meshgrid(x, y)
        
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        speeds = np.zeros_like(X)
        
        for i in range(resolution):
            for j in range(resolution):
                pos = Position(X[i,j], Y[i,j])
                wind = self.get_wind_state(pos)
                
                # Convert direction to vector components
                angle_rad = np.radians(wind.direction)
                U[i,j] = -np.sin(angle_rad)  # Negative because wind direction is where it's coming FROM
                V[i,j] = -np.cos(angle_rad)
                speeds[i,j] = wind.speed
        
        return X, Y, U, V, speeds

    def get_wind_states_list(self, boat_states: List) -> List[WindState]:
        """Generate list of wind states for visualization"""
        return [self.get_wind_state(state.position) for state in boat_states]