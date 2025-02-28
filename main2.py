import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from boat import Boat, Position, BoatState, get_tacking_penalty
from wind import WindState, WindModel
from polars import PolarData
import math

@dataclass
class WindBelief:
    """Represents belief about wind conditions"""
    direction_mean: float
    direction_std: float
    speed_mean: float
    speed_std: float
    trend: float  # degrees per hour

@dataclass
class GridCell:
    """Represents a position in our discretized course"""
    x: float  # meters from center
    y: float  # meters upwind
    value: float = float('inf')  # expected time to finish
    policy: str = ''  # optimal action at this point

class StochasticRouter:
    def __init__(self, 
                 course_width: float = 2000.0,
                 course_height: float = 3000.0,
                 grid_size: int = 20,  # cells across width
                 volatility: str = "medium"):
        
        # Grid setup
        self.dx = course_width / grid_size
        self.dy = course_height / grid_size
        self.grid_size = grid_size
        self.course_width = course_width
        self.course_height = course_height
        
        # Initialize grid of positions
        self.grid: List[List[GridCell]] = []
        for i in range(grid_size + 1):
            row = []
            for j in range(grid_size + 1):
                x = -course_width/2 + j * self.dx
                y = i * self.dy
                row.append(GridCell(x=x, y=y))
            self.grid.append(row)
        
        # Wind volatility parameters based on paper
        volatilities = {
            "low": (1.2, 0.7),      # 1.2° per 40s, 0.7 knots per 40s
            "medium": (2.0, 0.7),    # 2.0° per 40s
            "high": (4.0, 1.3)       # 4.0° per 40s, 1.3 knots per 40s
        }
        self.dir_vol, self.spd_vol = volatilities[volatility]
        
        # Initialize boat and polars
        self.polars = PolarData()
        self.boat = Boat(Position(0, 0), 0, 'starboard', self.polars)

    def get_transition_probs(self, wind_belief: WindBelief, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate wind transition probabilities for direction and speed
        Returns: (direction_probs, speed_probs) as numpy arrays
        """
        # Direction transitions (discretize into 24 angles, 15° apart)
        angles = np.linspace(-180, 180, 24)
        dir_std = self.dir_vol * np.sqrt(dt/40.0)  # Scale volatility to timestep
        dir_mean = wind_belief.direction_mean + wind_belief.trend * dt/3600.0
        
        # Calculate direction probabilities
        dir_probs = np.exp(-0.5 * ((angles - dir_mean)/dir_std)**2)
        dir_probs = dir_probs / dir_probs.sum()
        
        # Speed transitions (discretize into 10 speeds, 2 knots apart)
        speeds = np.linspace(6, 24, 10)
        spd_std = self.spd_vol * np.sqrt(dt/40.0)
        
        # Calculate speed probabilities
        spd_probs = np.exp(-0.5 * ((speeds - wind_belief.speed_mean)/spd_std)**2)
        spd_probs = spd_probs / spd_probs.sum()
        
        return dir_probs, spd_probs

    def compute_value_iteration(self, 
                              wind_belief: WindBelief,
                              max_iterations: int = 100,
                              convergence_threshold: float = 0.1) -> None:
        """
        Compute optimal policy using value iteration
        """
        # Available actions: angles relative to wind
        angles = [-40, -30, -20, 30, 40]  # Simplified angle set
        dt = 40.0  # Timestep in seconds
        
        # Get wind transition probabilities
        dir_probs, spd_probs = self.get_transition_probs(wind_belief, dt)
        
        # Initialize gate position value
        gate_x = 0
        gate_y = self.course_height
        self.grid[self.grid_size][self.grid_size//2].value = 0
        
        for iteration in range(max_iterations):
            max_delta = 0
            
            # Update each grid cell
            for i in range(self.grid_size):
                for j in range(self.grid_size + 1):
                    cell = self.grid[i][j]
                    old_value = cell.value
                    
                    # Skip if at finish
                    if i == self.grid_size:
                        continue
                    
                    best_value = float('inf')
                    best_action = None
                    
                    # Try each action
                    for angle in angles:
                        total_value = 0
                        
                        # Consider wind transitions
                        for d_idx, dir_prob in enumerate(dir_probs):
                            for s_idx, spd_prob in enumerate(spd_probs):
                                wind_dir = -180 + d_idx * 15
                                wind_speed = 6 + s_idx * 2
                                
                                # Calculate boat movement
                                boat_speed = self.polars.get_boat_speed(abs(angle), wind_speed)
                                heading = (wind_dir + angle) % 360
                                
                                # Movement in grid coordinates
                                dx = boat_speed * np.sin(np.radians(heading)) * dt
                                dy = boat_speed * np.cos(np.radians(heading)) * dt
                                
                                # New position
                                new_x = cell.x + dx
                                new_y = cell.y + dy
                                
                                # Convert to grid indices
                                grid_x = int((new_x + self.course_width/2) / self.dx)
                                grid_y = int(new_y / self.dy)
                                
                                # Bound checking
                                grid_x = max(0, min(grid_x, self.grid_size))
                                grid_y = max(0, min(grid_y, self.grid_size))
                                
                                # Add tacking penalty if changing angle sign
                                penalty = get_tacking_penalty(wind_speed) if (
                                    (cell.policy and angle * float(cell.policy) < 0)
                                ) else 0
                                
                                # Accumulate expected value
                                next_value = self.grid[grid_y][grid_x].value
                                total_value += (dt + penalty) * dir_prob * spd_prob
                                total_value += next_value * dir_prob * spd_prob
                        
                        # Update best action
                        if total_value < best_value:
                            best_value = total_value
                            best_action = str(angle)
                    
                    # Update cell
                    cell.value = best_value
                    cell.policy = best_action
                    max_delta = max(max_delta, abs(old_value - best_value))
            
            # Check convergence
            if max_delta < convergence_threshold:
                print(f"Converged after {iteration+1} iterations")
                break
    
    def get_recommended_action(self, 
                             position: Position, 
                             wind: WindState) -> Tuple[float, str]:
        """
        Get recommended action for current position and wind
        Returns: (angle_to_sail, action_description)
        """
        # Convert position to grid coordinates
        grid_x = int((position.x + self.course_width/2) / self.dx)
        grid_y = int(position.y / self.dy)
        
        # Bound checking
        grid_x = max(0, min(grid_x, self.grid_size))
        grid_y = max(0, min(grid_y, self.grid_size))
        
        # Get policy for this cell
        action = self.grid[grid_y][grid_x].policy
        if not action:
            return 0, "No policy available"
        
        angle = float(action)
        if angle > 0:
            return angle, "Continue on starboard tack"
        else:
            return angle, "Continue on port tack"

def test_router():
    """Test the stochastic router"""
    # Initialize router
    router = StochasticRouter(volatility="medium")
    
    # Initial wind belief
    wind_belief = WindBelief(
        direction_mean=0,
        direction_std=2.0,
        speed_mean=15,
        speed_std=0.7,
        trend=0
    )
    
    # Compute optimal policy
    router.compute_value_iteration(wind_belief)
    
    # Test some positions
    test_positions = [
        Position(0, 0),      # Start line center
        Position(-500, 1000), # Left side
        Position(500, 1000),  # Right side
        Position(0, 2000)     # Near gate
    ]
    
    print("\nTesting recommendations:")
    for pos in test_positions:
        angle, desc = router.get_recommended_action(pos, WindState(0, 15))
        print(f"Position ({pos.x:.0f}, {pos.y:.0f}): {desc} at {abs(angle):.1f}°")

if __name__ == "__main__":
    test_router()