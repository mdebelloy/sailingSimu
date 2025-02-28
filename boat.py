# boat.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Literal
from polars import PolarData
from wind import WindState
from copy import deepcopy

# Move penalty function outside the class to make it importable
def get_tacking_penalty(wind_speed: float) -> float:

    return 5

    """
    Get tacking penalty in seconds based on wind speed
    From Table 1 in the paper
    """
    # Interpolate between these values
    penalties = {
        6: 9.0,
        8: 7.0,
        10: 6.0,
        12: 5.5,
        14: 5.0,
        16: 5.0,
        18: 4.5,
        20: 4.5,
        22: 4.5,
        24: 4.5
    }
    
    # Find closest wind speeds and interpolate
    speeds = sorted(penalties.keys())
    if wind_speed <= speeds[0]:
        return penalties[speeds[0]]
    if wind_speed >= speeds[-1]:
        return penalties[speeds[-1]]
    
    # Linear interpolation
    for i in range(len(speeds)-1):
        if speeds[i] <= wind_speed <= speeds[i+1]:
            w1, w2 = speeds[i], speeds[i+1]
            p1, p2 = penalties[w1], penalties[w2]
            return p1 + (p2-p1)*(wind_speed-w1)/(w2-w1)

@dataclass
class Position:
    x: float  # meters east of start
    y: float  # meters north of start
    
    def copy(self):
        return Position(self.x, self.y)

@dataclass
class BoatState:
    position: Position
    heading: float     # degrees, 0 = north
    tack: str         # 'port' or 'starboard'
    leg: Literal["upwind", "downwind"] = "upwind"  # track if we're going upwind or downwind
    last_twa: float = 0.0  # track last true wind angle for detecting tacks/gybes
    
    def copy(self):
        return BoatState(
            self.position.copy(),
            self.heading,
            self.tack,
            self.leg,
            self.last_twa
        )


    
class Boat:
    def __init__(self, 
                 initial_position: Position,
                 initial_heading: float,
                 initial_tack: str,
                 polars: PolarData):
        
        self.state = BoatState(initial_position, initial_heading, initial_tack)
        self.polars = polars
        
    def get_velocity(self, wind: WindState, twa: float) -> Tuple[float, float]:
        """
        Get boat velocity components given wind and true wind angle
        Returns (vx, vy) in m/s
        """
        # Get boat speed in knots
        bsp = self.polars.get_boat_speed(twa, wind.speed)
        
        # Convert to m/s
        bsp_ms = bsp * 0.51444
        
        # Calculate heading (wind direction + true wind angle)
        heading = (wind.direction + twa) % 360
        
        # Convert to velocity components
        vx = bsp_ms * np.sin(np.radians(heading))
        vy = bsp_ms * np.cos(np.radians(heading))
        
        return vx, vy

    def check_for_maneuver(self, new_twa: float) -> Tuple[bool, str]:
        """
        Check if we're tacking or gybing
        Returns (is_maneuvering, maneuver_type)
        """
        old_twa = self.state.last_twa
        
        # Check for tack (crossing through 0 degrees)
        if (old_twa * new_twa < 0 and abs(old_twa) < 90 and abs(new_twa) < 90):
            return True, "tack"
        
        # Check for gybe (crossing through 180 degrees)
        if (abs(old_twa) > 90 and abs(new_twa) > 90 and 
            ((old_twa > 0 and new_twa < 0) or (old_twa < 0 and new_twa > 0))):
            return True, "gybe"
        
        return False, ""
    
    def step(self, dt: float, wind: WindState, twa: float):
        """
        Update boat state for time step dt (in seconds)
        Returns: new state and any time penalty incurred
        """
        # Check for tack/gybe
        is_maneuvering, maneuver = self.check_for_maneuver(twa)
        penalty = 0.0
        
        if is_maneuvering:
            penalty = get_tacking_penalty(wind.speed)
        
        # Update state
        vx, vy = self.get_velocity(wind, twa)
        
        # Update position (including any penalty time where we lose ground)
        effective_dt = dt + penalty
        self.state.position.x += vx * dt
        self.state.position.y += vy * dt
        
        # Update heading and last_twa
        self.state.heading = (wind.direction + twa) % 360
        self.state.last_twa = twa
        
        return self.state.copy(), penalty

    def get_optimal_angles(self, wind_speed: float) -> Tuple[float, float]:
        """
        Get the optimal angles for both upwind and downwind sailing
        Returns (upwind_twa, downwind_twa)
        """
        # Test angles for upwind (20-70 degrees)
        upwind_angles = np.arange(20, 71, 0.5)
        best_upwind_vmg = -float('inf')
        best_upwind_angle = 45  # fallback
        
        # Test angles for downwind (120-180 degrees)
        downwind_angles = np.arange(120, 181, 0.5)
        best_downwind_vmg = -float('inf')
        best_downwind_angle = 150  # fallback
        
        for twa in upwind_angles:
            bsp = self.polars.get_boat_speed(twa, wind_speed)
            vmg = bsp * np.cos(np.radians(twa))  # VMG upwind
            if vmg > best_upwind_vmg:
                best_upwind_vmg = vmg
                best_upwind_angle = twa
        
        for twa in downwind_angles:
            bsp = self.polars.get_boat_speed(twa, wind_speed)
            vmg = -bsp * np.cos(np.radians(twa))  # VMG downwind
            if vmg > best_downwind_vmg:
                best_downwind_vmg = vmg
                best_downwind_angle = twa
                
        return best_upwind_angle, best_downwind_angle

    def get_optimal_twa(self, wind_speed: float, leg: str = "upwind") -> float:
        """Get optimal true wind angle for the current leg"""
        upwind_angle, downwind_angle = self.get_optimal_angles(wind_speed)
        return upwind_angle if leg == "upwind" else downwind_angle