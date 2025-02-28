# wind.py
from dataclasses import dataclass
import numpy as np
from typing import Tuple, Literal
import random

@dataclass 
class WindState:
    direction: float  # degrees, 0 = from north
    speed: float     # knots
    
class WindModel:
    def __init__(self, 
                 initial_direction: float,
                 initial_speed: float,
                 direction_volatility: Literal["low", "medium", "high"] = "medium",
                 speed_volatility: Literal["low", "high"] = "low",
                 direction_trend: float = 0):        # degrees per hour
        
        self.state = WindState(initial_direction, initial_speed)
        
        # Direction volatility from paper: 1.2, 2.0, or 4.0 degrees per 40s
        direction_volatilities = {
            "low": 1.2,
            "medium": 2.0,
            "high": 4.0
        }
        self.direction_volatility = direction_volatilities[direction_volatility]
        
        # Speed volatility from paper: 0.7 or 1.3 knots per 40s
        speed_volatilities = {
            "low": 0.7,
            "high": 1.3
        }
        self.speed_volatility = speed_volatilities[speed_volatility]
        
        self.direction_trend = direction_trend
        
    def step(self, dt: float) -> WindState:
        """
        Update wind state for time step dt (in seconds)
        """
        # Scale volatilities for time step (paper uses 40s reference)
        dir_vol_scaled = self.direction_volatility * np.sqrt(dt/40) # * 3*np.sqrt(dt/40)
        spd_vol_scaled = self.speed_volatility * np.sqrt(dt/40)
        
        # Apply trend and random changes
        self.state.direction += (
            self.direction_trend * dt/3600 +  # Trend per hour
            np.random.normal(0, dir_vol_scaled)
        )
        
        self.state.speed = max(3, min(35,
            self.state.speed + np.random.normal(0, spd_vol_scaled)
        ))
        
        return self.state