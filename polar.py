import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd

@dataclass
class PolarPoint:
    twa: float  # True wind angle
    tws: float  # True wind speed
    bsp: float  # Boat speed

class PolarData:
    def __init__(self):
        # Initialize the IMOCA 60 polar data
        tws_vals = [5, 10, 15, 20, 25, 30, 35]
        twa_vals = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
        
        # Create the boat speed matrix from the data provided
        bsp_matrix = [
            [4.93, 7.88, 7.44, 7.27, 6.80, 6.73, 6.85],
            [6.68, 9.78, 9.63, 9.78, 9.52, 9.51, 9.72],
            [7.76, 10.74, 10.67, 10.95, 10.80, 10.92, 11.22],
            [8.48, 11.40, 11.52, 11.95, 11.93, 12.17, 12.56],
            [9.01, 11.99, 12.42, 13.16, 13.42, 13.85, 14.36],
            [9.27, 12.58, 13.55, 14.72, 15.34, 16.03, 16.70],
            [9.33, 13.12, 14.91, 16.67, 17.77, 18.72, 19.57],
            [9.06, 13.54, 16.19, 18.32, 19.62, 20.65, 21.57],
            [9.11, 13.66, 17.45, 20.04, 21.56, 22.67, 23.68],
            [9.03, 13.95, 18.76, 21.88, 23.72, 24.88, 25.98],
            [8.62, 14.04, 18.64, 22.51, 25.17, 26.23, 27.35],
            [7.84, 13.91, 19.28, 22.95, 25.24, 26.39, 27.54],
            [6.94, 13.38, 19.41, 23.27, 25.65, 26.82, 27.98],
            [5.93, 12.31, 17.56, 21.20, 23.52, 24.54, 25.60],
            [5.28, 11.40, 16.01, 19.23, 21.30, 22.24, 23.20],
            [4.43, 9.80, 14.38, 17.25, 19.01, 19.88, 20.74]
        ]
        
        self.tws_vals = np.array(tws_vals)
        self.twa_vals = np.array(twa_vals)
        self.bsp_matrix = np.array(bsp_matrix)
        
    def get_boat_speed(self, twa: float, tws: float) -> float:
        """
        Get interpolated boat speed for given true wind angle and speed
        """
        # Handle negative angles by mirroring
        twa = abs(twa)
        
        # Normalize angle to 0-180 range
        while twa > 180:
            twa = 360 - twa
            
        # Interpolate using numpy
        return float(
            np.interp2d(
                self.tws_vals, 
                self.twa_vals,
                self.bsp_matrix
            )(tws, twa)
        )