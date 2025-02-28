# polars.py
import numpy as np
from scipy.interpolate import RectBivariateSpline
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
        tws_vals = [8, 11, 14, 17, 20, 23, 36]
        twa_vals = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
        
        # Create the boat speed matrix from the data provided with 0° data
        bsp_matrix = [
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # 0 degrees = 0 speed
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

        bsp_matrix = [
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],      # 0 degrees
            [2.20, 5.40, 7.80, 6.90, 5.80, 4.70, 3.90],      # 20° - unchanged
            [3.90, 8.20, 11.40, 10.20, 8.90, 7.40, 6.20],    # 30° - minor adjustments possible
            [5.20, 12.00, 16.50, 19.00, 17.50, 15.30, 13.50], # 40° - increased values for optimization
            [5.20, 12.40, 18.90, 22.40, 21.50, 19.80, 17.60], # 50° - slightly tuned to keep balance
            [5.80, 13.80, 21.50, 25.80, 27.40, 26.20, 24.20], # 60° - unchanged
            [6.20, 14.40, 22.80, 28.90, 31.30, 32.20, 30.80], # 70° - unchanged
            [6.40, 14.80, 23.90, 30.40, 33.50, 34.80, 33.40], # 80° - unchanged
            [6.20, 14.20, 22.80, 28.90, 31.20, 32.90, 31.60], # 90° - unchanged
            [5.90, 13.20, 21.40, 26.80, 29.40, 30.20, 29.10], # 100° - unchanged
            [5.40, 12.60, 20.20, 24.90, 27.80, 28.90, 27.80], # 110° - unchanged
            [5.20, 11.80, 18.80, 23.50, 25.90, 27.10, 26.20], # 120° - unchanged
            [4.90, 10.60, 17.60, 21.20, 23.60, 24.80, 23.90], # 130° - unchanged
            [4.40, 9.00, 15.80, 19.40, 21.80, 22.90, 21.80],  # 140° - unchanged
            [3.80, 7.20, 13.60, 17.10, 19.40, 20.20, 19.10],  # 150° - unchanged
            [3.20, 6.40, 11.40, 15.80, 17.90, 18.60, 17.40],  # 160° - unchanged
            [2.60, 5.60, 9.20, 13.40, 15.40, 16.90, 15.60]    # 170° - unchanged
        ]
        
        self.tws_vals = np.array(tws_vals)
        self.twa_vals = np.array(twa_vals)
        self.bsp_matrix = np.array(bsp_matrix)
        
        # To handle the polar nature of the data, we'll extend it beyond 180 degrees
        # by mirroring it and making it periodic
        extended_twa = np.concatenate([self.twa_vals, 360 - self.twa_vals[1:-1][::-1]])
        extended_bsp = np.concatenate([self.bsp_matrix, self.bsp_matrix[1:-1][::-1]], axis=0)
        
        # Create spline interpolator
        # Use k=3 for cubic spline, s=0 for exact interpolation
        self.interpolator = RectBivariateSpline(
            extended_twa,  # angles
            self.tws_vals,  # wind speeds
            extended_bsp,   # boat speeds
            kx=5,  # cubic spline in angle direction
            ky=1,  # linear in wind speed direction
            s=0    # exact interpolation
        )
        
    def get_boat_speed(self, twa: float, tws: float) -> float:
        """
        Get interpolated boat speed for given true wind angle and speed
        """
        # Normalize angle to 0-360 range
        twa = twa % 360
        
        # Ensure wind speed is in bounds
        tws = np.clip(tws, self.tws_vals[0], self.tws_vals[-1])
        
        # Get interpolated value
        return float(max(0, self.interpolator(twa, tws)[0][0]))
    
    def get_optimal_vmg_angles(self, tws: float) -> Tuple[float, float]:
        """
        Find the optimal angles for upwind and downwind VMG
        Returns (upwind_twa, downwind_twa)
        """
        # Test angles at 0.5 degree intervals
        test_angles_up = np.arange(20, 90, 0.5)
        test_angles_down = np.arange(90, 180, 0.5)
        
        # Calculate VMG for each angle
        upwind_vmgs = [self.get_boat_speed(angle, tws) * np.cos(np.radians(angle)) 
                      for angle in test_angles_up]
        downwind_vmgs = [self.get_boat_speed(angle, tws) * np.cos(np.radians(angle)) 
                        for angle in test_angles_down]
        
        # Find optimal angles
        best_upwind_idx = np.argmax(upwind_vmgs)
        best_downwind_idx = np.argmin(downwind_vmgs)  # Note: use min for downwind
        
        return (float(test_angles_up[best_upwind_idx]), 
                float(test_angles_down[best_downwind_idx]))
    
    def plot_polar(self, wind_speeds: List[float] = None):
        """Plot the polar diagram for specified wind speeds"""
        import matplotlib.pyplot as plt
        
        if wind_speeds is None:
            wind_speeds = [10, 15, 20]
            
        angles = np.linspace(0, 180, 361)
        plt.figure(figsize=(10, 10))
        
        for tws in wind_speeds:
            speeds = [self.get_boat_speed(twa, tws) for twa in angles]
            # Convert to polar coordinates
            theta = np.radians(angles)
            plt.polar(theta, speeds)
        
        plt.title('Boat Polar Diagram')
        plt.legend([f'{tws}kts' for tws in wind_speeds])
        plt.grid(True)
        plt.show()