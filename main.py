# main.py
from typing import List, Tuple
from boat import Boat, Position, BoatState
from course import Course, create_standard_course
from wind import WindModel, WindState
from polars import PolarData
from visualization import plot_course
from copy import deepcopy
import numpy as np
import math
from static_wind import StaticWindField, WindState


def calculate_bearing_to_target(current_pos: Position, target: Position) -> float:
    """Calculate bearing from current position to target"""
    dx = target.x - current_pos.x
    dy = target.y - current_pos.y
    return math.degrees(math.atan2(dx, dy)) % 360

def distance_to_point(pos1: Position, pos2: Position) -> float:
    """Calculate distance between two positions"""
    return math.sqrt((pos2.x - pos1.x)**2 + (pos2.y - pos1.y)**2)

def get_optimal_starting_position(course: Course, wind_direction: float) -> Position:
    """Determine optimal starting position based on wind direction"""
    # If wind is coming from left side, start on right side and vice versa
    if wind_direction < 0:  # wind from left
        x = course.start_line[1].x  # right side
    else:
        x = course.start_line[0].x  # left side
    return Position(x, 0)

def predict_position(current_pos: Position, vx: float, vy: float, 
                    look_ahead_time: float) -> Position:
    """Predict future position based on current velocity"""
    future_x = current_pos.x + vx * look_ahead_time
    future_y = current_pos.y + vy * look_ahead_time
    return Position(future_x, future_y)

def should_tack(current_pos: Position, target: Position, wind: WindState, 
                course_width: float, current_twa: float, 
                boat: Boat, is_upwind: bool) -> bool:
    """
    Decide whether to tack based on position, boundaries, and predicted position
    Uses look-ahead to avoid unnecessary tacks
    """
    # Must tack if too close to absolute boundaries
    if current_pos.x < course_width * 0.05:
        return True
    if current_pos.x > course_width * 0.95:
        return True
    
    # Get current velocity
    vx, vy = boat.get_velocity(wind, current_twa)
    
    # Look ahead 30 seconds
    future_pos = predict_position(current_pos, vx, vy, 30.0)
    
    # Must tack if we'll hit boundaries
    if future_pos.x < 0 or future_pos.x > course_width:
        return True
    
    # Different strategy for upwind vs downwind
    if is_upwind:
        # Calculate VMG to target
        dx = target.x - current_pos.x
        dy = target.y - current_pos.y
        current_vmg = (dx * vx + dy * vy) / math.sqrt(dx*dx + dy*dy)
        
        # Look ahead to see if tacking would improve VMG
        opposite_twa = -current_twa
        vx_opp, vy_opp = boat.get_velocity(wind, opposite_twa)
        opposite_vmg = (dx * vx_opp + dy * vy_opp) / math.sqrt(dx*dx + dy*dy)
        
        # Only tack if significant improvement
        return opposite_vmg > current_vmg * 1.2
    
    else:  # Downwind
        # Much more conservative tacking downwind
        # Only tack if we're significantly off course
        bearing_to_target = calculate_bearing_to_target(current_pos, target)
        current_heading = (wind.direction + current_twa) % 360
        angle_diff = abs((bearing_to_target - current_heading + 180) % 360 - 180)
        
        # Only tack if we're more than 60 degrees off course
        return angle_diff > 60


def get_optimal_angles(boat: Boat, wind_speed: float) -> Tuple[float, float]:
    """Get optimal upwind and downwind angles for current wind speed"""
    return boat.get_optimal_angles(wind_speed)

def simulate_race(time_step: float = 1.0, max_time: float = 3600) -> Tuple[List[BoatState], List[WindState], List[float], List[float]]:
    """Simulate simplified race: just up to gate and back down"""
    course = create_standard_course()
    polars = PolarData()
    
    # Initialize wind
    wind = WindModel(
        initial_direction=0 + np.random.normal(0, 5),
        initial_speed=15 + np.random.normal(0, 2),
        direction_volatility="medium",
        speed_volatility="low",
        direction_trend=-10
    )
    
    
    # Start position and initial states
    start_pos = get_optimal_starting_position(course, wind.state.direction)
    boat = Boat(start_pos, 45, 'starboard', polars)
    
    # Racing states
    GOING_TO_GATE = 0
    RETURNING_TO_FINISH = 1
    current_state = GOING_TO_GATE
    
    # Store history
    boat_states = [boat.state.copy()]
    wind_states = [deepcopy(wind.state)]
    boat_speeds = [0.0]
    wind_angles = [0.0]
    total_penalties = 0.0
    last_tack_time = 0
    previous_position = boat.state.position
    
    def get_gate_target() -> Position:
        """Get target point for passing through gate"""
        return Position((course.top_marks[0].x + course.top_marks[1].x)/2, 
                       course.top_marks[0].y)
    
    # Track if we've been close to the gate
    approached_gate = False
    gate_y = course.top_marks[0].y
    
    time = 0
    while time < max_time:
        # Get current wind and compute optimal angles for these conditions
        current_wind = wind.step(time_step)
        upwind_twa, downwind_twa = get_optimal_angles(boat, current_wind.speed)
        
        if len(boat_states) % 100 == 0:  # Log wind changes periodically
            print(f"Wind speed: {current_wind.speed:.1f} kts - Optimal angles up/down: {upwind_twa:.1f}°/{downwind_twa:.1f}°")
        
        current_pos = boat.state.position
        
        # Check course boundaries
        if not course.is_valid_position(current_pos):
            print(f"Warning: Left course at ({current_pos.x:.1f}, {current_pos.y:.1f})")
            break
        
        # Enhanced gate passing detection
        gate_target = get_gate_target()
        dist_to_gate = distance_to_point(current_pos, gate_target)
        
        # Check if we're approaching the gate
        if current_state == GOING_TO_GATE and not approached_gate:
            if abs(current_pos.y - gate_y) < 100:
                if course.top_marks[0].x < current_pos.x < course.top_marks[1].x:
                    approached_gate = True
                    print(f"Approaching gate at time {time:.1f}")
        
        # Set target and TWA based on current state
        if current_state == GOING_TO_GATE:
            target = gate_target
            optimal_twa = upwind_twa
            current_twa = optimal_twa if boat.state.tack == 'starboard' else -optimal_twa
            is_upwind = True
            
            # Check if we've passed through the gate
            if (approached_gate and
                current_pos.y > gate_y and
                course.top_marks[0].x < current_pos.x < course.top_marks[1].x):
                print(f"Passed through gate at time {time:.1f}")
                current_state = RETURNING_TO_FINISH
                # Force a gybe to turn around
                boat.state.tack = 'port' if boat.state.tack == 'starboard' else 'starboard'
                last_tack_time = time
                approached_gate = False
                
        else:  # Returning to finish
            target = Position(course.width/2, 0)  # Middle of finish line
            optimal_twa = downwind_twa
            current_twa = optimal_twa if boat.state.tack == 'starboard' else -optimal_twa
            is_upwind = False
            
            # Check finish
            if course.is_finished(boat.state):
                print(f"Finished race at time {time:.1f}")
                break
        
        # Consider tacking if enough time has passed
        min_tack_interval = 60 if current_state == RETURNING_TO_FINISH else 30
        if time - last_tack_time > min_tack_interval:
            if should_tack(current_pos, target, current_wind, course.width, 
                          current_twa, boat, is_upwind):
                boat.state.tack = 'port' if boat.state.tack == 'starboard' else 'starboard'
                current_twa = -current_twa
                last_tack_time = time
                print(f"Tacking at time {time:.1f}, position: ({current_pos.x:.1f}, {current_pos.y:.1f})")
                print(f"Wind: {current_wind.direction:.1f}°, {current_wind.speed:.1f} kts")
        
        # Update boat
        new_state, penalty = boat.step(time_step, current_wind, current_twa)
        total_penalties += penalty
        
        # Calculate boat speed
        vx, vy = boat.get_velocity(current_wind, current_twa)
        boat_speed = math.sqrt(vx**2 + vy**2) / 0.51444  # Convert m/s to knots
        
        # Store state
        boat_states.append(new_state)
        wind_states.append(deepcopy(current_wind))
        boat_speeds.append(boat_speed)
        wind_angles.append(current_twa)
        
        # Update for next iteration
        previous_position = current_pos
        time += time_step
        
        # Debug print every 100 steps
        if len(boat_states) % 100 == 0:
            state_name = "GOING_TO_GATE" if current_state == GOING_TO_GATE else "RETURNING_TO_FINISH"
            print(f"Time: {time:.1f}, Position: ({current_pos.x:.1f}, {current_pos.y:.1f})")
            print(f"Wind: {current_wind.direction:.1f}°, {current_wind.speed:.1f} kts")
            print(f"Boat speed: {boat_speed:.1f} kts, TWA: {current_twa:.1f}°")
            print(f"Race state: {state_name}")
            print(f"Distance to gate: {dist_to_gate:.1f}m")
            print(f"Total tacking penalties: {total_penalties:.1f}s")
            print("---")
    
    print(f"\nRace summary:")
    print(f"Total time: {time:.1f}s")
    print(f"Total tacking penalties: {total_penalties:.1f}s")
    print(f"Net sailing time: {time - total_penalties:.1f}s")
    
    return boat_states, wind_states, boat_speeds, wind_angles

if __name__ == "__main__":
    # Run simulation
    boat_states, wind_states, boat_speeds, wind_angles = simulate_race()
    
    # Visualize results
    course = create_standard_course()
    plot_course(course, boat_states, wind_states, boat_speeds, wind_angles)