# course.py
from dataclasses import dataclass
from typing import List, Tuple
from boat import Position, BoatState
from wind import WindState

@dataclass
class Course:
    start_line: Tuple[Position, Position]  # Start line endpoints
    top_marks: Tuple[Position, Position]   # Two marks at the top
    length: float                          # Course length in meters
    width: float                           # Course width in meters
    extension: float                       # Extended area beyond marks

    def is_finished(self, boat_state: BoatState) -> bool:
        """Check if boat has crossed the finish line"""
        return (boat_state.position.y <= 0 and 
                self.start_line[0].x <= boat_state.position.x <= self.start_line[1].x)

    def is_valid_position(self, position: Position) -> bool:
        """Check if position is within course boundaries"""
        return (0 <= position.x <= self.width and
                0 <= position.y <= self.length + self.extension)

    def has_passed_gate(self, previous_pos: Position, current_pos: Position) -> bool:
        """Check if boat has passed between the marks"""
        # Gate y-position
        gate_y = self.top_marks[0].y
        # Gate x bounds
        left_x = min(self.top_marks[0].x, self.top_marks[1].x)
        right_x = max(self.top_marks[0].x, self.top_marks[1].x)
        
        # Check if we've crossed the gate line
        if (previous_pos.y < gate_y and current_pos.y >= gate_y):
            # Linear interpolation to find x-position at crossing
            if current_pos.y != previous_pos.y:  # Avoid division by zero
                t = (gate_y - previous_pos.y) / (current_pos.y - previous_pos.y)
                cross_x = previous_pos.x + t * (current_pos.x - previous_pos.x)
                return left_x <= cross_x <= right_x
        return False
    
def create_standard_course() -> Course:
    """Create a standard upwind course with two marks"""
    width = 2000   # meters
    length = 3000  # meters to marks
    extension = 500  # meters beyond marks
    gate_width = 1800  # meters between marks
    
    # Calculate mark positions
    center_x = width/2
    mark_y = length
    left_mark = Position(center_x - gate_width/2, mark_y)
    right_mark = Position(center_x + gate_width/2, mark_y)
    
    return Course(
        start_line=(Position(width/4, 0), Position(3*width/4, 0)),
        top_marks=(left_mark, right_mark),
        length=length,
        width=width,
        extension=extension
    )